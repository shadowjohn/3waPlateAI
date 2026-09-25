import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import threading
import time

import httpx
import pytest
from PIL import Image

from test_service_pipeline import make_config


def png():
    stream=io.BytesIO();Image.new('RGB',(20,10),'white').save(stream,format='PNG')
    return stream.getvalue()


class FixturePipeline:
    def __init__(self, config):
        self.config=config;self.initialized=0;self.calls=0;self.delay=0
        self.fail=False;self.thread=None
    def initialize(self):
        self.initialized+=1;self.thread=threading.get_ident()
    def metadata(self):return {'model_id':self.config.model_id,'ocr':{'effective_device':'cpu'}}
    def predict(self,bgr):
        assert threading.get_ident()==self.thread
        self.calls+=1;time.sleep(self.delay)
        if self.fail:raise RuntimeError('SECRET C:/private/model.pt ABC1234')
        return {'status':'no_plate','image':{'width':bgr.shape[1],'height':bgr.shape[0]},
                'detections':[],'model_id':self.config.model_id,'schema_version':1,
                'timings_ms':{'detect':0.,'ocr':0.}}


async def wait_ready(client,headers=None):
    for _ in range(100):
        result=await client.get('/api/ready',headers=headers)
        if result.status_code==200:return result.json()
        await asyncio.sleep(.01)
    raise AssertionError('not ready')


def run_app(tmp_path, scenario, **options):
    from plateai_service.app import create_app
    config=make_config(tmp_path);pipeline=FixturePipeline(config)
    app=create_app(config,pipeline_factory=lambda _:pipeline,**options)
    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                await scenario(client,pipeline,app)
    asyncio.run(run())


def test_three_inputs_reuse_one_pipeline_and_agree(tmp_path):
    async def scenario(client,pipeline,app):
        await wait_ready(client)
        data=png();encoded=base64.b64encode(data).decode()
        responses=[await client.post('/api/predict',files={'file':('a.png',data,'image/png')}),
                   await client.post('/api/predict',json={'image_base64':encoded}),
                   await client.post('/api/predict',json={'image_base64':'data:image/png;base64,'+encoded})]
        for response in responses:
            assert response.status_code==200,response.text
            assert response.json()['image']=={'width':20,'height':10}
            assert response.json()['status']=='no_plate'
        assert len({r.json()['request_id'] for r in responses})==3
        assert pipeline.initialized==1 and pipeline.calls==3
    run_app(tmp_path,scenario)


@pytest.mark.parametrize('payload,status,code',[
    ({},422,'missing_image'),({'image_base64':'!!!!'},400,'invalid_base64'),
    ({'file':'path','image_base64':'a'},422,'ambiguous_image_input'),
    ({'image_base64':base64.b64encode(b'no image').decode()},400,'invalid_image'),
    ({'url':'https://invalid/'},422,'missing_image'),
])
def test_json_errors_have_safe_contract(tmp_path,payload,status,code):
    async def scenario(client,pipeline,app):
        await wait_ready(client)
        response=await client.post('/api/predict',json=payload)
        assert response.status_code==status
        assert response.json()['error']['code']==code
        assert response.json()['request_id']
        assert pipeline.calls==0
    run_app(tmp_path,scenario)


def test_auth_covers_predict_ready_and_docs_not_health(tmp_path):
    async def scenario(client,pipeline,app):
        assert (await client.get('/api/health')).status_code==200
        for path in ('/api/ready','/docs','/openapi.json'):
            assert (await client.get(path)).status_code==401
        assert (await client.post('/api/predict',json={})).status_code==401
        headers={'Authorization':'Bearer '+('x'*32)}
        await wait_ready(client,headers)
        assert (await client.get('/openapi.json',headers=headers)).status_code==200
    run_app(tmp_path,scenario,token='x'*32)


def test_stream_limit_and_duplicate_upload_before_inference(tmp_path,monkeypatch):
    import plateai_service.app as service
    monkeypatch.setattr(service,'JSON_BODY_LIMIT',100)
    async def scenario(client,pipeline,app):
        await wait_ready(client)
        async def chunks():
            yield b' '*70
            yield b' '*70
        response=await client.post('/api/predict',content=chunks(),headers={'Content-Type':'application/json'})
        assert response.status_code==413
        response=await client.post('/api/predict',files=[('file',('a.png',png())),('file',('b.png',png()))])
        assert response.status_code==422
        assert response.json()['error']['code']=='ambiguous_image_input'
        assert pipeline.calls==0
    run_app(tmp_path,scenario)


def test_timeout_does_not_release_native_slot_and_health_remains_live(tmp_path):
    async def scenario(client,pipeline,app):
        await wait_ready(client);pipeline.delay=.20
        first=await client.post('/api/predict',files={'file':('a.png',png())})
        assert first.status_code==504
        assert (await client.get('/api/health')).status_code==200
        second=await client.post('/api/predict',files={'file':('a.png',png())})
        assert second.status_code==429 and second.headers['Retry-After']=='1'
        assert pipeline.calls==1
        await asyncio.sleep(.25)
        pipeline.delay=0
        assert (await client.post('/api/predict',files={'file':('a.png',png())})).status_code==200
        assert pipeline.initialized==1
    run_app(tmp_path,scenario,inference_timeout=.03)


def test_client_cancellation_keeps_slot_until_worker_done(tmp_path):
    async def scenario(client,pipeline,app):
        await wait_ready(client);pipeline.delay=.2
        request=asyncio.create_task(client.post('/api/predict',files={'file':('a.png',png())}))
        for _ in range(100):
            if pipeline.calls:break
            await asyncio.sleep(.002)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):await request
        assert (await client.post('/api/predict',files={'file':('a.png',png())})).status_code==429
        await asyncio.sleep(.25)
        assert not app.state.runtime.busy
    run_app(tmp_path,scenario)


def test_fatal_failure_revokes_ready_and_redacts_exception(tmp_path):
    async def scenario(client,pipeline,app):
        await wait_ready(client);pipeline.fail=True
        response=await client.post('/api/predict',files={'file':('a.png',png())})
        assert response.status_code==503 and response.json()['error']['code']=='inference_failed'
        assert 'SECRET' not in response.text and 'private' not in response.text
        ready=await client.get('/api/ready')
        assert ready.status_code==503 and 'SECRET' not in ready.text
        assert (await client.get('/api/health')).status_code==200
    run_app(tmp_path,scenario)


def test_failed_initialization_does_not_block_health(tmp_path):
    from plateai_service.app import create_app
    def factory(config):raise ValueError('private model path')
    app=create_app(make_config(tmp_path),pipeline_factory=factory)
    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                assert (await client.get('/api/health')).status_code==200
                await asyncio.sleep(.02)
                response=await client.get('/api/ready')
                assert response.status_code==503 and 'private model path' not in response.text
    asyncio.run(run())


def test_external_binding_requires_long_token():
    from plateai_service.config import validate_binding
    validate_binding('127.0.0.1',None)
    validate_binding('::1',None)
    with pytest.raises(ValueError):validate_binding('0.0.0.0',None)
    with pytest.raises(ValueError):validate_binding('192.168.1.2','short')
    validate_binding('0.0.0.0','x'*32)
