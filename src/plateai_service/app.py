"""Bounded HTTP adapter: no files written, URL fetches, or per-request model loads."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from starlette.exceptions import HTTPException

from .errors import ServiceError
from .images import MAX_IMAGE_BYTES, decode_base64
from .pipeline import PlatePipeline
from .runtime import InferenceRuntime

JSON_BODY_LIMIT = 17 * 1024 * 1024
MULTIPART_BODY_LIMIT = 13 * 1024 * 1024
LOG = logging.getLogger('plateai_service')


def request_id(request):
    return request.scope.setdefault('plate_request_id', uuid4().hex)


def error_response(request, error):
    return JSONResponse({'request_id':request_id(request), 'error':{'code':error.code,'message':error.code}},
                        status_code=error.status, headers={'Retry-After':'1'} if error.code=='busy' else None)


async def read_body(request, maximum):
    declared = request.headers.get('content-length')
    if declared is not None:
        try:
            length = int(declared)
        except ValueError as exc:
            raise ServiceError('invalid_request') from exc
        if length < 0:
            raise ServiceError('invalid_request')
        if length > maximum:
            raise ServiceError('image_too_large',413)
    body = bytearray()
    async for chunk in request.stream():
        if len(body)+len(chunk)>maximum:
            raise ServiceError('image_too_large',413)
        body.extend(chunk)
    return bytes(body)


def multipart_image(body: bytes, content_type: str):
    _, options = parse_options_header(content_type)
    boundary = options.get(b'boundary')
    if not boundary or len(boundary)>200:
        raise ServiceError('invalid_multipart')
    fields = {}
    state = {'ended':False, 'parts':0}
    def begin():
        state['parts'] += 1
        if state['parts']>2:
            raise ServiceError('ambiguous_image_input',422)
        state.update(headers={}, header_name=bytearray(), header_value=bytearray(), data=bytearray())
    def header_field(data,start,end):state['header_name'].extend(data[start:end])
    def header_value(data,start,end):state['header_value'].extend(data[start:end])
    def header_end():
        name=bytes(state['header_name']).lower()
        if name in state['headers']:
            raise ServiceError('invalid_multipart')
        state['headers'][name]=bytes(state['header_value'])
        state['header_name'].clear();state['header_value'].clear()
    def headers_finished():
        disposition, opts = parse_options_header(state['headers'].get(b'content-disposition',b''))
        if disposition != b'form-data':
            raise ServiceError('invalid_multipart')
        name=opts.get(b'name')
        if name not in (b'file',b'image_base64'):
            raise ServiceError('invalid_request',422)
        if name in fields:
            raise ServiceError('ambiguous_image_input',422)
        state['name']=name
        if name==b'file' and b'filename' not in opts:
            raise ServiceError('missing_image',422)
    def part_data(data,start,end):
        if len(state['data'])+end-start>MAX_IMAGE_BYTES:
            raise ServiceError('image_too_large',413)
        state['data'].extend(data[start:end])
    def part_end():fields[state['name']]=bytes(state['data'])
    def ended():state['ended']=True
    parser=MultipartParser(boundary,{
        'on_part_begin':begin,'on_header_field':header_field,'on_header_value':header_value,
        'on_header_end':header_end,'on_headers_finished':headers_finished,'on_part_data':part_data,
        'on_part_end':part_end,'on_end':ended,
    },max_size=MULTIPART_BODY_LIMIT)
    try:
        parser.write(body)
        parser.finalize()
    except MultipartParseError as exc:
        raise ServiceError('invalid_multipart') from exc
    if not state['ended']:
        raise ServiceError('invalid_multipart')
    if b'file' in fields and b'image_base64' in fields:
        raise ServiceError('ambiguous_image_input',422)
    if b'file' not in fields:
        raise ServiceError('missing_image',422)
    return fields[b'file'], None


def json_image(body):
    def unique_pairs(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ServiceError('ambiguous_image_input',422)
            result[key]=value
        return result
    try:
        obj=json.loads(body,object_pairs_hook=unique_pairs)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ServiceError('invalid_json') from exc
    if not isinstance(obj,dict):raise ServiceError('invalid_json')
    if 'file' in obj and 'image_base64' in obj:raise ServiceError('ambiguous_image_input',422)
    if 'image_base64' not in obj:raise ServiceError('missing_image',422)
    if set(obj)!={'image_base64'}:raise ServiceError('invalid_request',422)
    return decode_base64(obj['image_base64'])


def create_app(config, *, pipeline_factory=None, token=None, inference_timeout=10.) -> FastAPI:
    runtime=InferenceRuntime(config,pipeline_factory or PlatePipeline)
    @asynccontextmanager
    async def lifespan(app):
        runtime.start()
        yield
        runtime.close()
    app=FastAPI(title='3waPlateAI local plate API',version='1.0.0',lifespan=lifespan,
                docs_url=None,redoc_url=None,openapi_url=None)
    app.state.runtime=runtime

    def authorize(request):
        if token and not hmac.compare_digest(request.headers.get('authorization','').encode(),('Bearer '+token).encode()):
            raise ServiceError('unauthorized',401)

    @app.exception_handler(ServiceError)
    async def service_error(request,exc):return error_response(request,exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request,exc):return error_response(request,ServiceError('invalid_request',422))

    @app.exception_handler(HTTPException)
    async def http_error(request,exc):return error_response(request,ServiceError('http_error',exc.status_code))

    @app.get('/api/health')
    async def health():return {'status':'alive'}

    @app.get('/api/ready')
    async def ready(request:Request):
        authorize(request)
        if not runtime.ready:
            result=error_response(request,ServiceError('model_not_ready',503))
            return result
        return {'request_id':request_id(request),**runtime.metadata()}

    @app.get('/docs',include_in_schema=False)
    async def docs(request:Request):
        authorize(request)
        return get_swagger_ui_html(openapi_url='/openapi.json',title='3waPlateAI API')

    @app.get('/openapi.json',include_in_schema=False)
    async def openapi(request:Request):
        authorize(request)
        document=app.openapi()
        document['paths']['/api/predict']['post']['requestBody']={
            'required':True,'content':{
                'multipart/form-data':{'schema':{'type':'object','required':['file'],'properties':{'file':{'type':'string','format':'binary'}}}},
                'application/json':{'schema':{'type':'object','required':['image_base64'],'properties':{'image_base64':{'type':'string'}}}},
            }}
        return document

    @app.post('/api/predict')
    async def predict(request:Request):
        authorize(request)
        runtime.reserve()
        submitted=False
        started=time.perf_counter()
        try:
            content_type=request.headers.get('content-type','')
            media=content_type.split(';',1)[0].strip().lower()
            if media not in ('application/json','multipart/form-data'):
                raise ServiceError('unsupported_media_type',415)
            maximum=JSON_BODY_LIMIT if media=='application/json' else MULTIPART_BODY_LIMIT
            try:
                body=await asyncio.wait_for(read_body(request,maximum),timeout=10.)
            except asyncio.TimeoutError as exc:
                raise ServiceError('upload_timeout',408) from exc
            data,mime=json_image(body) if media=='application/json' else multipart_image(body,content_type)
            future=runtime.submit_reserved(data,mime)
            submitted=True
            # Consume late exceptions without cancelling native inference or freeing capacity.
            future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            try:
                result=await asyncio.wait_for(asyncio.shield(future),timeout=inference_timeout)
            except asyncio.TimeoutError as exc:
                raise ServiceError('inference_timeout',504) from exc
            except ServiceError:
                raise
            except Exception as exc:
                raise ServiceError('inference_failed',503) from exc
            result['request_id']=request_id(request)
            result['timings_ms']['total']=(time.perf_counter()-started)*1000
            LOG.info('request_id=%s status=%s count=%d total_ms=%.1f model_id=%s',
                     result['request_id'],result['status'],len(result['detections']),result['timings_ms']['total'],config.model_id)
            return result
        finally:
            if not submitted:
                runtime.busy=False
    return app
