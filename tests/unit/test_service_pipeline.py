import hashlib
import json

import numpy as np
import pytest


def make_config(tmp_path, **overrides):
    from plateai_service.config import ServiceConfig
    checkpoint=tmp_path/'detector.pt';checkpoint.write_bytes(b'trusted test asset')
    manifest={'model_id':'test-v3','detector':{'path':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
    for key in ('ocr_detection','ocr_recognition'):
        folder=tmp_path/key;folder.mkdir(exist_ok=True)
        hashes={}
        for name in ('inference.json','inference.pdiparams','inference.yml'):
            content=b'fixed model';(folder/name).write_bytes(content)
            hashes[name]=hashlib.sha256(content).hexdigest()
        manifest[key]={'path':str(folder),'files':hashes}
    manifest.update(overrides)
    path=tmp_path/'config.json';path.write_text(json.dumps(manifest),encoding='utf-8')
    return ServiceConfig.from_file(path)


def test_missing_or_corrupt_model_fails_before_any_loader(tmp_path,monkeypatch):
    from plateai_service.config import verify_assets
    config=make_config(tmp_path)
    verify_assets(config)
    config.detector_path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='asset_hash_mismatch'):
        verify_assets(config)


def test_manifest_requires_all_paddle_runtime_files(tmp_path):
    from plateai_service.config import ServiceConfig
    make_config(tmp_path)
    path=tmp_path/'config.json';obj=json.loads(path.read_text())
    del obj['ocr_recognition']['files']['inference.pdiparams']
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='model_manifest'):
        ServiceConfig.from_file(path)


def test_auto_gpu_warmup_failure_falls_back_once():
    from plateai_service.pipeline import initialize_component
    attempts=[]
    def loader(device):
        attempts.append(device)
        if device=='gpu':raise RuntimeError('private path')
        return 'cpu warmed instance'
    model,meta=initialize_component(loader,'auto',True)
    assert model=='cpu warmed instance'
    assert attempts==['gpu','cpu']
    assert meta['effective_device']=='cpu'
    assert meta['fallback_reason']=='gpu_initialization_failed:RuntimeError'
    assert 'private path' not in str(meta)


def test_force_cpu_does_not_probe_gpu_and_double_failure_propagates():
    from plateai_service.pipeline import initialize_component
    attempts=[]
    def okay(device):attempts.append(device);return object()
    initialize_component(okay,'cpu',True)
    assert attempts==['cpu']
    def fail(device):raise RuntimeError('failed')
    with pytest.raises(RuntimeError):initialize_component(fail,'auto',True)


def test_failed_gpu_probe_still_reaches_cpu_and_forced_cpu_skips_probe():
    from plateai_service.pipeline import initialize_component
    probes=[]
    def broken_probe():
        probes.append(1)
        raise RuntimeError('driver unavailable')
    devices=[]
    def loader(device):devices.append(device);return object()
    _, forced=initialize_component(loader,'cpu',broken_probe)
    assert devices==['cpu'] and probes==[]
    _, automatic=initialize_component(loader,'auto',broken_probe)
    assert devices==['cpu','cpu'] and probes==[1]
    assert automatic['effective_device']=='cpu'
    assert automatic['fallback_reason']=='gpu_probe_failed:RuntimeError'


class Boxes:
    def __init__(self, values):self.values=values
    def __call__(self,image):return self.values


def test_pipeline_preserves_unreadable_bbox_buffer_and_sorting(tmp_path):
    from plateai_service.pipeline import PlatePipeline
    pipeline=PlatePipeline(make_config(tmp_path))
    pipeline.detector=Boxes([[50,50,100,80],[0,1,40,21]])
    crops=[]
    def ocr(crop):
        crops.append(crop.shape)
        if len(crops)==1:return [],[]
        return ['EAB-5555'], [[[0,0],[40,0],[40,20],[0,20]]]
    pipeline.ocr=ocr
    result=pipeline.predict(np.zeros((80,100,3),np.uint8))
    assert result['status']=='partial'
    assert [d['id'] for d in result['detections']]==[1,2]
    assert result['detections'][0]['bbox']==[0,1,40,21]
    assert result['detections'][0]['crop_bbox']==[0,0,42,22]
    assert result['detections'][0]['status']=='unreadable'
    assert result['detections'][1]['crop_bbox']==[47,48,100,80]
    assert result['detections'][1]['text']=='EAB5555'
    assert crops==[(22,42,3),(32,53,3)]


def test_candidate_cap_and_native_failure_are_not_silent(tmp_path):
    from plateai_service.errors import ServiceError
    from plateai_service.pipeline import PlatePipeline
    pipeline=PlatePipeline(make_config(tmp_path))
    pipeline.detector=Boxes([[0,0,20,20]]*21)
    with pytest.raises(ServiceError,match='too_many_candidates'):
        pipeline.predict(np.zeros((30,30,3),np.uint8))
    pipeline.detector=Boxes([[0,0,20,20]])
    def failed(crop):raise RuntimeError('device failed')
    pipeline.ocr=failed
    with pytest.raises(RuntimeError):pipeline.predict(np.zeros((30,30,3),np.uint8))
    pipeline.detector=Boxes([])
    assert pipeline.predict(np.zeros((30,30,3),np.uint8))['status']=='no_plate'
