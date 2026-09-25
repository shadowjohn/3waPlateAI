from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest
from starlette.testclient import TestClient


def pose_training():
    assert importlib.util.find_spec('plateai_web.pose_training') is not None, 'Pose training is missing'
    return importlib.import_module('plateai_web.pose_training')


def test_preflight_lists_missing_assets_without_creating_them(tmp_path, monkeypatch):
    monkeypatch.delenv('PLATEAI_POSE_PYTHON', raising=False)
    status = pose_training().preflight(tmp_path)
    assert status['ready'] is False
    assert {'python', 'pretrained', 'dataset', 'annotations'} <= {item['id'] for item in status['checks'] if not item['ok']}
    assert list(tmp_path.iterdir()) == []


def test_pose_request_refuses_implicit_rights_and_changed_recipe(tmp_path):
    module = pose_training()
    for request in ({'epochs': 10}, {'epochs': 1, 'acknowledge_unreviewed_license': True},
                    {'epochs': 10, 'device': '../cuda', 'acknowledge_unreviewed_license': True}):
        with pytest.raises(ValueError):
            module.validate_request(tmp_path, request)


def test_training_uses_fixed_last_epoch_not_holdout_selection(tmp_path):
    module = pose_training()
    events = []

    class Model:
        def __init__(self, path):
            self.callbacks = {}
        def add_callback(self, event, callback):
            self.callbacks[event] = callback
        def train(self, **options):
            from types import SimpleNamespace
            from plateai_web.training_worker import _progress_for
            trainer = SimpleNamespace(epoch=4, tloss=None, train_loader=[1, 2])
            self.callbacks['on_train_epoch_start'](trainer)
            self.callbacks['on_train_batch_end'](trainer)
            self.callbacks['on_train_batch_end'](trainer)
            self.callbacks['on_train_epoch_end'](trainer)
            assert events[-2].batch == 2
            assert events[-2].total_batches == 2
            assert _progress_for(events[-1], 10)[1] >= 46
            assert options['epochs'] == 10
            assert options['val'] is False
            assert options['amp'] is False
            assert options['mosaic'] == 0
            assert options['seed'] == 42
            target = Path(options['project']) / options['name'] / 'weights'
            target.mkdir(parents=True)
            (target / 'last.pt').write_bytes(b'fixed-last')
            (target / 'best.pt').write_bytes(b'not-selected')

    checkpoint = tmp_path / 'pretrained.pt'
    checkpoint.write_bytes(b'initial')
    selected = module.train_prepared(tmp_path / 'dataset', checkpoint, tmp_path / 'run',
                                     device='cpu', on_progress=events.append, check_cancelled=lambda: None, model_factory=Model)
    assert selected.read_bytes() == b'fixed-last'


def test_export_package_is_allowlisted_and_cannot_overwrite(tmp_path):
    module = pose_training()
    checkpoint = tmp_path / 'last.pt'
    checkpoint.write_bytes(b'trusted-local-checkpoint')
    output = tmp_path / 'artifact'
    summary = {'preset': 'clean119-70-30-v1', 'train_count': 2080, 'holdout_count': 36,
               'metrics': {'bbox_recall_iou50': .5}, 'annotation_sha256': 'a' * 64}
    result = module.publish_artifact(checkpoint, output, summary)
    with zipfile.ZipFile(result['package']) as archive:
        assert set(archive.namelist()) == {'pose-pilot.pt', 'detector-manifest.json', 'training-summary.json', 'README.txt'}
        assert archive.read('pose-pilot.pt') == checkpoint.read_bytes()
        manifest = json.loads(archive.read('detector-manifest.json'))
        assert manifest['detector']['path'] == 'pose-pilot.pt'
        assert manifest['detector']['sha256'] == result['checkpoint_sha256']
        assert 'active' not in manifest
    with pytest.raises(FileExistsError):
        module.publish_artifact(checkpoint, output, summary)


def test_missing_pose_assets_do_not_create_a_ctc_task(tmp_path, monkeypatch):
    from plateai_web import app as app_module
    from plateai_web.training_store import TrainingStore
    store = TrainingStore(tmp_path / 'runs/.web-training')
    monkeypatch.setattr(app_module, 'ROOT', tmp_path)
    monkeypatch.setattr(app_module, 'launch_training_worker', lambda *args: None)
    app_module.app.dependency_overrides[app_module.get_training_store] = lambda: store
    try:
        with TestClient(app_module.app) as client:
            response = client.post('/api/train/start', json={'kind': 'pose', 'epochs': 10, 'acknowledge_unreviewed_license': True})
            assert response.status_code == 422
            assert store.latest() is None
            response = client.get('/api/train/pose/preflight')
            assert response.status_code == 200
            assert response.json()['ready'] is False
            assert client.get('/api/train/pose/artifacts/0123456789ab/package').status_code == 404
    finally:
        app_module.app.dependency_overrides.clear()


def test_artifact_endpoint_download_and_tampering(tmp_path, monkeypatch):
    from plateai_web import app as app_module
    from plateai_web.training_store import TrainingStore
    store = TrainingStore(tmp_path / 'runs/.web-training')
    task_id = '0123456789ab'
    store.create(task_id, 'pose', {'kind': 'pose', 'epochs': 10})
    store.claim(task_id, 1, 'test')
    source = tmp_path / 'last.pt'
    source.write_bytes(b'native-test-checkpoint')
    artifact = pose_training().publish_artifact(source, tmp_path / f'runs/pose-{task_id}/artifact', {})
    store.finish(task_id, 'completed', message='done', result=artifact)
    monkeypatch.setattr(app_module, 'ROOT', tmp_path)
    app_module.app.dependency_overrides[app_module.get_training_store] = lambda: store
    try:
        with TestClient(app_module.app) as client:
            url = f'/api/train/pose/artifacts/{task_id}/package'
            response = client.get(url)
            assert response.status_code == 200
            assert response.content == Path(artifact['package']).read_bytes()
            Path(artifact['package']).write_bytes(b'tampered')
            assert client.get(url).status_code == 404
            assert client.get('/api/train/pose/artifacts/bad/package').status_code == 404
    finally:
        app_module.app.dependency_overrides.clear()


@pytest.mark.parametrize('shift,rate', [(0, 1.), (9, 0.)])
def test_evaluation_adapter_semantic_corners_in_640_space(tmp_path, monkeypatch, shift, rate):
    import sys
    from types import SimpleNamespace
    import numpy as np
    import torch
    from PIL import Image
    from plateai_web.pose_data import sha256
    image = tmp_path / 'test.png'
    Image.new('RGB', (640, 640)).save(image)
    corners = np.array([[100, 100], [300, 100], [300, 200], [100, 200]], dtype=np.float32)
    result = SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[100, 100, 300, 200]]), conf=torch.tensor([.99])),
                             keypoints=SimpleNamespace(xy=torch.tensor((corners + [shift, 0])[None])))
    class Model:
        def __init__(self, path): pass
        def predict(self, *args, **kwargs): return [result]
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=Model))
    snapshot = {'rows': [{'split': 'holdout', 'image': 'test.png', 'image_sha256': sha256(image),
                           'width': 640, 'height': 640, 'corners': [corners.tolist()]}]}
    metrics = pose_training().evaluate_checkpoint(tmp_path / 'last.pt', tmp_path, snapshot, 'cpu', lambda: None)
    assert metrics['bbox_ap50'] == pytest.approx(1.)
    assert metrics['complete_quad_recall'] == pytest.approx(rate)
    assert metrics['corner_error_640px'] == pytest.approx(shift)


def test_worker_selects_pose_pipeline_not_ctc(tmp_path, monkeypatch):
    from plateai_web.training_store import TrainingStore
    from plateai_web.training_worker import run_worker
    from plateai_trainer.training.control import TrainingProgress
    store = TrainingStore(tmp_path / 'runs/.web-training')
    task_id = '0123456789ab'
    store.create(task_id, 'pose', {'kind': 'pose', 'epochs': 10})
    def pipeline(root, ident, request, *, on_progress, **kwargs):
        assert request['kind'] == 'pose'
        on_progress(TrainingProgress(kind='epoch', phase='training', epoch=1, total_epochs=10, train_loss=.5))
        return {'kind': 'pose', 'package_sha256': 'a' * 64}
    monkeypatch.setattr(pose_training(), 'run_pose_pipeline', pipeline)
    assert run_worker(tmp_path, task_id) == 0
    task = store.get(task_id)
    assert task['result']['history'][0]['val_acc'] is None
    assert task['result']['kind'] == 'pose'
    assert 'ONNX' not in task['message']


def test_frozen_input_identity_is_enforced(tmp_path):
    module = pose_training()
    source = {'dataset_id': 'test', 'revision': 'revision'}
    index = tmp_path / 'index.json'
    row = {'source_index': 17, 'image_sha256': 'a' * 64, 'corners': [[[10,10],[20,10],[20,20],[10,20]]]}
    fingerprint = {'source_index': 17, 'image_sha256': 'a' * 64, 'original_corners_sha256': module.corners_sha256(row['corners'])}
    index.write_text(json.dumps({'schema_version': 'pose-input-index-v1', 'source': source, 'train_images': [fingerprint]}))
    module.validate_input_index(index, source, [row])
    with pytest.raises(ValueError, match='frozen'):
        module.validate_input_index(index, source, [{**row, 'image_sha256': 'b' * 64}])
    with pytest.raises(ValueError, match='frozen'):
        module.validate_input_index(index, source, [{**row, 'corners': [[[11,10],[20,10],[20,20],[10,20]]]}])
