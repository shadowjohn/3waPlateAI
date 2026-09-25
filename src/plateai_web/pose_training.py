"""Fixed, local-only Pose pilot; never activates a runtime model."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from .pose_data import corners_sha256, dataset_records, load_annotations, prepare_dataset, safe_file, sha256, write_json
from .paths import package_data_root
from plateai_trainer.training.control import TrainingProgress

PRETRAIN_SHA256 = '869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0'
PRETRAIN_URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt'
ANNOTATIONS = 'annotations/pose-clean119-v1.json'
PRESET = 'clean119-70-30-v1'


def resource_path(root: Path, relative: str) -> Path:
    local = root / relative
    return local if local.is_file() else package_data_root() / relative


def validate_input_index(path: Path, source: dict, rows: list[dict]) -> None:
    index = json.loads(path.read_text(encoding='utf-8'))
    wanted = {(r['source_index'], r['image_sha256'], r['original_corners_sha256']) for r in index['train_images']}
    actual = {(r['source_index'], r['image_sha256'], corners_sha256(r['corners'])) for r in rows}
    if (index.get('schema_version') != 'pose-input-index-v1' or index['source'] != source
            or len(wanted) != len(index['train_images']) or len(actual) != len(rows) or actual != wanted):
        raise ValueError('train identities differ from frozen input index')


def python_path(root: Path) -> Path:
    configured = os.environ.get('PLATEAI_POSE_PYTHON')
    if configured:
        return Path(configured).resolve()
    candidates = [root / '.venv-pose/Scripts/python.exe', root / 'runs/yolo-pose-spike-20260925/env/Scripts/python.exe']
    return next((p for p in candidates if p.is_file()), candidates[0])


def pretrained_path(root: Path) -> Path:
    candidates = [root / 'models/private/yolo11n-pose.pt', root / 'runs/yolo-pose-spike-20260925/yolo11n-pose.pt']
    return next((p for p in candidates if p.is_file()), candidates[0])


def preflight(root: Path) -> dict:
    checks = []
    def check(key, action, remedy):
        try:
            detail = action()
            checks.append(dict(id=key, ok=True, detail=detail, remedy=remedy))
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            checks.append(dict(id=key, ok=False, detail=str(exc), remedy=remedy))

    def environment():
        python = python_path(root)
        if not python.is_file():
            raise ValueError('找不到獨立 Pose Python')
        probe = subprocess.run([str(python), '-c',
            "import json,torch,cv2,yaml,PIL,importlib.metadata as m; print(json.dumps({'torch':torch.__version__,'ultralytics':m.version('ultralytics'),'cuda':torch.cuda.is_available()}))"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
            env={**os.environ, 'YOLO_AUTOINSTALL': 'false'})
        if probe.returncode:
            raise ValueError('Pose 環境 import 失敗；請查核 setup_pose.ps1 的獨立環境')
        observed = json.loads(probe.stdout.strip().splitlines()[-1])
        if observed['ultralytics'] != '8.3.203':
            raise ValueError('此配方固定 ultralytics 8.3.203')
        return {**observed, 'python': str(python)}

    def annotations():
        doc = load_annotations(resource_path(root, ANNOTATIONS))
        counts = {s: sum(r['split'] == s for r in doc['records']) for s in ('train', 'holdout')}
        if counts != {'train': 83, 'holdout': 36} or doc['split_seed'] != 20260923:
            raise ValueError('此配方需要固定 83/36 標註分組')
        return {**counts, 'sha256': sha256(resource_path(root, ANNOTATIONS))}

    def weights():
        path = pretrained_path(root)
        if sha256(path) != PRETRAIN_SHA256:
            raise ValueError('初始 yolo11n-pose.pt hash 不符')
        return str(path)

    def inputs():
        doc = load_annotations(resource_path(root, ANNOTATIONS))
        base = dataset_records(root / 'out/ezcon-detector-v1/train', doc['source'], 'train')
        val = dataset_records(root / 'out/ezcon-detector-v1/validation', doc['source'], 'validation')
        if len(base) != 2000:
            raise ValueError('原始 train snapshot 應有 2,000 張（worker 再排除退化標註）')
        validate_input_index(resource_path(root, 'annotations/pose-pilot-inputs-v1.json'), doc['source'], base)
        hashes = {r['image_sha256']: r for r in val}
        if not all(r['image_sha256'] in hashes for r in doc['records']):
            raise ValueError('缺少與修正標註 hash 相符的 validation 圖片')
        for folder, rows in [('train', base), ('validation', [hashes[r['image_sha256']] for r in doc['records']])]:
            for row in rows:
                if not safe_file(root / 'out/ezcon-detector-v1' / folder, row['image_path']).is_file():
                    raise ValueError(f"缺少 {folder} 圖片 #{row['source_index']}")
        return '2,000 原始 train + 119 reviewed 圖片已找到；啟動後逐檔檢查 hash／pixel 重複／幾何'

    check('python', environment, '.\\setup_pose.ps1 -Profile cu118（1080）；50 系列用 cu128，無 GPU 用 cpu')
    check('annotations', annotations, '取得 repo 內 annotations/pose-clean119-v1.json；此包不含圖片')
    check('pretrained', weights, '.\\.venv-pose\\Scripts\\python.exe tools\\prepare_pose_inputs.py --download-pretrained --acknowledge-unreviewed-license')
    check('dataset', inputs, '.\\.venv-pose\\Scripts\\python.exe tools\\prepare_pose_inputs.py --download-dataset --acknowledge-unreviewed-license；或自行提供相符的 out/ezcon-detector-v1')
    if sys.platform != 'win32':
        checks.append(dict(id='worker', ok=False, detail='目前 Studio 背景訓練 worker 僅支援 Windows', remedy='使用 Windows Studio 訓練，再將成果搬到推論主機'))
    return dict(ready=all(c['ok'] for c in checks), checks=checks, preset=PRESET,
                recipe={'epochs': 10, 'imgsz': 640, 'batch': 8, 'train': '1997 + 83', 'holdout': 36, 'selection': 'fixed epoch 10 last.pt'},
                annotations_url='/api/train/pose/annotations', documentation='/api/train/pose/guide')


def validate_request(root: Path, request: dict) -> None:
    if request.get('acknowledge_unreviewed_license') is not True:
        raise ValueError('請確認資料與模型權利尚未完成部署審查，僅進行本機實驗')
    if request.get('epochs') != 10 or request.get('device', 'auto') not in ('auto', 'cpu', 'cuda'):
        raise ValueError('此配方固定 10 epochs，device 僅能為 auto / cpu / cuda')
    if any(request.get(k) is not None for k in ('run_name', 'train_dataset', 'validation_dataset', 'batch_size')):
        raise ValueError('Pose 配方不接受 OCR dataset / run / batch 覆寫')
    report = preflight(root)
    if not report['ready']:
        raise ValueError('Pose 尚未就緒：' + '；'.join(c['id'] + ': ' + c['detail'] for c in report['checks'] if not c['ok']))


def train_prepared(dataset: Path, initial: Path, run: Path, *, device: str, on_progress, check_cancelled, model_factory=None) -> Path:
    if model_factory is None:
        from ultralytics import YOLO, settings
        settings.update({k: False for k in ('sync', 'hub', 'wandb', 'mlflow', 'clearml', 'comet', 'neptune', 'tensorboard', 'dvc', 'raytune')})
        model_factory = YOLO
    model = model_factory(str(initial))
    batch_number = 0
    def epoch_start(trainer):
        nonlocal batch_number
        batch_number = 0
        check_cancelled()
    def batch_event(trainer):
        nonlocal batch_number
        check_cancelled()
        batch_number += 1
        on_progress(TrainingProgress(kind='batch', phase='training', epoch=trainer.epoch + 1,
                                     total_epochs=10, batch=batch_number, total_batches=len(trainer.train_loader)))
    def epoch_event(trainer):
        check_cancelled()
        loss = float(trainer.tloss.sum().item()) if trainer.tloss is not None else None
        on_progress(TrainingProgress(kind='epoch', phase='training', epoch=trainer.epoch + 1, total_epochs=10,
                                     batch=1, total_batches=1, train_loss=loss))
    model.add_callback('on_train_epoch_start', epoch_start)
    model.add_callback('on_train_batch_end', batch_event)
    model.add_callback('on_train_epoch_end', epoch_event)
    model.add_callback('on_train_start', lambda trainer: check_cancelled())
    check_cancelled()
    model.train(data=str(dataset / 'plate-pose.yaml'), epochs=10, imgsz=640, batch=8,
                device=device, workers=0, seed=42, deterministic=True, amp=False, cache=False,
                optimizer='AdamW', lr0=.001, lrf=.01, cos_lr=True, warmup_epochs=3,
                patience=10, close_mosaic=0, mosaic=0., mixup=0., copy_paste=0.,
                degrees=0., translate=0., scale=0., shear=0., perspective=0.,
                flipud=0., fliplr=0., hsv_h=0., hsv_s=0., hsv_v=0., plots=False,
                save=True, save_period=-1, val=False, project=str(run), name='pilot', exist_ok=False, verbose=False)
    check_cancelled()
    checkpoint = run / 'pilot/weights/last.pt'
    if not checkpoint.is_file():
        raise ValueError('訓練沒有產生固定 epoch 10 的 last.pt')
    return checkpoint


def evaluate_checkpoint(checkpoint: Path, dataset: Path, snapshot: dict, device: str, check_cancelled) -> dict:
    import cv2
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    from plateai_shared.detection import letterbox_rgb_v1, map_points_to_letterbox
    from plateai_trainer.detection.contracts import CompositeInstance
    from plateai_trainer.detection.engine import _prediction_metrics
    model = YOLO(str(checkpoint))
    predictions, truths = [], []
    for row in snapshot['rows']:
        if row['split'] != 'holdout':
            continue
        check_cancelled()
        path = safe_file(dataset, row['image'])
        if sha256(path) != row['image_sha256']:
            raise ValueError('holdout image changed')
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB'))
        _, transform = letterbox_rgb_v1(rgb)
        gt = []
        for corners in row['corners']:
            quad = map_points_to_letterbox(np.asarray(corners, dtype=np.float32), transform)
            gt.append(CompositeInstance(tuple(np.r_[quad.min(axis=0), quad.max(axis=0)]), quad, {}))
        truths.append(tuple(gt))
        result = model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), imgsz=640, rect=False,
                               conf=.05, iou=.45, max_det=100, device=device, half=False, verbose=False)[0]
        rows = []
        for box, score, corners in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.keypoints.xy.cpu().numpy(), strict=True):
            box = map_points_to_letterbox(box.reshape(2, 2), transform).reshape(4)
            corners = np.clip(corners, [0, 0], [row['width'] - 1, row['height'] - 1])
            quad = map_points_to_letterbox(corners, transform)
            rows.append(np.r_[(box[:2] + box[2:]) / 2, box[2:] - box[:2], score, quad.ravel()])
        predictions.append(np.asarray(rows, dtype=np.float32).reshape(-1, 13))
    return _prediction_metrics(predictions, truths)


def publish_artifact(checkpoint: Path, output: Path, summary: dict) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    target = output / 'pose-pilot.pt'
    shutil.copyfile(checkpoint, target)
    digest = sha256(target)
    manifest = dict(model_id='pose-' + digest[:12], detector={'path': 'pose-pilot.pt', 'sha256': digest},
                    purpose='detector-only; merge into a separately configured OCR service manifest', deployment_approved=False)
    write_json(output / 'detector-manifest.json', manifest)
    write_json(output / 'training-summary.json', summary)
    (output / 'README.txt').write_text('Private local experiment export. Not an official model release.\n'
        'Contains detector only; no OCR models, images or annotations.\n'
        'Check rights and quality before deployment. Configure OCR paths separately; never disable hash checks.\n'
        'New training produces a new hash; no active model was replaced.\n', encoding='utf-8')
    archive_path = output / 'pose-detector.zip'
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('pose-pilot.pt', 'detector-manifest.json', 'training-summary.json', 'README.txt'):
            archive.write(output / name, name)
    return dict(checkpoint_sha256=digest, package=str(archive_path), package_sha256=sha256(archive_path))


def run_pose_pipeline(root: Path, task_id: str, request: dict, *, on_progress, on_result, check_cancelled) -> dict:
    validate_request(root, request)
    import torch
    torch.set_num_threads(4)
    device = '0' if request.get('device', 'auto') != 'cpu' and torch.cuda.is_available() else 'cpu'
    if request.get('device') == 'cuda' and device == 'cpu':
        raise ValueError('指定 CUDA 但 torch.cuda.is_available() 為 false')
    run = root / 'runs' / f'pose-{task_id}'
    run.mkdir(parents=True, exist_ok=False)
    os.environ.update(YOLO_CONFIG_DIR=str(run / 'yolo-config'), YOLO_AUTOINSTALL='false', WANDB_DISABLED='true')
    on_result({'kind': 'pose', 'total_epochs': 10, 'device': device, 'run_dir': str(run)})
    on_progress(TrainingProgress(kind='stage', phase='preparing'))
    initial = run / 'yolo11n-pose.pt'
    shutil.copyfile(pretrained_path(root), initial)
    if sha256(initial) != PRETRAIN_SHA256:
        raise ValueError('initial checkpoint hash mismatch')
    snapshot = prepare_dataset(root / 'out/ezcon-detector-v1/train', root / 'out/ezcon-detector-v1/validation',
                               resource_path(root, ANNOTATIONS), run / 'dataset', check_cancelled=check_cancelled)
    if (snapshot['train_count'], snapshot['holdout_count'], len(snapshot['rejected'])) != (2080, 36, 3):
        raise ValueError('資料與既有 1997 + 83 / 36 配方不符；請檢查來源，不自動改分組')
    on_result({'train_count': 2080, 'holdout_count': 36})
    checkpoint = train_prepared(run / 'dataset', initial, run, device=device, on_progress=on_progress, check_cancelled=check_cancelled)
    on_progress(TrainingProgress(kind='stage', phase='validating', epoch=10, total_epochs=10))
    metrics = evaluate_checkpoint(checkpoint, run / 'dataset', snapshot, device, check_cancelled)
    check_cancelled()
    on_progress(TrainingProgress(kind='stage', phase='exporting'))
    summary = dict(preset=PRESET, train_count=2080, holdout_count=36, metrics=metrics, device=device,
                   annotation_sha256=snapshot['annotation_sha256'], initial_sha256=PRETRAIN_SHA256,
                   input_index_sha256=sha256(resource_path(root, 'annotations/pose-pilot-inputs-v1.json')),
                   source=load_annotations(resource_path(root, ANNOTATIONS))['source'],
                   epochs=10, imgsz=640, batch=8, seed=42, split_seed=20260923,
                   selection='fixed epoch10 last.pt; holdout not used for model selection',
                   runtime={p: importlib.metadata.version(p) for p in ('torch', 'ultralytics')},
                   scope='previously seen internal holdout, not independent blind accuracy',
                   deployment_approved=False)
    artifact = publish_artifact(checkpoint, run / 'artifact', summary)
    result = dict(kind='pose', **artifact, metrics=metrics, train_count=2080, holdout_count=36,
                  download_url=f'/api/train/pose/artifacts/{task_id}/package')
    on_result(result)
    return result
