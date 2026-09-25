"""Coordinates-only reviewed annotations and private, hash-checked YOLO snapshots."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from plateai_trainer.detection.real_dataset import canonical_quad, pixel_identity

CORNER_ORDER = ['left_top', 'right_top', 'right_bottom', 'left_bottom']


def corners_sha256(corners) -> str:
    """Fingerprint source coordinates without publishing the original labels."""
    array = np.asarray(corners, dtype='<f4')
    if array.ndim != 3 or array.shape[1:] != (4, 2) or not np.isfinite(array).all():
        raise ValueError('invalid source corners')
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(document, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def safe_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('invalid relative dataset path')
    root = root.resolve()
    candidate = root / relative
    candidate.resolve().relative_to(root)
    if any(part in ('.', '..') for part in relative.split('/')) or Path(relative).is_absolute():
        raise ValueError('invalid relative dataset path')
    for part in (candidate, *candidate.parents):
        if part == root:
            break
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError('redirected dataset path')
    return candidate


def label_lines(corners, width: int, height: int) -> list[str]:
    if type(width) is not int or type(height) is not int or min(width, height) < 2:
        raise ValueError('invalid image dimensions')
    if not isinstance(corners, list) or not corners:
        raise ValueError('empty corners')
    lines = []
    for points in corners:
        quad = np.asarray(points, dtype=np.float32)
        canonical = canonical_quad(quad, width, height)
        if not np.allclose(quad, canonical, atol=.001, rtol=0):
            raise ValueError('non-semantic corner order')
        scaled = quad * (640 / max(width, height))
        if (np.linalg.norm(np.roll(scaled, -1, axis=0) - scaled, axis=1).min() < 4
                or abs(cv2.contourArea(scaled)) < 16 or not cv2.isContourConvex(quad)):
            raise ValueError('degenerate quad')
        lo, hi = quad.min(axis=0), quad.max(axis=0)
        xywh = np.r_[(lo + hi) / 2, hi - lo] / [width, height, width, height]
        keypoints = np.column_stack((quad / [width, height], np.full(4, 2))).reshape(-1)
        lines.append('0 ' + ' '.join(f'{v:.10f}' for v in np.r_[xywh, keypoints]))
    return lines


def load_annotations(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding='utf-8'))
    if set(doc) != {'schema_version', 'source', 'corner_order', 'split_seed', 'records'}:
        raise ValueError('annotation fields must be coordinates-only')
    if doc['schema_version'] != 'pose-reviewed-v1' or doc['corner_order'] != CORNER_ORDER:
        raise ValueError('unsupported annotation contract')
    if set(doc['source']) != {'dataset_id', 'revision'} or not all(isinstance(v, str) and v for v in doc['source'].values()):
        raise ValueError('invalid source fields')
    if type(doc['split_seed']) is not int or not isinstance(doc['records'], list) or not doc['records']:
        raise ValueError('invalid annotation split')
    indices, hashes = set(), set()
    for row in doc['records']:
        if set(row) != {'validation_index', 'image_sha256', 'width', 'height', 'corners', 'split'}:
            raise ValueError('annotation record fields must be coordinates-only')
        index, digest = row['validation_index'], row['image_sha256']
        if (type(index) is not int or index < 0 or index in indices or not isinstance(digest, str)
                or not re.fullmatch('[0-9a-f]{64}', digest) or digest in hashes
                or row['split'] not in ('train', 'holdout')):
            raise ValueError('invalid/duplicate annotation identity or split')
        label_lines(row['corners'], row['width'], row['height'])
        indices.add(index)
        hashes.add(digest)
    if {r['split'] for r in doc['records']} != {'train', 'holdout'}:
        raise ValueError('both annotation splits are required')
    return doc


def export_annotations(review_root: Path, split_path: Path, output: Path) -> dict:
    """Whitelist reviewed data. Never copy image bytes, notes or original labels."""
    if output.exists():
        raise FileExistsError(output)
    review = json.loads((review_root / 'review_set.json').read_text(encoding='utf-8'))
    split = json.loads(split_path.read_text(encoding='utf-8'))
    train, holdout = set(split['train']), set(split['test'])
    if (train & holdout or len(train) != len(split['train']) or len(holdout) != len(split['test'])):
        raise ValueError('overlapping/duplicate split')
    rows = [json.loads(line) for line in (review_root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len({r['validation_index'] for r in rows}) != len(rows):
        raise ValueError('duplicate review index')
    records = []
    for row in rows:
        if row['review_status'] == 'excluded':
            continue
        if row['review_status'] != 'accepted':
            raise ValueError('unreviewed record')
        label = json.loads(safe_file(review_root, row['label_path']).read_text(encoding='utf-8'))
        index = row['validation_index']
        if label['validation_index'] != index or label['image_sha256'] != row['image_sha256']:
            raise ValueError('review identity mismatch')
        width, height = label['image_size_wh']
        label_lines(label['reviewed_corners'], width, height)
        records.append(dict(validation_index=index, image_sha256=row['image_sha256'], width=width,
                            height=height, corners=label['reviewed_corners'], split='train' if index in train else 'holdout'))
    if {r['validation_index'] for r in records} != train | holdout:
        raise ValueError('split does not match accepted reviews')
    doc = dict(schema_version='pose-reviewed-v1', source={k: review['source']['dataset'][k] for k in ('dataset_id', 'revision')},
               corner_order=CORNER_ORDER, split_seed=split['seed'], records=sorted(records, key=lambda r: r['validation_index']))
    write_json(output, doc)
    load_annotations(output)
    return doc


def dataset_records(directory: Path, source: dict, split: str) -> list[dict]:
    meta = json.loads((directory / 'dataset.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (directory / 'metadata.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if meta.get('source') != source or meta.get('split') != split or len(rows) != meta.get('count'):
        raise ValueError('dataset source/revision/count/split mismatch')
    if len({r['source_index'] for r in rows}) != len(rows) or len({r['image_sha256'] for r in rows}) != len(rows):
        raise ValueError('duplicate dataset identity')
    return rows


def prepare_dataset(train_root: Path, validation_root: Path, annotations: Path, output: Path, *, check_cancelled=lambda: None) -> dict:
    """Keep the frozen 36-image holdout out of both optimization and model selection."""
    if output.exists():
        raise FileExistsError(output)
    doc = load_annotations(annotations)
    base = dataset_records(train_root, doc['source'], 'train')
    validation = {r['image_sha256']: r for r in dataset_records(validation_root, doc['source'], 'validation')}
    candidates = [(train_root, r, 'train', f"base-{r['source_index']:06d}", False) for r in base]
    for row in doc['records']:
        original = validation.get(row['image_sha256'])
        if original is None or (original['width'], original['height']) != (row['width'], row['height']):
            raise ValueError('review image hash/dimensions not present in source')
        candidates.append((validation_root, {**original, **row}, row['split'], f"clean-{row['validation_index']:06d}", True))
    selected, rejected, pixels, byte_hashes = [], [], set(), set()
    for directory, row, split, name, reviewed in candidates:
        check_cancelled()
        path = safe_file(directory, row['image_path'])
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != row['image_sha256']:
            raise ValueError(f'image hash mismatch: {name}')
        with Image.open(io.BytesIO(payload)) as image:
            if image.size != (row['width'], row['height']) or image.getexif().get(274, 1) != 1:
                raise ValueError(f'image dimensions/orientation mismatch: {name}')
            if image.format not in ('JPEG', 'PNG'):
                raise ValueError('source must be JPEG or PNG')
            suffix = '.jpg' if image.format == 'JPEG' else '.png'
            identity = pixel_identity(np.asarray(image.convert('RGB')))
        if digest in byte_hashes or identity in pixels:
            raise ValueError('duplicate image across/within training and holdout')
        byte_hashes.add(digest)
        pixels.add(identity)
        try:
            lines = label_lines(row['corners'], row['width'], row['height'])
        except ValueError as exc:
            if reviewed:
                raise
            rejected.append({'source_index': row['source_index'], 'reason': str(exc)})
            continue
        selected.append((path, row, split, name + suffix, lines))
    output.mkdir(parents=True)
    manifest = []
    for path, row, split, name, lines in selected:
        check_cancelled()
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row['image_sha256']:
            raise ValueError('image changed while preparing snapshot')
        target = output / 'images' / split / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        label = output / 'labels' / split / (Path(name).stem + '.txt')
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        manifest.append(dict(image=str(target.relative_to(output)).replace('\\', '/'), image_sha256=row['image_sha256'],
                             width=row['width'], height=row['height'], corners=row['corners'], split=split))
    result = dict(train_count=sum(r['split'] == 'train' for r in manifest), holdout_count=sum(r['split'] == 'holdout' for r in manifest),
                  rejected=rejected, annotation_sha256=sha256(annotations), rows=manifest)
    write_json(output / 'snapshot.json', result)
    # No per-epoch holdout validation: this preset selects fixed epoch 10, not best.pt.
    (output / 'plate-pose.yaml').write_text('path: ' + json.dumps(output.resolve().as_posix()) +
        '\ntrain: images/train\nval: images/train\nnames: {0: plate}\nkpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\n', encoding='utf-8')
    return result
