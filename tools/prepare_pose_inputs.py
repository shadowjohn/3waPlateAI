"""Restore explicitly selected local Pose inputs. No downloads without opt-in."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from plateai_web.pose_data import corners_sha256, load_annotations, write_json, sha256
from plateai_web.pose_training import PRETRAIN_SHA256, PRETRAIN_URL, ANNOTATIONS
from plateai_shared.publication import publish_directory_no_replace


def export_input_index(train_root: Path, output: Path) -> dict:
    meta = json.loads((train_root / 'dataset.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (train_root / 'metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    doc = dict(schema_version='pose-input-index-v1', source=meta['source'],
               train_images=[{**{k: row[k] for k in ('source_index', 'image_sha256')},
                              'original_corners_sha256': corners_sha256(row['corners'])} for row in rows])
    write_json(output, doc)
    return doc


def restore_inputs(splits: dict, index_path: Path, annotations_path: Path, output: Path) -> None:
    from fetch_ezcon_detection import decode_row
    if output.exists():
        raise FileExistsError(f'保留現有目錄，不覆寫：{output}')
    doc = load_annotations(annotations_path)
    index = json.loads(index_path.read_text(encoding='utf-8'))
    if index.get('schema_version') != 'pose-input-index-v1' or index['source'] != doc['source']:
        raise ValueError('source revision mismatch')
    wanted_train = {r['source_index']: r['image_sha256'] for r in index['train_images']}
    fingerprints = {r['source_index']: r['original_corners_sha256'] for r in index['train_images']}
    wanted_validation = {r['image_sha256'] for r in doc['records']}
    if len(wanted_train) != len(index['train_images']) or len(set(wanted_train.values())) != len(wanted_train):
        raise ValueError('duplicate input index')
    selected = {'train': [], 'validation': []}
    for split in selected:
        for source_index, raw in enumerate(splits[split]):
            payload = raw['image']['bytes']
            digest = hashlib.sha256(payload).hexdigest()
            if split == 'train':
                if source_index not in wanted_train:
                    continue
                if digest != wanted_train[source_index]:
                    raise ValueError('train image hash differs from frozen input index')
            elif digest not in wanted_validation:
                continue
            payload, rgb, corners = decode_row(raw)
            if split == 'train':
                if corners_sha256(corners) != fingerprints[source_index]:
                    raise ValueError('source label differs from frozen input index')
            selected[split].append((source_index, digest, payload, rgb.shape[:2], corners))
    if len(selected['train']) != len(wanted_train) or {r[1] for r in selected['validation']} != wanted_validation:
        raise ValueError('missing images matching frozen hashes')
    if len(selected['validation']) != len(wanted_validation):
        raise ValueError('duplicate validation image hash')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f'.{output.name}.{uuid.uuid4().hex}.partial'
    staging.mkdir()
    # A failed preparation retains a clearly named partial directory for diagnosis.
    for split, rows in selected.items():
        directory = staging / split
        (directory / 'images').mkdir(parents=True)
        records = []
        for source_index, digest, payload, (height, width), corners in rows:
            name = f'images/{source_index:06d}.img'
            (directory / name).write_bytes(payload)
            records.append(dict(source_index=source_index, image_path=name, image_sha256=digest,
                                width=width, height=height, corners=corners))
        (directory / 'metadata.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8', newline='\n')
        write_json(directory / 'dataset.json', dict(schema_version='real-detection-v1', source=index['source'],
                                                   split=split, count=len(records), local_only=True, license_reviewed=False))
    publish_directory_no_replace(staging, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-pretrained', action='store_true')
    parser.add_argument('--download-dataset', action='store_true')
    parser.add_argument('--acknowledge-unreviewed-license', action='store_true')
    parser.add_argument('--export-input-index', type=Path, help='maintainer-only: export hashes, never images/labels')
    args = parser.parse_args(argv)
    if args.export_input_index:
        export_input_index(ROOT / 'out/ezcon-detector-v1/train', args.export_input_index)
        return 0
    if not (args.download_pretrained or args.download_dataset):
        parser.error('choose --download-pretrained and/or --download-dataset')
    if not args.acknowledge_unreviewed_license:
        parser.error('requires --acknowledge-unreviewed-license; local experiment only, not deployment approval')
    if args.download_pretrained:
        target = ROOT / 'models/private/yolo11n-pose.pt'
        if target.exists():
            if sha256(target) != PRETRAIN_SHA256:
                raise ValueError('existing pretrained hash mismatch; not overwritten')
        else:
            with urlopen(Request(PRETRAIN_URL, headers={'User-Agent': '3waPlateAI'}), timeout=60) as response:
                payload = response.read(32 * 1024 * 1024 + 1)
            if hashlib.sha256(payload).hexdigest() != PRETRAIN_SHA256:
                raise ValueError('downloaded pretrained SHA-256 mismatch')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(payload)
        print('Pinned pretrained ready. This is initialization, NOT a trained plate detector.', flush=True)
    if args.download_dataset:
        from fetch_ezcon_detection import fetch_snapshot, SOURCE
        index = ROOT / 'annotations/pose-pilot-inputs-v1.json'
        if json.loads(index.read_text(encoding='utf-8'))['source'] != SOURCE:
            raise ValueError('input index is not the pinned upstream revision')
        target = ROOT / 'out/ezcon-detector-v1'
        if target.exists():
            raise FileExistsError('Existing out/ezcon-detector-v1 preserved; run Web preflight instead')
        print('Fetching pinned EZCon snapshot; cache and photos stay local/ignored.', flush=True)
        splits = fetch_snapshot(ROOT / 'datasets/restricted/pose-ezcon-cache')
        restore_inputs(splits, index, ROOT / ANNOTATIONS, target)
        print('Restored 2,000 train and 119 validation-source photos; reviewed labels applied only when training.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
