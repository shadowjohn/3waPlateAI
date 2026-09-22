"""Pinned, local-only real-scene Detector data; never imports upstream code.

The upstream dataset has no declared license. Explicit acknowledgement is
required. Neither downloaded data nor derived model weights may be distributed
under this experiment's authorization.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import uuid
from urllib.request import Request, urlopen

import cv2
import numpy as np

from plateai_shared.publication import publish_directory_no_replace, remove_owned_staging
from plateai_trainer.detection.dataset import DetectionDataError
from plateai_trainer.detection.real_dataset import canonical_quad, pixel_identity

DATASET_ID = 'EZCon/taiwan-license-plate-detection'
REVISION = 'ab64ba1e86615c8371e1b5617792a130d45028e8'
SOURCE = {'dataset_id': DATASET_ID, 'revision': REVISION}
COUNTS = {'train': 2346, 'validation': 671, 'test': 336}


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _request(url):
    with urlopen(Request(url, headers={'User-Agent': '3waPlateAI-local-experiment/1'}), timeout=90) as response:
        return response.read()


def decode_row(row):
    """Use embedded bytes only; do not resolve source-controlled image paths."""
    try:
        payload = row['image']['bytes']
        if not isinstance(payload, bytes):
            raise DetectionDataError('missing embedded image bytes')
        bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise DetectionDataError('cannot decode source image')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        labels, polygons = row['label'], row['xyxyxyxyn']
        if len(labels) != len(polygons) or any(label != 0 for label in labels):
            raise DetectionDataError('invalid instance classes/count')
        corners = []
        for polygon in polygons:
            normalized = np.asarray(polygon, dtype=np.float32)
            if normalized.shape != (4, 2) or not np.isfinite(normalized).all() or np.any(normalized < 0) or np.any(normalized > 1):
                raise DetectionDataError('invalid normalized corners')
            # Source normalized endpoints may equal 1. Reader uses pixel centres.
            pixels = np.minimum(normalized * [width, height], [width - 1, height - 1])
            corners.append(canonical_quad(pixels, width, height).tolist())
        return payload, rgb, corners
    except (KeyError, TypeError, ValueError) as error:
        raise DetectionDataError(str(error)) from error


def _phash(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    low = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].ravel()[1:]
    return int.from_bytes(np.packbits(low > np.median(low)).tobytes(), 'big')


class HoldoutIndex:
    """Conservative duplicate exclusion, not a guarantee of independent scenes.

    Decoded-pixel SHA catches lossless re-encoding; 63-bit pHash distance <=4
    also excludes JPEG variants/near-identical scenes. False exclusions are
    preferable to test leakage. All exclusions retain their reason/source.
    """
    def __init__(self):
        self.exact, self.perceptual = {}, []

    def add(self, rgb, identity):
        self.exact[pixel_identity(rgb)] = identity
        self.perceptual.append((_phash(rgb), identity))

    def match(self, rgb):
        exact = self.exact.get(pixel_identity(rgb))
        if exact is not None:
            return exact, 'exact_pixels'
        value = _phash(rgb)
        for other, identity in self.perceptual:
            if (value ^ other).bit_count() <= 4:
                return identity, 'near_duplicate_phash'
        return None


def import_rows(splits, output, holdout_directories):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    index = HoldoutIndex()
    heldout = []
    for directory in holdout_directories:
        directory = Path(directory)
        if not directory.is_dir():
            raise DetectionDataError(f'held-out directory missing: {directory}')
        for path in sorted(directory.rglob('*')):
            if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                continue
            payload = path.read_bytes()
            bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise DetectionDataError(f'cannot decode held-out image: {path}')
            identity = f'holdout:{path.resolve()}'
            index.add(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), identity)
            heldout.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(payload).hexdigest()})
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f'.{output.name}.partial-{uuid.uuid4().hex}'
    staging.mkdir()
    report = {'source': SOURCE, 'local_only': True, 'license_reviewed': False,
              'redistribution_allowed': False, 'held_out_files': heldout,
              'exclusion_policy': 'external holdout > test > validation > train; exact RGB SHA256 or pHash Hamming<=4; no reassignment',
              'retained': {}, 'source_counts': {}, 'excluded': []}
    try:
        for split in ('test', 'validation', 'train'):
            root = staging / split
            (root / 'images').mkdir(parents=True)
            records = []
            source_count = 0
            for source_index, row in enumerate(splits[split]):
                source_count += 1
                try:
                    payload, rgb, corners = decode_row(row)
                except DetectionDataError as error:
                    report['excluded'].append({'split': split, 'source_index': source_index, 'reason': f'invalid_annotation:{error}'})
                    continue
                match = index.match(rgb)
                if match:
                    report['excluded'].append({'split': split, 'source_index': source_index, 'reason': match[1], 'matches': match[0]})
                    continue
                identity = f'{split}:{source_index}'
                index.add(rgb, identity)
                name = f'images/{source_index:06d}.img'
                (root / name).write_bytes(payload)
                records.append({'source_index': source_index, 'image_path': name,
                                'image_sha256': hashlib.sha256(payload).hexdigest(),
                                'width': rgb.shape[1], 'height': rgb.shape[0], 'corners': corners})
            if not records:
                raise DetectionDataError(f'no usable images remain in {split}')
            (root / 'metadata.jsonl').write_text(''.join(json.dumps(record, allow_nan=False) + '\n' for record in records), encoding='utf-8')
            _json(root / 'dataset.json', {'schema_version': 'real-detection-v1', 'source': SOURCE,
                'split': split, 'count': len(records), 'license_reviewed': False, 'local_only': True})
            report['retained'][split] = len(records)
            report['source_counts'][split] = source_count
            print(f'{split}: retained {len(records)} / {source_count}', flush=True)
        _json(staging / 'import_report.json', report)
        publish_directory_no_replace(staging, output)
    except BaseException:
        remove_owned_staging(staging, output)
        raise
    return report


def fetch_snapshot(cache):
    """Verify pinned LFS SHA-256 and size, including cached parquet files."""
    import pyarrow.parquet as pq
    cache = Path(cache)
    tree = json.loads(_request(f'https://huggingface.co/api/datasets/{DATASET_ID}/tree/{REVISION}?recursive=true'))
    files = [entry for entry in tree if re.fullmatch(r'data/(train|validation|test)-\d{5}-of-00012\.parquet', entry['path'])]
    if len(files) != 36:
        raise DetectionDataError('pinned source parquet inventory changed')
    cache.mkdir(parents=True, exist_ok=True)

    def download(entry):
        path = cache / Path(entry['path']).name
        expected = entry['lfs']['oid']
        payload = path.read_bytes() if path.is_file() else _request(f'https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/{entry["path"]}')
        if len(payload) != entry['size'] or hashlib.sha256(payload).hexdigest() != expected:
            raise DetectionDataError(f'source SHA/size mismatch: {path.name}')
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(payload)
        return path

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(download, files))
    _json(cache / 'snapshot.json', {'source': SOURCE, 'files': files, 'local_only': True, 'license_reviewed': False})
    splits = {}
    for split, count in COUNTS.items():
        rows = []
        for path in sorted(paths):
            if path.name.startswith(split + '-'):
                rows.extend(pq.read_table(path).to_pylist())
        if len(rows) != count:
            raise DetectionDataError(f'{split} row count differs from pinned card')
        splits[split] = rows
    return splits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--holdout-images', type=Path, action='append', required=True)
    parser.add_argument('--acknowledge-unreviewed-license', action='store_true', required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output exists; choose a new local directory')
    report = import_rows(fetch_snapshot(args.cache), args.output, args.holdout_images)
    print(json.dumps({'retained': report['retained'], 'excluded': len(report['excluded']), 'local_only': True}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
