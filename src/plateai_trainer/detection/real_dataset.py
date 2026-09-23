"""Local-only real-scene adapter for the existing detector training contract."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from plateai_reader.rectifier import InvalidCornersError, normalize_corners
from plateai_shared.detection import letterbox_rgb_v1, map_points_to_letterbox
from .contracts import CompositeInstance
from .dataset import DetectionDataError, DetectionDataset, DetectionSample, _json, _hash
from .targets import assign_detection_targets


def canonical_quad(points, width, height):
    """Validate a full-image quad without applying the runtime minimum-area gate."""
    points = np.asarray(points, dtype=np.float32)
    if (points.shape != (4, 2) or not np.isfinite(points).all()
            or np.any(points < 0) or np.any(points > [width - 1, height - 1])):
        raise DetectionDataError('invalid real-scene corner coordinates')
    origin = points.min(axis=0)
    span = np.ceil(points.max(axis=0) - origin).astype(int) + 1
    try:
        # Translate to a tight local canvas solely for semantic ordering.
        # Tiny full-scene plates remain valid training annotations.
        return normalize_corners(points - origin, tuple(int(v) for v in span)).points_xy + origin
    except (InvalidCornersError, ValueError) as error:
        raise DetectionDataError(f'invalid real-scene quad: {error}') from error


def pixel_identity(rgb):
    shape = np.asarray(rgb.shape, dtype=np.int64).tobytes()
    return hashlib.sha256(shape + rgb.tobytes()).hexdigest()


class RealDetectionDataset(DetectionDataset):
    """Snapshot verified RGB bytes; retain labels and unresolved license status.

    Deliberately separate from synthetic validation: real photos must not
    invent a synthetic font, plate text, rendering seed or homography.
    """

    def __init__(self, directory: Path, *, large_object_p3: bool = False):
        if type(large_object_p3) is not bool:
            raise DetectionDataError('large_object_p3 must be a boolean')
        self.root = Path(directory).resolve()
        self.large_object_p3 = large_object_p3
        self._files = {}
        self._images, self._instances = [], []
        identities = []
        try:
            metadata = _json(self._read('dataset.json'))
            if (not isinstance(metadata, dict)
                    or metadata.get('schema_version') != 'real-detection-v1'
                    or metadata.get('split') not in ('train', 'validation', 'test')
                    or type(metadata.get('count')) is not int or metadata['count'] < 1
                    or type(metadata.get('license_reviewed')) is not bool
                    or metadata.get('local_only') is not True
                    or not isinstance(metadata.get('source'), dict)):
                raise DetectionDataError('invalid real-detection metadata/provenance')
            source = metadata['source']
            if not isinstance(source.get('dataset_id'), str) or not source['dataset_id'] or not isinstance(source.get('revision'), str) or not source['revision']:
                raise DetectionDataError('missing real dataset identity/revision')
            self._records = tuple(_json(line) for line in self._read('metadata.jsonl').splitlines())
            if len(self._records) != metadata['count']:
                raise DetectionDataError('real metadata count mismatch')
            paths = set()
            for record in self._records:
                name = record['image_path']
                if not isinstance(name, str) or '\\' in name or len(name.split('/')) != 2 or name.split('/')[0] != 'images' or name in paths:
                    raise DetectionDataError('invalid/duplicate real image path')
                path = self._safe_path(name)
                if path.is_symlink():
                    raise DetectionDataError('real image must not be a symlink')
                payload = self._read(name)
                if not _hash(record['image_sha256']) or hashlib.sha256(payload).hexdigest() != record['image_sha256']:
                    raise DetectionDataError('real image SHA-256 mismatch')
                bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise DetectionDataError('real image cannot be decoded')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                if type(record['width']) is not int or type(record['height']) is not int or (width, height) != (record['width'], record['height']):
                    raise DetectionDataError('real image dimensions mismatch')
                if not isinstance(record['corners'], list):
                    raise DetectionDataError('real instances must be a list')
                instances = []
                for points in record['corners']:
                    quad = canonical_quad(points, width, height)
                    box = np.r_[quad.min(axis=0), quad.max(axis=0)]
                    instances.append(CompositeInstance(tuple(float(x) for x in box), quad, {}))
                rgb.setflags(write=False)
                self._images.append(rgb)
                self._instances.append(tuple(instances))
                identities.append(pixel_identity(rgb))
                paths.add(name)
            if len(set(identities)) != len(identities):
                raise DetectionDataError('duplicate real images within split')
            found = {p.relative_to(self.root).as_posix() for p in (self.root / 'images').rglob('*') if p.is_file()}
            if found != paths:
                raise DetectionDataError('unlisted or missing real image')
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise DetectionDataError(str(error)) from error
        self.background_sha256s = frozenset(identities)
        self.provenance = {'training_data': 'real', 'license_reviewed': metadata['license_reviewed'],
                           'local_only': True, 'source': source, 'split': metadata['split']}
        self.input_hashes = {'files': dict(self._files), 'pixel_sha256s': sorted(identities),
                             'provenance': self.provenance}

    def __getitem__(self, index):
        image, transform = letterbox_rgb_v1(self._images[index])
        instances = tuple(CompositeInstance(
            tuple(float(x) for x in map_points_to_letterbox(np.asarray(item.bbox_xyxy).reshape(2, 2), transform).ravel()),
            map_points_to_letterbox(item.corners_xy, transform), {},
        ) for item in self._instances[index])
        return DetectionSample(
            image,
            instances,
            assign_detection_targets(instances, large_object_p3=self.large_object_p3),
            transform,
        )


def load_detection_dataset(directory, *, large_object_p3=False):
    directory = Path(directory)
    dataset_type = RealDetectionDataset if (directory / 'dataset.json').exists() else DetectionDataset
    return dataset_type(directory, large_object_p3=large_object_p3)
