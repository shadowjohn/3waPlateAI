"""Real image adapter preserves whole plates, multiple instances and provenance."""
import hashlib
import json

import numpy as np
from PIL import Image
import pytest


def fixture(root, *, points=None, image_path='images/a.png'):
    folder = root / 'images'
    folder.mkdir(parents=True)
    Image.new('RGB', (800, 400), (20, 60, 100)).save(folder / 'a.png')
    payload = (folder / 'a.png').read_bytes()
    row = {'image_path': image_path, 'image_sha256': hashlib.sha256(payload).hexdigest(),
           'width': 800, 'height': 400, 'corners': points if points is not None else [
               [[100, 100], [300, 100], [300, 180], [100, 180]],
               [[450, 220], [700, 220], [700, 320], [450, 320]]], 'source_index': 0}
    (root / 'metadata.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    (root / 'dataset.json').write_text(json.dumps({'schema_version': 'real-detection-v1',
        'split': 'train', 'source': {'dataset_id': 'test/real', 'revision': 'a' * 40},
        'license_reviewed': False, 'local_only': True, 'count': 1}), encoding='utf-8')
    return root


def test_real_adapter_maps_whole_plates_and_all_corners(tmp_path):
    from plateai_trainer.detection.real_dataset import RealDetectionDataset
    data = RealDetectionDataset(fixture(tmp_path))
    sample = data[0]
    assert len(sample.instances) == 2
    # 800x400 -> 640x320 with 160 pixels top padding.
    np.testing.assert_allclose(sample.instances[0].bbox_xyxy, [80, 240, 240, 304])
    np.testing.assert_allclose(sample.instances[1].corners_xy,
        [[360, 336], [560, 336], [560, 416], [360, 416]])
    # Both fixtures have canonical P4 short sides, so each contributes its
    # nine P4 cells plus the next-finer P3 cells.
    assert len(sample.targets.positive_indices) == 36
    np.testing.assert_array_equal(
        np.bincount(sample.targets.positive_level_indices, minlength=3),
        [18, 18, 0],
    )
    assert data.provenance['license_reviewed'] is False
    assert data.provenance['training_data'] == 'real'


@pytest.mark.parametrize('points', [
    [[[100, 100], [300, 100], [300, 180], [100, 180]]],
    [[[300, 180], [100, 100], [100, 180], [300, 100]]],
])
def test_adapter_canonicalizes_unordered_corners(tmp_path, points):
    from plateai_trainer.detection.real_dataset import RealDetectionDataset
    sample = RealDetectionDataset(fixture(tmp_path, points=points))[0]
    np.testing.assert_allclose(sample.instances[0].corners_xy,
                              [[80, 240], [240, 240], [240, 304], [80, 304]])


def test_adapter_accepts_a_labeled_negative_image(tmp_path):
    from plateai_trainer.detection.real_dataset import RealDetectionDataset
    sample = RealDetectionDataset(fixture(tmp_path, points=[]))[0]
    assert not sample.instances
    assert sample.targets.positive_indices.size == 0


@pytest.mark.parametrize('points', [
    [[[0, 0], [800, 0], [100, 40], [0, 40]]],
    [[[10, 10], [30, 10], [30, 10], [10, 20]]],
    [[[float('nan'), 0], [30, 0], [30, 10], [0, 10]]],
])
def test_invalid_annotation_cannot_become_background(tmp_path, points):
    from plateai_trainer.detection.real_dataset import RealDetectionDataset
    from plateai_trainer.detection.dataset import DetectionDataError
    with pytest.raises(DetectionDataError):
        RealDetectionDataset(fixture(tmp_path, points=points))


def test_dataset_rejects_path_escape_and_changed_bytes(tmp_path):
    from plateai_trainer.detection.real_dataset import RealDetectionDataset
    from plateai_trainer.detection.dataset import DetectionDataError
    with pytest.raises(DetectionDataError):
        RealDetectionDataset(fixture(tmp_path / 'escape', image_path='../a.png'))
    data = RealDetectionDataset(fixture(tmp_path / 'valid'))
    (tmp_path / 'valid/images/a.png').write_bytes(b'changed')
    with pytest.raises(DetectionDataError):
        data.verify_unchanged()


def test_reencoded_duplicate_cannot_cross_training_validation(tmp_path):
    from plateai_trainer.detection.real_dataset import RealDetectionDataset
    from plateai_trainer.detection.dataset import DetectionDataError, validate_train_validation_pair
    train = fixture(tmp_path / 'train')
    val = fixture(tmp_path / 'validation')
    path = val / 'images/a.png'
    Image.open(path).save(path, compress_level=0)
    record = json.loads((val / 'metadata.jsonl').read_text(encoding='utf-8'))
    record['image_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (val / 'metadata.jsonl').write_text(json.dumps(record) + '\n', encoding='utf-8')
    assert path.read_bytes() != (train / 'images/a.png').read_bytes()
    with pytest.raises(DetectionDataError, match='disjoint'):
        validate_train_validation_pair(RealDetectionDataset(train), RealDetectionDataset(val))


@pytest.mark.parametrize('document', [[], None, 'not metadata'])
def test_metadata_must_be_an_object(tmp_path, document):
    from plateai_trainer.detection.real_dataset import RealDetectionDataset
    from plateai_trainer.detection.dataset import DetectionDataError
    fixture(tmp_path)
    (tmp_path / 'dataset.json').write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(DetectionDataError):
        RealDetectionDataset(tmp_path)
