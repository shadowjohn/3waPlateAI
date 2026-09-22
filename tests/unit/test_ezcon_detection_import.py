import cv2
import numpy as np
import pytest

from plateai_trainer.detection.dataset import DetectionDataError


def row():
    rgb = np.random.default_rng(1).integers(0, 256, (100, 200, 3), dtype=np.uint8)
    payload = cv2.imencode('.png', rgb)[1].tobytes()
    return {'image': {'bytes': payload, 'path': 'untrusted/elsewhere.jpg'},
            'label': [0, 0], 'xyxyxyxyn': [
                [[.1, .1], [.6, .1], [.6, .4], [.1, .4]],
                [[.2, .6], [1., .6], [1., 1.], [.2, 1.]]]}


def test_import_keeps_multi_plate_geometry_and_pixel_endpoint():
    from tools.fetch_ezcon_detection import decode_row
    payload, rgb, corners = decode_row(row())
    assert rgb.shape == (100, 200, 3)
    assert len(corners) == 2
    np.testing.assert_allclose(corners[0], [[20, 10], [120, 10], [120, 40], [20, 40]], atol=1e-5)
    np.testing.assert_allclose(corners[1], [[40, 60], [199, 60], [199, 99], [40, 99]], atol=1e-5)
    assert payload == row()['image']['bytes']


@pytest.mark.parametrize('bad', ['class', 'count', 'nan', 'bounds', 'duplicate'])
def test_bad_instance_rejects_whole_photo_not_just_one_label(bad):
    from tools.fetch_ezcon_detection import decode_row
    value = row()
    if bad == 'class': value['label'][0] = 1
    if bad == 'count': value['label'].pop()
    if bad == 'nan': value['xyxyxyxyn'][0][0][0] = float('nan')
    if bad == 'bounds': value['xyxyxyxyn'][0][0][0] = -0.1
    if bad == 'duplicate': value['xyxyxyxyn'][0][0] = value['xyxyxyxyn'][0][1]
    with pytest.raises(DetectionDataError): decode_row(value)


def test_holdout_index_detects_identical_pixels_and_jpeg_reencoding():
    from tools.fetch_ezcon_detection import HoldoutIndex
    rgb = np.random.default_rng(4).integers(0, 256, (640, 640, 3), dtype=np.uint8)
    rgb = cv2.GaussianBlur(rgb, (7, 7), 1.5)
    index = HoldoutIndex()
    index.add(rgb, 'held-out')
    assert index.match(rgb) == ('held-out', 'exact_pixels')
    recoded = cv2.imdecode(cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])[1], 1)
    assert index.match(recoded) == ('held-out', 'near_duplicate_phash')
    assert index.match(np.random.default_rng(8).integers(0, 256, rgb.shape, dtype=np.uint8)) is None


def test_import_preserves_official_splits_without_holdout_leakage(tmp_path):
    from tools.fetch_ezcon_detection import import_rows
    a = row()
    b = row()
    b['image']['bytes'] = cv2.imencode('.png', np.random.default_rng(9).integers(0, 256, (100, 200, 3), dtype=np.uint8))[1].tobytes()
    c = row()
    c['image']['bytes'] = cv2.imencode('.png', np.random.default_rng(10).integers(0, 256, (100, 200, 3), dtype=np.uint8))[1].tobytes()
    d = row()
    d['image']['bytes'] = cv2.imencode('.png', np.random.default_rng(11).integers(0, 256, (100, 200, 3), dtype=np.uint8))[1].tobytes()
    holdout = tmp_path / 'holdout'
    holdout.mkdir()
    (holdout / 'a.png').write_bytes(a['image']['bytes'])
    output = tmp_path / 'prepared'
    report = import_rows({'train': [a, b, c, d], 'validation': [c], 'test': [b]}, output, [holdout])
    assert report['retained'] == {'test': 1, 'validation': 1, 'train': 1}
    assert len(report['excluded']) == 3
    from plateai_trainer.detection.real_dataset import RealDetectionDataset
    train = RealDetectionDataset(output / 'train')
    assert train.provenance['local_only'] is True
    assert train.provenance['license_reviewed'] is False
    assert train._records[0]['source_index'] == 3
    with pytest.raises(FileExistsError):
        import_rows({'train': [d], 'validation': [c], 'test': [b]}, output, [holdout])


def test_fetch_requires_explicit_unreviewed_license_ack(tmp_path):
    from tools.fetch_ezcon_detection import main
    with pytest.raises(SystemExit):
        main(['--output', str(tmp_path / 'data'), '--cache', str(tmp_path / 'cache'),
              '--holdout-images', str(tmp_path)])
    assert not (tmp_path / 'cache').exists()
