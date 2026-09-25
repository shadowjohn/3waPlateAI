from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


def pose_data():
    assert importlib.util.find_spec('plateai_web.pose_data') is not None, 'Pose annotation workflow is missing'
    return importlib.import_module('plateai_web.pose_data')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture_sources(tmp_path):
    source = {'dataset_id': 'EZCon/taiwan-license-plate-detection', 'revision': 'test-revision'}
    review = tmp_path / 'review'
    review.mkdir()
    write_json(review / 'review_set.json', {'source': {'dataset': source}})
    manifests, originals = [], []
    for i in (1, 2):
        image = review / f'images/{i}.png'
        image.parent.mkdir(exist_ok=True)
        Image.new('RGB', (100, 60), (10 * i, 30, 50)).save(image)
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        corners = [[[10, 10], [90, 10], [90, 50], [10, 50]]]
        label = {'image_sha256': digest, 'validation_index': i, 'image_size_wh': [100, 60],
                 'reviewed_corners': corners, 'original_corners': [[[0, 0]]], 'secret_path': 'C:/private/a.jpg'}
        write_json(review / f'labels/{i}.json', label)
        manifests.append({'validation_index': i, 'review_status': 'accepted', 'image_sha256': digest,
                          'label_path': f'labels/{i}.json', 'image_path': f'images/{i}.png',
                          'reviewer_note': 'private-note'})
        originals.append({'source_index': i, 'image_path': f'images/{i}.png', 'image_sha256': digest,
                          'width': 100, 'height': 60, 'corners': corners})
    (review / 'manifest.jsonl').write_text('\n'.join(map(json.dumps, manifests)), encoding='utf-8')
    (review / 'metadata.jsonl').write_text('\n'.join(map(json.dumps, originals)), encoding='utf-8')
    write_json(review / 'dataset.json', {'source': source, 'count': 2, 'split': 'validation'})
    split = tmp_path / 'split.json'
    write_json(split, {'seed': 20260923, 'train': [1], 'test': [2]})
    base = tmp_path / 'base'
    image = base / 'images/0.png'
    image.parent.mkdir(parents=True)
    Image.new('RGB', (100, 60), (80, 20, 30)).save(image)
    base_row = dict(originals[0], source_index=0, image_path='images/0.png', image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    (base / 'metadata.jsonl').write_text(json.dumps(base_row), encoding='utf-8')
    write_json(base / 'dataset.json', {'source': source, 'count': 1, 'split': 'train'})
    return review, split, base


def test_export_is_coordinates_only_and_preserves_split(tmp_path):
    review, split, _ = fixture_sources(tmp_path)
    output = tmp_path / 'public.json'
    pose_data().export_annotations(review, split, output)
    doc = json.loads(output.read_text(encoding='utf-8'))
    assert doc['corner_order'] == ['left_top', 'right_top', 'right_bottom', 'left_bottom']
    assert [row['split'] for row in doc['records']] == ['train', 'holdout']
    assert set(doc['records'][0]) == {'validation_index', 'image_sha256', 'width', 'height', 'corners', 'split'}
    assert doc['records'][0]['corners'] == [[[10, 10], [90, 10], [90, 50], [10, 50]]]
    assert not any(word in output.read_text() for word in ('private-note', 'original_corners', 'image_path', 'base64', 'secret_path'))
    with pytest.raises(FileExistsError):
        pose_data().export_annotations(review, split, output)


def test_export_rejects_overlap_and_unreviewed_entries(tmp_path):
    review, split, _ = fixture_sources(tmp_path)
    write_json(split, {'seed': 20260923, 'train': [1, 2], 'test': [2]})
    with pytest.raises(ValueError, match='split'):
        pose_data().export_annotations(review, split, tmp_path / 'public.json')
    write_json(split, {'seed': 20260923, 'train': [1], 'test': [2]})
    text = (review / 'manifest.jsonl').read_text().replace('accepted', 'pending')
    (review / 'manifest.jsonl').write_text(text)
    with pytest.raises(ValueError, match='review'):
        pose_data().export_annotations(review, split, tmp_path / 'public.json')


def test_prepare_keeps_holdout_out_of_training_and_converts_keypoints(tmp_path):
    review, split, base = fixture_sources(tmp_path)
    annotations = tmp_path / 'public.json'
    data = pose_data()
    data.export_annotations(review, split, annotations)
    result = data.prepare_dataset(base, review, annotations, tmp_path / 'prepared')
    assert result['train_count'] == 2
    assert result['holdout_count'] == 1
    assert len(list((tmp_path / 'prepared/images/train').glob('*'))) == 2
    assert len(list((tmp_path / 'prepared/images/holdout').glob('*'))) == 1
    line = (tmp_path / 'prepared/labels/train/clean-000001.txt').read_text().split()
    assert len(line) == 17  # class, xywh, four xy/visibility triplets
    assert list(map(float, line[1:5])) == pytest.approx([.5, .5, .8, 2/3])
    assert list(map(float, line[5:8])) == pytest.approx([.1, 1/6, 2])
    assert not (tmp_path / 'prepared/labels/train/clean-000002.txt').exists()


def test_prepare_rejects_changed_image(tmp_path):
    review, split, base = fixture_sources(tmp_path)
    annotations = tmp_path / 'public.json'
    data = pose_data()
    data.export_annotations(review, split, annotations)
    (review / 'images/2.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        data.prepare_dataset(base, review, annotations, tmp_path / 'prepared')


def test_annotation_reader_rejects_hidden_payload_and_invalid_quad(tmp_path):
    review, split, _ = fixture_sources(tmp_path)
    output = tmp_path / 'public.json'
    data = pose_data()
    data.export_annotations(review, split, output)
    doc = json.loads(output.read_text())
    doc['records'][0]['image_base64'] = 'private'
    write_json(output, doc)
    with pytest.raises(ValueError, match='fields'):
        data.load_annotations(output)
    del doc['records'][0]['image_base64']
    doc['records'][0]['corners'][0][1] = [101, 10]
    write_json(output, doc)
    with pytest.raises(ValueError):
        data.load_annotations(output)


def test_prepare_rejects_cross_split_reencoded_pixels(tmp_path):
    review, split, base = fixture_sources(tmp_path)
    annotations = tmp_path / 'public.json'
    data = pose_data()
    data.export_annotations(review, split, annotations)
    with Image.open(review / 'images/2.png') as image:
        image.save(base / 'images/0.png', compress_level=0)
    row = json.loads((base / 'metadata.jsonl').read_text())
    row['image_sha256'] = data.sha256(base / 'images/0.png')
    (base / 'metadata.jsonl').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='duplicate image'):
        data.prepare_dataset(base, review, annotations, tmp_path / 'prepared')


def test_prepare_rejects_whole_base_image_if_one_quad_degenerate(tmp_path):
    review, split, base = fixture_sources(tmp_path)
    data = pose_data()
    annotations = tmp_path / 'public.json'
    data.export_annotations(review, split, annotations)
    row = json.loads((base / 'metadata.jsonl').read_text())
    row['corners'].append([[10, 10], [11, 10], [11, 10.1], [10, 10.1]])
    (base / 'metadata.jsonl').write_text(json.dumps(row))
    result = data.prepare_dataset(base, review, annotations, tmp_path / 'prepared')
    assert result['train_count'] == 1
    assert len(result['rejected']) == 1


def test_public_annotation_asset_contract():
    root = Path(__file__).resolve().parents[2]
    doc = pose_data().load_annotations(root / 'annotations/pose-clean119-v1.json')
    assert len(doc['records']) == 119
    assert sum(r['split'] == 'train' for r in doc['records']) == 83
    assert doc['split_seed'] == 20260923
    index = json.loads((root / 'annotations/pose-pilot-inputs-v1.json').read_text())
    assert set(index) == {'schema_version', 'source', 'train_images'}
    assert index['source'] == doc['source']
    assert len(index['train_images']) == 2000
    assert all(set(row) == {'source_index', 'image_sha256', 'original_corners_sha256'} for row in index['train_images'])
