import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def tool():
    path = Path(__file__).resolve().parents[2] / 'tools/prepare_pose_inputs.py'
    assert path.is_file(), 'Portable Pose preparation tool is missing'
    spec = importlib.util.spec_from_file_location('prepare_pose_inputs_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_input_index_contains_only_source_identity_not_images_or_labels(tmp_path):
    from test_pose_data import fixture_sources
    _, _, base = fixture_sources(tmp_path)
    result = tool().export_input_index(base, tmp_path / 'index.json')
    assert set(result) == {'schema_version', 'source', 'train_images'}
    assert set(result['train_images'][0]) == {'source_index', 'image_sha256', 'original_corners_sha256'}
    assert '"corners":' not in (tmp_path / 'index.json').read_text()


def test_restoration_requires_exact_bytes_and_publishes_no_holdout_as_train(tmp_path):
    from test_pose_data import fixture_sources
    from plateai_web.pose_data import export_annotations
    review, split, base = fixture_sources(tmp_path)
    annotations = tmp_path / 'annotations.json'
    export_annotations(review, split, annotations)
    module = tool()
    index = tmp_path / 'index.json'
    quad = [[.1, 1/6], [.9, 1/6], [.9, 5/6], [.1, 5/6]]
    def raw(path):
        return {'image': {'bytes': path.read_bytes()}, 'label': [0], 'xyxyxyxyn': [quad]}
    splits = {'train': [raw(base / 'images/0.png')],
              'validation': [raw(review / 'images/1.png'), raw(review / 'images/2.png')]}
    from fetch_ezcon_detection import decode_row
    row = json.loads((base / 'metadata.jsonl').read_text())
    row['corners'] = decode_row(splits['train'][0])[2]
    (base / 'metadata.jsonl').write_text(json.dumps(row))
    module.export_input_index(base, index)
    output = tmp_path / 'restored'
    module.restore_inputs(splits, index, annotations, output)
    rows = [json.loads(line) for line in (output / 'train/metadata.jsonl').read_text().splitlines()]
    assert len(rows) == 1
    assert hashlib.sha256((output / 'train' / rows[0]['image_path']).read_bytes()).hexdigest() == rows[0]['image_sha256']
    assert len((output / 'validation/metadata.jsonl').read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        module.restore_inputs(splits, index, annotations, output)
    splits['train'][0]['image']['bytes'] = b'wrong-file'
    with pytest.raises(ValueError, match='hash'):
        module.restore_inputs(splits, index, annotations, tmp_path / 'wrong')
