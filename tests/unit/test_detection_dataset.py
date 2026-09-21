from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from plateai_trainer.detection.composite import generate_composite_dataset
from plateai_trainer.detection.contracts import CompositeGenerationRequest
from plateai_trainer.detection.dataset import (
    DetectionDataError, DetectionDataset, validate_train_validation_pair,
)


def make_dataset(root, name="train", color=20, seed=7):
    background = root / f"{name}.png"
    Image.new("RGB", (800, 401), (color, 50, 80)).save(background)
    manifest = root / f"{name}.json"
    manifest.write_text(json.dumps({"schema_version": 1, "backgrounds": [{
        "image_path": background.name,
        "sha256": hashlib.sha256(background.read_bytes()).hexdigest(),
    }]}), encoding="utf-8")
    output = root / name
    generate_composite_dataset(CompositeGenerationRequest(
        output, 1, seed, manifest, instances_per_image=(1, 1),
    ))
    return output


def rewrite_record(root, mutate):
    path = root / "metadata.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    mutate(record)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_letterboxes_pixels_bbox_and_all_semantic_corners(tmp_path):
    root = make_dataset(tmp_path)
    def fixed(record):
        item = record["instances"][0]
        item["corners"] = [[100., 50.], [479., 50.], [479., 209.], [100., 209.]]
        item["bbox_xyxy"] = [100., 50., 479., 209.]
        item["transform"]["homography"] = [[1., 0., 100.], [0., 1., 50.], [0., 0., 1.]]
    rewrite_record(root, fixed)
    sample = DetectionDataset(root)[0]
    assert sample.image.shape == (3, 640, 640)
    assert sample.image.dtype == np.float32
    np.testing.assert_allclose(sample.image[:, 0, 0], [114 / 255] * 3)
    # 800x401 -> 640x321, top padding 159, affine scale exactly 0.8.
    np.testing.assert_allclose(sample.instances[0].bbox_xyxy, [80, 199, 383.2, 326.2])
    np.testing.assert_allclose(sample.instances[0].corners_xy,
                               [[80, 199], [383.2, 199], [383.2, 326.2], [80, 326.2]])
    assert len(sample.targets.positive_indices) > 0


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(schema_version=2),
    lambda r: r.update(index=2),
    lambda r: r.update(image_path="../outside.png"),
    lambda r: r.update(image_sha256="0" * 64),
    lambda r: r["background"].update(width=799),
    lambda r: r["background"].update(sha256="0" * 64),
    lambda r: r.update(instances=[]),
    lambda r: r.update(instances=r["instances"] * 4),
    lambda r: r["instances"][0]["corners"][0].__setitem__(0, 800),
    lambda r: r["instances"][0]["corners"].reverse(),
    lambda r: r["instances"][0]["corners"].append([1, 2]),
    lambda r: r["instances"][0]["bbox_xyxy"].__setitem__(0, 0),
    lambda r: r["instances"][0]["source_plate"].update(template_id="other"),
    lambda r: r["instances"][0]["source_plate"].update(canonical="AAA4444"),
    lambda r: r["instances"][0]["transform"]["homography"][0].__setitem__(0, float("nan")),
])
def test_rejects_tampered_metadata(tmp_path, mutation):
    root = make_dataset(tmp_path)
    rewrite_record(root, mutation)
    with pytest.raises(DetectionDataError):
        DetectionDataset(root)


@pytest.mark.parametrize("filename,field,value", [
    ("generation_config.json", "count", 2),
    ("generation_config.json", "schema_version", True),
    ("generation_config.json", "config_sha256", {}),
    ("summary.json", "generated", 2),
    ("summary.json", "background_sha256s", []),
])
def test_rejects_tampered_provenance(tmp_path, filename, field, value):
    root = make_dataset(tmp_path)
    path = root / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DetectionDataError):
        DetectionDataset(root)


def test_rechecks_image_hash_on_access(tmp_path):
    root = make_dataset(tmp_path)
    dataset = DetectionDataset(root)
    Image.new("RGB", (800, 401)).save(root / "images/000000.png")
    with pytest.raises(DetectionDataError, match="SHA-256"):
        dataset[0]


def test_rejects_non_rgb_even_with_correct_hash(tmp_path):
    root = make_dataset(tmp_path)
    path = root / "images/000000.png"
    Image.new("L", (800, 401)).save(path)
    rewrite_record(root, lambda r: r.update(image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    with pytest.raises(DetectionDataError, match="RGB"):
        DetectionDataset(root)


def test_rejects_unlisted_images(tmp_path):
    root = make_dataset(tmp_path)
    Image.new("RGB", (10, 10)).save(root / "images/unlisted.png")
    with pytest.raises(DetectionDataError, match="unlisted"):
        DetectionDataset(root)


def test_rejects_overlapping_instances_even_when_metadata_is_otherwise_valid(tmp_path):
    root = make_dataset(tmp_path)
    path = root / "generation_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["instances_per_image"] = [1, 3]
    path.write_text(json.dumps(config), encoding="utf-8")
    rewrite_record(root, lambda r: r.update(instances=r["instances"] * 2))
    with pytest.raises(DetectionDataError, match="overlap"):
        DetectionDataset(root)


def test_shared_background_rejected_even_for_different_generation_seeds(tmp_path):
    train = DetectionDataset(make_dataset(tmp_path))
    validation = DetectionDataset(make_dataset(tmp_path, "validation", seed=8))
    with pytest.raises(DetectionDataError, match="background SHA-256"):
        validate_train_validation_pair(train, validation)
    distinct = DetectionDataset(make_dataset(tmp_path, "distinct", color=21))
    validate_train_validation_pair(train, distinct)
