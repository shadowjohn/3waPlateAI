from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from plateai_shared.publication import OutputExistsError
from plateai_trainer.detection import (
    CompositeGenerationRequest,
    generate_composite_dataset,
)
from plateai_trainer.detection import contracts as contracts_module


@pytest.fixture
def background_manifest(tmp_path):
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    yy, xx = np.indices((480, 720), dtype=np.uint16)
    image_rgb = np.stack(
        (
            (xx % 256).astype(np.uint8),
            (yy % 256).astype(np.uint8),
            ((xx + yy) % 256).astype(np.uint8),
        ),
        axis=2,
    )
    ok, encoded = cv2.imencode(
        ".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    )
    assert ok
    image_path = backgrounds / "toy.png"
    image_path.write_bytes(encoded.tobytes())
    manifest = {
        "schema_version": 1,
        "backgrounds": [
            {
                "image_path": "backgrounds/toy.png",
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_path = tmp_path / "background-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _bbox_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection == 0.0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def max_pairwise_bbox_iou(instances):
    return max(
        (
            _bbox_iou(first["bbox_xyxy"], second["bbox_xyxy"])
            for index, first in enumerate(instances)
            for second in instances[index + 1 :]
        ),
        default=0.0,
    )


def is_semantic_quad(corners):
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        return False
    cross_products = []
    for index in range(4):
        first = points[(index + 1) % 4] - points[index]
        second = points[(index + 2) % 4] - points[(index + 1) % 4]
        cross_products.append(float(first[0] * second[1] - first[1] * second[0]))
    return all(value > 0.0 for value in cross_products)


@pytest.mark.parametrize("instances", [1, 2, 3])
def test_composite_records_exact_semantic_instances(
    tmp_path, background_manifest, instances
):
    output = tmp_path / f"composite-{instances}"
    summary = generate_composite_dataset(
        CompositeGenerationRequest(
            output=output,
            count=1,
            seed=7,
            background_manifest=background_manifest,
            instances_per_image=(instances, instances),
        )
    )
    record = json.loads((output / "metadata.jsonl").read_text(encoding="utf-8"))

    assert summary.generated == 1
    assert len(record["instances"]) == instances
    assert max_pairwise_bbox_iou(record["instances"]) == 0.0
    assert all(is_semantic_quad(item["corners"]) for item in record["instances"])
    for item in record["instances"]:
        homography = np.asarray(item["transform"]["homography"], dtype=np.float32)
        source = np.asarray(item["source_plate"]["corners"], dtype=np.float32)
        transformed = cv2.perspectiveTransform(source[None, :, :], homography)[0]
        np.testing.assert_allclose(transformed, item["corners"], atol=1e-4)


def test_composite_rejects_existing_output_without_deleting_it(
    tmp_path, background_manifest
):
    output = tmp_path / "exists"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(OutputExistsError):
        generate_composite_dataset(
            CompositeGenerationRequest(
                output=output,
                count=1,
                seed=1,
                background_manifest=background_manifest,
            )
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_same_request_writes_byte_identical_dataset(tmp_path, background_manifest):
    request = CompositeGenerationRequest(
        output=tmp_path / "first",
        count=3,
        seed=19,
        background_manifest=background_manifest,
    )
    generate_composite_dataset(request)
    generate_composite_dataset(replace(request, output=tmp_path / "second"))

    first_files = {
        path.relative_to(request.output): path.read_bytes()
        for path in request.output.rglob("*")
        if path.is_file()
    }
    second = tmp_path / "second"
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


def test_background_hash_mismatch_leaves_no_output_or_staging(
    tmp_path, background_manifest
):
    document = json.loads(background_manifest.read_text(encoding="utf-8"))
    document["backgrounds"][0]["sha256"] = "0" * 64
    background_manifest.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "dataset"

    with pytest.raises(ValueError, match="SHA-256"):
        generate_composite_dataset(
            CompositeGenerationRequest(
                output=output,
                count=1,
                seed=1,
                background_manifest=background_manifest,
            )
        )

    assert not output.exists()
    assert list(tmp_path.glob(".dataset.partial-*")) == []


def test_request_defaults_fall_back_to_installed_data_files(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    installed = tmp_path / "installed"
    relative_paths = (
        "configs/detection/composite_v1.json",
        "configs/charsets/tw_new_style_private_passenger_v1.txt",
        "configs/plate_rules/tw_new_style_private_passenger_v1.json",
        "configs/plate_templates/new_style_private_passenger_white_v1.json",
    )
    for relative_path in relative_paths:
        path = installed / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("installed", encoding="utf-8")
    monkeypatch.setattr(contracts_module, "_REPOSITORY_ROOT", checkout)
    monkeypatch.setattr(contracts_module, "_INSTALL_DATA_ROOT", installed)

    request = CompositeGenerationRequest(
        output=tmp_path / "out",
        count=1,
        seed=1,
        background_manifest=tmp_path / "manifest.json",
    )

    assert request.config_path == installed / relative_paths[0]
    assert request.charset_path == installed / relative_paths[1]
    assert request.rules_path == installed / relative_paths[2]
    assert request.template_path == installed / relative_paths[3]
