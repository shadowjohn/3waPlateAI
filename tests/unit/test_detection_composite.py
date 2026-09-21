from __future__ import annotations

import hashlib
import json
import shutil
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
from plateai_trainer.detection import composite as composite_module
from plateai_trainer.detection.composite import InvalidCompositeRequest

from tests.conftest import ROOT, V1_CHARSET, V1_RULES, V1_TEMPLATE


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
        assert source.tolist() == [
            [0.0, 0.0],
            [379.0, 0.0],
            [379.0, 159.0],
            [0.0, 159.0],
        ]
        transformed = cv2.perspectiveTransform(source[None, :, :], homography)[0]
        np.testing.assert_allclose(transformed, item["corners"], atol=1e-4)
        corners = np.asarray(item["corners"], dtype=np.float32)
        assert np.all(corners[:, 0] >= 0.0)
        assert np.all(corners[:, 0] <= record["background"]["width"] - 1)
        assert np.all(corners[:, 1] >= 0.0)
        assert np.all(corners[:, 1] <= record["background"]["height"] - 1)
        assert item["bbox_xyxy"] == [
            float(np.min(corners[:, 0])),
            float(np.min(corners[:, 1])),
            float(np.max(corners[:, 0])),
            float(np.max(corners[:, 1])),
        ]
        assert type(item["source_plate"]["plate_seed"]) is int
        assert item["source_plate"]["plate_seed"] >= 0


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


def test_installed_layout_generation_resolves_packaged_schemas(
    tmp_path, monkeypatch, background_manifest
):
    checkout = tmp_path / "checkout-without-data"
    installed = tmp_path / "installed"
    packaged_files = {
        "configs/detection/composite_v1.json": ROOT
        / "configs/detection/composite_v1.json",
        "configs/charsets/tw_new_style_private_passenger_v1.txt": V1_CHARSET,
        "configs/plate_rules/tw_new_style_private_passenger_v1.json": V1_RULES,
        "configs/plate_templates/new_style_private_passenger_white_v1.json": V1_TEMPLATE,
        "schemas/detection_background_manifest.schema.json": ROOT
        / "schemas/detection_background_manifest.schema.json",
        "schemas/detection_metadata.schema.json": ROOT
        / "schemas/detection_metadata.schema.json",
    }
    for relative_path, source in packaged_files.items():
        destination = installed / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    monkeypatch.setattr(contracts_module, "_REPOSITORY_ROOT", checkout)
    monkeypatch.setattr(contracts_module, "_INSTALL_DATA_ROOT", installed)
    monkeypatch.setattr(
        composite_module,
        "_BACKGROUND_SCHEMA",
        checkout / "schemas/detection_background_manifest.schema.json",
        raising=False,
    )
    monkeypatch.setattr(
        composite_module,
        "_METADATA_SCHEMA",
        checkout / "schemas/detection_metadata.schema.json",
        raising=False,
    )

    output = tmp_path / "installed-layout-output"
    generate_composite_dataset(
        CompositeGenerationRequest(
            output=output,
            count=1,
            seed=5,
            background_manifest=background_manifest,
        )
    )

    assert json.loads((output / "metadata.jsonl").read_text(encoding="utf-8"))[
        "schema_version"
    ] == 1


def test_composite_rejects_non_v1_template(tmp_path, background_manifest):
    template = json.loads(V1_TEMPLATE.read_text(encoding="utf-8"))
    template.update({"id": "not-m1-v1", "width": 320, "height": 96})
    template["text_box"] = [20, 20, 300, 76]
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")

    with pytest.raises(InvalidCompositeRequest, match="380x160|v1 template"):
        generate_composite_dataset(
            CompositeGenerationRequest(
                output=tmp_path / "wrong-template-output",
                count=1,
                seed=5,
                background_manifest=background_manifest,
                template_path=template_path,
            )
        )


@pytest.mark.parametrize(
    "rooted_path",
    [r"\rooted.png", r"C:drive-relative.png", r"C:\absolute.png", r"\\server\share\bg.png"],
)
def test_runtime_rejects_windows_rooted_background_paths(
    tmp_path, background_manifest, monkeypatch, rooted_path
):
    document = json.loads(background_manifest.read_text(encoding="utf-8"))
    document["backgrounds"][0]["image_path"] = rooted_path
    background_manifest.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(composite_module, "validate_document", lambda *_args: None)

    with pytest.raises(InvalidCompositeRequest, match="relative|root|anchor|drive"):
        generate_composite_dataset(
            CompositeGenerationRequest(
                output=tmp_path / "rooted-output",
                count=1,
                seed=5,
                background_manifest=background_manifest,
            )
        )


def test_input_change_during_generation_is_not_published(
    tmp_path, background_manifest, monkeypatch
):
    config_path = tmp_path / "composite.json"
    config_path.write_bytes((ROOT / "configs/detection/composite_v1.json").read_bytes())
    real_compose = composite_module._compose_one_image

    def mutate_config_then_compose(*args, **kwargs):
        record = real_compose(*args, **kwargs)
        config_path.write_text("{}", encoding="utf-8")
        return record

    monkeypatch.setattr(composite_module, "_compose_one_image", mutate_config_then_compose)
    output = tmp_path / "changed-input-output"

    with pytest.raises(InvalidCompositeRequest, match="changed during generation"):
        generate_composite_dataset(
            CompositeGenerationRequest(
                output=output,
                count=1,
                seed=5,
                background_manifest=background_manifest,
                config_path=config_path,
            )
        )

    assert not output.exists()
    assert list(tmp_path.glob(".changed-input-output.partial-*")) == []


def test_rules_parser_is_bound_to_snapshot_during_aba_mutation(
    tmp_path, background_manifest, monkeypatch
):
    rules_path = tmp_path / "rules.json"
    original_bytes = V1_RULES.read_bytes()
    rules_path.write_bytes(original_bytes)
    mutant = json.loads(original_bytes)
    mutant["id"] = "aba-mutant"
    mutant["rules"] = [
        {
            "id": "aba-all-a",
            "tokens": [{"literal": "A"}] * 7,
            "separator": "-",
            "separator_after": [3],
            "plate_type": "new-style-private-passenger",
            "weight": 1.0,
            "enabled": True,
        }
    ]
    mutant_bytes = json.dumps(mutant).encode("utf-8")
    real_load_ruleset = composite_module.load_ruleset

    def aba_load_ruleset(path, charset):
        rules_path.write_bytes(mutant_bytes)
        try:
            return real_load_ruleset(path, charset)
        finally:
            rules_path.write_bytes(original_bytes)

    monkeypatch.setattr(composite_module, "load_ruleset", aba_load_ruleset)
    output = tmp_path / "aba-output"
    generate_composite_dataset(
        CompositeGenerationRequest(
            output=output,
            count=1,
            seed=5,
            background_manifest=background_manifest,
            instances_per_image=(1, 1),
            rules_path=rules_path,
        )
    )

    record = json.loads((output / "metadata.jsonl").read_text(encoding="utf-8"))
    provenance = json.loads(
        (output / "generation_config.json").read_text(encoding="utf-8")
    )
    assert record["instances"][0]["source_plate"]["canonical"] != "AAAAAAA"
    assert provenance["config_sha256"]["rules"] == hashlib.sha256(
        original_bytes
    ).hexdigest()
    assert rules_path.read_bytes() == original_bytes
