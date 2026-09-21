from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from plateai_shared.schema_validation import (
    DocumentValidationError,
    validate_document,
    validate_model_manifest,
)
from plateai_shared.rules import load_character_set, load_ruleset
from plateai_trainer.synthetic.dataset import generate_dataset

from tests.conftest import (
    DEFAULT_AUGMENTATION,
    DEFAULT_RULES,
    DEFAULT_TEMPLATE,
    NONE_AUGMENTATION,
    ROOT,
    DEFAULT_CHARSET,
    write_json,
)


SCHEMAS = ROOT / "schemas"


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_generation_record(tmp_path, default_request) -> dict[str, object]:
    output = tmp_path / "schema-record"
    generate_dataset(
        replace(
            default_request,
            count=1,
            output=output,
            augmentation_path=NONE_AUGMENTATION,
        )
    )
    first_line = (output / "metadata.jsonl").read_text(encoding="utf-8").splitlines()[0]
    return json.loads(first_line)


def valid_crop_only_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract_version": "1.0",
        "model_id": "twplate-dev-rec-v001",
        "version": "0.1.0",
        "created_at": "2026-09-21T00:00:00Z",
        "capabilities": ["crop-recognition"],
        "plate_size": [320, 96],
        "charset": {
            "file": "charset.txt",
            "sha256": "0" * 64,
            "visible_symbols": 36,
        },
        "rules": {"file": "plate_rules.json", "sha256": "1" * 64},
        "components": {
            "recognizer": {
                "file": "recognizer.onnx",
                "format": "onnx",
                "sha256": "2" * 64,
                "inputs": [
                    {
                        "name": "image",
                        "dtype": "float32",
                        "shape": ["batch", 3, 96, 320],
                    }
                ],
                "outputs": [
                    {
                        "name": "logits",
                        "dtype": "float32",
                        "shape": ["batch", "time", 37],
                    }
                ],
                "batch": {"mode": "dynamic", "min": 1, "opt": 8, "max": 32},
            }
        },
        "decoder": {
            "type": "ctc",
            "blank_index": 0,
            "class_count": 37,
            "index_mapping": "charset-order-skipping-blank",
            "collapse_repeats": True,
        },
        "preprocess": {
            "color_order": "RGB",
            "dtype": "float32",
            "scale": 0.00392156862745098,
        },
        "provenance": {"training_data": "synthetic", "license_reviewed": True},
    }


@pytest.mark.parametrize(
    ("document_path", "schema_path"),
    [
        (DEFAULT_RULES, SCHEMAS / "plate_rules.schema.json"),
        (DEFAULT_TEMPLATE, SCHEMAS / "plate_template.schema.json"),
        (DEFAULT_AUGMENTATION, SCHEMAS / "augmentation_profile.schema.json"),
    ],
)
def test_default_documents_match_published_schemas(document_path, schema_path):
    validate_document(read_json(document_path), schema_path)


def test_disabled_rule_may_use_zero_weight_in_loader_and_schema(tmp_path):
    document = read_json(DEFAULT_RULES)
    document["rules"][0]["enabled"] = False
    document["rules"][0]["weight"] = 0
    path = write_json(tmp_path / "rules.json", document)
    load_ruleset(path, load_character_set(DEFAULT_CHARSET))
    validate_document(document, SCHEMAS / "plate_rules.schema.json")


def test_model_manifest_rejects_path_traversal():
    manifest = valid_crop_only_manifest()
    manifest["components"]["recognizer"]["file"] = "../recognizer.onnx"
    with pytest.raises(DocumentValidationError, match="components.recognizer.file"):
        validate_document(manifest, SCHEMAS / "model_manifest.schema.json")


def test_ctc_manifest_requires_explicit_blank_mapping():
    manifest = valid_crop_only_manifest()
    del manifest["decoder"]["blank_index"]
    with pytest.raises(DocumentValidationError, match="decoder.blank_index"):
        validate_model_manifest(
            manifest,
            SCHEMAS / "model_manifest.schema.json",
            visible_charset_symbol_count=36,
        )


@pytest.mark.parametrize(
    "batch",
    [
        {"mode": "dynamic", "min": 8, "opt": 4, "max": 2},
        {"mode": "fixed", "size": 8},
    ],
)
def test_manifest_rejects_invalid_batch_contract(batch):
    manifest = valid_crop_only_manifest()
    manifest["components"]["recognizer"]["batch"] = batch
    with pytest.raises(DocumentValidationError, match="batch"):
        validate_model_manifest(
            manifest,
            SCHEMAS / "model_manifest.schema.json",
            visible_charset_symbol_count=36,
        )


def test_manifest_rejects_class_count_that_disagrees_with_charset():
    manifest = valid_crop_only_manifest()
    manifest["decoder"]["class_count"] = 38
    with pytest.raises(DocumentValidationError, match="class_count"):
        validate_model_manifest(
            manifest,
            SCHEMAS / "model_manifest.schema.json",
            visible_charset_symbol_count=36,
        )


def test_plate_detection_capability_requires_rectifier_contract():
    manifest = valid_crop_only_manifest()
    manifest["capabilities"] = ["crop-recognition", "plate-detection"]
    with pytest.raises(DocumentValidationError, match="detector|rectifier"):
        validate_document(manifest, SCHEMAS / "model_manifest.schema.json")


def test_full_manifest_pins_detector_keypoints_and_corner_normalization():
    manifest = valid_crop_only_manifest()
    manifest["capabilities"] = ["crop-recognition", "plate-detection"]
    manifest["components"]["detector"] = {
        "file": "detector.onnx",
        "format": "onnx",
        "sha256": "3" * 64,
        "inputs": [
            {"name": "image", "dtype": "float32", "shape": ["batch", 3, 640, 640]}
        ],
        "outputs": [
            {"name": "poses", "dtype": "float32", "shape": ["batch", "detections", 12]}
        ],
        "keypoints": ["left_top", "right_top", "right_bottom", "left_bottom"],
    }
    manifest["rectifier"] = {
        "normalization_strategy": "convex-hull-semantic-v1"
    }
    validate_model_manifest(
        manifest,
        SCHEMAS / "model_manifest.schema.json",
        visible_charset_symbol_count=36,
    )


def test_valid_fixed_batch_contract_is_accepted():
    manifest = valid_crop_only_manifest()
    manifest["components"]["recognizer"]["batch"] = {
        "mode": "fixed",
        "size": 8,
        "padding": "neutral-image",
        "discard_padded_outputs": True,
    }
    validate_model_manifest(
        manifest,
        SCHEMAS / "model_manifest.schema.json",
        visible_charset_symbol_count=36,
    )


def test_metadata_requires_exactly_four_ordered_corners(tmp_path, default_request):
    metadata = valid_generation_record(tmp_path, default_request)
    metadata["corners"] = [[0, 0], [1, 0], [1, 1]]
    with pytest.raises(DocumentValidationError, match="corners"):
        validate_document(metadata, SCHEMAS / "generation_metadata.schema.json")


def test_metadata_filename_contract_scales_beyond_six_digit_indices(
    tmp_path, default_request
):
    metadata = valid_generation_record(tmp_path, default_request)
    metadata["index"] = 1_000_000
    metadata["image_path"] = "images/1000000.png"
    validate_document(metadata, SCHEMAS / "generation_metadata.schema.json")


def test_non_finite_numbers_are_rejected_before_schema_evaluation():
    manifest = valid_crop_only_manifest()
    manifest["preprocess"]["scale"] = float("nan")
    with pytest.raises(DocumentValidationError, match="preprocess.scale"):
        validate_document(manifest, SCHEMAS / "model_manifest.schema.json")
