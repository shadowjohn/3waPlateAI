from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from plateai_shared.schema_validation import (
    DocumentValidationError,
    validate_model_manifest,
)
from tests.conftest import ROOT


SCHEMA = ROOT / "schemas/model_manifest.schema.json"


def valid_v1_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_version": "1.0",
        "model_id": "twplate-v1-recognizer",
        "version": "1.0.0",
        "created_at": "2026-09-21T00:00:00Z",
        "capabilities": ["crop-recognition"],
        "plate_size": [380, 160],
        "charset": {
            "file": "charset.txt",
            "sha256": "0" * 64,
            "visible_symbols": 33,
        },
        "rules": {"file": "plate_rules.json", "sha256": "1" * 64},
        "components": {
            "recognizer": {
                "file": "recognizer.onnx",
                "format": "onnx",
                "sha256": "2" * 64,
                "inputs": [
                    {
                        "name": "input",
                        "dtype": "float32",
                        "shape": ["batch", 1, 64, 160],
                    }
                ],
                "outputs": [
                    {
                        "name": "logits",
                        "dtype": "float32",
                        "shape": ["batch", 80, 34],
                    }
                ],
                "batch": {"mode": "dynamic", "min": 1, "opt": 8, "max": 32},
            }
        },
        "decoder": {
            "type": "ctc",
            "blank_index": 0,
            "class_count": 34,
            "index_mapping": "charset-order-skipping-blank",
            "collapse_repeats": True,
        },
        "preprocess": {
            "source_size_wh": [380, 160],
            "color_space": "grayscale",
            "layout": "NCHW",
            "input_size_hw": [64, 160],
            "grayscale": {
                "reference": "pillow-image-convert-l",
                "library_version": "12.3.0",
            },
            "resize": {
                "mode": "letterbox",
                "reference": "pillow-image-resize",
                "interpolation": "Resampling.BILINEAR",
                "library_version": "12.3.0",
                "resized_size_hw": [64, 152],
                "padding_ltrb": [4, 0, 4, 0],
                "padding_raw_value": 255,
            },
            "normalization": {
                "type": "divide",
                "divisor": 255.0,
                "dtype": "float32",
            },
        },
        "provenance": {
            "training_data": "synthetic",
            "license_reviewed": True,
            "training_report": {
                "file": "training_report.json",
                "sha256": "3" * 64,
            },
        },
    }


def set_path(document: dict[str, Any], path: tuple[str | int, ...], value: Any) -> None:
    target: Any = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def test_v1_crop_manifest_accepts_exact_preprocess_and_tensor_contract():
    validate_model_manifest(valid_v1_manifest(), SCHEMA, visible_charset_symbol_count=33)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("components", "recognizer", "inputs", 0, "shape"), ["batch", 3, 64, 160], "inputs"),
        (("components", "recognizer", "outputs", 0, "shape"), ["batch", 80, 33], "outputs"),
        (("preprocess", "resize", "padding_ltrb"), [3, 0, 5, 0], "padding_ltrb"),
    ],
)
def test_v1_manifest_rejects_incompatible_cross_fields(path, value, message):
    manifest = valid_v1_manifest()
    set_path(manifest, path, value)
    with pytest.raises(DocumentValidationError, match=message):
        validate_model_manifest(manifest, SCHEMA, visible_charset_symbol_count=33)


def test_v1_manifest_rejects_nonzero_blank_and_static_batch():
    nonzero_blank = valid_v1_manifest()
    nonzero_blank["decoder"]["blank_index"] = 1
    with pytest.raises(DocumentValidationError, match="blank_index"):
        validate_model_manifest(nonzero_blank, SCHEMA, visible_charset_symbol_count=33)

    fixed_batch = valid_v1_manifest()
    fixed_batch["components"]["recognizer"]["batch"] = {
        "mode": "fixed",
        "size": 8,
        "padding": "neutral-image",
        "discard_padded_outputs": True,
    }
    with pytest.raises(DocumentValidationError, match="batch"):
        validate_model_manifest(fixed_batch, SCHEMA, visible_charset_symbol_count=33)
