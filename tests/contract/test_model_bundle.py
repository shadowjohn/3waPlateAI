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


def valid_full_manifest() -> dict[str, Any]:
    manifest = valid_v1_manifest()
    manifest["capabilities"] = ["crop-recognition", "plate-detection"]
    manifest["components"]["detector"] = {
        "file": "detector.onnx", "format": "onnx", "sha256": "4" * 64,
        "inputs": [{"name": "images", "dtype": "float32", "shape": ["batch", 3, 640, 640]}],
        "outputs": [{"name": "candidates", "dtype": "float32", "shape": ["batch", 8400, 13]}],
        "batch": {"mode": "dynamic", "min": 1, "opt": 8, "max": 32},
        "keypoints": ["left_top", "right_top", "right_bottom", "left_bottom"],
        "postprocess": {
            "candidate_format": "cxcywh-confidence-corners-letterbox-px-v1",
            "preprocess": "opencv-rgb-letterbox-640-v1", "nms": "numpy-nms-v1",
            "score_threshold": 0.25, "iou_threshold": 0.50, "max_detections": 100,
        },
    }
    manifest["rectifier"] = {"normalization_strategy": "convex-hull-semantic-v1"}
    manifest["provenance"]["detector_training_report"] = {"file": "detector_report.json", "sha256": "5" * 64}
    return manifest


def test_full_manifest_accepts_exact_detector_contract():
    validate_model_manifest(valid_full_manifest(), SCHEMA, visible_charset_symbol_count=33)


@pytest.mark.parametrize(("path", "value", "message"), [
    (("components", "detector", "inputs", 0, "name"), "image", "detector.inputs"),
    (("components", "detector", "inputs", 0, "shape"), [1, 3, 640, 640], "detector.inputs"),
    (("components", "detector", "outputs", 0, "shape"), ["batch", 8401, 13], "detector.outputs"),
    (("components", "detector", "outputs", 0, "dtype"), "float16", "detector.outputs"),
    (("components", "detector", "batch"), {"mode": "fixed", "size": 1}, "detector.batch"),
    (("components", "detector", "batch", "max"), 16, "detector.batch"),
    (("components", "detector", "keypoints"), ["right_top", "left_top", "right_bottom", "left_bottom"], "keypoints"),
    (("components", "detector", "postprocess", "candidate_format"), "xyxy", "postprocess"),
    (("components", "detector", "postprocess", "preprocess"), "other", "postprocess"),
    (("components", "detector", "postprocess", "nms"), "torchvision", "postprocess"),
    (("components", "detector", "postprocess", "score_threshold"), 0.3, "postprocess"),
    (("components", "detector", "postprocess", "iou_threshold"), 0.45, "postprocess"),
    (("components", "detector", "postprocess", "max_detections"), 10, "postprocess"),
    (("components", "detector", "postprocess", "extra"), True, "postprocess"),
    (("rectifier", "normalization_strategy"), "sort", "rectifier"),
    (("capabilities",), ["plate-detection"], "capabilities"),
    (("capabilities",), ["crop-recognition"], "capabilities"),
])
def test_full_manifest_rejects_incompatible_detector_contract(path, value, message):
    manifest = valid_full_manifest()
    set_path(manifest, path, value)
    with pytest.raises(DocumentValidationError, match=message):
        validate_model_manifest(manifest, SCHEMA, visible_charset_symbol_count=33)


@pytest.mark.parametrize(("section", "field", "message"), [
    ("provenance", "detector_training_report", "provenance.detector_training_report"),
    ("components", "detector", "components.detector"),
])
def test_full_manifest_requires_detector_and_report(section, field, message):
    manifest = valid_full_manifest()
    del manifest[section][field]
    with pytest.raises(DocumentValidationError, match=message):
        validate_model_manifest(manifest, SCHEMA, visible_charset_symbol_count=33)


def test_full_manifest_requires_postprocess():
    manifest = valid_full_manifest()
    del manifest["components"]["detector"]["postprocess"]
    with pytest.raises(DocumentValidationError, match="detector.postprocess"):
        validate_model_manifest(manifest, SCHEMA, visible_charset_symbol_count=33)


def test_crop_bundle_hash_validation_does_not_require_optional_onnx(tmp_path, monkeypatch):
    import builtins
    import hashlib
    import json
    from plateai_shared.bundle import validate_crop_bundle
    from tests.conftest import V1_CHARSET, V1_RULES

    manifest = valid_v1_manifest()
    declarations = [manifest["charset"], manifest["rules"],
                    manifest["components"]["recognizer"], manifest["provenance"]["training_report"]]
    contents = [V1_CHARSET.read_bytes(), V1_RULES.read_bytes(), b"opaque crop model", b"{}"]
    for item, content in zip(declarations, contents, strict=True):
        (tmp_path / item["file"]).write_bytes(content)
        item["sha256"] = hashlib.sha256(content).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    original_import = builtins.__import__

    def without_onnx(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("optional ONNX is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_onnx)
    assert validate_crop_bundle(tmp_path, SCHEMA) == manifest
