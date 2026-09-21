"""Deterministic validation helpers for public JSON contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .contracts import JsonValue


class DocumentValidationError(ValueError):
    """A stable, path-addressed schema or cross-field validation error."""


def _format_path(parts: Sequence[str | int]) -> str:
    result = ""
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        elif result:
            result += f".{part}"
        else:
            result = str(part)
    return result or "$"


def _reject_non_finite(value: Any, path: tuple[str | int, ...] = ()) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DocumentValidationError(
            f"{_format_path(path)}: number must be finite"
        )
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _reject_non_finite(value[key], path + (str(key),))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, path + (index,))


def load_schema(path: Path) -> Mapping[str, JsonValue]:
    """Load and meta-validate a Draft 2020-12 schema."""

    try:
        schema = json.loads(path.read_bytes())
    except OSError as exc:
        raise DocumentValidationError(f"$: cannot read schema {path}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DocumentValidationError(f"$: invalid UTF-8 JSON schema {path}") from exc
    if not isinstance(schema, dict):
        raise DocumentValidationError("$: schema must be a JSON object")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise DocumentValidationError(f"$: invalid schema: {exc.message}") from exc
    return schema


def _error_path(error) -> tuple[str | int, ...]:
    path = tuple(error.absolute_path)
    if error.validator == "required" and isinstance(error.instance, Mapping):
        required = error.validator_value
        if isinstance(required, list):
            missing = sorted(set(required) - set(error.instance))
            if missing:
                path += (missing[0],)
    return path


def _validation_sort_key(error) -> tuple[tuple[str, ...], str]:
    normalized = tuple(
        f"#{part:020d}" if isinstance(part, int) else f".{part}"
        for part in _error_path(error)
    )
    return normalized, error.message


def validate_document(
    document: Mapping[str, JsonValue], schema_path: Path
) -> None:
    """Validate a document and raise only the first deterministic error."""

    if not isinstance(document, Mapping):
        raise DocumentValidationError("$: document must be a JSON object")
    _reject_non_finite(document)
    schema = load_schema(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=_validation_sort_key,
    )
    if errors:
        error = errors[0]
        raise DocumentValidationError(
            f"{_format_path(_error_path(error))}: {error.message}"
        )


def validate_model_manifest(
    document: Mapping[str, JsonValue],
    schema_path: Path,
    *,
    visible_charset_symbol_count: int,
) -> None:
    """Validate schema plus CTC and dynamic-batch cross-field invariants."""

    validate_document(document, schema_path)
    has_detector = "detector" in document["components"]
    has_detection = "plate-detection" in document["capabilities"]
    if has_detector != has_detection:
        raise DocumentValidationError(
            "capabilities: detector requires crop-recognition and plate-detection"
        )
    if not has_detection and "detector_training_report" in document["provenance"]:
        raise DocumentValidationError(
            "provenance.detector_training_report: requires plate-detection"
        )
    if type(visible_charset_symbol_count) is not int or visible_charset_symbol_count < 1:
        raise DocumentValidationError(
            "charset.visible_symbols: visible symbol count must be positive"
        )

    if document["plate_size"] != [380, 160]:
        raise DocumentValidationError("plate_size: must equal [380, 160] for v1")

    recognizer = document["components"]["recognizer"]
    expected_inputs = [
        {"name": "input", "dtype": "float32", "shape": ["batch", 1, 64, 160]}
    ]
    if recognizer["inputs"] != expected_inputs:
        raise DocumentValidationError(
            "components.recognizer.inputs: must equal the v1 NCHW input tensor"
        )
    expected_class_count = visible_charset_symbol_count + 1
    expected_outputs = [
        {"name": "logits", "dtype": "float32", "shape": ["batch", 80, expected_class_count]}
    ]
    if recognizer["outputs"] != expected_outputs:
        raise DocumentValidationError(
            "components.recognizer.outputs: must equal the v1 80-step logits tensor"
        )
    expected_batch = {"mode": "dynamic", "min": 1, "opt": 8, "max": 32}
    if recognizer["batch"] != expected_batch:
        raise DocumentValidationError(
            "components.recognizer.batch: must equal the v1 dynamic batch contract"
        )

    decoder = document["decoder"]
    if decoder["blank_index"] != 0:
        raise DocumentValidationError("decoder.blank_index: must equal 0 for CTC")
    manifest_visible = document["charset"]["visible_symbols"]
    if manifest_visible != visible_charset_symbol_count:
        raise DocumentValidationError(
            "charset.visible_symbols: does not match the supplied charset"
        )
    if decoder["class_count"] != expected_class_count:
        raise DocumentValidationError(
            f"decoder.class_count: must equal {expected_class_count} ({visible_charset_symbol_count} visible symbols plus blank)"
        )
