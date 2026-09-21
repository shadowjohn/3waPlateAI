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
    if type(visible_charset_symbol_count) is not int or visible_charset_symbol_count < 1:
        raise DocumentValidationError(
            "charset.visible_symbols: visible symbol count must be positive"
        )

    components = document["components"]
    recognizer = components["recognizer"]
    batch = recognizer["batch"]
    if batch["mode"] == "dynamic" and not (
        batch["min"] <= batch["opt"] <= batch["max"]
    ):
        raise DocumentValidationError(
            "components.recognizer.batch: dynamic sizes must satisfy min <= opt <= max"
        )

    decoder = document["decoder"]
    blank_index = decoder["blank_index"]
    class_count = decoder["class_count"]
    if not 0 <= blank_index < class_count:
        raise DocumentValidationError(
            "decoder.blank_index: must be within the declared class range"
        )
    expected_class_count = visible_charset_symbol_count + 1
    if class_count != expected_class_count:
        raise DocumentValidationError(
            "decoder.class_count: must equal visible charset symbols plus one CTC blank"
        )
    manifest_visible = document["charset"]["visible_symbols"]
    if manifest_visible != visible_charset_symbol_count:
        raise DocumentValidationError(
            "charset.visible_symbols: does not match the supplied charset"
        )
