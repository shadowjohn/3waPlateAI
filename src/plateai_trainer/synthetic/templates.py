"""Loading and validation for plate rendering templates."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

from .models import PlateTemplate, PlateTemplateSelector


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _invalid(f"{context} must be an integer >= {minimum}")
    return value


def _rgb(value: Any, context: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(channel) is not int or not 0 <= channel <= 255 for channel in value)
    ):
        _invalid(f"{context} must contain three integer channels from 0 to 255")
    return (value[0], value[1], value[2])


def _load_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"cannot read plate template {path}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON plate template: {path}") from exc

    if not isinstance(document, dict):
        _invalid("plate template must be an object")
    return document


def _parse_template(document: dict[str, Any], *, context: str) -> PlateTemplate:
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        _invalid(f"{context} schema_version must be 1")
    template_id = document.get("id")
    if not isinstance(template_id, str) or not template_id:
        _invalid(f"{context} id must not be empty")

    width = _integer(document.get("width"), f"{context}.width", minimum=1)
    height = _integer(document.get("height"), f"{context}.height", minimum=1)
    border_width = _integer(document.get("border_width"), f"{context}.border_width")
    corner_radius = _integer(document.get("corner_radius"), f"{context}.corner_radius")
    if border_width * 2 > min(width, height):
        _invalid(f"{context}.border_width is too large for the canvas")
    if corner_radius * 2 > min(width, height):
        _invalid(f"{context}.corner_radius is too large for the canvas")

    raw_text_box = document.get("text_box")
    if (
        not isinstance(raw_text_box, list)
        or len(raw_text_box) != 4
        or any(type(coordinate) is not int for coordinate in raw_text_box)
    ):
        _invalid(f"{context}.text_box must contain four integer coordinates")
    left, top, right, bottom = raw_text_box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        _invalid(f"{context}.text_box must be ordered and inside the canvas")

    return PlateTemplate(
        id=template_id,
        width=width,
        height=height,
        background_rgb=_rgb(document.get("background_rgb"), f"{context}.background_rgb"),
        foreground_rgb=_rgb(document.get("foreground_rgb"), f"{context}.foreground_rgb"),
        border_rgb=_rgb(document.get("border_rgb"), f"{context}.border_rgb"),
        border_width=border_width,
        corner_radius=corner_radius,
        text_box=(left, top, right, bottom),
    )


def load_template(path: Path) -> PlateTemplate:
    """Read a validated single rendering template."""

    document = _load_document(path)
    if "templates" in document:
        _invalid("multi-template config requires load_template_selector")
    return _parse_template(document, context="plate-template")


def load_template_selector(path: Path) -> PlateTemplateSelector:
    """Load one fallback template or a strict plate-type template selector."""

    document = _load_document(path)
    if "templates" not in document:
        template = _parse_template(document, context="plate-template")
        return PlateTemplateSelector(
            id=template.id,
            fallback=template,
            by_plate_type=MappingProxyType({}),
            require_known_plate_type=False,
        )

    expected_keys = {"schema_version", "id", "default_plate_type", "templates"}
    if set(document) != expected_keys:
        _invalid("multi-template config must contain only schema_version, id, default_plate_type, and templates")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        _invalid("multi-template schema_version must be 1")
    selector_id = document.get("id")
    if not isinstance(selector_id, str) or not selector_id:
        _invalid("multi-template id must not be empty")
    default_plate_type = document.get("default_plate_type")
    if not isinstance(default_plate_type, str) or not default_plate_type:
        _invalid("multi-template default_plate_type must not be empty")
    raw_templates = document.get("templates")
    if not isinstance(raw_templates, dict) or not raw_templates:
        _invalid("multi-template templates must be a non-empty object")

    templates: dict[str, PlateTemplate] = {}
    for plate_type, raw_template in raw_templates.items():
        if not isinstance(plate_type, str) or not plate_type:
            _invalid("multi-template plate_type keys must not be empty")
        if not isinstance(raw_template, dict):
            _invalid(f"multi-template templates.{plate_type} must be an object")
        if "id" in raw_template or "schema_version" in raw_template:
            _invalid(f"multi-template templates.{plate_type} must not redefine id or schema_version")
        template_document = {
            "schema_version": 1,
            "id": f"{selector_id}/{plate_type}",
            **raw_template,
        }
        templates[plate_type] = _parse_template(
            template_document,
            context=f"multi-template templates.{plate_type}",
        )

    fallback = templates.get(default_plate_type)
    if fallback is None:
        _invalid("multi-template default_plate_type must name one configured template")
    return PlateTemplateSelector(
        id=selector_id,
        fallback=fallback,
        by_plate_type=MappingProxyType(templates),
        require_known_plate_type=True,
    )
