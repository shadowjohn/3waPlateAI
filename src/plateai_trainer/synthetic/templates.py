"""Loading and validation for plate rendering templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

from .models import PlateTemplate


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


def load_template(path: Path) -> PlateTemplate:
    """Read a validated version-one rendering template."""

    try:
        document = json.loads(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"cannot read plate template {path}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON plate template: {path}") from exc

    if not isinstance(document, dict):
        _invalid("plate template must be an object")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        _invalid("plate-template schema_version must be 1")
    template_id = document.get("id")
    if not isinstance(template_id, str) or not template_id:
        _invalid("plate-template id must not be empty")

    width = _integer(document.get("width"), "width", minimum=1)
    height = _integer(document.get("height"), "height", minimum=1)
    border_width = _integer(document.get("border_width"), "border_width")
    corner_radius = _integer(document.get("corner_radius"), "corner_radius")
    if border_width * 2 > min(width, height):
        _invalid("border_width is too large for the canvas")
    if corner_radius * 2 > min(width, height):
        _invalid("corner_radius is too large for the canvas")

    raw_text_box = document.get("text_box")
    if (
        not isinstance(raw_text_box, list)
        or len(raw_text_box) != 4
        or any(type(coordinate) is not int for coordinate in raw_text_box)
    ):
        _invalid("text_box must contain four integer coordinates")
    left, top, right, bottom = raw_text_box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        _invalid("text_box must be ordered and inside the canvas")

    return PlateTemplate(
        id=template_id,
        width=width,
        height=height,
        background_rgb=_rgb(document.get("background_rgb"), "background_rgb"),
        foreground_rgb=_rgb(document.get("foreground_rgb"), "foreground_rgb"),
        border_rgb=_rgb(document.get("border_rgb"), "border_rgb"),
        border_width=border_width,
        corner_radius=corner_radius,
        text_box=(left, top, right, bottom),
    )
