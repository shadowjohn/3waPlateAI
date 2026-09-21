"""CPU renderer for clean synthetic recognition crops."""

from __future__ import annotations

from types import MappingProxyType

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plateai_shared.contracts import GeneratedPlate

from .models import FontSpec, PlateTemplate, RenderedPlate


_HERSHEY_FACE = cv2.FONT_HERSHEY_SIMPLEX
_HERSHEY_MIN_SCALE = 0.1
_HERSHEY_MAX_SCALE = 10.0
_HERSHEY_THICKNESS = 2


def _draw_border(image: Image.Image, template: PlateTemplate) -> None:
    if template.border_width == 0:
        return
    draw = ImageDraw.Draw(image)
    inset = template.border_width // 2
    draw.rounded_rectangle(
        (inset, inset, template.width - 1 - inset, template.height - 1 - inset),
        radius=template.corner_radius,
        outline=template.border_rgb,
        width=template.border_width,
    )


def _hershey_metrics(text: str, scale: float) -> tuple[int, int, int]:
    (width, height), baseline = cv2.getTextSize(
        text, _HERSHEY_FACE, scale, _HERSHEY_THICKNESS
    )
    return width, height, baseline


def _draw_hershey(
    canvas_rgb: np.ndarray,
    text: str,
    template: PlateTemplate,
) -> None:
    left, top, right, bottom = template.text_box
    box_width = right - left
    box_height = bottom - top

    minimum = _hershey_metrics(text, _HERSHEY_MIN_SCALE)
    if minimum[0] > box_width or minimum[1] + minimum[2] > box_height:
        raise ValueError("display text does not fit template text_box")

    low = _HERSHEY_MIN_SCALE
    high = _HERSHEY_MAX_SCALE
    for _ in range(32):
        midpoint = (low + high) / 2.0
        width, height, baseline = _hershey_metrics(text, midpoint)
        if width <= box_width and height + baseline <= box_height:
            low = midpoint
        else:
            high = midpoint

    width, height, baseline = _hershey_metrics(text, low)
    origin_x = left + (box_width - width) // 2
    origin_y = top + (box_height - (height + baseline)) // 2 + height
    cv2.putText(
        canvas_rgb,
        text,
        (origin_x, origin_y),
        _HERSHEY_FACE,
        low,
        template.foreground_rgb,
        _HERSHEY_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def _load_truetype(path, size: int, variation_axes: tuple[int, ...]):
    try:
        font = ImageFont.truetype(str(path), size=size)
    except OSError as exc:
        raise ValueError(f"cannot load font: {path}") from exc
    if variation_axes:
        try:
            font.set_variation_by_axes(list(variation_axes))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot set font variation axes: {path}") from exc
    return font


def _draw_truetype(
    image: Image.Image,
    text: str,
    template: PlateTemplate,
    font: FontSpec,
) -> None:
    assert font.path is not None
    left, top, right, bottom = template.text_box
    box_width = right - left
    box_height = bottom - top
    draw = ImageDraw.Draw(image)

    def candidate(size: int):
        loaded = _load_truetype(font.path, size, font.variation_axes)
        bounds = draw.textbbox((0, 0), text, font=loaded)
        return loaded, bounds

    minimum_font, minimum_bounds = candidate(8)
    minimum_width = minimum_bounds[2] - minimum_bounds[0]
    minimum_height = minimum_bounds[3] - minimum_bounds[1]
    if minimum_width > box_width or minimum_height > box_height:
        raise ValueError("display text does not fit template text_box")

    best_font = minimum_font
    best_bounds = minimum_bounds
    low = 8
    high = template.height * 2
    while low <= high:
        size = (low + high) // 2
        loaded, bounds = candidate(size)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        if width <= box_width and height <= box_height:
            best_font = loaded
            best_bounds = bounds
            low = size + 1
        else:
            high = size - 1

    glyph_width = best_bounds[2] - best_bounds[0]
    glyph_height = best_bounds[3] - best_bounds[1]
    origin_x = left + (box_width - glyph_width) / 2 - best_bounds[0]
    origin_y = top + (box_height - glyph_height) / 2 - best_bounds[1]
    draw.text(
        (origin_x, origin_y),
        text,
        font=best_font,
        fill=template.foreground_rgb,
    )


def render_plate(
    sample: GeneratedPlate,
    template: PlateTemplate,
    font: FontSpec,
) -> RenderedPlate:
    """Render display text and return canonical source corners."""

    image = Image.new(
        "RGB",
        (template.width, template.height),
        color=template.background_rgb,
    )
    _draw_border(image, template)
    if font.kind == "hershey":
        canvas_rgb = np.asarray(image).copy()
        _draw_hershey(canvas_rgb, sample.display, template)
        image_rgb = np.ascontiguousarray(canvas_rgb, dtype=np.uint8)
    else:
        _draw_truetype(image, sample.display, template, font)
        image_rgb = np.ascontiguousarray(np.asarray(image), dtype=np.uint8)

    corners = np.asarray(
        [
            [0.0, 0.0],
            [template.width - 1.0, 0.0],
            [template.width - 1.0, template.height - 1.0],
            [0.0, template.height - 1.0],
        ],
        dtype=np.float32,
    )
    metadata = MappingProxyType(
        {
            "rendered_text": sample.display,
            "font_kind": font.kind,
            "font_name": font.name,
            "font_variation_axes": list(font.variation_axes),
            "template_id": template.id,
        }
    )
    return RenderedPlate(image_rgb=image_rgb, corners=corners, metadata=metadata)
