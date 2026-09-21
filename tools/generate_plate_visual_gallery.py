#!/usr/bin/env python3
"""Render supplied plate text as clean car and motorcycle visual comparisons.

This is intentionally a visual-inspection utility, not a dataset generator.
It does not claim a supplied number is valid for the selected vehicle class,
and its OFL font is an approximation rather than an official plate font.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np
from PIL import Image, ImageDraw

from plateai_shared.contracts import GeneratedPlate
from plateai_trainer.synthetic.fonts import resolve_font
from plateai_trainer.synthetic.models import PlateTemplate
from plateai_trainer.synthetic.renderer import render_plate


LABEL_PATTERN = re.compile(r"^[A-Z0-9]+-[A-Z0-9]+$")
GALLERY_SCHEMA_VERSION = 1


def _template(*, template_id: str, width: int, height: int, text_box: tuple[int, int, int, int]) -> PlateTemplate:
    return PlateTemplate(
        id=template_id,
        width=width,
        height=height,
        background_rgb=(245, 245, 242),
        foreground_rgb=(18, 18, 18),
        border_rgb=(18, 18, 18),
        border_width=3,
        corner_radius=10,
        text_box=text_box,
    )


CAR_TEMPLATE = _template(
    template_id="visual-private-passenger-380x160",
    width=380,
    height=160,
    text_box=(24, 24, 356, 136),
)
MOTORCYCLE_TEMPLATE = _template(
    template_id="visual-motorcycle-260x140",
    width=260,
    height=140,
    text_box=(16, 20, 244, 120),
)


def _parse_label(value: str) -> str:
    display = value.strip().upper()
    if not LABEL_PATTERN.fullmatch(display):
        raise argparse.ArgumentTypeError(
            "plate text must use uppercase A-Z/0-9 with one separator, for example LAB-6531"
        )
    return display


def _safe_stem(display: str) -> str:
    return display.replace("-", "_")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render(display: str, template: PlateTemplate, output: Path) -> dict[str, object]:
    sample = GeneratedPlate(
        canonical=display.replace("-", ""),
        display=display,
        rule_id="user-supplied-visual-comparison",
        plate_type="visual-comparison",
    )
    rendered = render_plate(sample, template, resolve_font(None))
    Image.fromarray(rendered.image_rgb, mode="RGB").save(output, format="PNG")
    return {
        "path": output.name,
        "template_id": template.id,
        "pixels_wh": [template.width, template.height],
        "sha256": _sha256(output),
    }


def _make_gallery(rows: list[dict[str, object]], output: Path) -> None:
    margin = 26
    row_height = 218
    car_x = margin
    motorcycle_x = car_x + CAR_TEMPLATE.width + 42
    canvas = Image.new(
        "RGB",
        (motorcycle_x + MOTORCYCLE_TEMPLATE.width + margin, margin + row_height * len(rows)),
        color=(233, 236, 241),
    )
    draw = ImageDraw.Draw(canvas)

    for index, row in enumerate(rows):
        y = margin + row_height * index
        display = str(row["display"])
        draw.text((car_x, y), f"{display}  |  private-car proportion 380 x 160", fill=(20, 20, 24))
        draw.text((motorcycle_x, y), "motorcycle proportion 260 x 140", fill=(20, 20, 24))
        car = Image.open(output.parent / str(row["car"]["path"])).convert("RGB")
        motorcycle = Image.open(output.parent / str(row["motorcycle"]["path"])).convert("RGB")
        canvas.paste(car, (car_x, y + 28))
        canvas.paste(motorcycle, (motorcycle_x, y + 38))

    canvas.save(output, format="PNG")


def _write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate(labels: list[str], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        rows: list[dict[str, object]] = []
        for index, display in enumerate(labels):
            stem = f"{index:02d}_{_safe_stem(display)}"
            car = _render(display, CAR_TEMPLATE, staging / f"{stem}_car.png")
            motorcycle = _render(display, MOTORCYCLE_TEMPLATE, staging / f"{stem}_motorcycle.png")
            rows.append(
                {
                    "display": display,
                    "canonical": display.replace("-", ""),
                    "car": car,
                    "motorcycle": motorcycle,
                }
            )
        _make_gallery(rows, staging / "gallery.png")
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": GALLERY_SCHEMA_VERSION,
                "purpose": "visual comparison only; not training or a claim of official visual fidelity",
                "font": {"name": resolve_font(None).name, "kind": "OFL approximation"},
                "templates": {
                    "private_car": {"pixels_wh": [380, 160], "basis": "current project private-passenger visual profile"},
                    "motorcycle": {"pixels_wh": [260, 140], "basis": "Taiwan Highway Bureau ordinary/light motorcycle external-size ratio"},
                },
                "rows": rows,
            },
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", action="append", required=True, type=_parse_label, help="Plate display text; repeat for each visual.")
    parser.add_argument("--output", type=Path, required=True, help="New ignored output directory.")
    args = parser.parse_args()
    try:
        generate(args.label, args.output.resolve())
    except (FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
