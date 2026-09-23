from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plateai_shared.contracts import GeneratedPlate
from plateai_trainer.synthetic.fonts import resolve_font
from plateai_trainer.synthetic.renderer import render_plate
from plateai_trainer.synthetic.templates import load_template

from tests.conftest import write_json


def test_renderer_returns_rgb_plate_and_ordered_corners(default_template):
    sample = GeneratedPlate("ABC1234", "ABC-1234", "standard-lll-dddd", "standard")
    rendered = render_plate(sample, default_template, resolve_font(None))
    assert rendered.image_rgb.shape == (96, 320, 3)
    assert rendered.image_rgb.dtype == np.uint8
    assert rendered.corners.tolist() == [
        [0.0, 0.0],
        [319.0, 0.0],
        [319.0, 95.0],
        [0.0, 95.0],
    ]
    assert np.count_nonzero(rendered.image_rgb < 80) > 500
    assert rendered.metadata["rendered_text"] == "ABC-1234"


def test_default_font_is_the_packaged_ofl_face():
    font = resolve_font(None)
    assert font.kind == "truetype"
    assert font.name == "NotoSansMono[wdth,wght].ttf"
    assert font.path is not None
    assert font.path.is_file()


def test_missing_explicit_font_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="font file does not exist"):
        resolve_font(tmp_path / "missing.ttf")


def test_template_rejects_out_of_bounds_text_box(
    tmp_path, valid_template_document
):
    valid_template_document["text_box"] = [0, 0, 999, 96]
    path = write_json(tmp_path / "template.json", valid_template_document)
    with pytest.raises(ValueError, match="text_box"):
        load_template(path)


@pytest.mark.parametrize(
    "value",
    ("taiwan_plate", Path("taiwan_plate"), "official", Path("official")),
)
def test_taiwan_plate_alias_requires_a_local_authorized_font(
    monkeypatch, tmp_path, value
):
    monkeypatch.setenv("PLATEAI_LOCAL_FONT_DIR", str(tmp_path / "missing-fonts"))

    with pytest.raises(FileNotFoundError, match="local Taiwan plate font is missing"):
        resolve_font(value)


def test_taiwan_plate_alias_accepts_a_local_authorized_font(
    default_template, monkeypatch, tmp_path
):
    local_font_dir = tmp_path / "authorized-fonts"
    local_font_dir.mkdir()
    source_font = resolve_font(None)
    assert source_font.path is not None
    local_font = local_font_dir / "TaiwanPlate-Regular.ttf"
    local_font.write_bytes(source_font.path.read_bytes())
    monkeypatch.setenv("PLATEAI_LOCAL_FONT_DIR", str(local_font_dir))

    font = resolve_font("taiwan_plate")
    assert font.kind == "truetype"
    assert font.name == "TaiwanPlate-Regular.ttf"
    assert font.path == local_font.resolve()

    sample = GeneratedPlate("AQ560", "AQ-560", "moto-red-2-3", "moto-red")
    rendered = render_plate(sample, default_template, font)
    assert rendered.image_rgb.shape == (96, 320, 3)
    assert rendered.metadata["rendered_text"] == "AQ-560"
