from __future__ import annotations

import numpy as np
import pytest

from plateai_shared.contracts import GeneratedPlate
from plateai_trainer.synthetic.fonts import resolve_font
from plateai_trainer.synthetic.renderer import render_plate
from plateai_trainer.synthetic.templates import load_template

from tests.conftest import write_json


def test_hershey_renderer_returns_rgb_plate_and_ordered_corners(default_template):
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
