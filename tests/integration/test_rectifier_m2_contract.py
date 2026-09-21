from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from plateai_reader.rectifier import InvalidCornersError, rectify_plate
from plateai_shared.contracts import GeneratedPlate
from plateai_shared.recognition import preprocess_v1_rgb
from plateai_trainer.synthetic.fonts import resolve_font
from plateai_trainer.synthetic.renderer import render_plate
from plateai_trainer.synthetic.templates import load_template


ROOT = Path(__file__).resolve().parents[2]
V1_TEMPLATE = ROOT / "configs/plate_templates/new_style_private_passenger_white_v1.json"


def _warped_m1_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample = GeneratedPlate(
        "ABC1234",
        "ABC-1234",
        "new-style-private-passenger-lll-dddd",
        "new-style-private-passenger",
    )
    rendered = render_plate(sample, load_template(V1_TEMPLATE), resolve_font(None))
    destination = np.float32([[35, 22], [463, 45], [430, 247], [55, 230]])
    transform = cv2.getPerspectiveTransform(rendered.corners, destination)
    warped = cv2.warpPerspective(
        rendered.image_rgb,
        transform,
        (520, 280),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rendered.image_rgb, warped, destination


def test_rectifier_warps_m1_perspective_fixture_to_m2_source_contract():
    canonical, warped_rgb, ground_truth_corners = _warped_m1_fixture()

    result = rectify_plate(warped_rgb, ground_truth_corners[[2, 0, 3, 1]])

    assert result.image_rgb.shape == (160, 380, 3)
    assert result.image_rgb.dtype == np.uint8
    assert preprocess_v1_rgb(result.image_rgb).shape == (1, 64, 160)
    assert np.mean(np.abs(result.image_rgb.astype(int) - canonical.astype(int))) < 8.0


def test_invalid_corners_never_call_perspective_transform(monkeypatch):
    rgb = np.full((160, 380, 3), 255, dtype=np.uint8)
    invalid_points = np.array([[0, 0], [379, 0], [379, 159], [379, 159]])
    monkeypatch.setattr(cv2, "getPerspectiveTransform", pytest.fail)

    with pytest.raises(InvalidCornersError, match="duplicate"):
        rectify_plate(rgb, invalid_points)
