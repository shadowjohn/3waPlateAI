from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from plateai_trainer.synthetic.augment import (
    apply_augmentations,
    derive_sample_seed,
    load_augment_profile,
)

from tests.conftest import DEFAULT_AUGMENTATION, write_json


def test_same_seed_reproduces_pixels_corners_and_metadata(
    rendered_plate, standard_profile
):
    first = apply_augmentations(rendered_plate, standard_profile, sample_seed=99)
    second = apply_augmentations(rendered_plate, standard_profile, sample_seed=99)
    assert np.array_equal(first.image_rgb, second.image_rgb)
    assert np.array_equal(first.corners, second.corners)
    assert first.metadata == second.metadata


def test_none_profile_is_identity(rendered_plate, none_profile):
    result = apply_augmentations(rendered_plate, none_profile, sample_seed=123)
    assert np.array_equal(result.image_rgb, rendered_plate.image_rgb)
    assert np.array_equal(result.corners, rendered_plate.corners)


def test_extreme_valid_profile_keeps_finite_clockwise_in_frame_corners(
    rendered_plate, extreme_profile
):
    result = apply_augmentations(rendered_plate, extreme_profile, sample_seed=4)
    height, width = result.image_rgb.shape[:2]
    assert result.image_rgb.dtype == np.uint8
    assert np.isfinite(result.corners).all()
    assert ((0 <= result.corners[:, 0]) & (result.corners[:, 0] < width)).all()
    assert ((0 <= result.corners[:, 1]) & (result.corners[:, 1] < height)).all()
    assert cv2.contourArea(result.corners.astype(np.float32), oriented=True) > 0


def test_derived_sample_seed_is_stable_and_indexed():
    assert derive_sample_seed(42, 0) == derive_sample_seed(42, 0)
    assert derive_sample_seed(42, 0) != derive_sample_seed(42, 1)


def test_profile_rejects_excessive_corner_jitter(tmp_path):
    document = json.loads(DEFAULT_AUGMENTATION.read_text(encoding="utf-8"))
    document["max_corner_jitter_ratio"] = 0.201
    path = write_json(tmp_path / "invalid-augmentation.json", document)
    with pytest.raises(ValueError, match="max_corner_jitter_ratio"):
        load_augment_profile(path)
