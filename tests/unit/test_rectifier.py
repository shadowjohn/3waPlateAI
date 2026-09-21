from __future__ import annotations

import itertools

import numpy as np
import pytest

from plateai_reader.rectifier import (
    InvalidCornersError,
    normalize_corners,
    rectify_plate,
)


def test_every_trapezoid_permutation_has_one_canonical_order():
    expected = np.array(
        [[30, 20], [350, 45], [330, 130], [50, 110]],
        dtype=np.float32,
    )

    for permutation in itertools.permutations(expected):
        normalized = normalize_corners(np.asarray(permutation), (380, 160))
        np.testing.assert_allclose(normalized.points_xy, expected)


@pytest.mark.parametrize(
    ("points", "image_size_wh", "reason"),
    [
        (
            np.array([[30, 20], [350, 45], [np.nan, 130], [50, 110]]),
            (380, 160),
            "non_finite",
        ),
        (
            np.array([[30, 20], [350, 45], [350, 45], [50, 110]]),
            (380, 160),
            "duplicate",
        ),
        (
            np.array([[120, 30], [190, 100], [120, 170], [50, 100]]),
            (240, 200),
            "ambiguous",
        ),
        (
            np.array([[30, 20], [350, 45], [330, 130], [200, 70]]),
            (380, 160),
            "non_convex",
        ),
        (
            np.array([[100, 50], [110, 50], [110, 55], [100, 55]]),
            (380, 160),
            "low_area",
        ),
        (
            np.array([[-1, 20], [350, 45], [330, 130], [50, 110]]),
            (380, 160),
            "out_of_frame",
        ),
    ],
)
def test_invalid_corner_sets_reject_with_stable_reason(
    points: np.ndarray,
    image_size_wh: tuple[int, int],
    reason: str,
):
    with pytest.raises(InvalidCornersError, match=reason):
        normalize_corners(points, image_size_wh)


def test_identity_rectification_preserves_every_outer_pixel_without_border_mixing():
    y, x = np.indices((160, 380), dtype=np.uint16)
    source = np.stack(
        ((x * 17 + y * 13) % 256, (x * 7 + y * 29) % 256, (x * 31 + y * 3) % 256),
        axis=-1,
    ).astype(np.uint8)
    corners = np.float32([[0, 0], [379, 0], [379, 159], [0, 159]])

    rectified = rectify_plate(source, corners).image_rgb

    np.testing.assert_array_equal(rectified[0, :, :], source[0, :, :])
    np.testing.assert_array_equal(rectified[-1, :, :], source[-1, :, :])
    np.testing.assert_array_equal(rectified[:, 0, :], source[:, 0, :])
    np.testing.assert_array_equal(rectified[:, -1, :], source[:, -1, :])
