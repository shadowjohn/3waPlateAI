from __future__ import annotations

import itertools

import numpy as np
import pytest

from plateai_reader.rectifier import InvalidCornersError, normalize_corners


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
