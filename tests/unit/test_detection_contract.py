import numpy as np
import pytest

from plateai_shared.detection import (
    inverse_and_clip_points,
    letterbox_rgb_v1,
    map_points_to_letterbox,
)


def test_letterbox_v1_has_golden_odd_padding_and_round_trip_corners():
    image = np.zeros((333, 1000, 3), dtype=np.uint8)
    tensor, transform = letterbox_rgb_v1(image)

    assert tensor.shape == (3, 640, 640)
    assert transform.resized_size_wh == (640, 213)
    assert transform.padding_ltrb == (0, 213, 0, 214)
    points = np.float32([[0, 0], [999, 0], [999, 332], [0, 332]])
    np.testing.assert_allclose(
        inverse_and_clip_points(map_points_to_letterbox(points, transform), transform),
        points,
        atol=1e-5,
    )


def test_letterbox_rejects_non_uint8_or_non_rgb_image():
    with pytest.raises(ValueError, match="uint8 RGB"):
        letterbox_rgb_v1(np.zeros((640, 640), dtype=np.uint8))
