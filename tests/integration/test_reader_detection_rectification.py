import numpy as np

from plateai_reader.detector import rectify_detections
from plateai_shared.detection import PlateDetection


def _detection(corners: list[list[float]]) -> PlateDetection:
    return PlateDetection(
        bbox_xyxy=np.float32([10, 10, 190, 80]),
        confidence=0.9,
        corners_xy=np.asarray(corners, dtype=np.float32),
    )


def test_one_invalid_rectification_does_not_discard_another_valid_detection():
    image_rgb = np.full((100, 200, 3), 255, dtype=np.uint8)
    invalid = _detection([[10, 10], [190, 10], [190, 10], [10, 80]])
    valid = _detection([[10, 10], [190, 10], [190, 80], [10, 80]])

    result = rectify_detections(image_rgb, [invalid, valid])

    assert len(result.rectified) == 1
    assert result.rectified[0].image_rgb.shape == (160, 380, 3)
    assert len(result.rejections) == 1
    assert result.rejections[0].detection is invalid
    assert result.rejections[0].reason == "duplicate"
