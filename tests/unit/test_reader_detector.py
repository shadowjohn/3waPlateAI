import numpy as np
import pytest

from plateai_reader.detector import numpy_nms_v1, postprocess_candidates
from plateai_shared.detection import LetterboxTransform


def _candidate(
    cx: float,
    cy: float,
    width: float,
    height: float,
    confidence: float,
    corners: list[list[float]] | None = None,
) -> np.ndarray:
    if corners is None:
        corners = [
            [cx - width / 2, cy - height / 2],
            [cx + width / 2, cy - height / 2],
            [cx + width / 2, cy + height / 2],
            [cx - width / 2, cy + height / 2],
        ]
    return np.asarray(
        [cx, cy, width, height, confidence, *np.asarray(corners).reshape(-1)],
        dtype=np.float32,
    )


def _odd_padding_transform() -> LetterboxTransform:
    return LetterboxTransform(
        source_size_wh=(1000, 333),
        resized_size_wh=(640, 213),
        padding_ltrb=(0, 213, 0, 214),
        scale=0.64,
    )


def _default_postprocess() -> dict[str, object]:
    return {
        "nms": "numpy-nms-v1",
        "score_threshold": 0.25,
        "iou_threshold": 0.50,
        "max_detections": 100,
    }


def test_postprocess_inverse_maps_and_clips_bbox_and_every_corner():
    candidates = np.stack(
        (
            _candidate(
                320.0,
                309.0,
                768.0,
                320.0,
                0.90,
                [[-64.0, 149.0], [704.0, 149.0], [704.0, 469.0], [-64.0, 469.0]],
            ),
            _candidate(
                160.0,
                309.0,
                64.0,
                32.0,
                0.80,
                [[128.0, 293.0], [192.0, 293.0], [192.0, 325.0], [128.0, 325.0]],
            ),
        )
    )

    detections = postprocess_candidates(candidates, _odd_padding_transform(), _default_postprocess())

    assert len(detections) == 2
    assert detections[0].bbox_xyxy.tolist() == [0.0, 0.0, 999.0, 332.0]
    assert detections[0].bbox_xyxy.dtype == np.float32
    assert detections[0].corners_xy.dtype == np.float32
    assert np.all((detections[0].corners_xy[:, 0] >= 0) & (detections[0].corners_xy[:, 0] <= 999))
    assert np.all((detections[0].corners_xy[:, 1] >= 0) & (detections[0].corners_xy[:, 1] <= 332))
    np.testing.assert_array_equal(
        detections[0].corners_xy,
        np.float32([[0, 0], [999, 0], [999, 332], [0, 332]]),
    )


def test_numpy_nms_breaks_equal_scores_by_candidate_index():
    candidates = np.stack(
        (
            _candidate(100.0, 100.0, 80.0, 40.0, 0.75),
            _candidate(100.0, 100.0, 80.0, 40.0, 0.75),
        )
    )

    assert numpy_nms_v1(candidates, 0.25, 0.50, 100) == [0]


def test_numpy_nms_orders_by_descending_confidence_before_candidate_index():
    candidates = np.stack(
        (
            _candidate(50.0, 50.0, 20.0, 10.0, 0.75),
            _candidate(150.0, 50.0, 20.0, 10.0, 0.90),
            _candidate(250.0, 50.0, 20.0, 10.0, 0.75),
        )
    )

    assert numpy_nms_v1(candidates, 0.25, 0.50, 100) == [1, 0, 2]


def test_numpy_nms_rejects_non_finite_and_non_positive_candidates():
    valid = _candidate(50.0, 50.0, 20.0, 10.0, 0.80)
    non_finite_corner = _candidate(150.0, 50.0, 20.0, 10.0, 0.99)
    non_finite_corner[12] = np.nan
    non_finite_confidence = _candidate(250.0, 50.0, 20.0, 10.0, np.inf)
    zero_width = _candidate(350.0, 50.0, 0.0, 10.0, 0.99)
    negative_height = _candidate(450.0, 50.0, 20.0, -1.0, 0.99)
    below_threshold = _candidate(550.0, 50.0, 20.0, 10.0, 0.24)

    assert numpy_nms_v1(
        np.stack((valid, non_finite_corner, non_finite_confidence, zero_width, negative_height, below_threshold)),
        0.25,
        0.50,
        100,
    ) == [0]


def test_postprocess_filters_candidate_whose_derived_bbox_is_non_finite():
    overflow = _candidate(
        3e38,
        100.0,
        3e38,
        10.0,
        0.90,
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    )

    assert postprocess_candidates(
        overflow.reshape(1, 13),
        _odd_padding_transform(),
        _default_postprocess(),
    ) == []


@pytest.mark.parametrize(
    ("score_threshold", "iou_threshold", "max_detections", "message"),
    (
        (-0.01, 0.50, 100, "score_threshold"),
        (1.01, 0.50, 100, "score_threshold"),
        (np.nan, 0.50, 100, "score_threshold"),
        (0.25, 0.0, 100, "iou_threshold"),
        (0.25, 1.01, 100, "iou_threshold"),
        (0.25, np.inf, 100, "iou_threshold"),
        (0.25, 0.50, 0, "max_detections"),
        (0.25, 0.50, 1.5, "max_detections"),
        (0.25, 0.50, True, "max_detections"),
    ),
)
def test_numpy_nms_validates_thresholds_before_candidate_shape(
    score_threshold: float,
    iou_threshold: float,
    max_detections: object,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        numpy_nms_v1(
            np.asarray([1.0, 2.0], dtype=np.float32),
            score_threshold,
            iou_threshold,
            max_detections,  # type: ignore[arg-type]
        )


def test_numpy_nms_requires_candidate_rows_with_exact_layout():
    with pytest.raises(ValueError, match=r"\[N, 13\]"):
        numpy_nms_v1(np.zeros((2, 12), dtype=np.float32), 0.25, 0.50, 100)


def test_postprocess_validates_thresholds_before_candidate_shape():
    options = _default_postprocess()
    options["score_threshold"] = -0.01

    with pytest.raises(ValueError, match="score_threshold"):
        postprocess_candidates(
            np.asarray([1.0, 2.0], dtype=np.float32),
            _odd_padding_transform(),
            options,
        )
