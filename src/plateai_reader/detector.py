"""Torch-free detector candidate postprocessing and M3a handoff."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from plateai_shared.detection import (
    LetterboxTransform,
    PlateDetection,
    inverse_and_clip_points,
)

from .rectifier import InvalidCornersError, RectifiedPlate, rectify_plate


@dataclass(frozen=True, slots=True)
class DetectionRejection:
    """One retained detection that M3a could not rectify."""

    detection: PlateDetection
    reason: str


@dataclass(frozen=True, slots=True)
class DetectionRectificationResult:
    """Independently rectified detections and their isolated rejections."""

    rectified: tuple[RectifiedPlate, ...]
    rejections: tuple[DetectionRejection, ...]


def _validate_nms_options(
    score_threshold: object,
    iou_threshold: object,
    max_detections: object,
) -> tuple[float, float, int]:
    if (
        isinstance(score_threshold, (bool, np.bool_))
        or not isinstance(score_threshold, Real)
        or not math.isfinite(float(score_threshold))
        or not 0.0 <= float(score_threshold) <= 1.0
    ):
        raise ValueError("score_threshold must be finite and in [0, 1]")
    if (
        isinstance(iou_threshold, (bool, np.bool_))
        or not isinstance(iou_threshold, Real)
        or not math.isfinite(float(iou_threshold))
        or not 0.0 < float(iou_threshold) <= 1.0
    ):
        raise ValueError("iou_threshold must be finite and in (0, 1]")
    if (
        isinstance(max_detections, (bool, np.bool_))
        or not isinstance(max_detections, (int, np.integer))
        or int(max_detections) <= 0
    ):
        raise ValueError("max_detections must be a positive integer")
    return float(score_threshold), float(iou_threshold), int(max_detections)


def _candidate_array(candidates: object) -> NDArray[np.float32]:
    raw = np.asarray(candidates)
    if raw.ndim != 2 or raw.shape[1] != 13:
        raise ValueError("candidates must have shape [N, 13]")
    try:
        return np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("candidates must contain numeric values") from error


def _candidate_xyxy(candidate: NDArray[np.float32]) -> NDArray[np.float32]:
    center_x, center_y, width, height = candidate[:4]
    half_width = width / np.float32(2.0)
    half_height = height / np.float32(2.0)
    return np.asarray(
        [
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ],
        dtype=np.float32,
    )


def _eligible_candidate_indices(
    candidates: NDArray[np.float32], score_threshold: float
) -> list[int]:
    return [
        index
        for index, candidate in enumerate(candidates)
        if np.isfinite(candidate).all()
        and candidate[2] > 0.0
        and candidate[3] > 0.0
        and candidate[4] >= score_threshold
    ]


def _xyxy_iou(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
    intersection_size = np.maximum(
        np.float32(0.0),
        np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]),
    )
    intersection = float(intersection_size[0] * intersection_size[1])
    left_area = float((left[2] - left[0]) * (left[3] - left[1]))
    right_area = float((right[2] - right[0]) * (right[3] - right[1]))
    return intersection / (left_area + right_area - intersection)


def numpy_nms_v1(
    candidates: NDArray[np.float32],
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[int]:
    """Return deterministic source row indices after class-agnostic bbox NMS."""

    score, overlap, maximum = _validate_nms_options(
        score_threshold, iou_threshold, max_detections
    )
    values = _candidate_array(candidates)
    eligible = _eligible_candidate_indices(values, score)
    ordered = sorted(eligible, key=lambda index: (-float(values[index, 4]), index))
    retained: list[int] = []
    for index in ordered:
        box = _candidate_xyxy(values[index])
        if all(
            _xyxy_iou(box, _candidate_xyxy(values[kept])) <= overlap
            for kept in retained
        ):
            retained.append(index)
        if len(retained) == maximum:
            break
    return retained


def postprocess_candidates(
    candidates: NDArray[np.float32],
    transform: LetterboxTransform,
    postprocess: Mapping[str, object],
) -> list[PlateDetection]:
    """Apply deterministic NMS and recover discrete source coordinates."""

    try:
        score_threshold = postprocess["score_threshold"]
        iou_threshold = postprocess["iou_threshold"]
        max_detections = postprocess["max_detections"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "postprocess must define score_threshold, iou_threshold, and max_detections"
        ) from error
    score, overlap, maximum = _validate_nms_options(
        score_threshold, iou_threshold, max_detections
    )
    retained = numpy_nms_v1(
        candidates,
        score,
        overlap,
        maximum,
    )
    values = _candidate_array(candidates)
    detections: list[PlateDetection] = []
    for index in retained:
        candidate = values[index]
        box = _candidate_xyxy(candidate).reshape(2, 2)
        source_box = inverse_and_clip_points(box, transform).reshape(4)
        source_corners = inverse_and_clip_points(candidate[5:].reshape(4, 2), transform)
        detections.append(
            PlateDetection(
                bbox_xyxy=np.asarray(source_box, dtype=np.float32),
                confidence=float(candidate[4]),
                corners_xy=np.asarray(source_corners, dtype=np.float32),
            )
        )
    return detections


def rectify_detections(
    image_rgb: NDArray[np.uint8], detections: Sequence[PlateDetection]
) -> DetectionRectificationResult:
    """Rectify every detection independently, preserving M3a rejection reasons."""

    accepted: list[RectifiedPlate] = []
    rejections: list[DetectionRejection] = []
    for detection in detections:
        try:
            accepted.append(rectify_plate(image_rgb, detection.corners_xy))
        except InvalidCornersError as error:
            rejections.append(DetectionRejection(detection=detection, reason=error.reason))
    return DetectionRectificationResult(tuple(accepted), tuple(rejections))
