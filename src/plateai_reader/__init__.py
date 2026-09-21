"""Deterministic reader components for Taiwan license-plate crops."""

from .detector import (
    DetectionRectificationResult,
    DetectionRejection,
    numpy_nms_v1,
    postprocess_candidates,
    rectify_detections,
)
from .rectifier import (
    InvalidCornersError,
    NormalizedCorners,
    RectifiedPlate,
    normalize_corners,
    rectify_plate,
)

__all__ = [
    "DetectionRectificationResult",
    "DetectionRejection",
    "InvalidCornersError",
    "NormalizedCorners",
    "RectifiedPlate",
    "numpy_nms_v1",
    "normalize_corners",
    "postprocess_candidates",
    "rectify_detections",
    "rectify_plate",
]
