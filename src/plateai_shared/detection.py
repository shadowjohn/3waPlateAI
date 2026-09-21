"""Shared M3b detector image and coordinate contracts."""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    source_size_wh: tuple[int, int]
    resized_size_wh: tuple[int, int]
    padding_ltrb: tuple[int, int, int, int]
    scale: float

    @property
    def pad_left(self) -> int:
        return self.padding_ltrb[0]

    @property
    def pad_top(self) -> int:
        return self.padding_ltrb[1]


@dataclass(frozen=True, slots=True)
class PlateDetection:
    bbox_xyxy: NDArray[np.float32]
    confidence: float
    corners_xy: NDArray[np.float32]


def _points(points_xy: object) -> NDArray[np.float32]:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates")
    return points


def letterbox_rgb_v1(image_rgb: object) -> tuple[NDArray[np.float32], LetterboxTransform]:
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be uint8 RGB")
    source_h, source_w = image.shape[:2]
    if source_w <= 0 or source_h <= 0:
        raise ValueError("image dimensions must be non-empty")
    scale = min(640.0 / source_w, 640.0 / source_h)
    resized_w = max(1, math.floor(source_w * scale + 0.5))
    resized_h = max(1, math.floor(source_h * scale + 0.5))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = 640 - resized_w, 640 - resized_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top
    boxed = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    tensor = np.ascontiguousarray(boxed.transpose(2, 0, 1), dtype=np.float32) / np.float32(255.0)
    transform = LetterboxTransform(
        source_size_wh=(source_w, source_h),
        resized_size_wh=(resized_w, resized_h),
        padding_ltrb=(left, top, right, bottom),
        scale=float(scale),
    )
    return tensor, transform


def map_points_to_letterbox(points_xy: object, transform: LetterboxTransform) -> NDArray[np.float32]:
    points = _points(points_xy).copy()
    points *= np.float32(transform.scale)
    points += np.float32([transform.pad_left, transform.pad_top])
    return points


def inverse_and_clip_points(points_xy: object, transform: LetterboxTransform) -> NDArray[np.float32]:
    points = (_points(points_xy) - np.float32([transform.pad_left, transform.pad_top])) / np.float32(transform.scale)
    source_w, source_h = transform.source_size_wh
    points = points.copy()
    points[:, 0] = np.clip(points[:, 0], 0.0, float(source_w - 1))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(source_h - 1))
    return points
