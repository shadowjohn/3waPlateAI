"""Hull-first, deterministic four-corner plate rectification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np


CANONICAL_WIDTH: Final = 380
CANONICAL_HEIGHT: Final = 160
_MIN_RELATIVE_HULL_AREA: Final = 0.001
_MIN_LONG_TO_SHORT_EDGE_RATIO: Final = 1.15
_AXIS_COMPONENT_EPSILON: Final = 1e-6


class InvalidCornersError(ValueError):
    """A candidate corner set cannot safely form a canonical plate crop."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class NormalizedCorners:
    """Validated source corners in left-top, right-top, right-bottom, left-bottom order."""

    points_xy: np.ndarray
    area: float
    reordered: bool
    long_axis_xy: np.ndarray


@dataclass(frozen=True)
class RectifiedPlate:
    """One canonical RGB plate crop and the source correspondence that made it."""

    image_rgb: np.ndarray
    corners: NormalizedCorners
    transform: np.ndarray


def normalize_corners(
    points_xy: np.ndarray,
    image_size_wh: tuple[int, int],
) -> NormalizedCorners:
    """Return an arbitrary four-point candidate in stable clockwise semantic order."""

    width, height = _validate_image_size(image_size_wh)
    points = _validate_points(points_xy, width, height)
    hull = cv2.convexHull(points.astype(np.float32), clockwise=False).reshape(-1, 2)
    if hull.shape != (4, 2):
        raise InvalidCornersError("non_convex")

    cyclic = _clockwise_cyclic_order(hull.astype(np.float64))
    area = abs(_signed_area(cyclic))
    if area < width * height * _MIN_RELATIVE_HULL_AREA:
        raise InvalidCornersError("low_area")

    long_edges = _long_edge_pair(cyclic)
    long_axis = _horizontal_axis(long_edges[0])
    normal = np.array([-long_axis[1], long_axis[0]], dtype=np.float64)
    top_edge, bottom_edge = _classify_top_bottom(long_edges, normal)
    left_top, right_top = _order_edge_endpoints(top_edge, long_axis)
    left_bottom, right_bottom = _order_edge_endpoints(bottom_edge, long_axis)
    canonical = np.array(
        [left_top, right_top, right_bottom, left_bottom],
        dtype=np.float32,
    )
    if _signed_area(canonical) <= 0:
        raise InvalidCornersError("non_clockwise")

    return NormalizedCorners(
        points_xy=canonical,
        area=float(area),
        reordered=not np.allclose(points, canonical, atol=1e-6, rtol=0.0),
        long_axis_xy=long_axis.astype(np.float32),
    )


def rectify_plate(image_rgb: np.ndarray, points_xy: np.ndarray) -> RectifiedPlate:
    """Warp a validated four-corner source candidate into M2's RGB crop contract."""

    height, width = _validate_rgb_image(image_rgb)
    corners = normalize_corners(points_xy, (width, height))
    destination = np.float32([[0, 0], [379, 0], [379, 159], [0, 159]])
    transform = cv2.getPerspectiveTransform(corners.points_xy, destination)
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise InvalidCornersError("invalid_homography")
    crop = cv2.warpPerspective(
        image_rgb,
        transform,
        (CANONICAL_WIDTH, CANONICAL_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    if crop.shape != (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3) or crop.dtype != np.uint8:
        raise RuntimeError("OpenCV returned an invalid canonical crop")
    return RectifiedPlate(image_rgb=crop, corners=corners, transform=transform)


def _validate_rgb_image(image_rgb: np.ndarray) -> tuple[int, int]:
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.dtype != np.uint8
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.shape[0] <= 0
        or image_rgb.shape[1] <= 0
    ):
        raise ValueError("image_rgb must be a non-empty uint8 RGB array")
    return int(image_rgb.shape[0]), int(image_rgb.shape[1])


def _validate_image_size(image_size_wh: tuple[int, int]) -> tuple[int, int]:
    try:
        width, height = image_size_wh
    except (TypeError, ValueError) as error:
        raise InvalidCornersError("invalid_image_size") from error
    if isinstance(width, (bool, np.bool_)) or isinstance(height, (bool, np.bool_)):
        raise InvalidCornersError("invalid_image_size")
    if not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)):
        raise InvalidCornersError("invalid_image_size")
    if width <= 0 or height <= 0:
        raise InvalidCornersError("invalid_image_size")
    return int(width), int(height)


def _validate_points(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        points = np.asarray(points_xy, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidCornersError("invalid_shape") from error
    if points.shape != (4, 2):
        raise InvalidCornersError("invalid_shape")
    if not np.isfinite(points).all():
        raise InvalidCornersError("non_finite")
    if (
        (points[:, 0] < 0).any()
        or (points[:, 0] >= width).any()
        or (points[:, 1] < 0).any()
        or (points[:, 1] >= height).any()
    ):
        raise InvalidCornersError("out_of_frame")
    distances = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    duplicate_mask = np.triu(distances <= _AXIS_COMPONENT_EPSILON, k=1)
    if duplicate_mask.any():
        raise InvalidCornersError("duplicate")
    return points


def _clockwise_cyclic_order(points_xy: np.ndarray) -> np.ndarray:
    center = points_xy.mean(axis=0)
    angles = np.arctan2(points_xy[:, 1] - center[1], points_xy[:, 0] - center[0])
    cyclic = points_xy[np.argsort(angles)]
    if _signed_area(cyclic) < 0:
        cyclic = cyclic[::-1]
    return cyclic


def _signed_area(points_xy: np.ndarray) -> float:
    return float(
        0.5
        * np.sum(
            points_xy[:, 0] * np.roll(points_xy[:, 1], -1)
            - points_xy[:, 1] * np.roll(points_xy[:, 0], -1)
        )
    )


def _long_edge_pair(cyclic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.roll(cyclic, -1, axis=0) - cyclic
    lengths = np.linalg.norm(edges, axis=1)
    first_pair = float((lengths[0] + lengths[2]) / 2.0)
    second_pair = float((lengths[1] + lengths[3]) / 2.0)
    if first_pair >= second_pair * _MIN_LONG_TO_SHORT_EDGE_RATIO:
        return cyclic[[0, 1]], cyclic[[2, 3]]
    if second_pair >= first_pair * _MIN_LONG_TO_SHORT_EDGE_RATIO:
        return cyclic[[1, 2]], cyclic[[3, 0]]
    raise InvalidCornersError("ambiguous_edge_ratio")


def _horizontal_axis(edge: np.ndarray) -> np.ndarray:
    axis = edge[1] - edge[0]
    length = float(np.linalg.norm(axis))
    if length <= _AXIS_COMPONENT_EPSILON:
        raise InvalidCornersError("degenerate_edge")
    axis /= length
    if abs(axis[0]) <= _AXIS_COMPONENT_EPSILON:
        raise InvalidCornersError("ambiguous_axis")
    if axis[0] < 0:
        axis = -axis
    return axis


def _classify_top_bottom(
    long_edges: tuple[np.ndarray, np.ndarray], normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first_projection = float(np.mean(long_edges[0], axis=0) @ normal)
    second_projection = float(np.mean(long_edges[1], axis=0) @ normal)
    if np.isclose(first_projection, second_projection, atol=_AXIS_COMPONENT_EPSILON):
        raise InvalidCornersError("ambiguous_top_bottom")
    if first_projection < second_projection:
        return long_edges[0], long_edges[1]
    return long_edges[1], long_edges[0]


def _order_edge_endpoints(edge: np.ndarray, long_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projections = edge @ long_axis
    if np.isclose(projections[0], projections[1], atol=_AXIS_COMPONENT_EPSILON):
        raise InvalidCornersError("ambiguous_left_right")
    if projections[0] < projections[1]:
        return edge[0], edge[1]
    return edge[1], edge[0]
