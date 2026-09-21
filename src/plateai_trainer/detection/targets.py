"""Deterministic anchor-free assignment in 640x640 letterbox coordinates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from plateai_trainer.detection.contracts import CompositeInstance


@dataclass(frozen=True, slots=True)
class DetectionTargets:
    """One image's targets, independent of Torch/device.

    matched_instance_indices is dense [8400], with -1 for negatives and the
    original input/metadata instance index for positives. Remaining arrays
    are sparse, aligned to sorted positive_indices. corners_xy is [P,4,2]
    in LT, RT, RB, LB order; positive_level_indices is [P] (0=P3, 1=P4,
    2=P5), so one instance can appear up to nine times.
    """

    matched_instance_indices: NDArray[np.int64]
    positive_indices: NDArray[np.int64]
    positive_level_indices: NDArray[np.int64]
    bbox_xyxy: NDArray[np.float32]
    corners_xy: NDArray[np.float32]


def assign_detection_targets(instances: Sequence[CompositeInstance]) -> DetectionTargets:
    """Select cell centres inside each box and within one centre-cell step.

    Centre cells use floor(cx/stride), floor(cy/stride). Both bbox bounds
    are inclusive. The short side chooses <64, [64,128), or >=128 pixels.
    Smaller box area wins conflicts, followed by original instance index.
    Inputs must already have been mapped to letterbox coordinates.
    """
    boxes = np.asarray([item.bbox_xyxy for item in instances], dtype=np.float32).reshape(-1, 4)
    corners = np.asarray([item.corners_xy for item in instances], dtype=np.float32)
    if not instances:
        corners = np.empty((0, 4, 2), dtype=np.float32)
    if not np.isfinite(boxes).all() or np.any(boxes < 0) or np.any(boxes > 640) or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("bbox must be finite, non-degenerate, and within the 640-pixel letterbox")
    if corners.shape != (len(instances), 4, 2) or not np.isfinite(corners).all():
        raise ValueError("corners must be finite [4,2] semantic LT,RT,RB,LB coordinates")
    matched = np.full(8400, -1, dtype=np.int64)
    levels = np.full(8400, -1, dtype=np.int64)
    widths_heights = boxes[:, 2:] - boxes[:, :2]
    areas = widths_heights[:, 0] * widths_heights[:, 1]
    for instance_index in sorted(range(len(instances)), key=lambda index: (float(areas[index]), index)):
        box = boxes[instance_index]
        short_side = min(widths_heights[instance_index])
        level = 0 if short_side < 64 else 1 if short_side < 128 else 2
        stride, offset = ((8, 0), (16, 6400), (32, 8000))[level]
        grid_size = 640 // stride
        cx, cy = (box[:2] + box[2:]) / 2
        centre_x, centre_y = int(cx // stride), int(cy // stride)
        for y in range(max(0, centre_y - 1), min(grid_size, centre_y + 2)):
            for x in range(max(0, centre_x - 1), min(grid_size, centre_x + 2)):
                px, py = (x + 0.5) * stride, (y + 0.5) * stride
                cell = offset + y * grid_size + x
                if box[0] <= px <= box[2] and box[1] <= py <= box[3] and matched[cell] == -1:
                    matched[cell] = instance_index
                    levels[cell] = level
    positives = np.flatnonzero(matched >= 0)
    owners = matched[positives]
    return DetectionTargets(matched, positives, levels[positives], boxes[owners], corners[owners])
