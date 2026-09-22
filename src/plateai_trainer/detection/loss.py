"""Objectness focal BCE + 5 CIoU + 2 normalized semantic-corner Smooth L1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from plateai_trainer.detection.targets import DetectionTargets


FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25


@dataclass(frozen=True, slots=True)
class DetectorLoss:
    total: torch.Tensor
    objectness: torch.Tensor
    box_ciou: torch.Tensor
    corner_smooth_l1: torch.Tensor


def _ciou_loss(predicted: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-7
    target_sizes = boxes[:, 2:] - boxes[:, :2]
    target_centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = predicted[:, 2:4].clamp_min(epsilon)
    lower, upper = predicted[:, :2] - sizes / 2, predicted[:, :2] + sizes / 2
    overlap = (torch.minimum(upper, boxes[:, 2:]) - torch.maximum(lower, boxes[:, :2])).clamp_min(0)
    intersection = overlap.prod(dim=1)
    union = sizes.prod(dim=1) + target_sizes.prod(dim=1) - intersection
    iou = intersection / union.clamp_min(epsilon)
    enclosing = torch.maximum(upper, boxes[:, 2:]) - torch.minimum(lower, boxes[:, :2])
    diagonal_squared = enclosing.square().sum(dim=1).clamp_min(epsilon)
    distance_squared = (predicted[:, :2] - target_centres).square().sum(dim=1)
    aspect = (4 / math.pi**2) * (
        torch.atan(target_sizes[:, 0] / target_sizes[:, 1]) - torch.atan(sizes[:, 0] / sizes[:, 1])
    ).square()
    alpha = (aspect / (1 - iou + aspect).clamp_min(epsilon)).detach()
    return (1 - iou + distance_squared / diagonal_squared + alpha * aspect).mean()


def detection_loss(
    predictions: torch.Tensor, targets: DetectionTargets | Sequence[DetectionTargets]
) -> DetectorLoss:
    """Sum focal BCE normalized by positive count; positive geometry means.

    Pass a single DetectionTargets for batch one, or one target object per
    image in batch order. Arrays are copied to predictions' device/dtype.
    Empty-positive batches retain a differentiable zero geometry loss.
    Objectness uses gamma=2.0, alpha=0.25 for positives and 1-alpha for
    negatives: -alpha_t * (1-p_t)**gamma * log(p_t).
    Normalizing by all 8,400 cells diluted objectness relative to geometry
    by hundreds-fold. Empty-positive batches use a denominator of one.
    """
    batch_targets = [targets] if isinstance(targets, DetectionTargets) else list(targets)
    if predictions.ndim != 3 or tuple(predictions.shape[1:]) != (8400, 13) or predictions.shape[0] != len(batch_targets) or not batch_targets:
        raise ValueError("predictions must be [batch,8400,13] with one target per batch image")
    objectness_targets = torch.zeros_like(predictions[..., 4])
    positives, boxes, corners = [], [], []
    for batch_index, target in enumerate(batch_targets):
        indices = torch.as_tensor(target.positive_indices, dtype=torch.long, device=predictions.device)
        objectness_targets[batch_index, indices] = 1
        positives.append(predictions[batch_index, indices])
        boxes.append(torch.as_tensor(target.bbox_xyxy, dtype=predictions.dtype, device=predictions.device))
        corners.append(torch.as_tensor(target.corners_xy, dtype=predictions.dtype, device=predictions.device))
    probabilities = predictions[..., 4].clamp(1e-7, 1 - 1e-7)
    bce = F.binary_cross_entropy(probabilities, objectness_targets, reduction="none")
    p_t = probabilities * objectness_targets + (1 - probabilities) * (1 - objectness_targets)
    alpha_t = FOCAL_ALPHA * objectness_targets + (1 - FOCAL_ALPHA) * (1 - objectness_targets)
    positive_count = sum(len(target.positive_indices) for target in batch_targets)
    objectness = (alpha_t * (1 - p_t).pow(FOCAL_GAMMA) * bce).sum() / max(1, positive_count)
    positive_predictions = torch.cat(positives)
    if positive_predictions.shape[0]:
        matched_boxes = torch.cat(boxes)
        matched_corners = torch.cat(corners)
        box_ciou = _ciou_loss(positive_predictions[:, :4], matched_boxes)
        scale = (matched_boxes[:, 2:] - matched_boxes[:, :2]).unsqueeze(1)
        residual = (positive_predictions[:, 5:].reshape(-1, 4, 2) - matched_corners) / scale
        corner_smooth_l1 = F.smooth_l1_loss(residual, torch.zeros_like(residual))
    else:
        box_ciou = positive_predictions[:, :4].sum() * 0
        corner_smooth_l1 = positive_predictions[:, 5:].sum() * 0
    total = objectness + 5.0 * box_ciou + 2.0 * corner_smooth_l1
    return DetectorLoss(total, objectness, box_ciou, corner_smooth_l1)
