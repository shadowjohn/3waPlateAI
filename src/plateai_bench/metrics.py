"""Metric calculation and failure attribution for license plate evaluation."""
from __future__ import annotations

from enum import Enum
from typing import Sequence
import cv2
import numpy as np


class AttributionCategory(str, Enum):
    SUCCESS = "SUCCESS"
    LOCATOR_MISSED = "LOCATOR_MISSED"
    LOCATOR_BAD_CROP = "LOCATOR_BAD_CROP"
    RECOGNIZER_MISREAD = "RECOGNIZER_MISREAD"
    RULE_FILTERED = "RULE_FILTERED"


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate character edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


def compute_cer(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Compute Character Error Rate (CER) across pairs of strings.
    
    CER = sum(Levenshtein(pred, target)) / sum(len(target))
    """
    total_dist = 0
    total_chars = 0
    for pred, target in zip(predictions, targets):
        total_dist += levenshtein_distance(pred, target)
        total_chars += len(target)
    if total_chars == 0:
        return 0.0
    return float(total_dist / total_chars)


def calculate_polygon_iou(
    poly1: np.ndarray,
    poly2: np.ndarray,
) -> float:
    """Compute exact continuous Intersection-over-Union (IoU) between two convex 4-point polygons."""
    p1 = cv2.convexHull(np.asarray(poly1, dtype=np.float32).reshape(-1, 2)).squeeze(1)
    p2 = cv2.convexHull(np.asarray(poly2, dtype=np.float32).reshape(-1, 2)).squeeze(1)
    if len(p1) < 3 or len(p2) < 3:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(p1, p2)
    area1 = cv2.contourArea(p1)
    area2 = cv2.contourArea(p2)
    union_area = area1 + area2 - inter_area
    return float(inter_area / union_area) if union_area > 0 else 0.0


def classify_failure(
    detected: bool,
    iou: float,
    pred_canonical: str,
    gt_canonical: str,
    greedy_text: str = "",
    iou_threshold: float = 0.5,
) -> AttributionCategory:
    """Classify the root cause of a recognition failure into an AttributionCategory."""
    if not detected:
        return AttributionCategory.LOCATOR_MISSED
    if iou < iou_threshold:
        return AttributionCategory.LOCATOR_BAD_CROP
    if pred_canonical == gt_canonical:
        return AttributionCategory.SUCCESS
    # At this point, localization was successful (iou >= threshold) but text didn't match
    clean_greedy = greedy_text.replace("-", "").replace(" ", "").upper()
    if clean_greedy == gt_canonical and pred_canonical != gt_canonical:
        return AttributionCategory.RULE_FILTERED
    return AttributionCategory.RECOGNIZER_MISREAD
