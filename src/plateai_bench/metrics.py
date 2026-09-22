"""Metric calculation and failure attribution for license plate evaluation."""
from __future__ import annotations

from enum import Enum

from plateai_eval.metrics import (
    calculate_polygon_iou,
    compute_cer,
    levenshtein_distance,
)


class AttributionCategory(str, Enum):
    SUCCESS = "SUCCESS"
    LOCATOR_MISSED = "LOCATOR_MISSED"
    LOCATOR_BAD_CROP = "LOCATOR_BAD_CROP"
    RECOGNIZER_MISREAD = "RECOGNIZER_MISREAD"
    RULE_FILTERED = "RULE_FILTERED"


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
