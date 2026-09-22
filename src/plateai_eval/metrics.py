"""Deterministic matching and metrics for strict M4.5 evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import cv2
import numpy as np

from plateai_shared.contracts import JsonValue

from .contracts import DetectionPrediction, EvaluationInputError, PlateTruth


AttributionCode = Literal[
    "DETECTOR_MISSED",
    "DETECTOR_FALSE_POSITIVE",
    "DETECTOR_BAD_BBOX",
    "DETECTOR_BAD_QUAD",
    "RECTIFIER_REJECTED",
    "RECOGNIZER_MISREAD",
    "RULE_FILTERED",
    "SUCCESS",
]
BBox: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PredictionTruthMatch:
    prediction_index: int
    truth_index: int
    prediction_id: str
    truth_id: str
    iou: float
    corner_errors: tuple[float, float, float, float]
    truth_canonical: str | None
    bbox_passed: bool = True
    quad_passed: bool = True


@dataclass(frozen=True, slots=True)
class MatchSet:
    matches: tuple[PredictionTruthMatch, ...]
    unmatched_prediction_ids: tuple[str, ...]
    unmatched_truth_ids: tuple[str, ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_truth_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SceneMetricInput:
    predictions: tuple[DetectionPrediction, ...]
    truths: tuple[PlateTruth, ...]


def levenshtein_distance(first: str, second: str) -> int:
    """Return the exact character edit distance using bounded working memory."""

    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for first_index, first_character in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_character in enumerate(second, start=1):
            current.append(
                min(
                    previous[second_index] + 1,
                    current[second_index - 1] + 1,
                    previous[second_index - 1]
                    + (first_character != second_character),
                )
            )
        previous = current
    return previous[-1]


def compute_cer(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Compatibility CER over zipped prediction/target pairs."""

    return float(ocr_metrics(tuple(zip(predictions, targets)))["micro_cer"])


def ocr_metrics(
    pairs: Sequence[tuple[str, str]],
) -> Mapping[str, JsonValue]:
    """Aggregate exact match and micro CER over ``(prediction, truth)`` pairs."""

    exact_match_count = 0
    total_edit_distance = 0
    total_truth_characters = 0
    for index, pair in enumerate(pairs):
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not isinstance(pair[1], str)
        ):
            raise EvaluationInputError(f"ocr pair {index}: expected two strings")
        prediction, truth = pair
        exact_match_count += prediction == truth
        total_edit_distance += levenshtein_distance(prediction, truth)
        total_truth_characters += len(truth)
    truth_count = len(pairs)
    return {
        "truth_count": truth_count,
        "exact_match_count": exact_match_count,
        "total_edit_distance": total_edit_distance,
        "total_truth_characters": total_truth_characters,
        "exact_match": exact_match_count / truth_count if truth_count else 0.0,
        "micro_cer": (
            total_edit_distance / total_truth_characters
            if total_truth_characters
            else 0.0
        ),
    }


def calculate_polygon_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Compatibility helper for exact continuous convex-polygon IoU."""

    first = np.asarray(poly1, dtype=np.float32).reshape(-1, 2)
    second = np.asarray(poly2, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise EvaluationInputError("polygon coordinates must be finite")
    first_hull = cv2.convexHull(first).reshape(-1, 2)
    second_hull = cv2.convexHull(second).reshape(-1, 2)
    if len(first_hull) < 3 or len(second_hull) < 3:
        return 0.0
    intersection_area, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    first_area = cv2.contourArea(first_hull)
    second_area = cv2.contourArea(second_hull)
    union_area = first_area + second_area - intersection_area
    return float(intersection_area / union_area) if union_area > 0 else 0.0


def _finite_values(values: object, label: str, *, count: int) -> tuple[float, ...]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError(f"{label}: coordinates must be finite") from exc
    if array.size != count or not np.isfinite(array).all():
        raise EvaluationInputError(f"{label}: coordinates must be finite")
    return tuple(float(value) for value in array.reshape(-1))


def _validate_prediction(
    prediction: DetectionPrediction, index: int
) -> tuple[BBox, np.ndarray]:
    if not math.isfinite(prediction.confidence):
        raise EvaluationInputError(
            f"prediction {index} confidence must be finite"
        )
    bbox_values = _finite_values(
        prediction.bbox_xyxy, f"prediction {index} bbox", count=4
    )
    bbox: BBox = (
        bbox_values[0],
        bbox_values[1],
        bbox_values[2],
        bbox_values[3],
    )
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        raise EvaluationInputError(f"prediction {index}: invalid bbox")
    corner_values = _finite_values(
        prediction.corners, f"prediction {index} corners", count=8
    )
    return bbox, np.asarray(corner_values, dtype=np.float64).reshape(4, 2)


def _validate_truth(truth: PlateTruth, index: int) -> tuple[BBox, np.ndarray]:
    corner_values = _finite_values(truth.corners, f"truth {index} corners", count=8)
    corners = np.asarray(corner_values, dtype=np.float64).reshape(4, 2)
    bbox: BBox = (
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].max()),
    )
    return bbox, corners


def bbox_iou(first: BBox, second: BBox) -> float:
    """Return continuous axis-aligned IoU for two validated ``xyxy`` boxes."""

    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions(
    predictions: Sequence[DetectionPrediction],
    truths: Sequence[PlateTruth],
    iou_min: float,
) -> MatchSet:
    """Score-order predictions into deterministic, one-to-one bbox matches."""

    if not math.isfinite(iou_min) or not 0.0 <= iou_min <= 1.0:
        raise EvaluationInputError("iou_min must be finite and within [0, 1]")
    prediction_ids = [prediction.prediction_id for prediction in predictions]
    truth_ids = [truth.instance_id for truth in truths]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise EvaluationInputError("duplicate prediction_id")
    if len(set(truth_ids)) != len(truth_ids):
        raise EvaluationInputError("duplicate truth instance_id")

    validated_predictions = [
        _validate_prediction(prediction, index)
        for index, prediction in enumerate(predictions)
    ]
    validated_truths = [
        _validate_truth(truth, index) for index, truth in enumerate(truths)
    ]
    ordered_prediction_indices = sorted(
        range(len(predictions)),
        key=lambda index: (
            -predictions[index].confidence,
            index,
            predictions[index].prediction_id,
        ),
    )
    remaining_truth_indices = set(range(len(truths)))
    matches: list[PredictionTruthMatch] = []
    unmatched_prediction_indices: list[int] = []

    for prediction_index in ordered_prediction_indices:
        prediction_bbox, prediction_corners = validated_predictions[prediction_index]
        candidates = [
            (bbox_iou(prediction_bbox, validated_truths[truth_index][0]), truth_index)
            for truth_index in sorted(remaining_truth_indices)
        ]
        if not candidates:
            unmatched_prediction_indices.append(prediction_index)
            continue
        best_iou, truth_index = min(candidates, key=lambda item: (-item[0], item[1]))
        if best_iou < iou_min:
            unmatched_prediction_indices.append(prediction_index)
            continue
        remaining_truth_indices.remove(truth_index)
        truth_corners = validated_truths[truth_index][1]
        errors = np.linalg.norm(prediction_corners - truth_corners, axis=1)
        matches.append(
            PredictionTruthMatch(
                prediction_index=prediction_index,
                truth_index=truth_index,
                prediction_id=predictions[prediction_index].prediction_id,
                truth_id=truths[truth_index].instance_id,
                iou=float(best_iou),
                corner_errors=tuple(float(value) for value in errors),  # type: ignore[arg-type]
                truth_canonical=truths[truth_index].canonical,
            )
        )

    unmatched_truth_indices = tuple(sorted(remaining_truth_indices))
    return MatchSet(
        matches=tuple(matches),
        unmatched_prediction_ids=tuple(
            predictions[index].prediction_id for index in unmatched_prediction_indices
        ),
        unmatched_truth_ids=tuple(
            truths[index].instance_id for index in unmatched_truth_indices
        ),
        unmatched_prediction_indices=tuple(unmatched_prediction_indices),
        unmatched_truth_indices=unmatched_truth_indices,
    )


def detector_metrics(
    scene_pairs: Sequence[SceneMetricInput],
    iou_min: float,
    max_corner_error: float,
) -> Mapping[str, JsonValue]:
    """Aggregate bbox and semantic complete-quad precision/recall."""

    if not math.isfinite(max_corner_error) or max_corner_error < 0:
        raise EvaluationInputError("max_corner_error must be finite and non-negative")
    prediction_count = 0
    truth_count = 0
    bbox_true_positives = 0
    complete_quad_true_positives = 0
    for scene in scene_pairs:
        prediction_count += len(scene.predictions)
        truth_count += len(scene.truths)
        matches = match_predictions(scene.predictions, scene.truths, iou_min)
        bbox_true_positives += len(matches.matches)
        complete_quad_true_positives += sum(
            all(error <= max_corner_error for error in match.corner_errors)
            for match in matches.matches
        )
    return {
        "truth_count": truth_count,
        "prediction_count": prediction_count,
        "bbox_true_positives": bbox_true_positives,
        "bbox_precision": (
            bbox_true_positives / prediction_count if prediction_count else 0.0
        ),
        "bbox_recall": bbox_true_positives / truth_count if truth_count else 0.0,
        "complete_quad_true_positives": complete_quad_true_positives,
        "complete_quad_precision": (
            complete_quad_true_positives / prediction_count
            if prediction_count
            else 0.0
        ),
        "complete_quad_recall": (
            complete_quad_true_positives / truth_count if truth_count else 0.0
        ),
    }


def attribute_e2e(
    match: PredictionTruthMatch | None,
    prediction: DetectionPrediction | None,
) -> AttributionCode:
    """Attribute the earliest stage that prevents one correct E2E result."""

    if match is None:
        return "DETECTOR_FALSE_POSITIVE" if prediction is not None else "DETECTOR_MISSED"
    if prediction is None or not match.bbox_passed:
        return "DETECTOR_BAD_BBOX"
    if not match.quad_passed:
        return "DETECTOR_BAD_QUAD"
    if prediction.rejection_stage == "rectifier":
        return "RECTIFIER_REJECTED"
    recognition = prediction.recognition
    if recognition is None:
        return "RECOGNIZER_MISREAD"
    if recognition.canonical != match.truth_canonical:
        if recognition.greedy == match.truth_canonical:
            return "RULE_FILTERED"
        return "RECOGNIZER_MISREAD"
    return "SUCCESS"


__all__ = [
    "AttributionCode",
    "BBox",
    "MatchSet",
    "PredictionTruthMatch",
    "SceneMetricInput",
    "attribute_e2e",
    "bbox_iou",
    "calculate_polygon_iou",
    "compute_cer",
    "detector_metrics",
    "levenshtein_distance",
    "match_predictions",
    "ocr_metrics",
]
