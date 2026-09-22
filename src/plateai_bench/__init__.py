"""3waPlateAI Benchmark & Real-Photo Evaluation Package."""
from __future__ import annotations

from .metrics import (
    AttributionCategory,
    calculate_polygon_iou,
    classify_failure,
    compute_cer,
    levenshtein_distance,
)
from .runner import BenchmarkRunner, BenchmarkSummary

__all__ = [
    "AttributionCategory",
    "calculate_polygon_iou",
    "classify_failure",
    "compute_cer",
    "levenshtein_distance",
    "BenchmarkRunner",
    "BenchmarkSummary",
]
