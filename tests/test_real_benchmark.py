"""Unit tests for plateai_bench metrics, attribution, and benchmark runner."""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from plateai_bench.metrics import (
    AttributionCategory,
    calculate_polygon_iou,
    classify_failure,
    compute_cer,
    levenshtein_distance,
)
from plateai_bench.runner import BenchmarkRunner


def test_levenshtein_and_cer():
    assert levenshtein_distance("MDX9717", "MDX9717") == 0
    assert levenshtein_distance("MDX9717", "MDX971") == 1
    assert levenshtein_distance("213NSK", "20MS") == 4
    assert levenshtein_distance("", "ABC") == 3

    # Test CER
    preds = ["MDX971", "20MS"]
    targets = ["MDX9717", "213NSK"]
    # distances: 1 + 4 = 5. total chars: 7 + 6 = 13.
    cer = compute_cer(preds, targets)
    assert pytest.approx(cer, rel=1e-3) == 5 / 13


def test_calculate_polygon_iou():
    # Identical axis-aligned box
    poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    poly2 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    assert pytest.approx(calculate_polygon_iou(poly1, poly2), rel=1e-2) == 1.0

    # Half overlap: poly2 shifted right by 5
    poly3 = np.array([[5, 0], [15, 0], [15, 10], [5, 10]], dtype=np.float32)
    # intersection: 5*10 = 50. union: 15*10 = 150. IoU = 50/150 = 0.333
    iou = calculate_polygon_iou(poly1, poly3)
    assert pytest.approx(iou, rel=1e-2) == 0.333

    # Disjoint boxes
    poly4 = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float32)
    assert calculate_polygon_iou(poly1, poly4) == 0.0


def test_classify_failure():
    # 1. Success
    assert classify_failure(True, 0.9, "ABC1234", "ABC1234") == AttributionCategory.SUCCESS

    # 2. Locator Missed
    assert classify_failure(False, 0.0, "", "ABC1234") == AttributionCategory.LOCATOR_MISSED

    # 3. Bad Crop (IoU < 0.5)
    assert classify_failure(True, 0.3, "ABC1234", "ABC1234") == AttributionCategory.LOCATOR_BAD_CROP

    # 4. Recognizer Misread
    assert classify_failure(True, 0.85, "ABC123", "ABC1234", greedy_text="ABC123") == AttributionCategory.RECOGNIZER_MISREAD

    # 5. Rule Filtered (Greedy got correct string, but constrained rule didn't)
    assert classify_failure(True, 0.85, "XYZ99", "ABC1234", greedy_text="ABC-1234") == AttributionCategory.RULE_FILTERED


def test_benchmark_runner_user_dataset(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    bundle_dir = root / "models" / "bundles" / "active-v1"
    if not (bundle_dir / "recognizer.onnx").exists():
        pytest.skip("active-v1 bundle not present")

    user_cases_file = root / "datasets" / "real_benchmarks" / "user_cases.jsonl"
    if not user_cases_file.exists():
        pytest.skip("user_cases.jsonl not present")

    runner = BenchmarkRunner(bundle_dir)
    records = [json.loads(line) for line in open(user_cases_file, encoding="utf-8")]

    summary = runner.run_benchmark(
        records=records,
        dataset_root=root / "datasets" / "real_benchmarks",
        dataset_name="test_user_cases",
        run_e2e=True,
    )

    assert summary.sample_count == 2
    assert summary.bundle_name == "active-v1"
    assert 0.0 <= summary.oracle_accuracy <= 1.0
    assert 0.0 <= summary.oracle_cer <= 1.0
    assert summary.oracle_avg_latency_ms > 0
    assert summary.e2e_avg_latency_ms > 0
    assert len(summary.attributions) == 5
