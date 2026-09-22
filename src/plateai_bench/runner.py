"""Benchmark runner for running Oracle OCR and End-to-End evaluation on real photos."""
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence
import cv2
import numpy as np
import onnxruntime as ort

from plateai_reader import decode_constrained_ctc_v1
from plateai_reader.rectifier import rectify_plate
from plateai_reader.runtime import PlateReader
from plateai_shared.recognition import CTCCodec, preprocess_v1_rgb
from plateai_shared.rules import load_character_set, load_ruleset

from .metrics import (
    AttributionCategory,
    calculate_polygon_iou,
    classify_failure,
    compute_cer,
    levenshtein_distance,
)


@dataclass
class EvalSampleResult:
    index: int
    image_path: str
    gt_canonical: str
    gt_display: str
    # Oracle OCR
    oracle_canonical: str = ""
    oracle_display: str = ""
    oracle_greedy: str = ""
    oracle_exact: bool = False
    oracle_cer: float = 1.0
    oracle_latency_ms: float = 0.0
    # End-to-End
    e2e_canonical: str = ""
    e2e_display: str = ""
    e2e_greedy: str = ""
    e2e_exact: bool = False
    e2e_cer: float = 1.0
    e2e_detected: bool = False
    e2e_iou: float = 0.0
    e2e_latency_ms: float = 0.0
    attribution: str = AttributionCategory.LOCATOR_MISSED.value
    # Timing details
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    dataset_name: str
    sample_count: int
    bundle_name: str
    # Oracle Metrics
    oracle_accuracy: float
    oracle_cer: float
    oracle_char_accuracy: float
    oracle_avg_latency_ms: float
    oracle_p50_latency_ms: float
    oracle_p95_latency_ms: float
    # End-to-End Metrics
    e2e_accuracy: float
    e2e_cer: float
    e2e_char_accuracy: float
    localization_recall: float
    e2e_avg_latency_ms: float
    # Attribution counts
    attributions: dict[str, int]
    # Sample lists
    failures: list[dict[str, Any]] = field(default_factory=list)
    successes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """Executes dual-mode benchmark evaluation on real-photo datasets."""

    def __init__(
        self,
        bundle_dir: Path | str,
        device: str = "cpu",
    ):
        self.bundle_dir = Path(bundle_dir)
        self.manifest_path = self.bundle_dir / "manifest.json"
        
        manifest_data = {}
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as f:
                manifest_data = json.load(f)
        self.manifest = manifest_data
        
        # Load recognizer
        recognizer_path = self.bundle_dir / "recognizer.onnx"
        if not recognizer_path.exists():
            raise FileNotFoundError(f"Missing recognizer ONNX at {recognizer_path}")
            
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(recognizer_path), providers=providers)
        
        charset_file = self.bundle_dir / "charset.txt"
        rules_file = self.bundle_dir / "plate_rules.json"
        self.charset = load_character_set(charset_file)
        self.codec = CTCCodec.from_charset(self.charset)
        self.ruleset = load_ruleset(rules_file, self.charset)

        # Check if full PlateReader (with detector) is available
        self.reader: PlateReader | None = None
        caps = self.manifest.get("capabilities", [])
        if "plate-detection" in caps and (self.bundle_dir / "detector.onnx").exists():
            try:
                self.reader = PlateReader(self.bundle_dir, providers=providers)
            except Exception:
                self.reader = None

    def _eval_single_crop(self, crop_rgb: np.ndarray) -> tuple[str, str, str, float, dict[str, float]]:
        """Run Recognizer ONNX and constrained decoding on a rectified RGB crop."""
        t_pre0 = time.perf_counter()
        tensor = preprocess_v1_rgb(crop_rgb)
        t_onnx0 = time.perf_counter()
        logits = self.session.run(["logits"], {"input": tensor[np.newaxis, ...]})[0]
        t_dec0 = time.perf_counter()
        greedy = self.codec.decode_greedy(logits[0].argmax(axis=1).tolist())
        decoded = decode_constrained_ctc_v1(logits[0], self.codec, self.ruleset)
        t_end = time.perf_counter()

        timings = {
            "preprocess_ms": round((t_onnx0 - t_pre0) * 1000, 2),
            "onnx_ms": round((t_dec0 - t_onnx0) * 1000, 2),
            "ctc_ms": round((t_end - t_dec0) * 1000, 2),
            "total_ms": round((t_end - t_pre0) * 1000, 2),
        }
        return decoded.canonical, decoded.display, greedy, decoded.log_probability, timings

    def run_benchmark(
        self,
        records: Sequence[dict[str, Any]],
        dataset_root: Path | str,
        dataset_name: str = "benchmark",
        run_e2e: bool = True,
        progress_cb: Callable[[int, str], None] | None = None,
    ) -> BenchmarkSummary:
        """Run evaluation over records."""
        if run_e2e and self.reader is None:
            raise RuntimeError(
                "diagnostic E2E requires a full bundle with a declared detector"
            )
        dataset_root = Path(dataset_root)
        total = len(records)
        sample_results: list[EvalSampleResult] = []

        oracle_exact_count = 0
        oracle_char_errors = 0
        oracle_total_chars = 0
        oracle_latencies = []

        e2e_exact_count = 0
        e2e_char_errors = 0
        e2e_total_chars = 0
        e2e_localized_count = 0
        e2e_latencies = []

        attribution_counts = {cat.value: 0 for cat in AttributionCategory}

        for idx, rec in enumerate(records):
            if progress_cb and (idx % max(1, total // 20) == 0 or idx == total - 1):
                progress_cb(int(idx / total * 100), f"評測中: [{idx+1}/{total}] {rec.get('ground_truth', {}).get('canonical', '')}")

            gt_info = rec["ground_truth"]
            gt_canon = gt_info["canonical"].strip().upper()
            gt_disp = gt_info.get("display", gt_canon)
            
            # Ground truth corners: accept 'corners', 'xywhr', or 'polygon'
            if "corners" in gt_info:
                gt_corners = np.asarray(gt_info["corners"], dtype=np.float32)
            elif "xywhr" in gt_info:
                cx, cy, w, h, rad = gt_info["xywhr"]
                gt_corners = cv2.boxPoints(((cx, cy), (w, h), math.degrees(rad))).astype(np.float32)
            elif "polygon" in gt_info:
                gt_corners = np.asarray(gt_info["polygon"], dtype=np.float32)
            else:
                raise ValueError(f"Record {idx} missing corners/xywhr/polygon")

            # Load image
            img_rel_path = rec["image"]["path"]
            img_file = dataset_root / img_rel_path
            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            sample = EvalSampleResult(
                index=idx,
                image_path=str(img_rel_path),
                gt_canonical=gt_canon,
                gt_display=gt_disp,
            )

            # ----------------------------------------------------
            # 1. Oracle Crop Evaluation
            # ----------------------------------------------------
            try:
                rect_res = rectify_plate(img_rgb, gt_corners)
                o_canon, o_disp, o_greedy, o_logp, o_timings = self._eval_single_crop(rect_res.image_rgb)
                sample.oracle_canonical = o_canon
                sample.oracle_display = o_disp
                sample.oracle_greedy = o_greedy
                sample.oracle_exact = (o_canon == gt_canon)
                sample.oracle_cer = float(levenshtein_distance(o_canon, gt_canon) / max(1, len(gt_canon)))
                sample.oracle_latency_ms = o_timings["total_ms"]
                sample.timings.update({f"oracle_{k}": v for k, v in o_timings.items()})

                if sample.oracle_exact:
                    oracle_exact_count += 1
                oracle_char_errors += levenshtein_distance(o_canon, gt_canon)
                oracle_total_chars += len(gt_canon)
                oracle_latencies.append(sample.oracle_latency_ms)
            except Exception as exc:
                sample.oracle_canonical = ""
                sample.oracle_exact = False
                sample.oracle_cer = 1.0
                oracle_char_errors += len(gt_canon)
                oracle_total_chars += len(gt_canon)

            # ----------------------------------------------------
            # 2. End-to-End Pipeline Evaluation
            # ----------------------------------------------------
            if run_e2e:
                t_e2e_start = time.perf_counter()
                best_iou = 0.0
                best_e2e_canon = ""
                best_e2e_disp = ""
                best_e2e_greedy = ""
                detected = False

                assert self.reader is not None
                reader_res = self.reader.read(img_rgb)
                for plate_read in reader_res.plates:
                    det_corners = plate_read.detection.corners_xy
                    iou = calculate_polygon_iou(det_corners, gt_corners)
                    if iou > best_iou:
                        best_iou = iou
                        best_e2e_canon = plate_read.decoded.canonical
                        best_e2e_disp = plate_read.decoded.display
                        best_e2e_greedy = plate_read.raw_greedy_text or ""
                        detected = True

                t_e2e_end = time.perf_counter()
                sample.e2e_detected = detected and (best_iou >= 0.5)
                sample.e2e_iou = round(float(best_iou), 3)
                sample.e2e_canonical = best_e2e_canon
                sample.e2e_display = best_e2e_disp
                sample.e2e_greedy = best_e2e_greedy
                sample.e2e_exact = (best_e2e_canon == gt_canon)
                sample.e2e_cer = float(levenshtein_distance(best_e2e_canon, gt_canon) / max(1, len(gt_canon)))
                sample.e2e_latency_ms = round((t_e2e_end - t_e2e_start) * 1000, 2)
                e2e_latencies.append(sample.e2e_latency_ms)

                if sample.e2e_detected:
                    e2e_localized_count += 1
                if sample.e2e_exact:
                    e2e_exact_count += 1
                e2e_char_errors += levenshtein_distance(best_e2e_canon, gt_canon)
                e2e_total_chars += len(gt_canon)

                # Attribute root cause
                attr = classify_failure(
                    detected=detected,
                    iou=best_iou,
                    pred_canonical=best_e2e_canon,
                    gt_canonical=gt_canon,
                    greedy_text=best_e2e_greedy or sample.oracle_greedy,
                )
                sample.attribution = attr.value
                attribution_counts[attr.value] += 1

            sample_results.append(sample)

        # Aggregate metrics
        oracle_acc = float(oracle_exact_count / total) if total > 0 else 0.0
        oracle_cer = float(oracle_char_errors / oracle_total_chars) if oracle_total_chars > 0 else 1.0
        oracle_char_acc = max(0.0, 1.0 - oracle_cer)

        e2e_acc = float(e2e_exact_count / total) if total > 0 else 0.0
        e2e_cer = float(e2e_char_errors / e2e_total_chars) if e2e_total_chars > 0 else 1.0
        e2e_char_acc = max(0.0, 1.0 - e2e_cer)
        loc_recall = float(e2e_localized_count / total) if total > 0 else 0.0

        failures = [
            asdict(s) for s in sample_results if not s.oracle_exact or (run_e2e and not s.e2e_exact)
        ]
        successes = [
            asdict(s) for s in sample_results if s.oracle_exact and (not run_e2e or s.e2e_exact)
        ]

        summary = BenchmarkSummary(
            dataset_name=dataset_name,
            sample_count=total,
            bundle_name=self.bundle_dir.name,
            oracle_accuracy=round(oracle_acc, 4),
            oracle_cer=round(oracle_cer, 4),
            oracle_char_accuracy=round(oracle_char_acc, 4),
            oracle_avg_latency_ms=round(statistics.mean(oracle_latencies), 1) if oracle_latencies else 0.0,
            oracle_p50_latency_ms=round(statistics.median(oracle_latencies), 1) if oracle_latencies else 0.0,
            oracle_p95_latency_ms=round(
                statistics.quantiles(oracle_latencies, n=20)[18] if len(oracle_latencies) >= 20 else max(oracle_latencies or [0]), 1
            ),
            e2e_accuracy=round(e2e_acc, 4),
            e2e_cer=round(e2e_cer, 4),
            e2e_char_accuracy=round(e2e_char_acc, 4),
            localization_recall=round(loc_recall, 4),
            e2e_avg_latency_ms=round(statistics.mean(e2e_latencies), 1) if e2e_latencies else 0.0,
            attributions=attribution_counts,
            failures=failures,
            successes=successes,
        )
        return summary
