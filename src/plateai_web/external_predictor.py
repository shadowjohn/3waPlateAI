"""Local-only A/B preview for the pinned external OCR over a native locator."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from plateai_reader.fpga_lpr import FpgaLprRecognizer
from plateai_reader.fpga_pipeline import FpgaSceneReader
from plateai_reader.runtime import PlateReader

from .predictor import _image_to_base64_jpeg


DETECTOR_BUNDLE = "candidate-detector-real-v1"
MODEL_ID = "fpga-lpr-mit-v1"
WARNINGS = (
    "作者 CPM + LPRNet 尚未經獨立實拍盲測；TLPD 訓練來源重播不是泛化準確率。",
    "本機 A/B 使用授權尚待確認的實驗 Detector；不得將其權重納入發行包。",
    "OCR 分數未校準；此候選不會取代 active-v1。",
)


class ExternalPredictorEngine:
    """Cache one vetted OCR pair and one fixed local detector bundle."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()
        self.scene_reader: FpgaSceneReader | None = None
        self.load_error: str | None = None
        self._load_models()

    def _load_models(self) -> None:
        try:
            recognizer = FpgaLprRecognizer(self.root / "third_party" / "fpga_lpr")
            detector = PlateReader(self.root / "models" / "bundles" / DETECTOR_BUNDLE)
            self.scene_reader = FpgaSceneReader(detector, recognizer)
            self.load_error = None
        except Exception as exc:
            self.scene_reader = None
            self.load_error = str(exc)

    def predict_image(self, image_bytes: bytes) -> dict[str, Any]:
        with self._lock:
            return self._predict_image(image_bytes)

    def _predict_image(self, image_bytes: bytes) -> dict[str, Any]:
        started = time.perf_counter()
        decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        height, width = (decoded.shape[:2] if decoded is not None else (0, 0))
        diagnostics: dict[str, Any] = {
            "bundle_name": "fpga-lpr-mit",
            "model_id": MODEL_ID,
            "recognizer_type": "fpga-lpr-mit",
            "detector_bundle": DETECTOR_BUNDLE,
            "locator_type": "plate_pose_net" if self.scene_reader else None,
            "detector_available": self.scene_reader is not None,
            "pipeline_mode": "neural_full_pipeline" if self.scene_reader else "unavailable",
            "score_kind": "uncalibrated",
            "source_url": "https://github.com/evan6007/FPGA-LPR",
            "warnings": list(WARNINGS),
            "error": self.load_error,
            "timing_breakdown": {
                "locator_ms": 0.0, "rectifier_ms": 0.0, "onnx_inference_ms": 0.0,
                "ctc_decoding_ms": 0.0, "total_ms": 0.0,
            },
        }
        detections: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        status = "no_plate"
        if self.scene_reader is None:
            status = "model_error"
        elif decoded is None:
            status = "inference_error"
            diagnostics["error"] = "invalid_image"
        else:
            try:
                scene = self.scene_reader.read(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
                diagnostics["timing_breakdown"].update(
                    locator_ms=round(scene.timings_ms.get("detector", 0.0), 1),
                    rectifier_ms=round(
                        sum(item.read.timings_ms.get("rectify", 0.0) for item in scene.plates), 1
                    ),
                    onnx_inference_ms=round(
                        sum(
                            item.read.timings_ms.get("cpm", 0.0)
                            + item.read.timings_ms.get("lprnet_decode", 0.0)
                            for item in scene.plates
                        ), 1
                    ),
                )
                diagnostics["candidate_count"] = len(scene.plates) + len(scene.rejections)
                for plate in scene.plates:
                    read = plate.read
                    record = {
                        "plate_text": read.normalized_text,
                        "canonical": read.normalized_text,
                        "raw_greedy_text": read.raw_text,
                        "confidence": None,
                        "recognition_score": None,
                        "score_kind": "uncalibrated",
                        "plate_type": "unclassified",
                        "rule_id": None,
                        "box": plate.detection.bbox_xyxy.tolist(),
                        "polygon": plate.detection.corners_xy.tolist(),
                        "detector_confidence": float(plate.detection.confidence),
                        "crop_base64": _image_to_base64_jpeg(read.aligned_rgb),
                        "timings": read.timings_ms,
                    }
                    if read.normalized_text:
                        detections.append(record)
                    else:
                        rejections.append(record | {"stage": "recognizer", "reason": "empty_ocr_text"})
                for rejected in scene.rejections:
                    rejections.append({
                        "stage": rejected.stage,
                        "reason": rejected.reason,
                        "box": rejected.detection.bbox_xyxy.tolist(),
                        "polygon": rejected.detection.corners_xy.tolist(),
                        "detector_confidence": float(rejected.detection.confidence),
                    })
                status = "ok" if detections else ("no_reliable_plate" if rejections else "no_plate")
            except Exception as exc:
                diagnostics["error"] = str(exc)
                status = "inference_error"
                detections = []
                rejections = []
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        diagnostics["rejected_count"] = len(rejections)
        diagnostics["timing_breakdown"]["total_ms"] = elapsed
        return {
            "image_width": width,
            "image_height": height,
            "detections": detections,
            "rejections": rejections,
            "count": len(detections),
            "status": status,
            "latency_ms": elapsed,
            "diagnostics": diagnostics,
            "model_status": "onnx_external" if self.scene_reader else "not_loaded",
        }
