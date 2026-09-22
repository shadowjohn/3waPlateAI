"""Live inference predictor module supporting full images and plate crops."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    from plateai_reader.runtime import PlateReader, decode_constrained_ctc_v1
    from plateai_reader.rectifier import rectify_plate
    from plateai_shared.recognition import CTCCodec, preprocess_v1_rgb
    from plateai_shared.rules import load_character_set, load_ruleset, format_display
    HAS_ENGINE = True
except Exception:
    HAS_ENGINE = False


def _box_iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    iou = inter / max(1e-6, float(union))
    # Overlap coefficient: ratio of intersection to the smaller box
    iomin = inter / max(1e-6, float(min(a1, a2)))
    return max(iou, iomin)


class PredictorEngine:
    """Predictor supporting both full Model Bundles and standalone ONNX Recognizer."""

    def __init__(self):
        self.root = Path(__file__).resolve().parent.parent.parent
        self.reader: PlateReader | None = None
        self.recognizer_session: ort.InferenceSession | None = None
        self.codec: CTCCodec | None = None
        self.ruleset: Any | None = None
        self._last_mtime: float = 0.0
        self._load_active_model()

    def _load_active_model(self):
        if not HAS_ENGINE:
            return

        bundle_dir = self.root / "models" / "bundles" / "active-v1"
        rec_onnx = bundle_dir / "recognizer.onnx"

        # Check if bundle has reader with detector
        if bundle_dir.exists() and (bundle_dir / "manifest.json").exists():
            try:
                self.reader = PlateReader.load(bundle_dir)
                print(f"[PredictorEngine] Loaded full bundle reader from {bundle_dir}")
            except Exception as e:
                self.reader = None

        # Load standalone recognizer
        if rec_onnx.exists():
            try:
                current_mtime = rec_onnx.stat().st_mtime
                if self.recognizer_session is None or current_mtime > self._last_mtime:
                    charset_path = self.root / "configs" / "charsets" / "tw_standard_v1.txt"
                    rules_path = self.root / "configs" / "plate_rules" / "tw_standard_v1.json"
                    charset = load_character_set(charset_path)
                    self.codec = CTCCodec.from_charset(charset)
                    self.ruleset = load_ruleset(rules_path, charset)
                    self.recognizer_session = ort.InferenceSession(
                        str(rec_onnx), providers=["CPUExecutionProvider"]
                    )
                    self._last_mtime = current_mtime
                    print(f"[PredictorEngine] Loaded ONNX recognizer from {rec_onnx}")
            except Exception as e:
                print(f"[PredictorEngine] Recognizer load failed: {e}")

    def predict_image(self, image_bytes: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("無法解析影像資料，請確認檔案格式是否為有效圖片 (PNG/JPG/WEBP)。")

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Refresh model if newer on disk
        self._load_active_model()

        detections = []

        # 1. If full PlateReader is operational, try it first
        if self.reader is not None:
            try:
                result = self.reader.read_image(img_rgb)
                for det in result.plates:
                    detections.append({
                        "plate_text": det.decoded.display,
                        "canonical": det.decoded.canonical,
                        "rule_id": det.decoded.rule_id,
                        "confidence": round(
                            float(np.clip(math.exp(det.decoded.log_probability / max(1, len(det.decoded.canonical))) * 100, 5.0, 99.9)),
                            1
                        ),
                        "box": [int(x) for x in det.detection.box],
                        "polygon": [[int(pt[0]), int(pt[1])] for pt in det.detection.polygon],
                        "plate_type": det.decoded.plate_type or "台灣標準號牌",
                    })
            except Exception as e:
                print(f"[PredictorEngine] Reader failed: {e}")

        # 2. If no full reader detections, use smart multi-candidate locator + ONNX recognizer
        if not detections and self.recognizer_session is not None and self.codec is not None:
            detections = self._detect_and_recognize(img, img_rgb, w, h)

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000, 1)

        return {
            "image_width": w,
            "image_height": h,
            "detections": detections,
            "latency_ms": latency_ms,
            "count": len(detections),
            "model_status": "onnx_active" if self.recognizer_session else "not_loaded",
        }

    def _extract_candidates(self, img_bgr: np.ndarray, w: int, h: int) -> list[tuple[int, int, int, int]]:
        """Multi-scale candidate box extraction for Taiwan license plates."""
        raw_boxes: list[tuple[int, int, int, int]] = []

        # Candidate A: Full image if aspect ratio matches a plate crop
        aspect = w / max(1.0, float(h))
        if 1.4 <= aspect <= 5.0:
            raw_boxes.append((0, 0, w, h))

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Candidate B: Vertical edge density (Sobel-X)
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_sobel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        closed_sobel = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_sobel)
        contours, _ = cv2.findContours(closed_sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            caspect = cw / max(1.0, float(ch))
            carea = cw * ch
            if 1.5 <= caspect <= 5.0 and carea >= 600 and carea <= (w * h * 0.85):
                raw_boxes.append((x, y, x + cw, y + ch))

        # Candidate C: High brightness / white and yellow color thresholding
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask_w = cv2.inRange(hsv, (0, 0, 100), (180, 85, 255))
        mask_y = cv2.inRange(hsv, (15, 60, 70), (38, 255, 255))
        mask = cv2.bitwise_or(mask_w, mask_y)
        kernel_color = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        closed_color = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_color)
        contours_c, _ = cv2.findContours(closed_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_c:
            x, y, cw, ch = cv2.boundingRect(c)
            caspect = cw / max(1.0, float(ch))
            carea = cw * ch
            if 1.5 <= caspect <= 5.0 and carea >= 600 and carea <= (w * h * 0.85):
                raw_boxes.append((x, y, x + cw, y + ch))

        # Non-Maximum Suppression to deduplicate overlapping candidate regions
        deduped: list[tuple[int, int, int, int]] = []
        for box in raw_boxes:
            if not any(_box_iou(box, ex) > 0.45 for ex in deduped):
                deduped.append(box)

        return deduped

    def _predict_crop(self, crop_rgb: np.ndarray) -> dict[str, Any] | None:
        """Run ONNX CTC recognizer on cropped plate image."""
        if crop_rgb.shape[0] < 12 or crop_rgb.shape[1] < 24:
            return None

        pil_img = Image.fromarray(crop_rgb)
        pil_resized = pil_img.resize((380, 160), Image.Resampling.BILINEAR)
        tensor = preprocess_v1_rgb(np.array(pil_resized))[np.newaxis, ...]
        logits = self.recognizer_session.run(["logits"], {"input": tensor})[0][0]  # shape: (80, C)

        # Softmax probabilities
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

        # Greedy CTC collapse
        best_indices = np.argmax(probs, axis=-1)
        greedy_chars = []
        greedy_confs = []
        prev = -1
        for t_idx, idx in enumerate(best_indices):
            if idx != prev:
                if idx != self.codec.blank_index and idx <= len(self.codec.symbols):
                    greedy_chars.append(self.codec.symbols[idx - 1])
                    greedy_confs.append(float(probs[t_idx, idx]))
                prev = idx
        greedy_str = "".join(greedy_chars)
        greedy_conf = float(np.mean(greedy_confs) * 100) if greedy_confs else 0.0

        # Try Constrained CTC Decode with rule enforcement
        if self.ruleset is not None:
            try:
                dec = decode_constrained_ctc_v1(logits, self.codec, self.ruleset)
                avg_logp = dec.log_probability / max(1, len(dec.canonical))
                conf = float(np.clip(math.exp(avg_logp) * 100, 5.0, 99.9))
                # Reasonable confidence threshold
                if dec.log_probability > -15.0 and len(dec.canonical) >= 4:
                    return {
                        "plate_text": dec.display,
                        "canonical": dec.canonical,
                        "rule_id": dec.rule_id,
                        "confidence": round(conf, 1),
                        "plate_type": "白牌自用車 (新式)" if "LLL-DDDD" in dec.rule_id else (dec.plate_type or "台灣標準號牌"),
                        "log_prob": dec.log_probability,
                    }
            except Exception:
                pass

        # Fallback to greedy if constrained decoding failed but characters are plausible
        if len(greedy_str) >= 4 and greedy_conf >= 35.0:
            # Format display string with dash if standard length
            if len(greedy_str) == 7:
                display = f"{greedy_str[:3]}-{greedy_str[3:]}"
            elif len(greedy_str) == 6:
                display = f"{greedy_str[:2]}-{greedy_str[2:]}"
            elif len(greedy_str) == 5:
                display = f"{greedy_str[:2]}-{greedy_str[2:]}"
            else:
                display = greedy_str

            return {
                "plate_text": display,
                "canonical": greedy_str,
                "rule_id": "ctc_unconstrained",
                "confidence": round(greedy_conf, 1),
                "plate_type": "台灣號牌 (未約束)",
                "log_prob": float(math.log(max(1e-6, greedy_conf / 100.0))),
            }

        return None

    def _detect_and_recognize(
        self, img_bgr: np.ndarray, img_rgb: np.ndarray, w: int, h: int
    ) -> list[dict[str, Any]]:
        """Locate plate candidates and recognize each with true ONNX inference."""
        candidates = self._extract_candidates(img_bgr, w, h)
        detected = []

        for x1, y1, x2, y2 in candidates:
            # Pad candidate 6% to ensure character boundaries are not cut off
            pad_w = int((x2 - x1) * 0.06)
            pad_h = int((y2 - y1) * 0.06)
            px1 = max(0, x1 - pad_w)
            py1 = max(0, y1 - pad_h)
            px2 = min(w, x2 + pad_w)
            py2 = min(h, y2 + pad_h)

            crop = img_rgb[py1:py2, px1:px2]
            res = self._predict_crop(crop)
            if res is not None:
                detected.append({
                    "plate_text": res["plate_text"],
                    "canonical": res["canonical"],
                    "rule_id": res["rule_id"],
                    "confidence": res["confidence"],
                    "box": [px1, py1, px2, py2],
                    "polygon": [[px1, py1], [px2, py1], [px2, py2], [px1, py2]],
                    "plate_type": res["plate_type"],
                    "log_prob": res["log_prob"],
                })

        if not detected:
            return []

        # Sort by log_prob or confidence descending
        detected.sort(key=lambda d: d["confidence"], reverse=True)

        # NMS on results to avoid duplicate boxes for same vehicle
        final_detections = []
        for det in detected:
            b = tuple(det["box"])
            if not any(_box_iou(b, tuple(ex["box"])) > 0.3 for ex in final_detections):
                # Clean up internal log_prob before returning
                det_clean = {k: v for k, v in det.items() if k != "log_prob"}
                final_detections.append(det_clean)

        return final_detections


# Global predictor instance
predictor = PredictorEngine()
