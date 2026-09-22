import base64
import io
import json
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


def _image_to_base64_jpeg(crop_rgb: np.ndarray, quality: int = 80) -> str:
    """Encode an RGB uint8 image as a base64 JPEG data URL."""
    try:
        pil_im = Image.fromarray(crop_rgb)
        buf = io.BytesIO()
        pil_im.save(buf, format="JPEG", quality=quality)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""


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
        self.manifest_data: dict[str, Any] = {}
        self._last_mtime: float = 0.0
        self._load_active_model()

    def _load_active_model(self):
        if not HAS_ENGINE:
            return

        bundle_dir = self.root / "models" / "bundles" / "active-v1"
        rec_onnx = bundle_dir / "recognizer.onnx"
        manifest_file = bundle_dir / "manifest.json"

        # Check bundle manifest and capabilities
        if bundle_dir.exists() and manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    self.manifest_data = json.load(f)
            except Exception as e:
                print(f"[PredictorEngine] Failed to read manifest: {e}")
                self.manifest_data = {}

            capabilities = self.manifest_data.get("capabilities", [])
            has_detector = "detector" in self.manifest_data.get("components", {})

            # Full pipeline requires both crop-recognition and plate-detection
            if "plate-detection" in capabilities and has_detector:
                try:
                    self.reader = PlateReader(bundle_dir)
                    print(f"[PredictorEngine] Loaded full bundle reader from {bundle_dir}")
                except Exception as e:
                    print(f"[PredictorEngine] PlateReader initialization failed: {e}")
                    self.reader = None
            else:
                self.reader = None

        # Load standalone recognizer
        if rec_onnx.exists():
            try:
                current_mtime = rec_onnx.stat().st_mtime
                if self.recognizer_session is None or current_mtime > self._last_mtime:
                    charset_file = self.manifest_data.get("charset", {}).get("file", "charset.txt")
                    charset_path = bundle_dir / charset_file if (bundle_dir / charset_file).exists() else (self.root / "configs" / "charsets" / "tw_standard_v1.txt")

                    rules_file = self.manifest_data.get("rules", {}).get("file", "plate_rules.json")
                    rules_path = bundle_dir / rules_file if (bundle_dir / rules_file).exists() else (self.root / "configs" / "plate_rules" / "tw_standard_v1.json")

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
        diagnostics: dict[str, Any] = {
            "bundle_name": "active-v1",
            "model_id": self.manifest_data.get("model_id", "twplate-v1-recognizer"),
            "capabilities": self.manifest_data.get("capabilities", ["crop-recognition"]),
            "pipeline_mode": "hybrid_heuristic",
            "locator_type": "opencv_contour_v1",
            "detector_available": self.reader is not None,
            "timing_breakdown": {
                "locator_ms": 0.0,
                "rectifier_ms": 0.0,
                "onnx_inference_ms": 0.0,
                "ctc_decoding_ms": 0.0,
                "total_ms": 0.0,
            },
        }

        # 1. If full PlateReader is operational, use neural detector + recognizer pipeline
        if self.reader is not None:
            diagnostics["pipeline_mode"] = "neural_full_pipeline"
            diagnostics["locator_type"] = "plate_pose_net"
            t_read_0 = time.perf_counter()
            try:
                result = self.reader.read(img_rgb)
                t_read_1 = time.perf_counter()

                timing = result.timing
                read_total_ms = (t_read_1 - t_read_0) * 1000.0
                ctc_dec_ms = max(
                    0.0,
                    read_total_ms
                    - timing.detector_ms
                    - timing.rectifier_ms
                    - timing.recognizer_ms,
                )

                diagnostics["timing_breakdown"] = {
                    "locator_ms": round(timing.detector_ms, 1),
                    "rectifier_ms": round(timing.rectifier_ms, 1),
                    "onnx_inference_ms": round(timing.recognizer_ms, 1),
                    "ctc_decoding_ms": round(ctc_dec_ms, 1),
                    "total_ms": round(read_total_ms, 1),
                }

                for plate in result.plates:
                    corners = plate.detection.corners_xy.tolist()
                    bbox = plate.detection.bbox_xyxy.tolist()
                    try:
                        rectified_crop = rectify_plate(img_rgb, plate.detection.corners_xy).image_rgb
                        crop_b64 = _image_to_base64_jpeg(rectified_crop)
                    except Exception:
                        crop_b64 = ""

                    detections.append({
                        "plate_text": plate.decoded.display,
                        "canonical": plate.decoded.canonical,
                        "rule_id": plate.decoded.rule_id,
                        "confidence": round(
                            float(
                                np.clip(
                                    math.exp(
                                        plate.decoded.log_probability
                                        / max(1, len(plate.decoded.canonical))
                                    )
                                    * 100,
                                    5.0,
                                    99.9,
                                )
                            ),
                            1,
                        ),
                        "box": [int(x) for x in bbox],
                        "polygon": [[int(pt[0]), int(pt[1])] for pt in corners],
                        "plate_type": plate.decoded.plate_type or "台灣標準號牌",
                        "raw_greedy_text": plate.decoded.canonical,
                        "crop_base64": crop_b64,
                        "timings": {
                            "rectifier_ms": round(timing.rectifier_ms, 1),
                            "onnx_ms": round(timing.recognizer_ms, 1),
                            "ctc_ms": round(ctc_dec_ms, 1),
                        },
                    })
            except Exception as e:
                print(f"[PredictorEngine] Reader failed: {e}")

        # 2. If no full reader (recognizer-only bundle), use contour locator + ONNX recognizer
        if not detections and self.recognizer_session is not None and self.codec is not None:
            detections, timing_stats = self._detect_and_recognize_with_timings(img, img_rgb, w, h)
            diagnostics["pipeline_mode"] = "hybrid_heuristic"
            diagnostics["locator_type"] = "opencv_contour_v1"
            diagnostics["timing_breakdown"] = timing_stats

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000, 1)
        diagnostics["timing_breakdown"]["total_ms"] = latency_ms

        return {
            "image_width": w,
            "image_height": h,
            "detections": detections,
            "latency_ms": latency_ms,
            "count": len(detections),
            "model_status": "onnx_active" if self.recognizer_session else "not_loaded",
            "diagnostics": diagnostics,
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
            if 1.4 <= caspect <= 5.0 and carea >= 600 and carea <= (w * h * 0.85):
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
            if 1.4 <= caspect <= 5.0 and carea >= 600 and carea <= (w * h * 0.85):
                raw_boxes.append((x, y, x + cw, y + ch))

        # Non-Maximum Suppression to deduplicate overlapping candidate regions
        deduped: list[tuple[int, int, int, int]] = []
        for box in raw_boxes:
            if not any(_box_iou(box, ex) > 0.45 for ex in deduped):
                deduped.append(box)

        return deduped

    def _predict_crop(self, crop_rgb: np.ndarray) -> dict[str, Any] | None:
        """Run ONNX CTC recognizer on cropped plate image with exact timing measurements."""
        if crop_rgb.shape[0] < 12 or crop_rgb.shape[1] < 24:
            return None

        # 1. Rectifier / Preprocessing timing
        t_rect_0 = time.perf_counter()
        pil_img = Image.fromarray(crop_rgb)
        pil_resized = pil_img.resize((380, 160), Image.Resampling.BILINEAR)
        tensor = preprocess_v1_rgb(np.array(pil_resized))[np.newaxis, ...]
        t_rect_1 = time.perf_counter()
        rectifier_ms = (t_rect_1 - t_rect_0) * 1000.0

        # 2. ONNX Forward Inference timing
        t_onnx_0 = time.perf_counter()
        logits = self.recognizer_session.run(["logits"], {"input": tensor})[0][0]  # shape: (80, C)
        t_onnx_1 = time.perf_counter()
        onnx_ms = (t_onnx_1 - t_onnx_0) * 1000.0

        # 3. CTC Decoding timing
        t_dec_0 = time.perf_counter()
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
        constrained_result = None
        if self.ruleset is not None:
            try:
                dec = decode_constrained_ctc_v1(logits, self.codec, self.ruleset)
                avg_logp = dec.log_probability / max(1, len(dec.canonical))
                conf = float(np.clip(math.exp(avg_logp) * 100, 5.0, 99.9))
                if dec.log_probability > -18.0 and len(dec.canonical) >= 4:
                    constrained_result = {
                        "plate_text": dec.display,
                        "canonical": dec.canonical,
                        "rule_id": dec.rule_id,
                        "confidence": round(conf, 1),
                        "plate_type": "白牌自用車 (新式)" if "lll-dddd" in dec.rule_id.lower() else (dec.plate_type or "台灣標準號牌"),
                        "log_prob": dec.log_probability,
                    }
            except Exception:
                pass

        t_dec_1 = time.perf_counter()
        ctc_ms = (t_dec_1 - t_dec_0) * 1000.0

        crop_b64 = _image_to_base64_jpeg(crop_rgb)

        if constrained_result is not None:
            constrained_result["raw_greedy_text"] = greedy_str
            constrained_result["crop_base64"] = crop_b64
            constrained_result["rectifier_ms"] = round(rectifier_ms, 1)
            constrained_result["onnx_ms"] = round(onnx_ms, 1)
            constrained_result["ctc_ms"] = round(ctc_ms, 1)
            return constrained_result

        # Fallback to greedy if constrained decoding failed but characters are plausible
        if len(greedy_str) >= 4 and greedy_conf >= 30.0:
            if len(greedy_str) == 7:
                display = f"{greedy_str[:3]}-{greedy_str[3:]}"
            elif len(greedy_str) == 6:
                display = f"{greedy_str[:3]}-{greedy_str[3:]}"
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
                "raw_greedy_text": greedy_str,
                "crop_base64": crop_b64,
                "rectifier_ms": round(rectifier_ms, 1),
                "onnx_ms": round(onnx_ms, 1),
                "ctc_ms": round(ctc_ms, 1),
            }

        return None

    def _detect_and_recognize_with_timings(
        self, img_bgr: np.ndarray, img_rgb: np.ndarray, w: int, h: int
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Locate plate candidates and recognize each with true ONNX inference and stage timings."""
        t_loc_0 = time.perf_counter()
        candidates = self._extract_candidates(img_bgr, w, h)
        t_loc_1 = time.perf_counter()
        locator_ms = (t_loc_1 - t_loc_0) * 1000.0

        detected = []
        total_rect_ms = 0.0
        total_onnx_ms = 0.0
        total_ctc_ms = 0.0

        for x1, y1, x2, y2 in candidates:
            pad_w = int((x2 - x1) * 0.06)
            pad_h = int((y2 - y1) * 0.06)
            px1 = max(0, x1 - pad_w)
            py1 = max(0, y1 - pad_h)
            px2 = min(w, x2 + pad_w)
            py2 = min(h, y2 + pad_h)

            crop = img_rgb[py1:py2, px1:px2]
            res = self._predict_crop(crop)
            if res is not None:
                total_rect_ms += res["rectifier_ms"]
                total_onnx_ms += res["onnx_ms"]
                total_ctc_ms += res["ctc_ms"]
                detected.append({
                    "plate_text": res["plate_text"],
                    "canonical": res["canonical"],
                    "rule_id": res["rule_id"],
                    "confidence": res["confidence"],
                    "box": [px1, py1, px2, py2],
                    "polygon": [[px1, py1], [px2, py1], [px2, py2], [px1, py2]],
                    "plate_type": res["plate_type"],
                    "log_prob": res["log_prob"],
                    "raw_greedy_text": res.get("raw_greedy_text", ""),
                    "crop_base64": res.get("crop_base64", ""),
                    "timings": {
                        "rectifier_ms": res["rectifier_ms"],
                        "onnx_ms": res["onnx_ms"],
                        "ctc_ms": res["ctc_ms"],
                    },
                })

        timing_stats = {
            "locator_ms": round(locator_ms, 1),
            "rectifier_ms": round(total_rect_ms, 1),
            "onnx_inference_ms": round(total_onnx_ms, 1),
            "ctc_decoding_ms": round(total_ctc_ms, 1),
            "total_ms": round(locator_ms + total_rect_ms + total_onnx_ms + total_ctc_ms, 1),
        }

        if not detected:
            return [], timing_stats

        # Sort by confidence descending
        detected.sort(key=lambda d: d["confidence"], reverse=True)

        # NMS on results to avoid duplicate boxes for same vehicle
        final_detections = []
        for det in detected:
            b = tuple(det["box"])
            if not any(_box_iou(b, tuple(ex["box"])) > 0.3 for ex in final_detections):
                det_clean = {k: v for k, v in det.items() if k != "log_prob"}
                final_detections.append(det_clean)

        return final_detections, timing_stats


# Global predictor instance
predictor = PredictorEngine()
