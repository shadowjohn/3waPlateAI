"""Validated, cached Web inference with explicit acceptance and diagnostics."""
from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    from plateai_reader.runtime import PlateReader, decode_constrained_ctc_v1, _default_schema_path
    from plateai_shared.bundle import validate_model_bundle
    from plateai_shared.recognition import CTCCodec, preprocess_v1_rgb
    from plateai_shared.rules import load_character_set, load_ruleset
    HAS_ENGINE = True
    ENGINE_IMPORT_ERROR = None
except ImportError as error:
    HAS_ENGINE = False
    ENGINE_IMPORT_ERROR = str(error)


def _image_to_base64_jpeg(crop_rgb: np.ndarray, quality: int = 80) -> str:
    buf = io.BytesIO()
    Image.fromarray(crop_rgb).save(buf, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _box_iou(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return max(inter / max(1e-6, a1 + a2 - inter), inter / max(1e-6, min(a1, a2)))


class PredictorEngine:
    """One cached bundle; a full-reader error/empty result never invokes fallback.

    The recognition cutoff is an operating policy, not a calibrated probability
    of correctness. Passing it does not establish real-world model accuracy.
    """

    def __init__(self, root: Path | None = None, *, minimum_recognition_score: float = 0.65):
        if not 0 <= minimum_recognition_score <= 1:
            raise ValueError('minimum_recognition_score must be in [0,1]')
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
        self.minimum_recognition_score = minimum_recognition_score
        self._lock = threading.RLock()
        self._bundle_stamp = None
        self.reader = self.recognizer_session = self.codec = self.ruleset = None
        self.manifest_data: dict[str, Any] = {}
        self.load_error = None
        self.model_warnings: list[str] = []
        self._load_active_model()

    @staticmethod
    def _stamp(bundle: Path):
        return tuple((p.relative_to(bundle).as_posix(), p.stat().st_mtime_ns, p.stat().st_size)
                     for p in sorted(bundle.rglob('*')) if p.is_file())

    def _load_active_model(self):
        with self._lock:
            if not HAS_ENGINE:
                self.load_error = ENGINE_IMPORT_ERROR
                return
            bundle = self.root / 'models/bundles/active-v1'
            try:
                stamp = self._stamp(bundle)
                if stamp == self._bundle_stamp:
                    return
                self._bundle_stamp = stamp
                self.reader = self.recognizer_session = self.codec = self.ruleset = None
                self.manifest_data = {}
                self.model_warnings = []
                self.load_error = None
                declaration = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
                if 'plate-detection' in declaration.get('capabilities', []):
                    reader = PlateReader(bundle)
                    manifest = reader.manifest
                    session, codec, ruleset = reader._recognizer_session, reader.codec, reader.ruleset
                else:
                    manifest = validate_model_bundle(bundle, _default_schema_path())
                    charset = load_character_set(bundle / manifest['charset']['file'])
                    codec = CTCCodec.from_charset(charset)
                    ruleset = load_ruleset(bundle / manifest['rules']['file'], charset)
                    session = ort.InferenceSession(
                        str(bundle / manifest['components']['recognizer']['file']),
                        providers=['CPUExecutionProvider'])
                    reader = None
                if self._stamp(bundle) != stamp:
                    raise ValueError('bundle changed while loading; retry after activation finishes')
                self.reader, self.recognizer_session = reader, session
                self.codec, self.ruleset, self.manifest_data = codec, ruleset, manifest
                self._read_model_warnings(bundle)
            except Exception as error:
                self.reader = self.recognizer_session = self.codec = self.ruleset = None
                self.load_error = str(error)

    def _read_model_warnings(self, bundle: Path):
        provenance = self.manifest_data.get('provenance', {})
        if provenance.get('training_data') == 'synthetic':
            self.model_warnings.append('此模型以合成資料訓練，合成驗證分數不代表實拍辨識率。')
        report = provenance.get('detector_training_report')
        if report:
            validation = json.loads((bundle / report['file']).read_text(encoding='utf-8')).get('validation', {})
            precision = validation.get('complete_quad_precision')
            if isinstance(precision, (int, float)) and precision < 0.5:
                self.model_warnings.append(
                    f"Detector 報告的完整四角 precision 僅 {precision:.2%}；定位品質不足，需重新訓練與實拍驗收。")

    def _recognition_record(self, decoded, raw, crop, timings):
        logp = float(decoded.log_probability)
        score = math.exp(min(0.0, logp) / max(1, len(decoded.canonical))) if math.isfinite(logp) else 0.0
        reason = None
        if score < self.minimum_recognition_score:
            reason = 'low_recognition_score'
        elif raw is None:
            reason = 'missing_greedy_evidence'
        elif raw != decoded.canonical:
            reason = 'decoder_disagreement'
        return {
            'plate_text': decoded.display, 'canonical': decoded.canonical,
            'rule_id': decoded.rule_id, 'plate_type': decoded.plate_type,
            'confidence': round(score * 100, 1), 'recognition_score': score,
            'score_kind': 'ctc_viterbi_per_character_uncalibrated',
            'raw_greedy_text': raw, 'ctc_log_probability': logp if math.isfinite(logp) else None,
            'crop_base64': _image_to_base64_jpeg(crop) if crop is not None else '',
            'timings': timings, 'accepted': reason is None, 'reason': reason,
        }

    def predict_image(self, image_bytes: bytes) -> dict[str, Any]:
        with self._lock:
            return self._predict_image(image_bytes)

    def _predict_image(self, image_bytes: bytes) -> dict[str, Any]:
        started = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('無法解析影像資料，請使用有效 PNG/JPG/WEBP。')
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        loading_started = time.perf_counter()
        self._load_active_model()
        loading_ms = (time.perf_counter() - loading_started) * 1000
        full = self.reader is not None
        diag = {
            'bundle_name': 'active-v1', 'model_id': self.manifest_data.get('model_id'),
            'capabilities': self.manifest_data.get('capabilities', []),
            'pipeline_mode': 'neural_full_pipeline' if full else 'hybrid_heuristic',
            'locator_type': 'plate_pose_net' if full else 'opencv_contour_v1',
            'detector_available': full, 'warnings': list(self.model_warnings),
            'minimum_recognition_score': self.minimum_recognition_score,
            'error': self.load_error,
            'timing_breakdown': dict(locator_ms=0.0, rectifier_ms=0.0,
                onnx_inference_ms=0.0, ctc_decoding_ms=0.0, preprocess_ms=0.0,
                model_load_ms=round(loading_ms, 1), total_ms=0.0),
        }
        detections, rejections = [], []
        status = 'no_plate'
        if self.load_error or self.recognizer_session is None:
            status = 'model_error'
            diag['pipeline_mode'] = 'unavailable'
            diag['locator_type'] = None
        elif full:
            try:
                result = self.reader.read(rgb)
                timing = result.timing
                diag['timing_breakdown'].update(
                    locator_ms=round(timing.detector_ms, 1), rectifier_ms=round(timing.rectifier_ms, 1),
                    onnx_inference_ms=round(timing.onnx_inference_ms, 1),
                    ctc_decoding_ms=round(timing.ctc_decoding_ms, 1),
                    preprocess_ms=round(timing.preprocess_ms, 1))
                diag['candidate_count'] = timing.retained_detection_count
                diag['rectified_count'] = timing.rectified_plate_count
                diag['recognition_batch_sizes'] = list(timing.recognition_batch_sizes)
                for plate in result.plates:
                    record = self._recognition_record(
                        plate.decoded, plate.raw_greedy_text, plate.crop_rgb,
                        {'onnx_ms': None, 'ctc_ms': round(plate.ctc_decoding_ms, 1),
                         'batch_index': plate.recognition_batch_index, 'onnx_scope': 'shared_batch'})
                    record.update(box=plate.detection.bbox_xyxy.tolist(),
                        polygon=plate.detection.corners_xy.tolist(),
                        detector_confidence=float(plate.detection.confidence))
                    (detections if record['accepted'] else rejections).append(record)
                for rejected in result.rejections:
                    rejections.append({'stage': rejected.stage, 'reason': rejected.reason,
                        'box': rejected.detection.bbox_xyxy.tolist(),
                        'polygon': rejected.detection.corners_xy.tolist(),
                        'detector_confidence': float(rejected.detection.confidence)})
            except Exception as error:
                status = 'inference_error'
                diag['error'] = str(error)
                detections = []
        else:
            try:
                detections, stats = self._detect_and_recognize_with_timings(img, rgb, w, h)
                rejections = stats.pop('rejections')
                diag['candidate_count'] = stats.pop('candidate_count')
                diag['timing_breakdown'].update(stats)
            except Exception as error:
                status = 'inference_error'
                diag['error'] = str(error)
        if status not in ('model_error', 'inference_error'):
            status = 'ok' if detections else ('no_reliable_plate' if rejections else 'no_plate')
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        diag['timing_breakdown']['total_ms'] = elapsed
        diag['rejected_count'] = len(rejections)
        return {'image_width': w, 'image_height': h, 'detections': detections,
                'rejections': rejections, 'count': len(detections), 'status': status,
                'latency_ms': elapsed, 'diagnostics': diag,
                'model_status': 'onnx_active' if self.recognizer_session is not None else 'not_loaded'}

    def _extract_candidates(self, img_bgr, w, h):
        """Bound contour work; map coordinates back for original-resolution crops."""
        if max(w, h) > 1280:
            scale = 1280 / max(w, h)
            small = cv2.resize(img_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            sh, sw = small.shape[:2]
            return [(max(0, int(x1 * w / sw)), max(0, int(y1 * h / sh)),
                     min(w, int(math.ceil(x2 * w / sw))), min(h, int(math.ceil(y2 * h / sh))))
                    for x1, y1, x2, y2 in self._extract_candidates(small, sw, sh)]
        boxes = [(0, 0, w, h)] if 1.4 <= w / h <= 5.0 else []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, threshold = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        colors = cv2.bitwise_or(cv2.inRange(hsv, (0, 0, 100), (180, 85, 255)),
                               cv2.inRange(hsv, (15, 60, 70), (38, 255, 255)))
        for mask, size in ((threshold, (21, 5)), (colors, (25, 7))):
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, size))
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                if 1.4 <= cw / ch <= 5 and 600 <= cw * ch <= w * h * 0.85:
                    boxes.append((x, y, x + cw, y + ch))
        deduped = []
        for box in boxes:
            if not any(_box_iou(box, other) > 0.45 for other in deduped):
                deduped.append(box)
        return deduped

    def _recognize_crop(self, crop_rgb):
        t0 = time.perf_counter()
        canonical = np.array(Image.fromarray(crop_rgb).resize((380, 160), Image.Resampling.BILINEAR))
        tensor = preprocess_v1_rgb(canonical)[None, ...]
        t1 = time.perf_counter()
        logits = self.recognizer_session.run(['logits'], {'input': tensor})[0][0]
        t2 = time.perf_counter()
        raw = self.codec.decode_greedy(logits.argmax(axis=1).tolist())
        decoded = decode_constrained_ctc_v1(logits, self.codec, self.ruleset)
        t3 = time.perf_counter()
        return self._recognition_record(decoded, raw, canonical,
            {'rectifier_ms': 0.0, 'preprocess_ms': (t1-t0)*1000,
             'onnx_ms': (t2-t1)*1000, 'ctc_ms': (t3-t2)*1000})

    def _predict_crop(self, crop_rgb):
        record = self._recognize_crop(crop_rgb)
        return record if record['accepted'] else None

    def _detect_and_recognize_with_timings(self, img_bgr, img_rgb, w, h):
        t0 = time.perf_counter()
        candidates = self._extract_candidates(img_bgr, w, h)
        stats = dict(locator_ms=(time.perf_counter()-t0)*1000, rectifier_ms=0.0,
                     preprocess_ms=0.0, onnx_inference_ms=0.0, ctc_decoding_ms=0.0,
                     candidate_count=len(candidates), rejections=[])
        detected = []
        for x1, y1, x2, y2 in candidates:
            pw, ph = int((x2-x1)*0.06), int((y2-y1)*0.06)
            x1, y1, x2, y2 = max(0, x1-pw), max(0, y1-ph), min(w, x2+pw), min(h, y2+ph)
            record = self._recognize_crop(img_rgb[y1:y2, x1:x2])
            record.update(box=[x1, y1, x2, y2], polygon=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                          detector_confidence=None)
            stats['preprocess_ms'] += record['timings']['preprocess_ms']
            stats['onnx_inference_ms'] += record['timings']['onnx_ms']
            stats['ctc_decoding_ms'] += record['timings']['ctc_ms']
            (detected if record['accepted'] else stats['rejections']).append(record)
        final = []
        for record in sorted(detected, key=lambda r: r['recognition_score'], reverse=True):
            if not any(_box_iou(record['box'], other['box']) > 0.3 for other in final):
                final.append(record)
        return final, stats


predictor = PredictorEngine()
