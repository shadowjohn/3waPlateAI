"""Single-owner native inference. All initialization and calls run on one thread."""
from __future__ import annotations

import importlib.metadata
import math
import time

import numpy as np

from .config import ServiceConfig, verify_assets
from .errors import ServiceError
from .text import read_plate_line


def initialize_component(loader, requested: str, gpu_available):
    metadata = {'requested_device': requested, 'effective_device': 'cpu', 'fallback_reason': None,
                'initialization_attempts': 0, 'warmed': False}
    available = False
    if requested == 'auto':
        try:
            available = bool(gpu_available() if callable(gpu_available) else gpu_available)
        except Exception as exc:
            metadata['fallback_reason'] = 'gpu_probe_failed:' + type(exc).__name__
    if requested == 'auto' and available:
        metadata['initialization_attempts'] += 1
        try:
            instance = loader('gpu')
            metadata.update(effective_device='gpu:0', warmed=True)
            return instance, metadata
        except Exception as exc:
            metadata['fallback_reason'] = 'gpu_initialization_failed:' + type(exc).__name__
    elif requested == 'auto' and metadata['fallback_reason'] is None:
        metadata['fallback_reason'] = 'cuda_unavailable'
    metadata['initialization_attempts'] += 1
    instance = loader('cpu')
    metadata['warmed'] = True
    return instance, metadata


class PlatePipeline:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.detector = None
        self.ocr = None
        self._metadata = {'model_id': config.model_id}

    def initialize(self):
        # Validate before importing code which can load weights or try CUDA.
        verify_assets(self.config)
        # Windows: reversing this order reproduced torch shm.dll WinError 127.
        import torch
        import cv2
        torch.set_num_threads(self.config.threads)
        cv2.setNumThreads(1)
        from ultralytics import YOLO

        warm_scene = np.full((640, 640, 3), 255, dtype=np.uint8)
        warm_crop = np.full((96, 320, 3), 255, dtype=np.uint8)
        cv2.putText(warm_crop, 'ABC-1234', (8, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,0,0), 3)

        def detector_loader(device):
            model = YOLO(str(self.config.detector_path))
            target = 'cuda:0' if device == 'gpu' else 'cpu'
            def predict(image):
                result = model.predict(image, imgsz=640, conf=.25, iou=.45,
                                       device=target, half=False, max_det=21, verbose=False)[0]
                return result.boxes.xyxy.detach().cpu().numpy().tolist()
            for _ in range(3):
                predict(warm_scene)
            return predict
        self.detector, det_meta = initialize_component(detector_loader, self.config.device, torch.cuda.is_available)

        import paddle
        from paddleocr import PaddleOCR
        def ocr_loader(device):
            target = 'gpu:0' if device == 'gpu' else 'cpu'
            model = PaddleOCR(
                text_detection_model_name='PP-OCRv3_mobile_det',
                text_recognition_model_name='en_PP-OCRv3_mobile_rec',
                text_detection_model_dir=str(self.config.ocr_detection_dir),
                text_recognition_model_dir=str(self.config.ocr_recognition_dir),
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False, device=target, cpu_threads=self.config.threads,
                enable_mkldnn=False, enable_hpi=False, text_det_limit_side_len=640,
                text_det_limit_type='max', text_det_thresh=.3, text_det_box_thresh=.6,
                text_det_unclip_ratio=1.5, text_rec_score_thresh=0.0,
            )
            for name in ('text_det_model', 'text_rec_model'):
                observed = getattr(model.paddlex_pipeline, name).pp_option.device_type
                if observed != device:
                    raise RuntimeError('ocr_device_mismatch')
            def predict(image):
                output = list(model.predict(image))[0]
                if device == 'gpu':
                    paddle.device.cuda.synchronize()
                return list(output['rec_texts']), np.asarray(output['rec_polys']).tolist()
            for _ in range(3):
                predict(warm_crop)
            return predict
        paddle_gpu = lambda: paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        self.ocr, ocr_meta = initialize_component(ocr_loader, self.config.device, paddle_gpu)
        versions = {}
        for name in ('torch', 'torchvision', 'ultralytics', 'paddleocr', 'paddlepaddle-gpu', 'paddlepaddle', 'paddlex'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        self._metadata.update(detector=det_meta, ocr=ocr_meta, runtime=versions,
                              model_id=self.config.model_id, rotation_rectification=False,
                              deployment_scope='local_validation',
                              runtime_warnings=['Paddle cuDNN patch-version warning observed in Phase A; deployment qualification pending'])

    def metadata(self):
        return dict(self._metadata)

    def predict(self, bgr: np.ndarray) -> dict:
        started = time.perf_counter()
        height, width = bgr.shape[:2]
        boxes = self.detector(bgr)
        detect_ms = (time.perf_counter()-started)*1000
        if len(boxes) > 20:
            raise ServiceError('too_many_candidates', 422)
        if any(len(box) != 4 or not np.isfinite(box).all() for box in boxes):
            raise RuntimeError('invalid_detector_output')
        boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
        detections = []
        ocr_started = time.perf_counter()
        for index, box in enumerate(boxes, 1):
            x1, y1, x2, y2 = [float(value) for value in box]
            x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
            y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
            bw, bh = x2-x1, y2-y1
            crop_box = [max(0, math.floor(x1-bw*.05)), max(0, math.floor(y1-bh*.05)),
                        min(width, math.ceil(x2+bw*.05)), min(height, math.ceil(y2+bh*.05))]
            detection = dict(id=index, bbox=[x1,y1,x2,y2], crop_bbox=crop_box)
            if bw <= 0 or bh <= 0:
                detection.update(read_plate_line([], []), status='error', reason='crop_invalid')
            else:
                left, top, right, bottom = crop_box
                try:
                    texts, polygons = self.ocr(np.ascontiguousarray(bgr[top:bottom,left:right]))
                    detection.update(read_plate_line(texts, polygons))
                except ValueError:
                    detection.update(read_plate_line([], []), status='error', reason='ocr_failed')
            detections.append(detection)
        status = 'no_plate' if not detections else ('ok' if all(d['status']=='recognized' for d in detections) else 'partial')
        return dict(schema_version=1, status=status, model_id=self.config.model_id,
                    image={'width':width,'height':height}, detections=detections,
                    timings_ms={'detect':detect_ms,'ocr':(time.perf_counter()-ocr_started)*1000})
