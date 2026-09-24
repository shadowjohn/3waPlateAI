"""Compose the existing native plate locator with independent external OCR."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from plateai_shared.detection import PlateDetection

from .fpga_lpr import FpgaLprError, FpgaLprRead
from .runtime import DetectionResult


class FpgaPipelineError(ValueError):
    """The scene-level path cannot return a trustworthy detection result."""


class Detector(Protocol):
    def detect(self, image_rgb: object) -> DetectionResult: ...


class Recognizer(Protocol):
    def recognize(self, roi_rgb: NDArray[np.uint8]) -> FpgaLprRead: ...


@dataclass(frozen=True, slots=True)
class FpgaScenePlate:
    detection: PlateDetection
    read: FpgaLprRead

    @property
    def text(self) -> str:
        return self.read.normalized_text


@dataclass(frozen=True, slots=True)
class FpgaSceneRejection:
    detection: PlateDetection
    stage: str
    reason: str


@dataclass(frozen=True, slots=True)
class FpgaSceneResult:
    plates: tuple[FpgaScenePlate, ...]
    rejections: tuple[FpgaSceneRejection, ...]
    detector_id: str
    recognizer_id: str
    providers: tuple[str, ...]
    timings_ms: dict[str, float]


def crop_box(
    image_rgb: NDArray[np.uint8], bbox_xyxy: object, margin: float
) -> NDArray[np.uint8]:
    """Expand a detection by its own size, then round outward and clip."""

    image = np.asarray(image_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise FpgaLprError("invalid_roi")
    if not isinstance(margin, (int, float)) or not math.isfinite(float(margin)) or not 0 <= margin <= 1:
        raise FpgaLprError("invalid_bbox")
    try:
        box = np.asarray(bbox_xyxy, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FpgaLprError("invalid_bbox") from exc
    if box.shape != (4,) or not np.isfinite(box).all():
        raise FpgaLprError("invalid_bbox")
    left, top, right, bottom = (float(value) for value in box)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise FpgaLprError("invalid_bbox")
    image_height, image_width = image.shape[:2]
    x0 = max(0, math.floor(left - width * margin))
    y0 = max(0, math.floor(top - height * margin))
    x1 = min(image_width, math.ceil(right + width * margin))
    y1 = min(image_height, math.ceil(bottom + height * margin))
    if x0 >= x1 or y0 >= y1:
        raise FpgaLprError("invalid_bbox")
    return image[y0:y1, x0:x1]


class FpgaSceneReader:
    def __init__(self, detector: Detector, recognizer: Recognizer, roi_margin: float = 0.06) -> None:
        if not isinstance(roi_margin, (int, float)) or not math.isfinite(float(roi_margin)) or not 0 <= roi_margin <= 1:
            raise ValueError("roi_margin must be finite and between 0 and 1")
        self.detector = detector
        self.recognizer = recognizer
        self.roi_margin = float(roi_margin)

    def read(self, image_rgb: NDArray[np.uint8]) -> FpgaSceneResult:
        image = np.asarray(image_rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise FpgaPipelineError("invalid_image")
        started = time.perf_counter()
        try:
            detected = self.detector.detect(image)
        except Exception as exc:
            raise FpgaPipelineError("detector_failed") from exc
        detector_done = time.perf_counter()
        plates: list[FpgaScenePlate] = []
        rejections: list[FpgaSceneRejection] = []
        for detection in detected.detections:
            try:
                roi = crop_box(image, detection.bbox_xyxy, self.roi_margin)
                plates.append(FpgaScenePlate(detection, self.recognizer.recognize(roi)))
            except FpgaLprError as exc:
                stage = "roi" if exc.reason == "invalid_bbox" else "recognizer"
                rejections.append(FpgaSceneRejection(detection, stage, exc.reason))
        finished = time.perf_counter()
        return FpgaSceneResult(
            plates=tuple(plates),
            rejections=tuple(rejections),
            detector_id=getattr(self.detector, "model_id", "plate_pose_net"),
            recognizer_id=getattr(getattr(self.recognizer, "manifest", None), "model_id", "fpga-lpr-mit-v1"),
            providers=detected.providers,
            timings_ms={
                "detector": detected.detector_ms,
                "roi_and_recognizer": (finished - detector_done) * 1000,
                "total": (finished - started) * 1000,
            },
        )
