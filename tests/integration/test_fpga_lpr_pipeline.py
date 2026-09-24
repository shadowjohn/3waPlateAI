"""A failed plate OCR must not erase another detection in the same scene."""

from __future__ import annotations

import numpy as np
import pytest

from plateai_reader.fpga_lpr import FpgaLprError, FpgaLprRead
from plateai_reader.fpga_pipeline import FpgaPipelineError, FpgaSceneReader, crop_box
from plateai_reader.runtime import DetectionResult
from plateai_shared.detection import PlateDetection


def _detection(left: float, top: float, right: float, bottom: float) -> PlateDetection:
    return PlateDetection(
        bbox_xyxy=np.array([left, top, right, bottom], np.float32),
        confidence=0.8,
        corners_xy=np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]], np.float32
        ),
    )


class FakeDetector:
    def __init__(self, detections: tuple[PlateDetection, ...]):
        self.detections = detections

    def detect(self, _image: np.ndarray) -> DetectionResult:
        return DetectionResult(self.detections, ("CPUExecutionProvider",), 2.0)


class FakeRecognizer:
    def __init__(self, reject_first: bool = False):
        self.reject_first = reject_first
        self.calls: list[np.ndarray] = []

    def recognize(self, roi_rgb: np.ndarray) -> FpgaLprRead:
        self.calls.append(roi_rgb.copy())
        if self.reject_first and len(self.calls) == 1:
            raise FpgaLprError("invalid_corners")
        return FpgaLprRead(
            raw_text="ABC1234",
            normalized_text="ABC1234",
            aligned_rgb=np.zeros((48, 94, 3), np.uint8),
            roi_corners_xy=np.zeros((4, 2), np.float32),
            timings_ms={"total": 1.5},
        )


def test_bad_first_plate_keeps_second() -> None:
    first = _detection(5, 5, 35, 20)
    second = _detection(50, 30, 90, 50)
    recognizer = FakeRecognizer(reject_first=True)
    image = np.full((60, 100, 3), 255, np.uint8)
    result = FpgaSceneReader(FakeDetector((first, second)), recognizer).read(image)
    assert [plate.text for plate in result.plates] == ["ABC1234"]
    assert result.plates[0].detection is second
    assert [rejection.reason for rejection in result.rejections] == ["invalid_corners"]
    assert result.rejections[0].detection is first
    assert len(recognizer.calls) == 2
    assert result.detector_id == "plate_pose_net"
    assert result.recognizer_id == "fpga-lpr-mit-v1"


def test_bbox_margin_clips_to_image_edges_and_rejects_empty() -> None:
    image = np.zeros((10, 20, 3), np.uint8)
    edge = crop_box(image, np.array([0, 0, 5, 5], np.float32), margin=0.5)
    assert edge.shape == (8, 8, 3)
    with pytest.raises(FpgaLprError, match="invalid_bbox"):
        crop_box(image, np.array([5, 1, 5, 4], np.float32), margin=0.06)
    with pytest.raises(FpgaLprError, match="invalid_bbox"):
        crop_box(image, np.array([np.nan, 1, 5, 4], np.float32), margin=0.06)


def test_empty_scene_does_not_invoke_ocr_or_fallback() -> None:
    recognizer = FakeRecognizer()
    result = FpgaSceneReader(FakeDetector(()), recognizer).read(np.zeros((30, 40, 3), np.uint8))
    assert result.plates == ()
    assert result.rejections == ()
    assert recognizer.calls == []


def test_detector_failure_is_explicit() -> None:
    class BrokenDetector:
        def detect(self, _image: np.ndarray) -> DetectionResult:
            raise RuntimeError("provider failure")

    with pytest.raises(FpgaPipelineError, match="detector_failed"):
        FpgaSceneReader(BrokenDetector(), FakeRecognizer()).read(np.zeros((30, 40, 3), np.uint8))
