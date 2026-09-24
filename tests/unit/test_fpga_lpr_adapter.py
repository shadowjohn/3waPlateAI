"""The attributed OCR adapter is not a native-v1 CTC bundle."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plateai_reader.fpga_lpr import (
    FPGA_CHARS,
    FpgaLprError,
    FpgaLprRecognizer,
    decode_fpga_logits,
)
from plateai_reader.fpga_pipeline import FpgaSceneReader
from plateai_reader.runtime import DetectionResult
from plateai_shared.detection import PlateDetection


ASSETS = Path(__file__).resolve().parents[2] / "third_party" / "fpga_lpr"


def logits_for(indices: list[int]) -> np.ndarray:
    logits = np.full((37, 18), -10.0, dtype=np.float32)
    logits[36, :] = 1.0
    for time_index, class_index in enumerate(indices):
        logits[class_index, time_index] = 10.0
    return logits


class FakeSession:
    def __init__(self, model: str, points: tuple[tuple[int, int], ...] | None = None):
        self.model = model
        self.inputs: list[np.ndarray] = []
        self.points = points or ((5, 10), (45, 10), (45, 30), (5, 30))

    def get_inputs(self) -> list[SimpleNamespace]:
        shape = ["batch", 3, 100, 100] if self.model == "cpm" else ["batch", 3, 48, 94]
        return [SimpleNamespace(name="input", shape=shape, type="tensor(float)")]

    def get_outputs(self) -> list[SimpleNamespace]:
        if self.model == "cpm":
            return [
                SimpleNamespace(name=name, shape=["batch", 4, 50, 50], type="tensor(float)")
                for name in ("stage", "heatmap")
            ]
        return [SimpleNamespace(name="logits", shape=["batch", 37, 18], type="tensor(float)")]

    def run(self, _output_names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.inputs.append(feed["input"].copy())
        if self.model == "cpm":
            heatmap = np.zeros((1, 4, 50, 50), dtype=np.float32)
            for channel, (x, y) in enumerate(self.points):
                heatmap[0, channel, y, x] = 10.0
            return [heatmap.copy(), heatmap]
        return [logits_for([10, 36, 10, 36])[None]]


def _recognizer(
    points: tuple[tuple[int, int], ...] | None = None,
) -> tuple[FpgaLprRecognizer, FakeSession, FakeSession]:
    cpm = FakeSession("cpm", points)
    lpr = FakeSession("lprnet")

    def factory(path: Path, _providers: object) -> FakeSession:
        return cpm if path.name == "cpm.onnx" else lpr

    return FpgaLprRecognizer(ASSETS, session_factory=factory), cpm, lpr


def test_repeat_requires_blank_and_dash_is_blank() -> None:
    assert decode_fpga_logits(logits_for([10, 36, 10, 36]), FPGA_CHARS) == "AA"
    assert decode_fpga_logits(logits_for([10, 10, 36]), FPGA_CHARS) == "A"
    assert "-" not in decode_fpga_logits(logits_for([36] * 18), FPGA_CHARS)


def test_rgb_input_is_converted_to_author_bgr_and_returns_rgb_crop() -> None:
    reader, cpm, lpr = _recognizer()
    image_rgb = np.zeros((80, 160, 3), dtype=np.uint8)
    image_rgb[:, :] = [255, 0, 0]
    result = reader.recognize(image_rgb)
    assert np.allclose(cpm.inputs[0][0, :, 50, 50], [0.0, 0.0, 1.0])
    assert lpr.inputs[0].shape == (1, 3, 48, 94)
    assert result.raw_text == "AA"
    assert result.normalized_text == "AA"
    assert result.score_kind == "uncalibrated"
    assert result.aligned_rgb.shape == (48, 94, 3)
    assert result.aligned_rgb[24, 47, 0] == 255
    assert result.timings_ms["lprnet"] >= 0
    assert result.timings_ms["decode"] >= 0


def test_invalid_roi_is_rejected_before_inference() -> None:
    reader, cpm, _ = _recognizer()
    with pytest.raises(FpgaLprError, match="invalid_roi"):
        reader.recognize(np.zeros((50, 50), np.uint8))
    with pytest.raises(FpgaLprError, match="invalid_roi"):
        reader.recognize(np.zeros((50, 50, 3), np.float32))
    assert cpm.inputs == []


def test_safe_policy_rejects_duplicate_corners_but_accepts_shuffled_valid_shape() -> None:
    bad_points = ((5, 10), (5, 10), (45, 30), (5, 30))
    bad_reader, _, _ = _recognizer(bad_points)
    with pytest.raises(FpgaLprError, match="invalid_corners"):
        bad_reader.recognize(np.full((80, 160, 3), 255, np.uint8), corner_policy="safe")
    shuffled = ((45, 30), (5, 30), (5, 10), (45, 10))
    good_reader, _, _ = _recognizer(shuffled)
    assert good_reader.recognize(np.full((80, 160, 3), 255, np.uint8), corner_policy="safe").raw_text == "AA"


def test_compat_and_scene_reject_degenerate_real_cpm_corners() -> None:
    bad_reader, _, lpr = _recognizer(((5, 10), (5, 10), (45, 30), (5, 30)))
    image = np.full((80, 160, 3), 255, np.uint8)
    with pytest.raises(FpgaLprError, match="invalid_corners"):
        bad_reader.recognize(image, corner_policy="compat")
    detection = PlateDetection(
        bbox_xyxy=np.array([5, 5, 100, 50], np.float32), confidence=0.9,
        corners_xy=np.array([[5, 5], [100, 5], [100, 50], [5, 50]], np.float32),
    )

    class FakeDetector:
        def detect(self, _image):
            return DetectionResult((detection,), ("CPUExecutionProvider",), 1.0)

    scene = FpgaSceneReader(FakeDetector(), bad_reader).read(image)
    assert scene.plates == ()
    assert [item.reason for item in scene.rejections] == ["invalid_corners"]
    assert lpr.inputs == []


def test_runtime_rejects_wrong_onnx_metadata() -> None:
    cpm = FakeSession("cpm")
    cpm.get_outputs = lambda: [SimpleNamespace(name="heatmap", shape=["batch", 4, 50, 49], type="tensor(float)")]
    lpr = FakeSession("lprnet")

    def factory(path: Path, _providers: object) -> FakeSession:
        return cpm if path.name == "cpm.onnx" else lpr

    with pytest.raises(FpgaLprError, match="invalid_cpm_session"):
        FpgaLprRecognizer(ASSETS, session_factory=factory)


def test_cpm_runtime_failure_is_a_per_plate_rejection() -> None:
    reader, cpm, _ = _recognizer()

    def fail_run(_outputs: object, _feed: object) -> list[np.ndarray]:
        raise RuntimeError("provider failed")

    cpm.run = fail_run
    with pytest.raises(FpgaLprError, match="cpm_inference_failed"):
        reader.recognize(np.full((80, 160, 3), 255, np.uint8))
