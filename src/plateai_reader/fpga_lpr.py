"""Attributed ONNX CPM + LPRNet reader for one plate-centric RGB image."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Protocol, Sequence

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from .fpga_assets import FPGA_CHARS, FpgaLprManifest, load_fpga_manifest
from .rectifier import InvalidCornersError, normalize_corners


class FpgaLprError(ValueError):
    """An ROI cannot be processed under the selected external OCR policy."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class OrtSession(Protocol):
    def get_inputs(self) -> Sequence[object]: ...

    def get_outputs(self) -> Sequence[object]: ...

    def run(self, output_names: Sequence[str] | None, feed: dict[str, NDArray[np.float32]]) -> Sequence[object]: ...


SessionFactory = Callable[[Path, Sequence[str]], OrtSession]


@dataclass(frozen=True, slots=True)
class FpgaLprRead:
    raw_text: str
    normalized_text: str
    aligned_rgb: NDArray[np.uint8] = field(repr=False, compare=False)
    roi_corners_xy: NDArray[np.float32] = field(repr=False, compare=False)
    timings_ms: dict[str, float]
    score_kind: Literal["uncalibrated"] = "uncalibrated"


def collapse_ctc(indices: Sequence[int], *, blank_index: int) -> tuple[int, ...]:
    result: list[int] = []
    previous = -1
    for index in indices:
        current = int(index)
        if current != blank_index and current != previous:
            result.append(current)
        previous = current
    return tuple(result)


def decode_fpga_logits(logits: np.ndarray, chars: Sequence[str] = FPGA_CHARS) -> str:
    """Match the author's greedy blank-last collapse, without v1 rule filtering."""

    if (
        not isinstance(logits, np.ndarray)
        or logits.shape != (37, 18)
        or len(chars) != 37
        or not np.isfinite(logits).all()
    ):
        raise FpgaLprError("invalid_lpr_logits")
    return "".join(chars[index] for index in collapse_ctc(logits.argmax(axis=0), blank_index=36))


def _default_session_factory(path: Path, providers: Sequence[str]) -> OrtSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=list(providers))


def _check_session(session: OrtSession, expected_inputs: object, expected_outputs: object, label: str) -> None:
    try:
        inputs = [(item.name, item.shape, item.type) for item in session.get_inputs()]
        outputs = [(item.name, item.shape, item.type) for item in session.get_outputs()]
    except (AttributeError, TypeError) as exc:
        raise FpgaLprError(f"invalid_{label}_session") from exc
    expected_in = [(name, list(shape), "tensor(float)") for name, shape in expected_inputs.items()]
    expected_out = [(name, list(shape), "tensor(float)") for name, shape in expected_outputs.items()]
    if inputs != expected_in or outputs != expected_out:
        raise FpgaLprError(f"invalid_{label}_session")


def _validate_roi(image: object) -> NDArray[np.uint8]:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] < 1
        or image.shape[1] < 1
    ):
        raise FpgaLprError("invalid_roi")
    return image


def _heatmap_corners(heatmap: NDArray[np.float32]) -> NDArray[np.float32]:
    """Author sum/difference order on 100x100 upsampled CPM channels."""

    upsampled = np.stack(
        [cv2.resize(channel, (100, 100), interpolation=cv2.INTER_LINEAR) for channel in heatmap],
        axis=0,
    )
    points = np.array(
        [
            (int(np.argmax(channel) % 100), int(np.argmax(channel) // 100))
            for channel in upsampled
        ],
        dtype=np.float32,
    )
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    return points[[np.argmin(sums), np.argmin(diffs), np.argmax(sums), np.argmax(diffs)]]


def _validate_author_corners(corners: NDArray[np.float32]) -> None:
    """Reject impossible warps without changing the author's valid point order."""

    if (
        corners.shape != (4, 2)
        or not np.isfinite(corners).all()
        or (corners < 0).any()
        or (corners > 99).any()
        or np.unique(corners, axis=0).shape[0] != 4
        or not cv2.isContourConvex(corners)
        or cv2.contourArea(corners) < 4.0
    ):
        raise FpgaLprError("invalid_corners")


class FpgaLprRecognizer:
    """One fixed external model pair; never a native v1 model bundle."""

    def __init__(
        self,
        assets_dir: Path,
        providers: Sequence[str] | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self.assets_dir = Path(assets_dir)
        self.manifest: FpgaLprManifest = load_fpga_manifest(self.assets_dir)
        requested = tuple(providers or ("CPUExecutionProvider",))
        if not requested or any(provider not in ort.get_available_providers() for provider in requested):
            raise FpgaLprError("unavailable_onnx_provider")
        factory = session_factory or _default_session_factory
        try:
            self.cpm = factory(self.assets_dir / "cpm.onnx", requested)
            self.lprnet = factory(self.assets_dir / "lprnet.onnx", requested)
        except Exception as exc:
            raise FpgaLprError("onnx_session_load_failed") from exc
        _check_session(
            self.cpm,
            self.manifest.components["cpm"].inputs,
            self.manifest.components["cpm"].outputs,
            "cpm",
        )
        _check_session(
            self.lprnet,
            self.manifest.components["lprnet"].inputs,
            self.manifest.components["lprnet"].outputs,
            "lprnet",
        )
        self.providers = requested

    def recognize(
        self,
        roi_rgb: NDArray[np.uint8],
        *,
        corner_policy: Literal["compat", "safe"] = "compat",
    ) -> FpgaLprRead:
        if corner_policy not in ("compat", "safe"):
            raise FpgaLprError("invalid_corner_policy")
        image_rgb = _validate_roi(roi_rgb)
        started = time.perf_counter()
        # The public image contract is RGB; the author's OpenCV notebook used BGR.
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        resized_bgr = cv2.resize(bgr, (100, 100), interpolation=cv2.INTER_LINEAR)
        normalized = resized_bgr.astype(np.float64) / 255.0
        cpm_input = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
        preprocessed = time.perf_counter()
        try:
            cpm_outputs = self.cpm.run(None, {"input": cpm_input})
        except Exception as exc:
            raise FpgaLprError("cpm_inference_failed") from exc
        if len(cpm_outputs) != 2 or any(
            not isinstance(value, np.ndarray)
            or value.shape != (1, 4, 50, 50)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
            for value in cpm_outputs
        ):
            raise FpgaLprError("invalid_cpm_heatmap")
        cpm_done = time.perf_counter()
        corners = _heatmap_corners(cpm_outputs[1][0])
        if corner_policy == "safe":
            try:
                corners = normalize_corners(corners, (100, 100)).points_xy
            except InvalidCornersError as exc:
                raise FpgaLprError("invalid_corners") from exc
        else:
            _validate_author_corners(corners)
        destination = np.float32([[0, 0], [93, 0], [93, 47], [0, 47]])
        try:
            transform = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
            if not np.isfinite(transform).all():
                raise FpgaLprError("invalid_corners")
            # Match the original float64 divide/multiply/cast path.
            warped_bgr = cv2.warpPerspective((normalized * 255).astype(np.uint8), transform, (94, 48))
        except cv2.error as exc:
            raise FpgaLprError("invalid_corners") from exc
        lpr_input = np.ascontiguousarray(
            (((warped_bgr.astype(np.float32) - 127.5) * 0.0078125).transpose(2, 0, 1))[None]
        )
        rectified = time.perf_counter()
        try:
            lpr_outputs = self.lprnet.run(None, {"input": lpr_input})
        except Exception as exc:
            raise FpgaLprError("lprnet_inference_failed") from exc
        if len(lpr_outputs) != 1 or not isinstance(lpr_outputs[0], np.ndarray) or lpr_outputs[0].shape != (1, 37, 18):
            raise FpgaLprError("invalid_lpr_logits")
        lpr_done = time.perf_counter()
        raw_text = decode_fpga_logits(lpr_outputs[0][0], FPGA_CHARS)
        finished = time.perf_counter()
        return FpgaLprRead(
            raw_text=raw_text,
            normalized_text=re.sub(r"[^A-Z0-9]", "", raw_text.upper()),
            aligned_rgb=cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB),
            roi_corners_xy=corners,
            timings_ms={
                "preprocess": (preprocessed - started) * 1000,
                "cpm": (cpm_done - preprocessed) * 1000,
                "rectify": (rectified - cpm_done) * 1000,
                "lprnet": (lpr_done - rectified) * 1000,
                "decode": (finished - lpr_done) * 1000,
                "lprnet_decode": (finished - rectified) * 1000,
                "total": (finished - started) * 1000,
            },
        )
