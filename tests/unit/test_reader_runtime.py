from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from plateai_reader import runtime
from plateai_reader.runtime import (
    PlateReader,
    ReaderError,
    decode_constrained_ctc_v1,
    select_ort_providers,
)
from plateai_shared.recognition import CTCCodec
from plateai_shared.rules import load_character_set, load_ruleset
from tests.conftest import V1_CHARSET, V1_RULES
from tests.contract.test_model_bundle import valid_full_manifest


def _ctc_logits(codec: CTCCodec, text: str) -> np.ndarray:
    logits = np.full((80, codec.class_count), -20.0, dtype=np.float32)
    logits[:, codec.blank_index] = 5.0
    for offset, symbol in enumerate(text):
        logits[1 + offset * 2, codec.index_by_symbol[symbol]] = 20.0
    return logits


def _full_bundle(tmp_path: Path) -> tuple[Path, dict[str, object], CTCCodec]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "charset.txt").write_bytes(V1_CHARSET.read_bytes())
    (bundle / "plate_rules.json").write_bytes(V1_RULES.read_bytes())
    manifest = valid_full_manifest()
    return bundle, manifest, CTCCodec.from_charset(load_character_set(bundle / "charset.txt"))


def _candidate(corners: list[list[float]], confidence: float = 0.9) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float32)
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    return np.asarray(
        [
            (left + right) / 2,
            (top + bottom) / 2,
            right - left,
            bottom - top,
            confidence,
            *points.reshape(-1),
        ],
        dtype=np.float32,
    )


class _FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.calls: list[tuple[list[str] | None, dict[str, object]]] = []

    def run(self, output_names, input_feed):
        self.calls.append((output_names, input_feed))
        if output_names == ["logits"]:
            batch = np.asarray(input_feed["input"]).shape[0]
            return [np.repeat(self.output[np.newaxis, ...], batch, axis=0)]
        return [self.output]


def test_constrained_ctc_decoder_accepts_blank_separated_repeat_heavy_rule(v1_charset):
    codec = CTCCodec.from_charset(v1_charset)
    ruleset = load_ruleset(V1_RULES, v1_charset)

    decoded = decode_constrained_ctc_v1(
        _ctc_logits(codec, "AAA8888"), codec, ruleset
    )

    assert decoded.canonical == "AAA8888"
    assert decoded.display == "AAA-8888"
    assert decoded.rule_id == "new-style-private-passenger-lll-dddd"


def test_constrained_ctc_decoder_rejects_wrong_logit_shape(v1_charset):
    codec = CTCCodec.from_charset(v1_charset)
    ruleset = load_ruleset(V1_RULES, v1_charset)

    with pytest.raises(ReaderError, match=r"\[80, 34\]"):
        decode_constrained_ctc_v1(
            np.zeros((79, codec.class_count), dtype=np.float32), codec, ruleset
        )


def test_provider_selection_prefers_tensorrt_then_cuda_then_cpu():
    assert select_ort_providers(
        ["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"]
    ) == (
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    with pytest.raises(ReaderError, match="none of"):
        select_ort_providers(["OpenVINOExecutionProvider"])


def test_reader_runs_validated_bundle_with_persistent_sessions(tmp_path, monkeypatch):
    bundle, manifest, codec = _full_bundle(tmp_path)
    monkeypatch.setattr(runtime, "validate_model_bundle", lambda *_: manifest)

    candidates = np.zeros((1, 8400, 13), dtype=np.float32)
    # Source image is 200x100, letterboxed to 640x640 with y padding 160.
    candidates[0, 0] = _candidate(
        [[32, 224], [608, 224], [608, 416], [32, 416]]
    )
    detector = _FakeSession(candidates)
    recognizer = _FakeSession(_ctc_logits(codec, "AAA8888"))
    created: list[Path] = []

    def factory(path, _providers):
        created.append(path)
        return detector if path.name == "detector.onnx" else recognizer

    reader = PlateReader(
        bundle,
        providers=("CPUExecutionProvider",),
        session_factory=factory,
    )
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    first = reader.read(image)
    second = reader.read(image)

    assert [path.name for path in created] == ["detector.onnx", "recognizer.onnx"]
    assert len(first.plates) == len(second.plates) == 1
    assert first.plates[0].decoded.canonical == "AAA8888"
    assert first.plates[0].detection.corners_xy.tolist() == [
        [10.0, 20.0], [190.0, 20.0], [190.0, 80.0], [10.0, 80.0]
    ]
    assert first.timing.recognition_batch_sizes == (1,)
    assert len(detector.calls) == 2
    assert len(recognizer.calls) == 2


def test_reader_isolates_one_invalid_rectification(tmp_path, monkeypatch):
    bundle, manifest, codec = _full_bundle(tmp_path)
    monkeypatch.setattr(runtime, "validate_model_bundle", lambda *_: deepcopy(manifest))

    candidates = np.zeros((1, 8400, 13), dtype=np.float32)
    candidates[0, 0] = _candidate([[32, 224], [608, 224], [608, 224], [32, 416]], 0.95)
    candidates[0, 1] = _candidate([[32, 224], [288, 224], [288, 300], [32, 300]], 0.90)
    detector = _FakeSession(candidates)
    recognizer = _FakeSession(_ctc_logits(codec, "AAA8888"))

    reader = PlateReader(
        bundle,
        providers=("CPUExecutionProvider",),
        session_factory=lambda path, _: detector if path.name == "detector.onnx" else recognizer,
    )
    result = reader.read(np.full((100, 200, 3), 255, dtype=np.uint8))

    assert len(result.plates) == 1
    assert [(item.stage, item.reason) for item in result.rejections] == [
        ("rectifier", "duplicate")
    ]
