"""The redistributable ONNX pair must match the pinned author models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from plateai_reader.fpga_assets import load_fpga_manifest
from tools.build_fpga_lpr_onnx import verify_original_weights


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "third_party" / "fpga_lpr"


def test_wrong_original_digest_is_rejected_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "best_val_loss.pth").write_bytes(b"wrong cpm")
    (tmp_path / "lpr_model_weight.pth").write_bytes(b"wrong lpr")

    def must_not_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("torch.load must not run on a mismatched source")

    monkeypatch.setattr("torch.load", must_not_load)
    with pytest.raises(ValueError, match="CPM source sha256 mismatch"):
        verify_original_weights(tmp_path)


def test_exported_onnx_signatures() -> None:
    manifest = load_fpga_manifest(ASSETS)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    cpm_session = ort.InferenceSession(
        str(ASSETS / manifest.components["cpm"].filename),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    lpr_session = ort.InferenceSession(
        str(ASSETS / manifest.components["lprnet"].filename),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    assert [(item.name, item.shape) for item in cpm_session.get_inputs()] == [
        ("input", ["batch", 3, 100, 100])
    ]
    assert [(item.name, item.shape) for item in cpm_session.get_outputs()] == [
        ("stage", ["batch", 4, 50, 50]),
        ("heatmap", ["batch", 4, 50, 50]),
    ]
    assert [(item.name, item.shape) for item in lpr_session.get_inputs()] == [
        ("input", ["batch", 3, 48, 94])
    ]
    assert [(item.name, item.shape) for item in lpr_session.get_outputs()] == [
        ("logits", ["batch", 37, 18])
    ]
    for batch in (1, 2):
        cpm_output = cpm_session.run(None, {"input": np.zeros((batch, 3, 100, 100), np.float32)})
        lpr_output = lpr_session.run(None, {"input": np.zeros((batch, 3, 48, 94), np.float32)})
        assert [item.shape for item in cpm_output] == [(batch, 4, 50, 50)] * 2
        assert [item.shape for item in lpr_output] == [(batch, 37, 18)]
