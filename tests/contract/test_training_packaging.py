from __future__ import annotations

from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement


def test_installed_distribution_exposes_the_isolated_training_extra():
    """Fails if the installable package drops M2's CPU export dependencies."""

    requirements = metadata.distribution("3wa-plate-ai").requires or []
    training = {
        requirement.name: str(requirement.specifier)
        for raw in requirements
        for requirement in [Requirement(raw)]
        if requirement.marker is not None
        and requirement.marker.evaluate({"extra": "training"})
    }

    assert training == {
        "torch": "==2.14.0",
        "onnx": "==1.23.0",
        "onnxruntime": "==1.30.0",
        "onnxscript": "==0.7.2",
    }


def test_installed_distribution_exposes_the_lightweight_reader_extra():
    requirements = metadata.distribution("3wa-plate-ai").requires or []
    reader = {
        requirement.name: str(requirement.specifier)
        for raw in requirements
        for requirement in [Requirement(raw)]
        if requirement.marker is not None
        and requirement.marker.evaluate({"extra": "reader"})
    }

    assert reader == {
        "onnx": "==1.23.0",
        "onnxruntime": "==1.30.0",
    }


def test_training_docs_publish_the_v1_input_and_onnx_parity_contract():
    text = (Path(__file__).resolve().parents[2] / "docs/training.md").read_text(encoding="utf-8")
    assert "[batch, 1, 64, 160]" in text
    assert "blank_index = 0" in text
    assert "native-versus-ONNX parity" in text
    assert "not committed" in text
