from __future__ import annotations

from importlib import metadata

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
