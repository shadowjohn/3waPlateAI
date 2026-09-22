from __future__ import annotations

from pathlib import Path
import pytest
import torch

from plateai_trainer.training.cli import build_parser
from plateai_trainer.training.dataset import TrainingDataError
from plateai_trainer.training.engine import (
    TrainingConfig,
    resolve_training_device,
    _validate_config,
)


def test_training_config_defaults_to_auto_device():
    config = TrainingConfig(
        train_directory=Path("train"),
        validation_directory=Path("val"),
        output_directory=Path("out"),
    )
    assert config.device == "auto"


def test_cli_parser_defaults_to_auto_device():
    parser = build_parser()
    args = parser.parse_args(["--train", "t", "--validation", "v", "--output", "o"])
    assert args.device == "auto"


def test_resolve_device_auto_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = resolve_training_device("auto")
    assert device.type == "cuda"


def test_resolve_device_auto_falls_back_to_cpu_when_no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = resolve_training_device("auto")
    assert device.type == "cpu"


def test_resolve_device_none_falls_back_to_cpu_when_no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = resolve_training_device(None)
    assert device.type == "cpu"


def test_resolve_device_cuda_falls_back_to_cpu_when_no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = resolve_training_device("cuda")
    assert device.type == "cpu"


def test_resolve_device_explicit_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = resolve_training_device("cpu")
    assert device.type == "cpu"


def test_resolve_device_invalid_raises_training_data_error():
    with pytest.raises(TrainingDataError, match="invalid device"):
        resolve_training_device("totally_nonexistent_device_xyz")


def test_validate_config_resolves_device(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    config = TrainingConfig(
        train_directory=tmp_path / "t",
        validation_directory=tmp_path / "v",
        output_directory=tmp_path / "out_does_not_exist",
        device="auto",
    )
    dev = _validate_config(config)
    assert dev.type == "cuda"
