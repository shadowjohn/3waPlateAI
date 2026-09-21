from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from plateai_trainer.training.cli import main


def test_train_cli_writes_checkpoint_and_synthetic_metrics(
    v1_train_dir, v1_validation_dir, tmp_path
):
    output = tmp_path / "run"
    result = main(
        [
            "--train",
            str(v1_train_dir),
            "--validation",
            str(v1_validation_dir),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--seed",
            "7",
        ]
    )

    assert result == 0
    assert (output / "best.pt").is_file()
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["validation"]["samples"] == 4


def test_installed_plateai_train_command_shows_help():
    executable = Path(sys.executable).with_name("plateai-train.exe")
    result = subprocess.run(
        [str(executable), "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0
    assert "--train" in result.stdout
