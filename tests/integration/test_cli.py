from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from plateai_trainer.synthetic import cli


def venv_scripts() -> Path:
    return Path(sys.executable).parent


def detector_executable(command: str) -> Path:
    return venv_scripts() / (command + (".exe" if sys.platform == "win32" else ""))


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "plateai_trainer.synthetic", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_generate_command_creates_requested_count(tmp_path):
    output = tmp_path / "demo"
    result = run_cli(
        "generate",
        "--count",
        "3",
        "--seed",
        "42",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["generated"] == 3
    assert payload["output"] == str(output)


def test_default_cli_profile_generates_new_style_private_passenger_crops(tmp_path):
    output = tmp_path / "new-style-private-passenger"
    result = run_cli("generate", "--count", "1", "--seed", "42", "--output", str(output))
    assert result.returncode == 0, result.stderr
    with Image.open(output / "images/000000.png") as image:
        assert image.size == (380, 160)
    label = (output / "labels.txt").read_text(encoding="utf-8").split("\t", 1)[1]
    assert not ({"I", "O", "4"} & set(label))
    configuration = json.loads((output / "generation_config.json").read_text(encoding="utf-8"))
    assert configuration["font"] == "NotoSansMono[wdth,wght].ttf"
    assert configuration["font_variation_axes"] == [700, 62]
    assert configuration["font_sha256"] == "2cb2adb378a8f574213e23df697050b83c54c27df465a2015552740b2769a081"


def test_zero_count_returns_usage_error_without_output(tmp_path):
    output = tmp_path / "zero"
    result = run_cli("generate", "--count", "0", "--output", str(output))
    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert not output.exists()


def test_existing_output_returns_stable_exit_code(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    result = run_cli("generate", "--count", "1", "--output", str(output))
    assert result.returncode == 3
    assert "output already exists" in result.stderr


def test_missing_configuration_is_concise_validation_error(tmp_path):
    output = tmp_path / "missing-config"
    result = run_cli(
        "generate",
        "--count",
        "1",
        "--output",
        str(output),
        "--rules",
        str(tmp_path / "missing.json"),
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_default_config_path_falls_back_to_installed_data(monkeypatch, tmp_path):
    checkout_root = tmp_path / "checkout-without-assets"
    installed_root = tmp_path / "installed-data"
    asset = installed_root / "configs/charsets/tw_plate_latin_v1.txt"
    asset.parent.mkdir(parents=True)
    asset.write_text("A\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_REPOSITORY_ROOT", checkout_root)
    monkeypatch.setattr(cli, "_INSTALL_DATA_ROOT", installed_root)
    assert cli._default_config_path("configs/charsets/tw_plate_latin_v1.txt") == asset


@pytest.mark.parametrize(
    "executable",
    ("plateai-compose", "plateai-detect-train", "plateai-detect-export"),
)
def test_detector_cli_help_is_installed(executable):
    result = subprocess.run([detector_executable(executable), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("platform,suffix", [("win32", ".exe"), ("linux", "")])
@pytest.mark.parametrize("command", ["plateai-compose", "plateai-detect-train", "plateai-detect-export"])
def test_detector_command_path_uses_platform_entry_point(monkeypatch, platform, suffix, command):
    with monkeypatch.context() as context:
        context.setattr(sys, "platform", platform)
        executable = detector_executable(command)
    assert executable == Path(sys.executable).with_name(command + suffix)
