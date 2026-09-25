from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def test_linux_api_launcher_exposes_local_microservice_options() -> None:
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        ["bash", "run_api_1788.sh", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--host" in result.stdout
    assert "127.0.0.1" in result.stdout


def test_linux_api_launcher_forwards_its_local_defaults_to_the_service() -> None:
    project_root = Path(__file__).resolve().parents[2]
    echo = shutil.which("echo")
    assert echo is not None

    result = subprocess.run(
        [
            "bash",
            "run_api_1788.sh",
            "--python",
            echo,
            "--config",
            str(Path(__file__).resolve()),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.split() == [
        "-m",
        "plateai_service.cli",
        "--config",
        str(Path(__file__).resolve()),
        "--device",
        "auto",
        "--host",
        "127.0.0.1",
        "--port",
        "1788",
    ]
