from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows detached worker contract")


def test_detached_worker_outlives_its_launcher_and_honors_cancellation(tmp_path: Path) -> None:
    from plateai_web.training_store import TrainingStore

    root = tmp_path / "project"
    state_dir = root / "runs" / ".web-training"
    store = TrainingStore(state_dir)
    task_id = "abcdef987654"
    store.create(task_id, "train", {"epochs": 1})
    helper = Path(__file__).parents[1] / "helpers" / "background_training_probe.py"
    log_path = store.log_path(task_id)
    environment = os.environ.copy()
    source = Path(__file__).resolve().parents[2] / "src"
    environment["PYTHONPATH"] = str(source) + os.pathsep + environment.get("PYTHONPATH", "")

    parent = subprocess.run(
        [
            sys.executable,
            "-u",
            str(helper),
            "parent",
            "--root",
            str(root),
            "--task-id",
            task_id,
            "--log",
            str(log_path),
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert parent.returncode == 0

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        task = store.get(task_id)
        if task and task["status"] == "running" and task["heartbeat_at"] is not None:
            break
        time.sleep(0.05)
    else:
        pytest.fail("detached worker never claimed the task")

    assert store.request_cancel(task_id)
    while time.monotonic() < deadline:
        task = store.get(task_id)
        if task and task["status"] == "cancelled":
            break
        time.sleep(0.05)
    else:
        pytest.fail("detached worker did not observe cancellation")

    lines, _total = store.read_logs(task_id)
    assert any("probe worker started" in line for line in lines)
