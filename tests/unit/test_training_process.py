from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows file lock contract")


def _child_environment() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


def _start_lock_holder(path: Path, mode: str) -> subprocess.Popen[str]:
    helper = Path(__file__).parents[1] / "helpers" / "background_training_probe.py"
    return subprocess.Popen(
        [sys.executable, "-u", str(helper), mode, "--lock", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True,
        env=_child_environment(),
    )


@pytest.mark.parametrize("mode", ["lock", "lock-crash"])
def test_two_processes_cannot_hold_same_lock_and_release_after_exit(
    tmp_path: Path, mode: str
) -> None:
    from plateai_web.training_process import TrainingBusy, training_lock

    lock_path = tmp_path / "worker.lock"
    child = _start_lock_holder(lock_path, mode)
    assert child.stdout is not None
    try:
        assert child.stdout.readline().strip() == "LOCKED"
        with pytest.raises(TrainingBusy):
            with training_lock(lock_path):
                raise AssertionError("should not acquire another process lock")
        assert child.stdin is not None
        child.stdin.write('release\n')
        child.stdin.flush()
        assert child.wait(timeout=10) == (0 if mode == "lock" else 9)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)

    with training_lock(lock_path):
        pass


def test_process_identity_does_not_treat_a_pid_reuse_as_alive() -> None:
    from plateai_web.training_process import current_process_identity, process_state

    process_id, start_token = current_process_identity()
    assert process_state(process_id, start_token) == "alive"
    assert process_state(process_id, start_token + "different") == "dead"


def test_reconcile_only_fails_confirmed_dead_or_stale_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plateai_web import training_process
    from plateai_web.training_store import TrainingStore

    store = TrainingStore(tmp_path / "runs" / ".web-training", clock=lambda: 0.0)
    pending_id = "0123456789ab"
    running_id = "abcdef012345"
    store.create(pending_id, "train", {"epochs": 1})
    store.create(running_id, "train", {"epochs": 1})
    assert store.claim(running_id, 999, "old-process")

    monkeypatch.setattr(training_process.time, "time", lambda: 31.0)
    monkeypatch.setattr(
        training_process,
        "process_state",
        lambda pid, token: "dead" if pid == 999 else "unknown",
    )
    training_process.reconcile_training_tasks(store)
    assert store.get(pending_id)["status"] == "failed"
    assert store.get(running_id)["status"] == "failed"
