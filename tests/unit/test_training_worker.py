from __future__ import annotations

import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(__import__("sys").platform != "win32", reason="Windows worker contract")


def _store(root: Path):
    from plateai_web.training_store import TrainingStore

    return TrainingStore(root / "runs" / ".web-training")


def test_cancelled_before_claim_never_trains(tmp_path: Path) -> None:
    from plateai_web.training_worker import run_worker

    store = _store(tmp_path)
    task_id = "0123456789ab"
    store.create(task_id, "train", {"epochs": 1})
    assert store.request_cancel(task_id)

    def forbidden_pipeline(*args, **kwargs):
        raise AssertionError("cancelled task reached training")

    assert run_worker(tmp_path, task_id, pipeline=forbidden_pipeline) == 0
    assert store.get(task_id)["status"] == "cancelled"


def test_worker_persists_real_callback_result_before_completing(tmp_path: Path) -> None:
    from plateai_trainer.training.control import TrainingProgress
    from plateai_web.training_worker import run_worker

    store = _store(tmp_path)
    task_id = "abcdef012345"
    store.create(task_id, "train", {"epochs": 1})

    def pipeline(_root, _task_id, _request, *, on_progress, on_result, check_cancelled):
        check_cancelled()
        on_progress(
            TrainingProgress(
                kind="batch",
                phase="training",
                epoch=1,
                total_epochs=1,
                batch=1,
                total_batches=1,
                train_loss=0.25,
            )
        )
        on_progress(
            TrainingProgress(
                kind="epoch",
                phase="validating",
                epoch=1,
                total_epochs=1,
                train_loss=0.25,
                val_loss=0.5,
                val_acc=0.75,
            )
        )
        result = {"bundle_dir": "bundles/train", "history": [{"epoch": 1, "val_acc": 75.0}]}
        on_result(result)
        return result

    assert run_worker(tmp_path, task_id, pipeline=pipeline) == 0
    task = store.get(task_id)
    assert task["status"] == "completed"
    assert task["progress"] == 100
    assert task["result"]["bundle_dir"] == "bundles/train"


def test_cancel_after_pipeline_result_preserves_published_paths(tmp_path: Path) -> None:
    from plateai_web.training_worker import run_worker

    store = _store(tmp_path)
    task_id = "123456abcdef"
    store.create(task_id, "train", {"epochs": 1})

    def pipeline(_root, _task_id, _request, *, on_progress, on_result, check_cancelled):
        result = {"bundle_dir": "bundles/already-published", "run_dir": "runs/kept"}
        on_result(result)
        assert store.request_cancel(task_id)
        return result

    assert run_worker(tmp_path, task_id, pipeline=pipeline) == 0
    task = store.get(task_id)
    assert task["status"] == "cancelled"
    assert task["result"]["bundle_dir"] == "bundles/already-published"


def test_heartbeat_storage_error_stops_at_the_next_safe_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plateai_web import training_worker
    from plateai_web.training_store import TrainingStore
    from plateai_web.training_worker import run_worker

    class FaultyHeartbeatStore(TrainingStore):
        def heartbeat(self, task_id: str) -> bool:
            raise OSError("simulated SQLite write failure")

    monkeypatch.setattr(training_worker, "TrainingStore", FaultyHeartbeatStore)
    monkeypatch.setattr(training_worker, "HEARTBEAT_INTERVAL_SECONDS", 0.01)
    store = FaultyHeartbeatStore(tmp_path / "runs" / ".web-training")
    task_id = "fedcba654321"
    store.create(task_id, "train", {"epochs": 1})

    def pipeline(_root, _task_id, _request, *, on_progress, on_result, check_cancelled):
        time.sleep(0.05)
        check_cancelled()
        raise AssertionError("storage failure should have stopped the pipeline")

    assert run_worker(tmp_path, task_id, pipeline=pipeline) == 1
    assert store.get(task_id)["status"] == "failed"
