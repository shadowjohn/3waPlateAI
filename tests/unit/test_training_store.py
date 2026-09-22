from __future__ import annotations

import itertools
import math

import pytest

from plateai_web.training_store import TrainingStore


TASK_ID = "0123456789ab"


def test_autocommit_is_visible_and_cancel_blocks_claim(tmp_path):
    """Removing autocommit or ignoring cancellation would let a stale worker train."""
    statements: list[str] = []
    state_dir = tmp_path / "任務 記錄"
    writer = TrainingStore(state_dir, trace=statements.append)
    reader = TrainingStore(state_dir)

    assert not state_dir.exists()
    writer.create(TASK_ID, "模型訓練", {"epochs": 1})

    assert reader.get(TASK_ID)["status"] == "pending"
    assert writer.request_cancel(TASK_ID)
    assert not reader.claim(TASK_ID, 123, "process-start-1")
    assert reader.get(TASK_ID)["cancel_requested"] is True
    assert not any(
        statement.lstrip().split()[0].upper()
        in {"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT"}
        for statement in statements
        if statement.strip()
    )


def test_snapshot_heartbeat_and_terminal_states_preserve_real_task_state(tmp_path):
    """A heartbeat or stale terminal write must not erase progress or revive a task."""
    ticks = itertools.count(10.0)
    store = TrainingStore(tmp_path / "state", clock=lambda: next(ticks))
    store.create(TASK_ID, "模型訓練", {"epochs": 1})
    assert store.claim(TASK_ID, 99, "process-start-99")
    assert store.snapshot(
        TASK_ID,
        phase="training",
        progress=42,
        message="正在訓練",
        result={"history": []},
        advanced=True,
    )
    before_heartbeat = store.get(TASK_ID)

    assert store.heartbeat(TASK_ID)
    after_heartbeat = store.get(TASK_ID)
    assert after_heartbeat["progress"] == 42
    assert after_heartbeat["last_progress_at"] == before_heartbeat["last_progress_at"]
    assert store.request_cancel(TASK_ID)
    assert not store.finish(TASK_ID, "completed", message="完成")
    assert store.finish(TASK_ID, "cancelled", message="已停止")
    assert not store.finish(TASK_ID, "failed", message="不應覆寫")
    assert store.get(TASK_ID)["status"] == "cancelled"
    assert store.get(TASK_ID)["result"] == {"history": []}


def test_log_reader_returns_only_last_complete_lines_and_rejects_bad_task_ids(tmp_path):
    """Reading a task log must not expose a partial write or form a path from user input."""
    store = TrainingStore(tmp_path / "state")
    store.create(TASK_ID, "模型訓練", {"epochs": 1})
    log_path = store.log_path(TASK_ID)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    complete_lines = [f"第 {number} 行" for number in range(105)]
    log_path.write_text("\n".join(complete_lines) + "\n尚未寫完", encoding="utf-8")

    lines, total = store.read_logs(TASK_ID)

    assert total == 105
    assert lines == complete_lines[-100:]
    with pytest.raises(ValueError):
        store.log_path("../outside")
    with pytest.raises(ValueError):
        store.create("BADID", "模型訓練", {"epochs": 1})
    with pytest.raises(ValueError):
        store.create("abcdefabcdef", "模型訓練", {"loss": math.nan})
