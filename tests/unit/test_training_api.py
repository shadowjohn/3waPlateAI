from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def training_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, object]]:
    from plateai_web import app as app_module
    from plateai_web.tasks import TaskManager
    from plateai_web.training_store import TrainingStore

    store = TrainingStore(tmp_path / "runs" / ".web-training")
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    monkeypatch.setattr(app_module, "task_manager", TaskManager())
    monkeypatch.setattr(app_module, "launch_training_worker", lambda _root, _task_id: None)
    app_module.app.dependency_overrides[app_module.get_training_store] = lambda: store
    try:
        with TestClient(app_module.app) as client:
            yield client, store
    finally:
        app_module.app.dependency_overrides.clear()


def test_training_task_is_readable_after_store_reopens(training_client) -> None:
    client, store = training_client
    task_id = "0123456789ab"
    store.create(task_id, "模型訓練", {"epochs": 1})

    response = client.get(f"/api/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "pending"
    assert type(store)(store.state_dir).get(task_id)["name"] == "模型訓練"


def test_start_creates_persisted_task_and_rejects_pending_duplicate(training_client) -> None:
    client, store = training_client

    started = client.post("/api/train/start", json={"epochs": 1})
    assert started.status_code == 200
    task_id = started.json()["task_id"]
    assert store.get(task_id)["status"] == "pending"

    busy = client.post("/api/train/start", json={"epochs": 1})
    assert busy.status_code == 409
    assert busy.json()["task_id"] == task_id


def test_start_failure_marks_the_persisted_row_failed(
    training_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, store = training_client
    from plateai_web import app as app_module

    def fail_launch(_root, _task_id):
        raise OSError("detached worker cannot start")

    monkeypatch.setattr(app_module, "launch_training_worker", fail_launch)
    response = client.post("/api/train/start", json={"epochs": 1})
    assert response.status_code == 503
    failed = store.latest()
    assert failed is not None
    assert failed["status"] == "failed"
    assert "cannot start" in failed["error"]


def test_stop_only_records_a_cancellation_request(training_client) -> None:
    client, store = training_client
    task_id = "abcdef012345"
    store.create(task_id, "模型訓練", {"epochs": 1})

    response = client.post("/api/train/stop")
    assert response.status_code == 200
    assert response.json()["cancel_requested"] is True
    assert response.json()["cancelled"] is False
    assert client.get(f"/api/tasks/{task_id}").json()["status"] == "pending"


def test_unknown_and_memory_tasks_remain_correctly_routed(training_client) -> None:
    client, _store = training_client
    from plateai_web import app as app_module

    memory_id = app_module.task_manager.create_task("環境建置")
    memory = client.get(f"/api/tasks/{memory_id}")
    assert memory.status_code == 200
    assert memory.json()["name"] == "環境建置"
    assert client.get("/api/tasks/111111111111").status_code == 404


def test_invalid_training_request_is_not_spawned(training_client) -> None:
    client, store = training_client
    response = client.post("/api/train/start", json={"epochs": 1, "run_name": "../escape"})
    assert response.status_code == 422
    assert store.latest() is None
