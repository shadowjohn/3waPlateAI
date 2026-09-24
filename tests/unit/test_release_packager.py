"""The standalone scaffold must never publish a local model bundle."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from plateai_web import release_packager
from plateai_web.tasks import TaskManager, TaskStatus


def test_release_scaffold_omits_active_bundle_and_fails_closed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(release_packager, "workspace_root", lambda: tmp_path)
    (tmp_path / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (tmp_path / "THIRD_PARTY_NOTICES.md").write_text("# Notices\n", encoding="utf-8")
    active = tmp_path / "models" / "bundles" / "active-v1"
    active.mkdir(parents=True)
    (active / "detector.onnx").write_bytes(b"local-only model")

    tasks = TaskManager()
    task_id = tasks.create_task("release")
    release_packager.build_release_task(task_id, tasks)

    release_dir = tmp_path / "release" / "3wa_plate_api"
    assert not list((release_dir / "models").iterdir())
    assert (active / "detector.onnx").read_bytes() == b"local-only model"
    assert tasks.get_task(task_id).status is TaskStatus.COMPLETED
    assert tasks.get_task(task_id).result["status"] == "scaffold"
    assert (release_dir / "LICENSE").read_text(encoding="utf-8") == "MIT License\n"
    assert (release_dir / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8") == "# Notices\n"
    assert "python-multipart" in (release_dir / "requirements.txt").read_text(encoding="utf-8")

    module_globals = {"__name__": "standalone_release", "__file__": str(release_dir / "app.py")}
    exec((release_dir / "app.py").read_text(encoding="utf-8"), module_globals)
    assert TestClient(module_globals["app"]).get("/api/health").json()["inference_ready"] is False
    image = np.full((8, 12, 3), 255, dtype=np.uint8)
    encoded, png = cv2.imencode(".png", image)
    assert encoded
    response = TestClient(module_globals["app"]).post(
        "/api/predict", files={"file": ("plate.png", png.tobytes(), "image/png")}
    )
    assert response.status_code == 503
    assert "3WA-8888" not in response.text


def test_release_scaffold_rejects_existing_model_files_without_removing_them(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(release_packager, "workspace_root", lambda: tmp_path)
    release_dir = tmp_path / "release" / "3wa_plate_api"
    models_dir = release_dir / "models"
    models_dir.mkdir(parents=True)
    stale = models_dir / "old.onnx"
    stale.write_bytes(b"user model")

    tasks = TaskManager()
    task_id = tasks.create_task("release")
    with pytest.raises(RuntimeError, match="models"):
        release_packager.build_release_task(task_id, tasks)

    assert stale.read_bytes() == b"user model"
    assert not (release_dir / "app.py").exists()
