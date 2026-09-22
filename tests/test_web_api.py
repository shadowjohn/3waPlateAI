"""Automated tests for 3waPlateAI Web Studio FastAPI backend."""
import os
import sys
import time
from pathlib import Path
import pytest
from starlette.testclient import TestClient

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))

from plateai_web.app import app

client = TestClient(app)


def test_index_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "3waPlateAI Studio" in resp.text
    assert "老司機" in resp.text


def test_static_assets():
    resp_css = client.get("/css/app.css")
    assert resp_css.status_code == 200
    resp_js = client.get("/js/app.js")
    assert resp_js.status_code == 200


def test_system_status():
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "system" in data
    assert "datasets" in data
    assert "model" in data


def test_train_datasets():
    resp = client.get("/api/train/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert "datasets" in data
    assert isinstance(data["datasets"], list)
    # Check if demo-10000 is present
    names = [d["name"] for d in data["datasets"]]
    assert any("demo-10000" in n for n in names)


def test_gpu_memory_api():
    resp = client.get("/api/system/gpu_memory")
    assert resp.status_code == 200
    data = resp.json()
    assert "used_gb" in data or "used_mb" in data
    assert "percent" in data
    assert "device_name" in data


def test_train_active_and_stop():
    # Active train status
    resp = client.get("/api/train/active")
    assert resp.status_code == 200
    data = resp.json()
    assert "active" in data

    # Stop even when stopped
    stop_resp = client.post("/api/train/stop")
    assert stop_resp.status_code == 200
    stop_data = stop_resp.json()
    assert stop_data["status"] == "ok"



def test_env_build_task():
    resp = client.post("/api/env/build")
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    
    # Wait for completion or check status
    time.sleep(1)
    task_resp = client.get(f"/api/tasks/{task_id}")
    assert task_resp.status_code == 200
    data = task_resp.json()
    assert data["id"] == task_id
    assert len(data["logs"]) >= 0


def test_predict_fixture_image():
    fixture_png = root / "tests" / "fixtures" / "synthetic" / "000000.png"
    assert fixture_png.exists()
    
    with open(fixture_png, "rb") as f:
        resp = client.post("/api/predict", files={"file": ("plate.png", f, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert "detections" in data
    assert len(data["detections"]) > 0
    assert "latency_ms" in data


def test_release_build_task():
    resp = client.post("/api/release/build")
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    
    # Poll until completed
    for _ in range(10):
        time.sleep(0.5)
        task_resp = client.get(f"/api/tasks/{task_id}")
        data = task_resp.json()
        if data["status"] == "completed":
            break
            
    release_dir = root / "release" / "3wa_plate_api"
    assert release_dir.exists()
    assert (release_dir / "run_api_1788.bat").exists()
    assert (release_dir / "app.py").exists()
    assert (release_dir / "requirements.txt").exists()
