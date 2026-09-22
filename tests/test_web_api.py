"""Isolated FastAPI checks for 3waPlateAI Web Studio."""
from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def web_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from plateai_web import app as app_module
    from plateai_web.tasks import TaskManager
    from plateai_web.training_store import TrainingStore

    sample = tmp_path / "out" / "sample" / "images"
    sample.mkdir(parents=True)
    (sample / "000000.png").write_bytes(b"not-used-by-dataset-listing")
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    monkeypatch.setattr(app_module, "task_manager", TaskManager())
    store = TrainingStore(tmp_path / "runs" / ".web-training")
    app_module.app.dependency_overrides[app_module.get_training_store] = lambda: store
    try:
        with TestClient(app_module.app) as client:
            yield client
    finally:
        app_module.app.dependency_overrides.clear()


def _wait_for_task(client: TestClient, task_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/tasks/{task_id}")
        task = response.json()
        if task["status"] in {"completed", "failed"}:
            return task
        time.sleep(0.02)
    pytest.fail("stub task did not reach a terminal state")


def test_index_page(web_client: TestClient):
    response = web_client.get("/")
    assert response.status_code == 200
    assert "3waPlateAI Studio" in response.text
    assert "老司機" in response.text


def test_static_assets(web_client: TestClient):
    assert web_client.get("/css/app.css").status_code == 200
    assert web_client.get("/js/app.js").status_code == 200


def test_system_status_and_sample_dataset(web_client: TestClient):
    status = web_client.get("/api/status")
    assert status.status_code == 200
    assert {"system", "datasets", "model"} <= status.json().keys()
    datasets = web_client.get("/api/train/datasets").json()["datasets"]
    assert len(datasets) == 1
    assert datasets[0]["name"] == "sample"
    assert datasets[0]["count"] == 1


def test_gpu_memory_api_reports_nvidia_gpu_utilization(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    from plateai_web import app as app_module

    calls: list[tuple[list[str], dict]] = []

    class NvidiaSmiResult:
        stdout = "73, 19\r\n"

    def fake_nvidia_smi(command: list[str], **kwargs):
        calls.append((command, kwargs))
        return NvidiaSmiResult()

    monkeypatch.setattr("subprocess.run", fake_nvidia_smi)
    data = web_client.get("/api/system/gpu_memory").json()
    assert "percent" in data
    assert "device_name" in data
    assert data["gpu_utilization_percent"] == 73.0
    assert data["memory_utilization_percent"] == 19.0
    assert data["metrics_source"] == "nvidia-smi"
    assert calls[0][0] == [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]


def test_train_active_and_stop_without_a_worker(web_client: TestClient):
    assert web_client.get("/api/train/active").json()["active"] is False
    stop = web_client.post("/api/train/stop")
    assert stop.status_code == 200
    assert stop.json()["cancel_requested"] is False


def test_env_build_task_uses_a_stubbed_runner(web_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from plateai_web import app as app_module

    def fake_env(task_id, manager):
        manager.complete_task(task_id, result={"status": "stubbed"}, message="stubbed env")

    monkeypatch.setattr(app_module, "_run_build_env_task", fake_env)
    task_id = web_client.post("/api/env/build").json()["task_id"]
    task = _wait_for_task(web_client, task_id)
    assert task["result"] == {"status": "stubbed"}


def test_predict_fixture_image(web_client: TestClient):
    fixture_png = PROJECT_ROOT / "tests" / "fixtures" / "synthetic" / "000000.png"
    with fixture_png.open("rb") as stream:
        response = web_client.post("/api/predict", files={"file": ("plate.png", stream, "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "diagnostics" in data
    diag = data["diagnostics"]
    # The public source checkout deliberately ships without an ONNX Bundle.
    assert data["status"] == "model_error"
    assert diag["pipeline_mode"] == "unavailable"
    assert diag["detector_available"] is False
    assert "timing_breakdown" in diag
    tb = diag["timing_breakdown"]
    for key in ("locator_ms", "rectifier_ms", "onnx_inference_ms", "ctc_decoding_ms", "total_ms"):
        assert key in tb


def test_compare_api_returns_named_active_and_fixed_candidate_without_activation(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from plateai_web import app as app_module

    calls: list[tuple[str, bytes]] = []

    class StubPredictor:
        def __init__(self, bundle_name: str):
            self.bundle_name = bundle_name

        def predict_image(self, image_bytes: bytes) -> dict:
            calls.append((self.bundle_name, image_bytes))
            return {
                "status": "no_plate", "detections": [], "rejections": [], "count": 0,
                "latency_ms": 1.0,
                "diagnostics": {"bundle_name": self.bundle_name, "warnings": [],
                                "timing_breakdown": {}},
            }

    active = StubPredictor("active-v1")
    candidate = StubPredictor("candidate-detector-real-v1")
    monkeypatch.setattr(app_module, "predictor", active)
    monkeypatch.setattr(app_module, "_candidate_predictor", lambda: candidate)
    response = web_client.post(
        "/api/predict/compare", files={"file": ("plate.png", b"image-payload", "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["active"]["diagnostics"]["bundle_name"] == "active-v1"
    assert data["candidate"]["diagnostics"]["bundle_name"] == "candidate-detector-real-v1"
    assert data["preview"] == {
        "active_bundle": "active-v1", "candidate_bundle": "candidate-detector-real-v1",
        "activation_changed": False,
    }
    assert calls == [("active-v1", b"image-payload"), ("candidate-detector-real-v1", b"image-payload")]
    assert not (tmp_path / "models" / "bundles" / "active-v1").exists()


def test_predict_plate_reader_contract_alignment(monkeypatch: pytest.MonkeyPatch):
    import numpy as np
    from plateai_web.predictor import predictor
    from plateai_reader.runtime import ReaderResult, ReaderTiming, PlateRead, DecodedPlate
    from plateai_shared.detection import PlateDetection

    fake_detection = PlateDetection(
        bbox_xyxy=np.array([10.0, 20.0, 100.0, 60.0], dtype=np.float32),
        confidence=0.98,
        corners_xy=np.array([[10.0, 20.0], [100.0, 20.0], [100.0, 60.0], [10.0, 60.0]], dtype=np.float32),
    )
    fake_decoded = DecodedPlate(
        canonical="ABC5678",
        display="ABC-5678",
        rule_id="new-style-lll-dddd",
        plate_type="standard",
        log_probability=-0.05,
    )
    fake_read = PlateRead(detection=fake_detection, decoded=fake_decoded,
                         raw_greedy_text="ABC5678", crop_rgb=np.full((160, 380, 3), 255, np.uint8))
    fake_timing = ReaderTiming(
        detector_ms=12.5,
        rectifier_ms=1.2,
        recognizer_ms=3.4,
        retained_detection_count=1,
        rectified_plate_count=1,
        recognition_batch_sizes=(1,),
    )
    fake_result = ReaderResult(
        plates=(fake_read,),
        rejections=(),
        providers=("CPUExecutionProvider",),
        timing=fake_timing,
    )

    class FakePlateReader:
        def read(self, img_rgb):
            return fake_result

    monkeypatch.setattr(predictor, "reader", FakePlateReader())
    monkeypatch.setattr(predictor, "recognizer_session", object())
    monkeypatch.setattr(predictor, "load_error", None)
    monkeypatch.setattr(predictor, "_load_active_model", lambda: None)
    
    # Run prediction on a dummy white 100x200 image
    import cv2
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    _, enc = cv2.imencode(".png", img)
    res = predictor.predict_image(enc.tobytes())

    assert res["count"] == 1
    assert res["diagnostics"]["pipeline_mode"] == "neural_full_pipeline"
    assert res["diagnostics"]["locator_type"] == "plate_pose_net"
    assert res["diagnostics"]["detector_available"] is True
    assert res["diagnostics"]["timing_breakdown"]["locator_ms"] == 12.5
    det = res["detections"][0]
    assert det["plate_text"] == "ABC-5678"
    assert det["canonical"] == "ABC5678"
    assert det["box"] == [10, 20, 100, 60]
    assert len(det["polygon"]) == 4
    assert det["crop_base64"].startswith("data:image/jpeg;base64,")


def test_release_build_task_uses_a_stubbed_runner(web_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from plateai_web import app as app_module

    def fake_release(task_id, manager):
        manager.complete_task(task_id, result={"status": "stubbed-release"}, message="stubbed release")

    monkeypatch.setattr(app_module, "build_release_task", fake_release)
    task_id = web_client.post("/api/release/build").json()["task_id"]
    task = _wait_for_task(web_client, task_id)
    assert task["result"] == {"status": "stubbed-release"}
