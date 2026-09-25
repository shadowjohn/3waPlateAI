"""Image-backed benchmark reports must preserve scoring and bounded reads."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from plateai_web import evaluator
from plateai_web.tasks import TaskManager


def test_browser_bbox_text_uses_source_labels_and_original_pixel_coordinates():
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser formatter regression")
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
// Skip DOM-ready setup, but execute the actual production formatter.
const context = { $: () => {} };
vm.runInNewContext(fs.readFileSync('web/js/benchmark-details.js', 'utf8'), context);
assert.equal(typeof context.benchmarkBboxText, 'function', 'report needs a bbox formatter');
assert.equal(context.benchmarkBboxText({bbox_source:'detector', predicted_boxes:[[1.24,2.26,30,40],[5,6,70,80]]}),
    'P1 [1.2,2.3,30,40]\nP2 [5,6,70,80]');
assert.equal(context.benchmarkBboxText({bbox_source:'gt_crop', gt_box:[10,20,300,400], predicted_boxes:[]}),
    'GT [10,20,300,400]');
assert.equal(context.benchmarkBboxText({bbox_source:'detector', gt_box:[10,20,300,400], predicted_boxes:[]}),
    '未定位');
"""
    result = subprocess.run([node, "-e", script], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr


def make_dataset(root, count=2):
    folder = root / "datasets/real_benchmarks"
    folder.mkdir(parents=True)
    image = np.full((80, 160, 3), 230, np.uint8)
    cv2.imwrite(str(folder / "plate.jpg"), image)
    rows = [{"image": {"path": "plate.jpg"}, "ground_truth": {
        "canonical": f"ABC{i:04}", "corners": [[20, 20], [140, 20], [140, 60], [20, 60]]}}
        for i in range(count)]
    (folder / "user_cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return folder


def test_report_one_row_per_sample_red_text_and_geometry_are_independent(tmp_path, monkeypatch):
    make_dataset(tmp_path)
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)

    class Adapter:
        model_id = "test-reader"
        identity = {"manifest": "a" * 64}

        def __init__(self, root, kind):
            self.calls = 0

        def predict(self, image, corners, mode):
            self.calls += 1
            # Matching text, wrong geometry: not a string mismatch, still a failure.
            return evaluator.Prediction("ABC0000", False, crop=image[20:60, 20:140],
                predicted_boxes=[[0, 0, 10, 10]], reason="no_gt_match", iou=0.0)

    monkeypatch.setattr(evaluator, "ModelAdapter", Adapter)
    tm = TaskManager()
    tid = tm.create_task("benchmark")
    evaluator.run_benchmark_task(tid, tm, "user", rounds=3, warmup=1, mode="scene")
    task = tm.get_task(tid)
    assert task.status.value == "completed", task.error
    assert task.result["results"][0]["success"] == "0/6"
    assert "report_id" in task.result
    from plateai_web import app as app_module
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    with TestClient(app_module.app) as client:
        base = f"/api/benchmark/reports/{task.result['report_id']}"
        response = client.get(base + "/rows", params={"group": "active-v1-scene"})
        assert response.status_code == 200
        rows = response.json()["rows"]
        assert len(rows) == 2  # not 6 trials or the warmup
        assert rows[0]["text_match"] is True
        assert rows[0]["correct"] is False
        assert rows[1]["text_match"] is False
        assert rows[0]["ocr_score"] is None  # never invented from detector score
        assert rows[0]["predicted_boxes"] == [[0, 0, 10, 10]]
        assert rows[0]["gt_box"] == [20.0, 20.0, 140.0, 60.0]
        assert rows[0]["bbox_source"] == "detector"
        thumb = client.get(rows[0]["image_url"])
        assert thumb.status_code == 200
        rgb = cv2.imdecode(np.frombuffer(thumb.content, np.uint8), cv2.IMREAD_COLOR)
        assert rgb is not None and max(rgb.shape[:2]) <= 640
        assert "base64" not in response.text
        page = client.get(base + "/rows", params={"group": "active-v1-scene", "offset": 1, "limit": 1}).json()
        assert page["total"] == 2 and len(page["rows"]) == 1
        assert page["rows"][0]["expected"] == "ABC0001"
        assert client.get(base + "/rows", params={"group": "active-v1-scene", "limit": 101}).status_code == 422
        assert client.get(base + "/images/manifest.json").status_code == 404
        assert client.get("/api/benchmark/reports/not-a-report/rows").status_code == 404


def test_thousands_of_samples_are_not_retained_as_decoded_images(tmp_path):
    folder = make_dataset(tmp_path, count=1001)
    samples, fingerprint = evaluator.load_samples(tmp_path, "user", 1001)
    assert len(samples) == 1001
    assert not isinstance(samples[0], tuple), "keep lazy image references, not full RGB arrays"
    image, corners, expected = samples[0]
    assert image.shape == (80, 160, 3) and expected == "ABC0000"
    (folder / "plate.jpg").write_bytes(b"changed after input snapshot")
    with pytest.raises(ValueError, match="changed|變更"):
        tuple(samples[0])


def test_measure_saves_details_outside_timing_and_without_extra_inference(monkeypatch):
    ticks = iter([10.0, 10.01, 20.0, 20.01])
    monkeypatch.setattr(evaluator.time, "perf_counter", lambda: next(ticks))
    calls, details = [], []
    class Adapter:
        def predict(self, image, corners, mode):
            calls.append(mode)
            return evaluator.Prediction("ABC0000", True)
    sample = (np.zeros((2, 2, 3), np.uint8), np.zeros((4, 2)), "ABC0000")
    result = evaluator.measure(Adapter(), [sample], "crop", 2, 1,
                               on_detail=lambda *args: details.append(args))
    assert len(calls) == 3 and len(details) == 1
    assert result["avg_ms"] == 10.0 and result["success"] == "2/2"


def test_actual_adapter_preserves_wrong_location_text_and_external_score_is_missing():
    adapter = object.__new__(evaluator.ModelAdapter)
    adapter.external = True
    detection = SimpleNamespace(bbox_xyxy=np.array([0, 0, 5, 5]),
        corners_xy=np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=np.float32))
    crop = np.ones((4, 8, 3), np.uint8)
    plate = SimpleNamespace(detection=detection, read=SimpleNamespace(normalized_text="ABC0000", aligned_rgb=crop))
    adapter.scene = SimpleNamespace(read=lambda _: SimpleNamespace(plates=[plate], rejections=[]))
    prediction = adapter.predict(crop, np.array([[20, 20], [40, 20], [40, 30], [20, 30]]), "scene")
    assert prediction.text == "ABC0000" and prediction.matched is False
    assert prediction.reason == "no_gt_match" and prediction.iou == 0.0
    assert prediction.score is None and prediction.crop is crop


def test_thousand_report_rows_are_paged_and_errors_filtered(tmp_path, monkeypatch):
    from plateai_web.benchmark_reports import ReportWriter
    from plateai_web import app as app_module
    writer = ReportWriter(tmp_path)
    group = "fpga-lpr-mit-crop"
    # Rendering is covered by the two-image integration above. Large row sets
    # exercise the real paged reader without generating redundant test JPEGs.
    rows = [{"index": i, "correct": i % 2 == 0} for i in range(1001)]
    (writer.staging / f"{group}.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    with TestClient(app_module.app) as client:
        url = f"/api/benchmark/reports/{writer.report_id}/rows"
        assert client.get(url, params={"group": group}).status_code == 404  # not yet published
        writer.finish({"groups": [group]})
        page = client.get(url, params={"group": group}).json()
        assert page["total"] == 1001 and len(page["rows"]) == 100
        tail = client.get(url, params={"group": group, "offset": 1000}).json()
        assert [row["index"] for row in tail["rows"]] == [1000]
        errors = client.get(url, params={"group": group, "errors_only": True}).json()
        assert errors["total"] == 500 and len(errors["rows"]) == 100
        assert [row["index"] for row in errors["rows"][:3]] == [1, 3, 5]
