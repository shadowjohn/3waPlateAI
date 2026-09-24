"""Regression checks for data preservation and trustworthy Studio results."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plateai_web.tasks import TaskManager


def test_generation_refuses_existing_data_and_escape(tmp_path, monkeypatch):
    from plateai_web import generator
    monkeypatch.setattr(generator, "workspace_root", lambda: tmp_path)
    old = tmp_path / "out/keep"
    old.mkdir(parents=True)
    (old / "photo.png").write_bytes(b"user data")
    tm = TaskManager()
    tid = tm.create_task("generation")
    with pytest.raises(FileExistsError):
        generator.generate_dataset_task(tid, tm, count=1, output_dir_name="keep")
    with pytest.raises(ValueError):
        generator.generate_dataset_task(tid, tm, count=1, output_dir_name="../escape")
    assert (old / "photo.png").read_bytes() == b"user data"


def test_environment_failure_cannot_become_ready(monkeypatch):
    from plateai_web.app import _run_build_env_task
    tm = TaskManager()
    tid = tm.create_task("check")
    monkeypatch.setattr(tm, "run_subprocess_command", lambda *a: 1)
    _run_build_env_task(tid, tm)
    assert tm.get_task(tid).status.value == "failed"
    assert tm.get_task(tid).result.get("status") != "ready"


def test_legacy_config_basenames_resolve_by_exact_hash(v1_train_dir):
    from plateai_web.trainer import _dataset_contract
    p = v1_train_dir / "generation_config.json"
    config = json.loads(p.read_text())
    config["config_paths"] = {k: Path(v).name for k, v in config["config_paths"].items()}
    p.write_text(json.dumps(config))
    contract = _dataset_contract(v1_train_dir.parent, v1_train_dir)
    assert contract["rules"].is_file()
    config["config_sha256"]["rules"] = "0" * 64
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="找不到"):
        _dataset_contract(v1_train_dir.parent, v1_train_dir)


@pytest.mark.parametrize("fail_reload", [False, True])
def test_activation_preserves_previous_bytes_and_rolls_back(tmp_path, monkeypatch, fail_reload):
    from plateai_web import activation
    bundles = tmp_path / "models/bundles"
    for name, content in [("active-v1", b"old"), ("candidate", b"new")]:
        directory = bundles / name
        directory.mkdir(parents=True)
        (directory / "model").write_bytes(content)
    (bundles / "active-v1/old-only").write_bytes(b"must not remain in new active")

    class FakePredictor:
        def __init__(self, root, bundle_name="active-v1"):
            self._lock = threading.RLock()
            self.load_error = None
            self.recognizer_session = object()

        def _load_active_model(self):
            new = (bundles / "active-v1/model").read_bytes() == b"new"
            self.load_error = "reload failed" if new and fail_reload else None
            self.recognizer_session = None if self.load_error else object()

    monkeypatch.setattr(activation, "PredictorEngine", FakePredictor)
    predictor = FakePredictor(tmp_path)
    if fail_reload:
        with pytest.raises(RuntimeError):
            activation.activate_bundle(tmp_path, "candidate", predictor)
        assert (bundles / "active-v1/model").read_bytes() == b"old"
        assert predictor.load_error is None
    else:
        result = activation.activate_bundle(tmp_path, "candidate", predictor)
        assert (Path(result["backup_path"]) / "model").read_bytes() == b"old"
        assert (bundles / "active-v1/model").read_bytes() == b"new"
        assert not (bundles / "active-v1/old-only").exists()
    assert (bundles / "candidate/model").read_bytes() == b"new"


def test_invalid_candidate_never_touches_active(tmp_path):
    from plateai_web.activation import activate_bundle
    from plateai_web.predictor import PredictorEngine
    bundles = tmp_path / "models/bundles"
    (bundles / "active-v1").mkdir(parents=True)
    (bundles / "active-v1/keep").write_bytes(b"original")
    (bundles / "candidate").mkdir()
    (bundles / "candidate/manifest.json").write_text('{}')
    with pytest.raises(ValueError, match="驗證"):
        activate_bundle(tmp_path, "candidate", PredictorEngine(tmp_path))
    assert (bundles / "active-v1/keep").read_bytes() == b"original"
    assert not list(bundles.glob(".active-v1.partial-*"))


def test_benchmark_rounds_warmup_and_geometry_gate():
    from plateai_web.evaluator import measure
    calls = []
    class Adapter:
        def predict(self, image, corners, mode):
            calls.append(mode)
            # A matching string without a spatial match must remain a failure.
            return "ABC1234", False
    sample = (np.zeros((20, 40, 3), dtype=np.uint8), np.zeros((4, 2)), "ABC1234")
    result = measure(Adapter(), [sample, sample], "scene", rounds=3, warmup=2)
    assert len(calls) == 8
    assert result["sample_count"] == 2
    assert result["trials"] == 6
    assert result["success"] == "0/6"
    assert result["max_ms"] is not None


def test_missing_benchmark_data_fails_instead_of_empty_success(tmp_path, monkeypatch):
    from plateai_web import evaluator
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    tm = TaskManager()
    tid = tm.create_task("benchmark")
    evaluator.run_benchmark_task(tid, tm, target_dataset="user")
    assert tm.get_task(tid).status.value == "failed"
