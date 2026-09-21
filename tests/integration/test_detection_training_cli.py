from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from plateai_shared.publication import OutputExistsError
from plateai_trainer.detection import engine
from plateai_trainer.detection.dataset import DetectionDataError
from plateai_trainer.detection.engine import DetectorTrainingConfig, train_detector
from plateai_trainer.detection.train_cli import main
from tests.unit.test_detection_dataset import make_dataset


def test_training_is_deterministic_publishes_reloadable_best_and_finite_report(tmp_path):
    train = make_dataset(tmp_path)
    validation = make_dataset(tmp_path, "validation", color=21, seed=8)
    config = DetectorTrainingConfig(train, validation, tmp_path / "run", epochs=2, batch_size=1, seed=11)
    run = train_detector(config)
    second = train_detector(replace(config, output_directory=tmp_path / "repeat"))
    assert run.best_checkpoint.is_file()
    assert json.loads(run.report_path.read_text(encoding="utf-8")) == run.report
    assert math.isfinite(run.report["train"]["loss"])
    assert run.report["overfit"]["last_loss"] < run.report["overfit"]["first_loss"]
    for field in ("train", "validation", "overfit", "input_hashes"):
        assert run.report[field] == second.report[field]
    metrics = run.report["validation"]
    for key in ("bbox_ap50", "complete_quad_precision", "complete_quad_recall", "rectifier_acceptance"):
        assert 0 <= metrics[key] <= 1
    assert metrics["corner_error_640px"] is None or math.isfinite(metrics["corner_error_640px"])
    assert sum(metrics["strata"]["instance_count"].values()) == 1
    assert run.report["input_hashes"]["train"]["files"]["images/000000.png"]
    assert run.report["runtime_provenance"]["dependency_locks"]["py311.training.lock.txt"]
    assert run.report["runtime_provenance"]["source_sha256"]["model"]
    checkpoint = torch.load(run.best_checkpoint, weights_only=True)
    other = torch.load(second.best_checkpoint, weights_only=True)
    for key, value in checkpoint["model_state_dict"].items():
        assert torch.equal(value, other["model_state_dict"][key])
    model = engine.PlatePoseNet().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    assert checkpoint["epoch"] == run.report["best_epoch"]
    assert torch.get_num_threads() == 1
    with pytest.raises(OutputExistsError):
        train_detector(config)


def test_evaluation_metrics_score_perfect_duplicate_and_bad_semantic_quads():
    from plateai_trainer.detection.contracts import CompositeInstance
    corners = np.float32([[100, 100], [200, 100], [200, 140], [100, 140]])
    gt = [CompositeInstance((100, 100, 200, 140), corners, {})]
    good = np.r_[150, 120, 100, 40, .9, corners.ravel()].astype(np.float32)
    metrics = engine._prediction_metrics([np.stack([good, good])], [gt])
    assert metrics["bbox_ap50"] == 1
    assert metrics["complete_quad_precision"] == 1
    assert metrics["complete_quad_recall"] == 1
    assert metrics["corner_error_640px"] == 0
    assert metrics["rectifier_acceptance"] == 1
    assert metrics["strata"]["projected_shortest_edge_640px"] == {"lt16": 0, "16to31": 0, "32to63": 1, "ge64": 0}
    bad = good.copy()
    bad[5:] = corners[::-1].ravel()
    metrics = engine._prediction_metrics([bad[None]], [gt])
    assert metrics["bbox_ap50"] == 1
    assert metrics["complete_quad_recall"] == 0
    assert metrics["corner_error_640px"] == 40


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"batch_size": True}, {"learning_rate": float("nan")}, {"seed": -1}, {"device": "cuda"}])
def test_invalid_config_creates_no_output(tmp_path, kwargs):
    config = DetectorTrainingConfig(tmp_path / "missing", tmp_path / "missing", tmp_path / "run")
    with pytest.raises(DetectionDataError):
        train_detector(replace(config, **kwargs))
    assert not config.output_directory.exists()


def test_training_failure_removes_owned_staging(tmp_path, monkeypatch):
    train = make_dataset(tmp_path)
    validation = make_dataset(tmp_path, "validation", color=21)
    def fail(*args, **kwargs):
        raise RuntimeError("injected save failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(RuntimeError, match="injected"):
        train_detector(DetectorTrainingConfig(train, validation, tmp_path / "run", epochs=1))
    assert not (tmp_path / "run").exists()
    assert list(tmp_path.glob(".run.partial-*")) == []


def test_cli_errors_and_installed_help(tmp_path):
    assert main(["--train", str(tmp_path), "--validation", str(tmp_path), "--output", str(tmp_path / "run")]) == 2
    executable = Path(sys.executable).with_name("plateai-detect-train.exe" if sys.platform == "win32" else "plateai-detect-train")
    result = subprocess.run([str(executable), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--train" in result.stdout


def test_score_ranked_ap_counts_false_positives_and_empty_matches():
    from plateai_trainer.detection.contracts import CompositeInstance
    corners = np.float32([[100, 100], [200, 100], [200, 140], [100, 140]])
    gt = [CompositeInstance((100, 100, 200, 140), corners, {})]
    good = np.r_[150, 120, 100, 40, .8, corners.ravel()].astype(np.float32)
    false = good.copy()
    false[:2] += 300
    false[5:] += 300
    false[4] = .9
    metrics = engine._prediction_metrics([np.stack([false, good])], [gt])
    assert metrics["bbox_ap50"] == .5
    assert metrics["complete_quad_precision"] == .5
    assert metrics["complete_quad_recall"] == 1
    empty = engine._prediction_metrics([np.empty((0, 13), dtype=np.float32)], [gt])
    assert empty["bbox_ap50"] == 0
    assert empty["corner_error_640px"] is None


def test_per_size_stratum_results_use_global_matches_and_count_false_positives():
    from plateai_trainer.detection.contracts import CompositeInstance
    small = np.float32([[30, 30], [90, 30], [90, 54], [30, 54]])
    medium = np.float32([[140, 100], [240, 100], [240, 140], [140, 140]])
    large = np.float32([[300, 200], [500, 200], [500, 280], [300, 280]])
    truths = [CompositeInstance((30, 30, 90, 54), small, {}),
              CompositeInstance((140, 100, 240, 140), medium, {}),
              CompositeInstance((300, 200, 500, 280), large, {})]
    good_small = np.r_[60, 42, 60, 24, .8, small.ravel()]
    # Perfect medium bbox but corners predict a large, valid quad. Its result
    # belongs to the matched medium GT, never to the predicted large stratum.
    bad_medium = np.r_[190, 120, 100, 40, .7, (medium + [[0, 0], [0, 0], [0, 40], [0, 40]]).ravel()]
    # High-score false positive with a small edge; M3a rejects its off-frame quad.
    false_small = np.r_[20, 200, 60, 24, .9, (small - [40, 0]).ravel()]
    rows = np.float32([false_small, good_small, good_small, bad_medium])
    metrics = engine._prediction_metrics([rows], [truths])
    sizes = metrics["stratum_results"]["projected_shortest_edge_640px"]
    assert sizes["16to31"] == {
        "instances": 1, "bbox_ap50": .5, "corner_error_640px": 0., "corner_matched_instances": 1,
        "complete_quad_precision": .5, "complete_quad_recall": 1., "complete_quad_true_positives": 1,
        "nms_predictions": 2, "rectifier_acceptance": .5, "rectifier_accepted": 1,
    }
    assert sizes["32to63"]["bbox_ap50"] == 1
    assert sizes["32to63"]["corner_error_640px"] == 20
    assert sizes["32to63"]["complete_quad_precision"] == 0
    assert sizes["32to63"]["complete_quad_recall"] == 0
    assert sizes["32to63"]["rectifier_acceptance"] == 1
    assert sizes["ge64"]["instances"] == 1
    assert sizes["ge64"]["bbox_ap50"] == 0
    assert sizes["ge64"]["corner_error_640px"] is None
    assert sizes["ge64"]["nms_predictions"] == 0
    assert sizes["lt16"]["instances"] == 0
    assert sizes["lt16"]["bbox_ap50"] == 0
    assert sizes["lt16"]["corner_error_640px"] is None
    assert sum(part["nms_predictions"] for part in sizes.values()) == metrics["nms_predictions"]


@pytest.mark.parametrize("shortest,stratum", [(15, "lt16"), (16, "16to31"), (31.5, "16to31"), (32, "32to63"), (63.5, "32to63"), (64, "ge64")])
def test_per_size_stratum_boundary_and_no_operating_predictions(shortest, stratum):
    from plateai_trainer.detection.contracts import CompositeInstance
    corners = np.float32([[100, 100], [300, 100], [300, 100 + shortest], [100, 100 + shortest]])
    gt = CompositeInstance((100, 100, 300, 100 + shortest), corners, {})
    row = np.float32([200, 100 + shortest / 2, 200, shortest, .1, *corners.ravel()])
    metrics = engine._prediction_metrics([row[None]], [[gt]])
    result = metrics["stratum_results"]["projected_shortest_edge_640px"][stratum]
    assert result["instances"] == 1
    assert result["bbox_ap50"] == 1
    assert result["corner_error_640px"] is None
    assert result["complete_quad_precision"] == result["complete_quad_recall"] == 0
    assert result["rectifier_acceptance"] == result["nms_predictions"] == 0


def test_publication_race_preserves_winner_and_removes_only_staging(tmp_path, monkeypatch):
    train = make_dataset(tmp_path)
    validation = make_dataset(tmp_path, "validation", color=21)
    real_publish = engine.publish_directory_no_replace
    def race(staging, output):
        output.mkdir()
        (output / "winner.txt").write_text("keep", encoding="utf-8")
        real_publish(staging, output)
    monkeypatch.setattr(engine, "publish_directory_no_replace", race)
    with pytest.raises(OutputExistsError):
        train_detector(DetectorTrainingConfig(train, validation, tmp_path / "run", epochs=1))
    assert (tmp_path / "run/winner.txt").read_text(encoding="utf-8") == "keep"
    assert list(tmp_path.glob(".run.partial-*")) == []


def test_metadata_change_during_training_is_not_published(tmp_path, monkeypatch):
    train = make_dataset(tmp_path)
    validation = make_dataset(tmp_path, "validation", color=21)
    real_save = torch.save
    def mutate(*args, **kwargs):
        real_save(*args, **kwargs)
        (train / "metadata.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(torch, "save", mutate)
    with pytest.raises(DetectionDataError, match="SHA-256"):
        train_detector(DetectorTrainingConfig(train, validation, tmp_path / "run", epochs=1))
    assert not (tmp_path / "run").exists()
    assert list(tmp_path.glob(".run.partial-*")) == []
