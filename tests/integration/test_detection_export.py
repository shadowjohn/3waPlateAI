from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import onnx
import pytest
import torch

from plateai_reader.detector import numpy_nms_v1
from plateai_shared import bundle as bundle_validation
from plateai_shared.detection import letterbox_rgb_v1
from plateai_shared.publication import OutputExistsError
from plateai_shared.schema_validation import DocumentValidationError
from plateai_trainer.detection.engine import DetectorTrainingConfig, train_detector
from plateai_trainer.detection.model import PlatePoseNet
from plateai_trainer.export.bundle import ExportParityError, ExportRequest, create_cpu_session, export_crop_bundle
from plateai_trainer.training.engine import TrainingConfig, train_recognizer
from tests.conftest import ROOT, _generate_v1_dataset
from tests.unit.test_detection_dataset import make_dataset


SCHEMA = ROOT / "schemas/model_manifest.schema.json"


@pytest.fixture(scope="module")
def export_api():
    # Assert the missing public boundary as a test failure during the RED run.
    assert hasattr(bundle_validation, "validate_model_bundle"), "full bundle validator is missing"
    return importlib.import_module("plateai_trainer.detection.export")


@pytest.fixture(scope="module")
def local_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("detection-export")
    crop_run = train_recognizer(TrainingConfig(
        _generate_v1_dataset(root, "crop-train", count=2, seed=10),
        _generate_v1_dataset(root, "crop-validation", count=2, seed=20),
        root / "crop-run", epochs=1, batch_size=2,
    ))
    crop = export_crop_bundle(ExportRequest(crop_run.best_checkpoint, crop_run.report_path, root / "crop"))
    detector_run = train_detector(DetectorTrainingConfig(
        make_dataset(root, "detector-train"),
        make_dataset(root, "detector-validation", color=21, seed=8),
        # A one-step network has near-flat scores at float32 rounding precision;
        # use a learned score distribution for exact native/ORT NMS parity.
        root / "detector-run", epochs=40, batch_size=1, seed=11,
    ))
    return crop, detector_run


@pytest.fixture(scope="module")
def full_bundle(export_api, local_runs, tmp_path_factory):
    crop, run = local_runs
    return export_api.export_full_bundle(export_api.DetectionExportRequest(
        crop, run.best_checkpoint, run.report_path, tmp_path_factory.mktemp("full-export") / "full",
    ))


def _copy_bundle(full_bundle, tmp_path):
    return Path(shutil.copytree(full_bundle, tmp_path / "full"))


def _update_hash(bundle, section, name):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    declaration = manifest[section][name]
    declaration["sha256"] = hashlib.sha256((bundle / declaration["file"]).read_bytes()).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_detector_export_merges_valid_crop_bundle_and_matches_onnx(full_bundle, local_runs):
    manifest = bundle_validation.validate_model_bundle(full_bundle, SCHEMA)
    assert manifest["components"]["detector"]["outputs"][0]["shape"] == ["batch", 8400, 13]
    assert manifest["capabilities"] == ["crop-recognition", "plate-detection"]
    crop, run = local_runs
    for name in ("recognizer.onnx", "charset.txt", "plate_rules.json", "report.json"):
        assert (full_bundle / name).read_bytes() == (crop / name).read_bytes()
    assert (full_bundle / "detector_report.json").read_bytes() == run.report_path.read_bytes()
    model = onnx.load(full_bundle / "detector.onnx", load_external_data=False)
    assert next(item.version for item in model.opset_import if item.domain == "") == 17
    assert (full_bundle / "detector.onnx").stat().st_size <= 8 * 1024 * 1024
    assert bundle_validation.validate_model_bundle(crop, SCHEMA) == bundle_validation.validate_crop_bundle(crop, SCHEMA)
    with pytest.raises(DocumentValidationError, match="crop-only"):
        bundle_validation.validate_crop_bundle(full_bundle, SCHEMA)


def test_native_and_onnx_candidates_keep_the_same_nms_survivors(full_bundle, local_runs):
    model = PlatePoseNet().eval()
    model.load_state_dict(torch.load(local_runs[1].best_checkpoint, weights_only=True)["model_state_dict"])
    session = create_cpu_session(full_bundle / "detector.onnx")
    assert session.get_inputs()[0].shape == ["batch", 3, 640, 640]
    assert session.get_outputs()[0].shape == ["batch", 8400, 13]
    rng = np.random.default_rng(42)
    images = [letterbox_rgb_v1(rng.integers(0, 256, (241, 503, 3), dtype=np.uint8))[0],
              letterbox_rgb_v1(rng.integers(0, 256, (399, 251, 3), dtype=np.uint8))[0]]
    for batch in (1, 2):
        values = np.stack(images[:batch])
        with torch.no_grad():
            native = model(torch.from_numpy(values)).numpy()
        exported = session.run(["candidates"], {"images": values})[0]
        np.testing.assert_allclose(native, exported, rtol=1e-4, atol=1e-5)
        for left, right in zip(native, exported, strict=True):
            retained = numpy_nms_v1(left, .25, .50, 100)
            assert retained, "the fixture must exercise non-empty NMS"
            assert retained == numpy_nms_v1(right, .25, .50, 100)


@pytest.mark.parametrize("filename", ["extra.onnx", "nested/extra.onnx"])
def test_full_bundle_rejects_every_undeclared_regular_file(full_bundle, tmp_path, filename):
    bundle = _copy_bundle(full_bundle, tmp_path)
    extra = bundle / filename
    extra.parent.mkdir(exist_ok=True)
    extra.write_bytes(b"unexpected")
    with pytest.raises(DocumentValidationError, match="undeclared"):
        bundle_validation.validate_model_bundle(bundle, SCHEMA)


@pytest.mark.parametrize("filename", ["recognizer.onnx", "report.json", "detector.onnx", "detector_report.json", "charset.txt", "plate_rules.json"])
def test_full_bundle_checks_every_declared_hash(full_bundle, tmp_path, filename):
    bundle = _copy_bundle(full_bundle, tmp_path)
    with (bundle / filename).open("ab") as target:
        target.write(b"tampered")
    with pytest.raises(DocumentValidationError, match="hash mismatch"):
        bundle_validation.validate_model_bundle(bundle, SCHEMA)


@pytest.mark.parametrize("component", ["recognizer", "detector"])
def test_full_bundle_rejects_external_onnx_data_even_with_correct_hash(full_bundle, tmp_path, component):
    bundle = _copy_bundle(full_bundle, tmp_path)
    path = bundle / f"{component}.onnx"
    model = onnx.load(path)
    tensor = model.graph.initializer[0]
    onnx.external_data_helper.set_external_data(tensor, "../unread-external.bin")
    tensor.ClearField("raw_data")
    path.write_bytes(model.SerializeToString())
    _update_hash(bundle, "components", component)
    with pytest.raises(DocumentValidationError, match="external"):
        bundle_validation.validate_model_bundle(bundle, SCHEMA)


def test_full_bundle_rejects_detector_over_eight_mib(full_bundle, tmp_path):
    bundle = _copy_bundle(full_bundle, tmp_path)
    (bundle / "detector.onnx").write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    _update_hash(bundle, "components", "detector")
    with pytest.raises(DocumentValidationError, match="8 MiB"):
        bundle_validation.validate_model_bundle(bundle, SCHEMA)


def test_export_refuses_candidate_parity_failure_without_publishing(export_api, local_runs, tmp_path):
    seen_batches = []
    class PerturbedSession:
        def __init__(self, path):
            self.session = create_cpu_session(path)
        def get_inputs(self):
            return self.session.get_inputs()
        def get_outputs(self):
            return self.session.get_outputs()
        def run(self, names, feed):
            result = self.session.run(names, feed)
            size = feed["images"].shape[0]
            seen_batches.append(size)
            if size == 2:
                result[0][0, 0, 5] += 1.0
            return result
    crop, run = local_runs
    request = export_api.DetectionExportRequest(crop, run.best_checkpoint, run.report_path, tmp_path / "bad")
    with pytest.raises(ExportParityError, match="candidates"):
        export_api.export_full_bundle(request, session_factory=PerturbedSession)
    assert seen_batches == [1, 2]
    assert not request.output.exists()
    assert list(tmp_path.glob(".bad.partial-*")) == []


def test_export_preserves_existing_or_raced_output(export_api, local_runs, tmp_path, monkeypatch):
    crop, run = local_runs
    output = tmp_path / "winner"
    request = export_api.DetectionExportRequest(crop, run.best_checkpoint, run.report_path, output)
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(OutputExistsError):
        export_api.export_full_bundle(request)
    request = replace(request, output=tmp_path / "race")
    publish = export_api.publish_directory_no_replace
    def race(staging, target):
        target.mkdir()
        (target / "keep.txt").write_text("raced", encoding="utf-8")
        publish(staging, target)
    monkeypatch.setattr(export_api, "publish_directory_no_replace", race)
    with pytest.raises(OutputExistsError):
        export_api.export_full_bundle(request)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (request.output / "keep.txt").read_text(encoding="utf-8") == "raced"
    assert list(tmp_path.glob(".race.partial-*")) == []


def test_export_rejects_incompatible_checkpoint_and_unrelated_report(export_api, local_runs, tmp_path):
    crop, run = local_runs
    checkpoint = torch.load(run.best_checkpoint, weights_only=True)
    checkpoint["preprocess"] = "other"
    bad_checkpoint = tmp_path / "bad.pt"
    torch.save(checkpoint, bad_checkpoint)
    request = export_api.DetectionExportRequest(crop, bad_checkpoint, run.report_path, tmp_path / "bad")
    with pytest.raises(ValueError, match="checkpoint"):
        export_api.export_full_bundle(request)
    report = tmp_path / "report.json"
    report.write_text('{"schema_version":1,"checkpoint_sha256":"wrong"}', encoding="utf-8")
    with pytest.raises(ValueError, match="report"):
        export_api.export_full_bundle(replace(request, detector_checkpoint=run.best_checkpoint, detector_report=report))
    assert not request.output.exists()


def test_installed_detection_export_help_and_error(export_api, tmp_path):
    executable = Path(sys.executable).with_name("plateai-detect-export.exe" if sys.platform == "win32" else "plateai-detect-export")
    result = subprocess.run([str(executable), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--recognizer-bundle" in result.stdout
    cli = importlib.import_module("plateai_trainer.detection.export_cli")
    assert cli.main(["--recognizer-bundle", str(tmp_path), "--checkpoint", "missing.pt", "--report", "missing.json", "--output", str(tmp_path / "full")]) == 2


def test_export_load_and_report_hash_share_snapshot_during_checkpoint_aba(export_api, local_runs, tmp_path, monkeypatch):
    crop, run = local_runs
    checkpoint_path = tmp_path / "checkpoint.pt"
    original = run.best_checkpoint.read_bytes()
    checkpoint_path.write_bytes(original)
    real_load = torch.load
    expected = real_load(io.BytesIO(original), weights_only=True)
    mutant = real_load(io.BytesIO(original), weights_only=True)
    first_weight = next(iter(mutant["model_state_dict"]))
    mutant["model_state_dict"][first_weight] += .01
    stream = io.BytesIO()
    torch.save(mutant, stream)

    def aba_load(source, *args, **kwargs):
        checkpoint_path.write_bytes(stream.getvalue())
        try:
            return real_load(source, *args, **kwargs)
        finally:
            checkpoint_path.write_bytes(original)

    real_export = export_api._export_detector_onnx
    def export_original_snapshot(model, path):
        assert torch.equal(model.state_dict()[first_weight], expected["model_state_dict"][first_weight]), "export loaded B although report hashes A"
        real_export(model, path)

    monkeypatch.setattr(torch, "load", aba_load)
    monkeypatch.setattr(export_api, "_export_detector_onnx", export_original_snapshot)
    request = export_api.DetectionExportRequest(crop, checkpoint_path, run.report_path, tmp_path / "aba")
    output = export_api.export_full_bundle(request)
    manifest = bundle_validation.validate_model_bundle(output, SCHEMA)
    assert manifest["capabilities"] == ["crop-recognition", "plate-detection"]
    assert (output / "detector_report.json").read_bytes() == run.report_path.read_bytes()
    assert checkpoint_path.read_bytes() == original
    assert not list(tmp_path.glob(".aba.partial-*"))
