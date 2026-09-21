"""Verify native detector ONNX parity, then atomically publish a full bundle."""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import onnx
import torch

from plateai_reader.detector import numpy_nms_v1
from plateai_shared.bundle import validate_crop_bundle, validate_model_bundle
from plateai_shared.detection import letterbox_rgb_v1
from plateai_shared.publication import OutputExistsError, publish_directory_no_replace, remove_owned_staging
from plateai_trainer.export.bundle import ExportParityError, OrtSession, create_cpu_session
from plateai_trainer.synthetic.cli import _default_config_path
from .model import PlatePoseNet


@dataclass(frozen=True, slots=True)
class DetectionExportRequest:
    recognizer_bundle: Path
    detector_checkpoint: Path
    detector_report: Path
    output: Path


_KEYPOINTS = ["left_top", "right_top", "right_bottom", "left_bottom"]
_POSTPROCESS = {
    "candidate_format": "cxcywh-confidence-corners-letterbox-px-v1",
    "preprocess": "opencv-rgb-letterbox-640-v1",
    "nms": "numpy-nms-v1", "score_threshold": 0.25,
    "iou_threshold": 0.50, "max_detections": 100,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_detector_checkpoint(path: Path) -> PlatePoseNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"schema_version": 1, "architecture": "PlatePoseNet",
                "input_shape": [3, 640, 640], "output_shape": [8400, 13],
                "preprocess": "letterbox_rgb_v1", "corner_order": _KEYPOINTS}
    if not isinstance(checkpoint, dict) or any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint is not compatible with the v1 detector contract")
    model = PlatePoseNet()
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError("checkpoint has incompatible detector weights") from exc
    model.eval()
    return model


def _export_detector_onnx(model: PlatePoseNet, path: Path) -> None:
    torch.onnx.export(
        model, (torch.zeros((2, 3, 640, 640), dtype=torch.float32),), path,
        input_names=["images"], output_names=["candidates"], opset_version=17,
        dynamo=True, external_data=False,
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=32)},),
    )
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("detector model exceeds 8 MiB")
    onnx.checker.check_model(str(path), full_check=True)
    graph = onnx.load(path, load_external_data=False)
    if next((item.version for item in graph.opset_import if item.domain == ""), None) != 17:
        raise ExportParityError("detector ONNX opset must equal 17")


def _assert_detector_onnx_parity(model: PlatePoseNet, session: OrtSession) -> None:
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or len(outputs) != 1
            or inputs[0].name != "images" or outputs[0].name != "candidates"
            or inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)"
            or list(inputs[0].shape) != ["batch", 3, 640, 640]
            or list(outputs[0].shape) != ["batch", 8400, 13]):
        raise ExportParityError("detector ONNX IO contract differs")
    rng = np.random.default_rng(42)
    images = [letterbox_rgb_v1(rng.integers(0, 256, shape, dtype=np.uint8))[0]
              for shape in ((241, 503, 3), (399, 251, 3))]
    for batch in (1, 2):
        values = np.stack(images[:batch])
        with torch.no_grad():
            native = model(torch.from_numpy(values)).numpy()
        exported = session.run(["candidates"], {"images": values})[0]
        if (native.shape != (batch, 8400, 13) or exported.shape != native.shape
                or exported.dtype != np.float32 or not np.isfinite(native).all()
                or not np.isfinite(exported).all()
                or not np.allclose(native, exported, rtol=1e-4, atol=1e-5)):
            raise ExportParityError(f"detector ONNX candidates differ at batch {batch}")
        for left, right in zip(native, exported, strict=True):
            if numpy_nms_v1(left, .25, .50, 100) != numpy_nms_v1(right, .25, .50, 100):
                raise ExportParityError(f"detector ONNX candidates NMS survivors differ at batch {batch}")


def export_full_bundle(
    request: DetectionExportRequest,
    session_factory: Callable[[Path], OrtSession] = create_cpu_session,
) -> Path:
    """Merge a validated crop bundle and matching detector run without clobbering."""
    if request.output.exists():
        raise OutputExistsError(f"output already exists: {request.output}")
    schema = _default_config_path("schemas/model_manifest.schema.json")
    crop_manifest = validate_crop_bundle(request.recognizer_bundle, schema)
    torch.set_num_threads(1)
    model = _load_detector_checkpoint(request.detector_checkpoint)
    report_bytes = request.detector_report.read_bytes()
    try:
        report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("detector report is not valid JSON") from exc
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("checkpoint_sha256") != _sha(request.detector_checkpoint)):
        raise ValueError("detector report does not match checkpoint")
    declarations = [crop_manifest["charset"], crop_manifest["rules"],
                    crop_manifest["components"]["recognizer"], crop_manifest["provenance"]["training_report"]]
    if {item["file"] for item in declarations} & {"detector.onnx", "detector_report.json"}:
        raise ValueError("crop bundle filenames conflict with detector files")
    request.output.parent.mkdir(parents=True, exist_ok=True)
    staging = request.output.parent / f".{request.output.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    try:
        for item in declarations:
            shutil.copyfile(request.recognizer_bundle / item["file"], staging / item["file"])
        detector_path = staging / "detector.onnx"
        _export_detector_onnx(model, detector_path)
        _assert_detector_onnx_parity(model, session_factory(detector_path))
        (staging / "detector_report.json").write_bytes(report_bytes)
        manifest = deepcopy(crop_manifest)
        manifest["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest["capabilities"] = ["crop-recognition", "plate-detection"]
        manifest["components"]["detector"] = {
            "file": "detector.onnx", "format": "onnx", "sha256": _sha(detector_path),
            "inputs": [{"name": "images", "dtype": "float32", "shape": ["batch", 3, 640, 640]}],
            "outputs": [{"name": "candidates", "dtype": "float32", "shape": ["batch", 8400, 13]}],
            "batch": {"mode": "dynamic", "min": 1, "opt": 8, "max": 32},
            "keypoints": list(_KEYPOINTS), "postprocess": dict(_POSTPROCESS),
        }
        manifest["rectifier"] = {"normalization_strategy": "convex-hull-semantic-v1"}
        manifest["provenance"]["detector_training_report"] = {
            "file": "detector_report.json", "sha256": _sha(staging / "detector_report.json"),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        validate_model_bundle(staging, schema)
        publish_directory_no_replace(staging, request.output)
    except BaseException:
        remove_owned_staging(staging, request.output)
        raise
    return request.output
