"""Validation for self-contained local v1 Model Bundles."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Mapping
from .contracts import JsonValue
from .rules import load_character_set
from .schema_validation import DocumentValidationError, validate_document, validate_model_manifest

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _reject_external_tensors(message) -> None:
    # Include nested graphs, tensor attributes, and function bodies.
    if message.DESCRIPTOR.full_name == "onnx.TensorProto":
        if message.data_location == 1 or message.external_data:
            raise DocumentValidationError("ONNX external data is not allowed")
    for field, value in message.ListFields():
        if field.message_type is not None:
            for child in value if field.is_repeated else (value,):
                _reject_external_tensors(child)


def _validate_onnx(path: Path, component: Mapping[str, JsonValue]) -> None:
    # Shared contract imports remain independent of optional ONNX dependencies.
    import onnx

    try:
        model = onnx.load(path, load_external_data=False)
        _reject_external_tensors(model)
        onnx.checker.check_model(model, full_check=True)
        for field in ("inputs", "outputs"):
            tensors = model.graph.input if field == "inputs" else model.graph.output
            actual = [
                {"name": value.name,
                 "dtype": "float32" if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT else "unsupported",
                 "shape": [dim.dim_param if dim.HasField("dim_param") else dim.dim_value
                           for dim in value.type.tensor_type.shape.dim]}
                for value in tensors
            ]
            if actual != component[field]:
                raise DocumentValidationError(f"{path.name}: ONNX {field} contract differs")
    except DocumentValidationError:
        raise
    except Exception as exc:
        raise DocumentValidationError(f"{path.name}: invalid ONNX model: {exc}") from exc


def _validate_bundle_files(bundle_dir: Path, schema_path: Path) -> Mapping[str, JsonValue]:
    root = Path(bundle_dir).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DocumentValidationError("manifest.json: invalid or missing") from exc
    # Validate declaration structure and filenames before using them on disk.
    validate_document(manifest, schema_path)
    declarations = [manifest["charset"], manifest["rules"],
                    manifest["components"]["recognizer"], manifest["provenance"]["training_report"]]
    if "detector" in manifest["components"]:
        declarations.append(manifest["components"]["detector"])
    if "detector_training_report" in manifest["provenance"]:
        declarations.append(manifest["provenance"]["detector_training_report"])
    names = [item["file"] for item in declarations]
    if len(set(names + ["manifest.json"])) != len(names) + 1:
        raise DocumentValidationError("bundle: file declarations must be distinct")
    for declaration in declarations:
        path = root / declaration["file"]
        if not path.is_file() or path.is_symlink() or path.resolve().parent != root or _sha(path) != declaration["sha256"]:
            raise DocumentValidationError(f"bundle file hash mismatch: {path.name}")
    charset = load_character_set(root / manifest["charset"]["file"])
    validate_model_manifest(manifest, schema_path, visible_charset_symbol_count=len(charset.symbols))
    expected = {"manifest.json", *names}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        raise DocumentValidationError("bundle: contains undeclared files")
    return manifest


def validate_model_bundle(bundle_dir: Path, schema_path: Path) -> Mapping[str, JsonValue]:
    """Validate all bundle files and ONNX graphs, including external-data checks."""
    manifest = _validate_bundle_files(bundle_dir, schema_path)
    root = Path(bundle_dir).resolve()
    for name, component in manifest["components"].items():
        path = root / component["file"]
        if name == "detector" and path.stat().st_size > 8 * 1024 * 1024:
            raise DocumentValidationError("detector model exceeds 8 MiB")
        if component["format"] == "onnx":
            _validate_onnx(path, component)
    return manifest


def validate_crop_bundle(bundle_dir: Path, schema_path: Path) -> Mapping[str, JsonValue]:
    """Retain M2 hash validation without requiring optional ONNX dependencies."""
    manifest = _validate_bundle_files(bundle_dir, schema_path)
    if manifest["capabilities"] != ["crop-recognition"] or "detector" in manifest["components"]:
        raise DocumentValidationError("bundle: expected a crop-only bundle")
    return manifest
