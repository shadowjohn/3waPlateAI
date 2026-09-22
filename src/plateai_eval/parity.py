"""Native checkpoint to deployed ONNX parity on a frozen real-scene subset."""

from __future__ import annotations

import hashlib
import json
import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from plateai_reader import ReaderError, decode_constrained_ctc_v1
from plateai_reader.detector import numpy_nms_v1, postprocess_candidates
from plateai_reader.rectifier import InvalidCornersError, normalize_corners, rectify_plate
from plateai_shared.bundle import validate_model_bundle
from plateai_shared.contracts import JsonValue
from plateai_shared.detection import LetterboxTransform, letterbox_rgb_v1
from plateai_shared.recognition import CTCCodec, preprocess_v1_rgb
from plateai_shared.rules import load_character_set, load_ruleset
from plateai_shared.schema_validation import DocumentValidationError
from plateai_trainer.detection.export import _load_detector_checkpoint
from plateai_trainer.export.bundle import _load_recognizer_checkpoint

from .contracts import (
    EvaluationInputError,
    EvaluationRuntimeError,
    ParityResult,
    Scene,
    StrictProfile,
    SuiteSnapshot,
)
from .suite import load_scene_rgb


def _default_model_schema() -> Path:
    checkout = Path(__file__).resolve().parents[2] / "schemas/model_manifest.schema.json"
    if checkout.is_file():
        return checkout
    return (
        Path(sysconfig.get_path("data"))
        / "share/3wa-plate-ai/schemas/model_manifest.schema.json"
    )


def _read_regular(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise EvaluationInputError(f"{label}: missing regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise EvaluationInputError(f"{label}: cannot read file") from exc


def _reject_duplicate_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise EvaluationInputError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _read_report(
    bundle: Path,
    declaration: Mapping[str, JsonValue],
    label: str,
) -> Mapping[str, JsonValue]:
    name = declaration.get("file")
    expected = declaration.get("sha256")
    if not isinstance(name, str) or Path(name).name != name:
        raise EvaluationInputError(f"{label}: unsafe report path")
    path = bundle / name
    payload = _read_regular(path, label)
    if not isinstance(expected, str) or hashlib.sha256(payload).hexdigest() != expected:
        raise EvaluationInputError(f"{label}: bundle report hash mismatch")
    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except EvaluationInputError as exc:
        raise EvaluationInputError(f"{label}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"{label}: invalid JSON") from exc
    if not isinstance(document, dict):
        raise EvaluationInputError(f"{label}: report must be an object")
    return document


def _create_session(path: Path, provider: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise EvaluationRuntimeError("ONNX Runtime is unavailable") from exc
    available = tuple(ort.get_available_providers())
    if provider not in available:
        raise EvaluationRuntimeError(f"{provider} unavailable")
    try:
        session = ort.InferenceSession(str(path), providers=[provider])
    except Exception as exc:
        raise EvaluationRuntimeError(
            f"failed to create {path.name} session on {provider}: {exc}"
        ) from exc
    actual = tuple(session.get_providers())
    if not actual or actual[0] != provider:
        rendered = actual[0] if actual else "unknown"
        raise EvaluationRuntimeError(
            f"provider substitution: requested {provider}, actual {rendered}"
        )
    return session


def _actual_provider(session, requested: str) -> None:
    getter = getattr(session, "get_providers", None)
    actual = tuple(getter()) if callable(getter) else ()
    if not actual or actual[0] != requested:
        rendered = actual[0] if actual else "unknown"
        raise EvaluationRuntimeError(
            f"provider substitution: requested {requested}, actual {rendered}"
        )


def _device_for_provider(provider: str) -> torch.device:
    if provider == "CPUExecutionProvider":
        return torch.device("cpu")
    if provider not in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
        raise EvaluationRuntimeError(f"unsupported evaluation provider: {provider}")
    if not torch.cuda.is_available():
        raise EvaluationRuntimeError(
            f"{provider} requires an available PyTorch CUDA device for native parity"
        )
    return torch.device("cuda")


def _failure_result(
    failures: Sequence[str],
    profile: StrictProfile,
    recognizer_sha256: str,
    detector_sha256: str,
    provider: str,
    device: str,
    *,
    diagnostic_count: int = 0,
) -> ParityResult:
    return ParityResult(
        passed=False,
        failures=tuple(failures),
        batch_sizes=tuple(int(value) for value in profile.parity["batch_sizes"]),
        diagnostic_raw_difference_count=diagnostic_count,
        recognizer_checkpoint_sha256=recognizer_sha256,
        detector_checkpoint_sha256=detector_sha256,
        provider=provider,
        pytorch_device=device,
    )


def _numpy_output(value: object, label: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationRuntimeError(f"{label}: output is not numeric") from exc
    return array


def _run_native(model, values: np.ndarray, device: torch.device, label: str) -> np.ndarray:
    try:
        tensor = torch.from_numpy(values).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            output = model(tensor)
        return _numpy_output(output.detach().cpu().numpy(), label)
    except EvaluationRuntimeError:
        raise
    except Exception as exc:
        raise EvaluationRuntimeError(f"{label}: native inference failed: {exc}") from exc


def _run_onnx(session, output: str, input_name: str, values: np.ndarray, label: str) -> np.ndarray:
    try:
        outputs = session.run([output], {input_name: values})
    except Exception as exc:
        raise EvaluationRuntimeError(f"{label}: ONNX inference failed: {exc}") from exc
    if len(outputs) != 1:
        raise EvaluationRuntimeError(f"{label}: ONNX returned the wrong output count")
    return _numpy_output(outputs[0], label)


def _valid_output_pair(
    native: np.ndarray,
    exported: np.ndarray,
    expected_shape: tuple[int, ...],
    label: str,
    failures: list[str],
) -> bool:
    valid = True
    if native.shape != expected_shape or exported.shape != expected_shape:
        failures.append(
            f"{label} shape differs: native={list(native.shape)} onnx={list(exported.shape)}"
        )
        valid = False
    if native.dtype != np.float32 or exported.dtype != np.float32:
        failures.append(
            f"{label} dtype differs: native={native.dtype} onnx={exported.dtype}"
        )
        valid = False
    if not np.isfinite(native).all() or not np.isfinite(exported).all():
        failures.append(f"{label} output is non-finite")
        valid = False
    return valid


def _acceptance_signature(
    rows: np.ndarray,
    transform: LetterboxTransform,
    postprocess: Mapping[str, JsonValue],
) -> tuple[str, ...]:
    detections = postprocess_candidates(rows, transform, postprocess)
    signature: list[str] = []
    for detection in detections:
        try:
            normalize_corners(detection.corners_xy, transform.source_size_wh)
        except InvalidCornersError as exc:
            signature.append(f"rejected:{exc.reason}")
        else:
            signature.append("accepted")
    return tuple(signature)


def _compare_detector(
    native_model,
    session,
    tensors: Sequence[np.ndarray],
    transforms: Sequence[LetterboxTransform],
    scenes: Sequence[Scene],
    profile: StrictProfile,
    postprocess: Mapping[str, JsonValue],
    device: torch.device,
    failures: list[str],
) -> int:
    rtol = float(profile.parity["rtol"])
    atol = float(profile.parity["atol"])
    score = float(postprocess["score_threshold"])
    overlap = float(postprocess["iou_threshold"])
    maximum = int(postprocess["max_detections"])
    diagnostic_rows: set[tuple[str, int]] = set()
    for batch_size in tuple(int(value) for value in profile.parity["batch_sizes"]):
        values = np.stack(tensors[:batch_size]).astype(np.float32, copy=False)
        native = _run_native(native_model, values, device, f"detector batch {batch_size}")
        exported = _run_onnx(
            session,
            "candidates",
            "images",
            values,
            f"detector batch {batch_size}",
        )
        if not _valid_output_pair(
            native,
            exported,
            (batch_size, 8400, 13),
            f"detector batch {batch_size}",
            failures,
        ):
            continue
        for item_index, (native_rows, exported_rows) in enumerate(
            zip(native, exported, strict=True)
        ):
            native_retained = numpy_nms_v1(native_rows, score, overlap, maximum)
            exported_retained = numpy_nms_v1(exported_rows, score, overlap, maximum)
            close_rows = np.all(
                np.isclose(native_rows, exported_rows, rtol=rtol, atol=atol), axis=1
            )
            below_threshold = (
                (native_rows[:, 4] < score) & (exported_rows[:, 4] < score)
            )
            for row_index in np.flatnonzero(~close_rows & below_threshold):
                diagnostic_rows.add((scenes[item_index].scene_id, int(row_index)))
            if native_retained != exported_retained:
                failures.append(
                    f"detector retained indices differ at batch {batch_size} item {item_index}"
                )
                continue
            if native_retained and not np.allclose(
                native_rows[native_retained],
                exported_rows[exported_retained],
                rtol=rtol,
                atol=atol,
            ):
                failures.append(
                    f"detector retained rows differ at batch {batch_size} item {item_index}"
                )
            try:
                native_signature = _acceptance_signature(
                    native_rows, transforms[item_index], postprocess
                )
                exported_signature = _acceptance_signature(
                    exported_rows, transforms[item_index], postprocess
                )
            except Exception as exc:
                failures.append(
                    f"detector rectifier parity failed at batch {batch_size} item {item_index}: {exc}"
                )
            else:
                if native_signature != exported_signature:
                    failures.append(
                        f"detector rectifier acceptance differs at batch {batch_size} item {item_index}"
                    )
    return len(diagnostic_rows)


def _decoded_identity(logits: np.ndarray, codec: CTCCodec, ruleset) -> tuple[str, str, str, str]:
    greedy = codec.decode_greedy(logits.argmax(axis=1).tolist())
    decoded = decode_constrained_ctc_v1(logits, codec, ruleset)
    return greedy, decoded.canonical, decoded.display, decoded.rule_id


def _compare_recognizer(
    native_model,
    session,
    tensors: Sequence[np.ndarray],
    profile: StrictProfile,
    codec: CTCCodec,
    ruleset,
    device: torch.device,
    failures: list[str],
) -> None:
    rtol = float(profile.parity["rtol"])
    atol = float(profile.parity["atol"])
    for batch_size in tuple(int(value) for value in profile.parity["batch_sizes"]):
        values = np.stack(tensors[:batch_size]).astype(np.float32, copy=False)
        native = _run_native(
            native_model, values, device, f"recognizer batch {batch_size}"
        )
        exported = _run_onnx(
            session,
            "logits",
            "input",
            values,
            f"recognizer batch {batch_size}",
        )
        expected_shape = (batch_size, 80, codec.class_count)
        if not _valid_output_pair(
            native,
            exported,
            expected_shape,
            f"recognizer batch {batch_size}",
            failures,
        ):
            continue
        if not np.allclose(native, exported, rtol=rtol, atol=atol):
            failures.append(f"recognizer logits differ at batch {batch_size}")
        for item_index, (native_logits, exported_logits) in enumerate(
            zip(native, exported, strict=True)
        ):
            try:
                native_identity = _decoded_identity(native_logits, codec, ruleset)
                exported_identity = _decoded_identity(exported_logits, codec, ruleset)
            except ReaderError as exc:
                failures.append(
                    f"recognizer decode failed at batch {batch_size} item {item_index}: {exc}"
                )
                continue
            if native_identity != exported_identity:
                failures.append(
                    f"recognizer decode differs at batch {batch_size} item {item_index}"
                )


def run_parity(
    snapshot: SuiteSnapshot,
    profile: StrictProfile,
    bundle_dir: Path,
    recognizer_checkpoint: Path,
    detector_checkpoint: Path,
    provider: str = "CPUExecutionProvider",
    schema_path: Path | None = None,
) -> ParityResult:
    """Re-run native and ONNX models over the frozen parity subset."""

    bundle = Path(bundle_dir).resolve()
    try:
        manifest = validate_model_bundle(bundle, schema_path or _default_model_schema())
    except DocumentValidationError as exc:
        raise EvaluationInputError(f"bundle validation failed: {exc}") from exc
    if manifest.get("capabilities") != ["crop-recognition", "plate-detection"]:
        raise EvaluationInputError("parity requires a full Model Bundle")

    recognizer_bytes = _read_regular(
        Path(recognizer_checkpoint), "recognizer checkpoint"
    )
    detector_bytes = _read_regular(Path(detector_checkpoint), "detector checkpoint")
    recognizer_sha256 = hashlib.sha256(recognizer_bytes).hexdigest()
    detector_sha256 = hashlib.sha256(detector_bytes).hexdigest()
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise EvaluationInputError("bundle provenance is missing")
    recognizer_declaration = provenance.get("training_report")
    detector_declaration = provenance.get("detector_training_report")
    if not isinstance(recognizer_declaration, Mapping) or not isinstance(
        detector_declaration, Mapping
    ):
        raise EvaluationInputError("bundle native training report identity is missing")
    recognizer_report = _read_report(
        bundle, recognizer_declaration, "recognizer training report"
    )
    detector_report = _read_report(
        bundle, detector_declaration, "detector training report"
    )
    device = _device_for_provider(provider)
    failures: list[str] = []
    if recognizer_report.get("schema_version") != 1:
        failures.append("recognizer report schema identity is missing")
    if detector_report.get("schema_version") != 1:
        failures.append("detector report schema identity is missing")
    if recognizer_report.get("checkpoint_sha256") != recognizer_sha256:
        failures.append("recognizer checkpoint hash does not match bundle report")
    if detector_report.get("checkpoint_sha256") != detector_sha256:
        failures.append("detector checkpoint hash does not match bundle report")
    batch_sizes = tuple(int(value) for value in profile.parity["batch_sizes"])
    parity_scene_ids = snapshot.lane_scene_ids.get("parity", ())
    needed_scenes = int(profile.minimums["parity_scenes"])
    if len(parity_scene_ids) < needed_scenes:
        failures.append(
            f"parity scene population is {len(parity_scene_ids)}, requires {needed_scenes}"
        )
    if failures:
        return _failure_result(
            failures,
            profile,
            recognizer_sha256,
            detector_sha256,
            provider,
            str(device),
        )

    by_id = {scene.scene_id: scene for scene in snapshot.scenes}
    selected_scenes = tuple(
        by_id[scene_id] for scene_id in parity_scene_ids[:needed_scenes]
    )
    if max(batch_sizes, default=0) > len(selected_scenes):
        return _failure_result(
            ["parity subset does not contain every profile batch size"],
            profile,
            recognizer_sha256,
            detector_sha256,
            provider,
            str(device),
        )

    charset_declaration = manifest["charset"]
    rules_declaration = manifest["rules"]
    assert isinstance(charset_declaration, Mapping)
    assert isinstance(rules_declaration, Mapping)
    try:
        charset = load_character_set(bundle / str(charset_declaration["file"]))
        ruleset = load_ruleset(bundle / str(rules_declaration["file"]), charset)
    except Exception as exc:
        raise EvaluationInputError(f"bundle recognition contract is unreadable: {exc}") from exc
    codec = CTCCodec.from_charset(charset)
    try:
        recognizer_model = _load_recognizer_checkpoint(
            recognizer_bytes, charset, str(rules_declaration["sha256"])
        )
    except ValueError as exc:
        failures.append(f"recognizer checkpoint identity invalid: {exc}")
        recognizer_model = None
    try:
        detector_model = _load_detector_checkpoint(detector_bytes)
    except ValueError as exc:
        failures.append(f"detector checkpoint identity invalid: {exc}")
        detector_model = None
    if failures:
        return _failure_result(
            failures,
            profile,
            recognizer_sha256,
            detector_sha256,
            provider,
            str(device),
        )
    assert recognizer_model is not None and detector_model is not None
    try:
        recognizer_model = recognizer_model.to(device).eval()
        detector_model = detector_model.to(device).eval()
    except Exception as exc:
        raise EvaluationRuntimeError(f"failed to place native models on {device}: {exc}") from exc

    components = manifest["components"]
    assert isinstance(components, Mapping)
    recognizer_component = components["recognizer"]
    detector_component = components["detector"]
    assert isinstance(recognizer_component, Mapping)
    assert isinstance(detector_component, Mapping)
    recognizer_session = _create_session(
        bundle / str(recognizer_component["file"]), provider
    )
    detector_session = _create_session(
        bundle / str(detector_component["file"]), provider
    )
    _actual_provider(recognizer_session, provider)
    _actual_provider(detector_session, provider)

    detector_tensors: list[np.ndarray] = []
    transforms: list[LetterboxTransform] = []
    recognizer_tensors: list[np.ndarray] = []
    for scene in selected_scenes:
        image = load_scene_rgb(scene)
        detector_tensor, transform = letterbox_rgb_v1(image)
        detector_tensors.append(detector_tensor)
        transforms.append(transform)
        audited = [
            plate
            for plate in scene.plates
            if plate.geometry_audited and plate.text_audited and plate.canonical
        ]
        if not audited:
            failures.append(
                f"{scene.scene_id}: parity recognizer crop requires audited text and geometry"
            )
            continue
        try:
            crop = rectify_plate(
                image, np.asarray(audited[0].corners, dtype=np.float32)
            ).image_rgb
        except InvalidCornersError as exc:
            failures.append(
                f"{scene.scene_id}: parity recognizer crop rejected ({exc.reason})"
            )
            continue
        recognizer_tensors.append(preprocess_v1_rgb(crop))
    if len(recognizer_tensors) < max(batch_sizes, default=0):
        failures.append("parity recognizer crops do not cover every profile batch size")
    if failures:
        return _failure_result(
            failures,
            profile,
            recognizer_sha256,
            detector_sha256,
            provider,
            str(device),
        )

    postprocess = detector_component.get("postprocess")
    if not isinstance(postprocess, Mapping):
        raise EvaluationInputError("detector postprocess contract is missing")
    diagnostic_count = _compare_detector(
        detector_model,
        detector_session,
        detector_tensors,
        transforms,
        selected_scenes,
        profile,
        postprocess,
        device,
        failures,
    )
    _compare_recognizer(
        recognizer_model,
        recognizer_session,
        recognizer_tensors,
        profile,
        codec,
        ruleset,
        device,
        failures,
    )
    snapshot.verify_unchanged()
    return ParityResult(
        passed=not failures,
        failures=tuple(failures),
        batch_sizes=batch_sizes,
        diagnostic_raw_difference_count=diagnostic_count,
        recognizer_checkpoint_sha256=recognizer_sha256,
        detector_checkpoint_sha256=detector_sha256,
        provider=provider,
        pytorch_device=str(device),
    )


__all__ = ["run_parity"]
