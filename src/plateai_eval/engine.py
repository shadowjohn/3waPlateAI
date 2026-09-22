"""Exact-provider ONNX engine and strict deployed-behavior evaluation lanes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from plateai_reader import PlateReader, ReaderError, decode_constrained_ctc_v1
from plateai_reader.rectifier import InvalidCornersError, rectify_plate
from plateai_reader.runtime import SessionFactory, _single_output
from plateai_shared.contracts import JsonValue
from plateai_shared.detection import letterbox_rgb_v1, map_points_to_letterbox
from plateai_shared.recognition import preprocess_v1_rgb
from plateai_shared.schema_validation import DocumentValidationError

from .contracts import (
    DetectionPrediction,
    EvaluationInputError,
    EvaluationRuntimeError,
    LaneName,
    LaneResult,
    PlateTruth,
    RecognitionPrediction,
    Scene,
    SuiteSnapshot,
)
from .metrics import (
    SceneMetricInput,
    attribute_e2e,
    detector_metrics,
    match_predictions,
    ocr_metrics,
)
from .suite import load_scene_rgb


PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


class EvaluationEngine(Protocol):
    def recognize(
        self, scene: Scene, plate: PlateTruth, crop_rgb: NDArray[np.uint8]
    ) -> tuple[RecognitionPrediction, Mapping[str, JsonValue]]: ...

    def detect(
        self, scene: Scene, image_rgb: NDArray[np.uint8]
    ) -> tuple[tuple[DetectionPrediction, ...], Mapping[str, JsonValue]]: ...

    def read(
        self, scene: Scene, image_rgb: NDArray[np.uint8]
    ) -> tuple[tuple[DetectionPrediction, ...], Mapping[str, JsonValue]]: ...


def available_ort_providers() -> tuple[str, ...]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise EvaluationRuntimeError("ONNX Runtime is unavailable") from exc
    return tuple(ort.get_available_providers())


def _default_exact_session_factory(path: Path, providers: Sequence[str]):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise EvaluationRuntimeError("ONNX Runtime is unavailable") from exc
    return ort.InferenceSession(str(path), providers=list(providers))


def _declared_artifact_names(manifest: Mapping[str, JsonValue]) -> tuple[str, ...]:
    names: list[str] = []
    for key in ("charset", "rules"):
        declaration = manifest[key]
        assert isinstance(declaration, Mapping)
        names.append(str(declaration["file"]))
    components = manifest["components"]
    assert isinstance(components, Mapping)
    for declaration in components.values():
        assert isinstance(declaration, Mapping)
        names.append(str(declaration["file"]))
    provenance = manifest["provenance"]
    assert isinstance(provenance, Mapping)
    for key in ("training_report", "detector_training_report"):
        if key in provenance:
            declaration = provenance[key]
            assert isinstance(declaration, Mapping)
            names.append(str(declaration["file"]))
    return tuple(sorted(names))


def _bundle_identity(
    bundle_dir: Path, manifest: Mapping[str, JsonValue]
) -> tuple[str, str, Mapping[str, str]]:
    manifest_bytes = (bundle_dir / "manifest.json").read_bytes()
    digest = hashlib.sha256()
    digest.update(b"manifest.json\0")
    digest.update(manifest_bytes)
    artifact_hashes: dict[str, str] = {}
    for name in _declared_artifact_names(manifest):
        artifact_hash = hashlib.sha256((bundle_dir / name).read_bytes()).hexdigest()
        artifact_hashes[name] = artifact_hash
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(artifact_hash))
    return (
        digest.hexdigest(),
        hashlib.sha256(manifest_bytes).hexdigest(),
        MappingProxyType(artifact_hashes),
    )


class OnnxEvaluationEngine:
    """One full validated Bundle pinned to exactly one requested provider."""

    def __init__(
        self,
        bundle_dir: Path,
        provider: str,
        schema_path: Path,
        session_factory: SessionFactory | None = None,
    ) -> None:
        available = available_ort_providers()
        if provider not in available:
            raise EvaluationRuntimeError(f"{provider} unavailable")
        bundle = Path(bundle_dir).resolve()
        factory = session_factory or _default_exact_session_factory
        created_providers: list[str] = []

        def exact_factory(path: Path, providers: Sequence[str]):
            if tuple(providers) != (provider,):
                raise EvaluationRuntimeError(
                    "evaluation session requested more than the exact provider"
                )
            try:
                session = factory(path, (provider,))
            except EvaluationRuntimeError:
                raise
            except Exception as exc:
                raise EvaluationRuntimeError(
                    f"failed to create {path.name} session on {provider}: {exc}"
                ) from exc
            get_providers = getattr(session, "get_providers", None)
            actual = tuple(get_providers()) if callable(get_providers) else ()
            if not actual or actual[0] != provider:
                rendered = actual[0] if actual else "unknown"
                raise EvaluationRuntimeError(
                    f"provider substitution: requested {provider}, actual {rendered}"
                )
            created_providers.append(actual[0])
            return session

        try:
            reader = PlateReader(
                bundle,
                providers=(provider,),
                session_factory=exact_factory,
                schema_path=schema_path,
            )
        except EvaluationRuntimeError:
            raise
        except (DocumentValidationError, ReaderError, OSError, ValueError) as exc:
            raise EvaluationInputError(f"bundle validation failed: {exc}") from exc
        if len(created_providers) != 2:
            raise EvaluationRuntimeError(
                "full evaluation bundle did not create detector and recognizer sessions"
            )
        manifest = reader.manifest
        if manifest["capabilities"] != ["crop-recognition", "plate-detection"]:
            raise EvaluationInputError(
                "bundle must provide exactly crop-recognition and plate-detection"
            )
        try:
            bundle_sha256, manifest_sha256, artifact_hashes = _bundle_identity(
                bundle, manifest
            )
        except OSError as exc:
            raise EvaluationInputError("bundle identity files became unavailable") from exc

        self.bundle_dir = bundle
        self.reader = reader
        self.manifest = manifest
        self.requested_provider = provider
        self.actual_provider = provider
        self.bundle_sha256 = bundle_sha256
        self.manifest_sha256 = manifest_sha256
        self.artifact_hashes = artifact_hashes

    def recognize(
        self, scene: Scene, plate: PlateTruth, crop_rgb: NDArray[np.uint8]
    ) -> tuple[RecognitionPrediction, Mapping[str, JsonValue]]:
        started = perf_counter()
        preprocessing_started = perf_counter()
        tensor = preprocess_v1_rgb(crop_rgb)
        preprocess_ms = (perf_counter() - preprocessing_started) * 1000.0
        inference_started = perf_counter()
        try:
            logits = _single_output(
                self.reader._recognizer_session,
                "logits",
                "input",
                tensor[np.newaxis, ...],
                (1, 80, self.reader.codec.class_count),
            )[0]
            onnx_ms = (perf_counter() - inference_started) * 1000.0
            decoding_started = perf_counter()
            greedy = self.reader.codec.decode_greedy(logits.argmax(axis=1).tolist())
            decoded = decode_constrained_ctc_v1(
                logits, self.reader.codec, self.reader.ruleset
            )
            decoding_ms = (perf_counter() - decoding_started) * 1000.0
        except Exception as exc:
            raise EvaluationRuntimeError(
                f"{scene.scene_id}/{plate.instance_id}: recognizer failed: {exc}"
            ) from exc
        return (
            RecognitionPrediction(
                canonical=decoded.canonical,
                display=decoded.display,
                greedy=greedy,
                rule_id=decoded.rule_id,
            ),
            {
                "preprocess_ms": preprocess_ms,
                "onnx_ms": onnx_ms,
                "ctc_decoding_ms": decoding_ms,
                "total_ms": (perf_counter() - started) * 1000.0,
            },
        )

    def detect(
        self, scene: Scene, image_rgb: NDArray[np.uint8]
    ) -> tuple[tuple[DetectionPrediction, ...], Mapping[str, JsonValue]]:
        try:
            result = self.reader.detect(image_rgb)
        except Exception as exc:
            raise EvaluationRuntimeError(
                f"{scene.scene_id}: detector failed: {exc}"
            ) from exc
        predictions = tuple(
            _detection_prediction(
                detection,
                prediction_id=f"{scene.scene_id}-prediction-{index + 1:04d}",
            )
            for index, detection in enumerate(result.detections)
        )
        return predictions, {"detector_ms": result.detector_ms}

    def read(
        self, scene: Scene, image_rgb: NDArray[np.uint8]
    ) -> tuple[tuple[DetectionPrediction, ...], Mapping[str, JsonValue]]:
        try:
            result = self.reader.read(image_rgb)
        except Exception as exc:
            raise EvaluationRuntimeError(f"{scene.scene_id}: reader failed: {exc}") from exc
        retained: list[tuple[object, RecognitionPrediction | None, str | None]] = []
        for plate in result.plates:
            retained.append(
                (
                    plate.detection,
                    RecognitionPrediction(
                        canonical=plate.decoded.canonical,
                        display=plate.decoded.display,
                        greedy=plate.raw_greedy_text or "",
                        rule_id=plate.decoded.rule_id,
                    ),
                    None,
                )
            )
        for rejection in result.rejections:
            retained.append((rejection.detection, None, rejection.stage))
        retained.sort(
            key=lambda item: (
                -float(item[0].confidence),
                tuple(float(value) for value in item[0].bbox_xyxy),
                item[2] or "",
            )
        )
        predictions = tuple(
            _detection_prediction(
                detection,
                prediction_id=f"{scene.scene_id}-prediction-{index + 1:04d}",
                recognition=recognition,
                rejection_stage=rejection_stage,
            )
            for index, (detection, recognition, rejection_stage) in enumerate(retained)
        )
        timing = result.timing
        return predictions, {
            "detector_ms": timing.detector_ms,
            "rectifier_ms": timing.rectifier_ms,
            "recognizer_ms": timing.recognizer_ms,
            "onnx_ms": timing.onnx_inference_ms,
            "ctc_decoding_ms": timing.ctc_decoding_ms,
            "preprocess_ms": timing.preprocess_ms,
            "total_ms": timing.detector_ms + timing.rectifier_ms + timing.recognizer_ms,
        }


def _detection_prediction(
    detection,
    *,
    prediction_id: str,
    recognition: RecognitionPrediction | None = None,
    rejection_stage: str | None = None,
) -> DetectionPrediction:
    bbox = tuple(float(value) for value in detection.bbox_xyxy)
    corners = tuple(
        (float(point[0]), float(point[1])) for point in detection.corners_xy
    )
    return DetectionPrediction(
        prediction_id=prediction_id,
        confidence=float(detection.confidence),
        bbox_xyxy=bbox,  # type: ignore[arg-type]
        corners=corners,  # type: ignore[arg-type]
        recognition=recognition,
        rejection_stage=rejection_stage,
    )


def _lane_scenes(snapshot: SuiteSnapshot, lane: LaneName) -> tuple[Scene, ...]:
    by_id = {scene.scene_id: scene for scene in snapshot.scenes}
    return tuple(by_id[scene_id] for scene_id in snapshot.lane_scene_ids[lane])


def _call_with_context(label: str, function, *args):
    try:
        return function(*args)
    except EvaluationInputError:
        raise
    except EvaluationRuntimeError as exc:
        raise EvaluationRuntimeError(f"{label}: {exc}") from exc
    except Exception as exc:
        raise EvaluationRuntimeError(f"{label}: unexpected runtime failure: {exc}") from exc


def _sample(
    *,
    lane: LaneName,
    scene: Scene,
    instance_id: str,
    prediction_id: str,
    prediction: DetectionPrediction | None,
    iou: float | None,
    corner_errors: Sequence[float] | None,
    attribution: str,
    timings: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    recognition = prediction.recognition if prediction is not None else None
    return {
        "lane": lane,
        "source_id": scene.source_id,
        "scene_id": scene.scene_id,
        "instance_id": instance_id,
        "prediction_id": prediction_id,
        "image_sha256": scene.image_sha256,
        "predicted_canonical": recognition.canonical if recognition else None,
        "greedy": recognition.greedy if recognition else None,
        "score": prediction.confidence if prediction else None,
        "iou": iou,
        "corner_errors": (
            list(corner_errors) if corner_errors is not None else [None, None, None, None]
        ),
        "attribution": attribution,
        "timings_ms": dict(timings),
    }


def run_oracle_lane(
    snapshot: SuiteSnapshot, engine: EvaluationEngine
) -> LaneResult:
    scenes = _lane_scenes(snapshot, "oracle_ocr")
    pairs: list[tuple[str, str]] = []
    samples: list[Mapping[str, JsonValue]] = []
    processed_instances = 0
    for scene in scenes:
        for plate in scene.plates:
            image = load_scene_rgb(scene)
            try:
                crop = rectify_plate(
                    image, np.asarray(plate.corners, dtype=np.float32)
                ).image_rgb
            except InvalidCornersError as exc:
                raise EvaluationRuntimeError(
                    f"{scene.scene_id}/{plate.instance_id}: oracle rectifier failed: {exc.reason}"
                ) from exc
            prediction, timings = _call_with_context(
                f"{scene.scene_id}/{plate.instance_id}",
                engine.recognize,
                scene,
                plate,
                crop,
            )
            truth = plate.canonical or ""
            pairs.append((prediction.canonical, truth))
            samples.append(
                _sample(
                    lane="oracle_ocr",
                    scene=scene,
                    instance_id=plate.instance_id,
                    prediction_id=f"{scene.scene_id}-{plate.instance_id}-oracle",
                    prediction=DetectionPrediction(
                        prediction_id=f"{scene.scene_id}-{plate.instance_id}-oracle",
                        confidence=1.0,
                        bbox_xyxy=_bbox_from_corners(plate.corners),
                        corners=plate.corners,
                        recognition=prediction,
                    ),
                    iou=1.0,
                    corner_errors=(0.0, 0.0, 0.0, 0.0),
                    attribution=(
                        "SUCCESS"
                        if prediction.canonical == truth
                        else (
                            "RULE_FILTERED"
                            if prediction.greedy == truth
                            else "RECOGNIZER_MISREAD"
                        )
                    ),
                    timings=timings,
                )
            )
            processed_instances += 1
    snapshot.verify_unchanged()
    instance_count = sum(len(scene.plates) for scene in scenes)
    return LaneResult(
        lane="oracle_ocr",
        scene_count=len(scenes),
        instance_count=instance_count,
        processed_scene_count=len(scenes),
        processed_instance_count=processed_instances,
        metrics=ocr_metrics(pairs),
        samples=tuple(samples),
    )


def _bbox_from_corners(corners) -> tuple[float, float, float, float]:
    values = np.asarray(corners, dtype=np.float64)
    return (
        float(values[:, 0].min()),
        float(values[:, 1].min()),
        float(values[:, 0].max()),
        float(values[:, 1].max()),
    )


def _letterbox_truth(plate: PlateTruth, transform) -> PlateTruth:
    corners_array = map_points_to_letterbox(plate.corners, transform)
    corners = tuple((float(x), float(y)) for x, y in corners_array)
    return replace(plate, corners=corners)  # type: ignore[arg-type]


def _letterbox_prediction(
    prediction: DetectionPrediction, transform
) -> DetectionPrediction:
    corners_array = map_points_to_letterbox(prediction.corners, transform)
    x1, y1, x2, y2 = prediction.bbox_xyxy
    bbox_points = map_points_to_letterbox(((x1, y1), (x2, y2)), transform)
    corners = tuple((float(x), float(y)) for x, y in corners_array)
    bbox = tuple(float(value) for value in bbox_points.reshape(-1))
    return replace(
        prediction,
        bbox_xyxy=bbox,  # type: ignore[arg-type]
        corners=corners,  # type: ignore[arg-type]
    )


def _matched_samples(
    *,
    lane: LaneName,
    scene: Scene,
    predictions: tuple[DetectionPrediction, ...],
    truths: tuple[PlateTruth, ...],
    iou_min: float,
    max_corner_error: float,
    timings: Mapping[str, JsonValue],
) -> tuple[list[Mapping[str, JsonValue]], list[tuple[str, str]], int]:
    matches = match_predictions(predictions, truths, iou_min)
    samples: list[Mapping[str, JsonValue]] = []
    text_pairs: list[tuple[str, str]] = []
    by_truth = {match.truth_index: match for match in matches.matches}
    for truth_index, truth in enumerate(truths):
        match = by_truth.get(truth_index)
        if match is None:
            text_pairs.append(("", truth.canonical or ""))
            samples.append(
                _sample(
                    lane=lane,
                    scene=scene,
                    instance_id=truth.instance_id,
                    prediction_id="",
                    prediction=None,
                    iou=None,
                    corner_errors=None,
                    attribution="DETECTOR_MISSED",
                    timings=timings,
                )
            )
            continue
        prediction = predictions[match.prediction_index]
        quad_passed = all(
            error <= max_corner_error for error in match.corner_errors
        )
        attributed_match = replace(match, quad_passed=quad_passed)
        recognition = prediction.recognition
        text_pairs.append(
            (recognition.canonical if recognition else "", truth.canonical or "")
        )
        samples.append(
            _sample(
                lane=lane,
                scene=scene,
                instance_id=truth.instance_id,
                prediction_id=prediction.prediction_id,
                prediction=prediction,
                iou=match.iou,
                corner_errors=match.corner_errors,
                attribution=(
                    ("SUCCESS" if quad_passed else "DETECTOR_BAD_QUAD")
                    if lane == "detector"
                    else attribute_e2e(attributed_match, prediction)
                ),
                timings=timings,
            )
        )
    for prediction_index in matches.unmatched_prediction_indices:
        prediction = predictions[prediction_index]
        samples.append(
            _sample(
                lane=lane,
                scene=scene,
                instance_id="",
                prediction_id=prediction.prediction_id,
                prediction=prediction,
                iou=None,
                corner_errors=None,
                attribution=attribute_e2e(None, prediction),
                timings=timings,
            )
        )
    return samples, text_pairs, len(matches.matches)


def run_detector_lane(
    snapshot: SuiteSnapshot,
    engine: EvaluationEngine,
    *,
    iou_min: float = 0.5,
    max_corner_error: float = 8.0,
) -> LaneResult:
    scenes = _lane_scenes(snapshot, "detector")
    scene_inputs: list[SceneMetricInput] = []
    samples: list[Mapping[str, JsonValue]] = []
    processed_instances = 0
    for scene in scenes:
        image = load_scene_rgb(scene)
        predictions, timings = _call_with_context(
            scene.scene_id, engine.detect, scene, image
        )
        _, transform = letterbox_rgb_v1(image)
        mapped_predictions = tuple(
            _letterbox_prediction(prediction, transform)
            for prediction in predictions
        )
        mapped_truths = tuple(
            _letterbox_truth(plate, transform) for plate in scene.plates
        )
        scene_inputs.append(SceneMetricInput(mapped_predictions, mapped_truths))
        one_samples, _, _ = _matched_samples(
            lane="detector",
            scene=scene,
            predictions=mapped_predictions,
            truths=mapped_truths,
            iou_min=iou_min,
            max_corner_error=max_corner_error,
            timings=timings,
        )
        samples.extend(one_samples)
        processed_instances += len(scene.plates)
    snapshot.verify_unchanged()
    instance_count = sum(len(scene.plates) for scene in scenes)
    return LaneResult(
        lane="detector",
        scene_count=len(scenes),
        instance_count=instance_count,
        processed_scene_count=len(scenes),
        processed_instance_count=processed_instances,
        metrics=detector_metrics(scene_inputs, iou_min, max_corner_error),
        samples=tuple(samples),
    )


def run_e2e_lane(
    snapshot: SuiteSnapshot,
    engine: EvaluationEngine,
    *,
    iou_min: float = 0.5,
    max_corner_error: float = 8.0,
) -> LaneResult:
    scenes = _lane_scenes(snapshot, "end_to_end")
    samples: list[Mapping[str, JsonValue]] = []
    pairs: list[tuple[str, str]] = []
    prediction_count = 0
    localization_true_positives = 0
    processed_instances = 0
    for scene in scenes:
        image = load_scene_rgb(scene)
        predictions, timings = _call_with_context(
            scene.scene_id, engine.read, scene, image
        )
        _, transform = letterbox_rgb_v1(image)
        mapped_predictions = tuple(
            _letterbox_prediction(prediction, transform)
            for prediction in predictions
        )
        mapped_truths = tuple(
            _letterbox_truth(plate, transform) for plate in scene.plates
        )
        one_samples, one_pairs, matched = _matched_samples(
            lane="end_to_end",
            scene=scene,
            predictions=mapped_predictions,
            truths=mapped_truths,
            iou_min=iou_min,
            max_corner_error=max_corner_error,
            timings=timings,
        )
        samples.extend(one_samples)
        pairs.extend(one_pairs)
        prediction_count += len(mapped_predictions)
        localization_true_positives += matched
        processed_instances += len(scene.plates)
    snapshot.verify_unchanged()
    instance_count = sum(len(scene.plates) for scene in scenes)
    metrics = dict(ocr_metrics(pairs))
    metrics.update(
        {
            "prediction_count": prediction_count,
            "localization_true_positives": localization_true_positives,
            "localization_recall": (
                localization_true_positives / instance_count
                if instance_count
                else 0.0
            ),
        }
    )
    return LaneResult(
        lane="end_to_end",
        scene_count=len(scenes),
        instance_count=instance_count,
        processed_scene_count=len(scenes),
        processed_instance_count=processed_instances,
        metrics=metrics,
        samples=tuple(samples),
    )


__all__ = [
    "PROVIDERS",
    "EvaluationEngine",
    "OnnxEvaluationEngine",
    "available_ort_providers",
    "run_detector_lane",
    "run_e2e_lane",
    "run_oracle_lane",
]
