"""Manifest-validated ONNX Runtime inference for local full Model Bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
import sysconfig
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from plateai_shared.bundle import validate_model_bundle
from plateai_shared.detection import PlateDetection, letterbox_rgb_v1
from plateai_shared.recognition import CTCCodec, preprocess_v1_rgb
from plateai_shared.rules import format_display, load_character_set, load_ruleset

from .detector import postprocess_candidates
from .rectifier import InvalidCornersError, rectify_plate


class ReaderError(ValueError):
    """A local bundle or inference result cannot satisfy the Reader contract."""


class OrtSession(Protocol):
    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, object]
    ) -> Sequence[object]: ...


SessionFactory = Callable[[Path, Sequence[str]], OrtSession]

_PROVIDER_PRIORITY = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


@dataclass(frozen=True, slots=True)
class DecodedPlate:
    """The highest-log-probability CTC sequence permitted by one plate rule."""

    canonical: str
    display: str
    rule_id: str
    plate_type: str
    log_probability: float


@dataclass(frozen=True, slots=True)
class PlateRead:
    """One successful full-pipeline plate result in retained detection order."""

    detection: PlateDetection
    decoded: DecodedPlate
    raw_greedy_text: str | None = None
    crop_rgb: NDArray[np.uint8] | None = field(default=None, repr=False, compare=False)
    ctc_decoding_ms: float = 0.0
    recognition_batch_index: int = 0


@dataclass(frozen=True, slots=True)
class ReaderRejection:
    """A single detection that failed without stopping other retained plates."""

    detection: PlateDetection
    stage: str
    reason: str


@dataclass(frozen=True, slots=True)
class ReaderTiming:
    """Observed per-call timings; they are measurements, not performance claims."""

    detector_ms: float
    rectifier_ms: float
    recognizer_ms: float
    retained_detection_count: int
    rectified_plate_count: int
    recognition_batch_sizes: tuple[int, ...]
    onnx_inference_ms: float = 0.0
    ctc_decoding_ms: float = 0.0
    preprocess_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class ReaderResult:
    plates: tuple[PlateRead, ...]
    rejections: tuple[ReaderRejection, ...]
    providers: tuple[str, ...]
    timing: ReaderTiming


def _default_schema_path() -> Path:
    checkout_path = Path(__file__).resolve().parents[2] / "schemas/model_manifest.schema.json"
    if checkout_path.is_file():
        return checkout_path
    return Path(sysconfig.get_path("data")) / "share/3wa-plate-ai/schemas/model_manifest.schema.json"


def select_ort_providers(available_providers: Sequence[str]) -> tuple[str, ...]:
    """Select supported providers in the documented TensorRT, CUDA, CPU order."""

    available = set(available_providers)
    selected = tuple(name for name in _PROVIDER_PRIORITY if name in available)
    if not selected:
        raise ReaderError(
            "ONNX Runtime provides none of TensorRT, CUDA, or CPU execution providers"
        )
    return selected


def _default_session_factory(path: Path, providers: Sequence[str]) -> OrtSession:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ReaderError(
            "ONNX Runtime is required; install the reader extra or the training environment"
        ) from error
    return ort.InferenceSession(str(path), providers=list(providers))


_format_display = format_display


def _log_softmax(logits: NDArray[np.float32]) -> NDArray[np.float64]:
    if not np.isfinite(logits).all():
        raise ReaderError("recognizer logits must be finite")
    values = np.asarray(logits, dtype=np.float64)
    maximum = np.max(values)
    return values - maximum - math.log(float(np.exp(values - maximum).sum()))


def _replace_if_better(
    states: dict[tuple[int, int, int], tuple[float, str]],
    key: tuple[int, int, int],
    candidate: tuple[float, str],
) -> None:
    current = states.get(key)
    if current is None or candidate[0] > current[0] or (
        candidate[0] == current[0] and candidate[1] < current[1]
    ):
        states[key] = candidate


def decode_constrained_ctc_v1(
    logits: object,
    codec: CTCCodec,
    ruleset: object,
) -> DecodedPlate:
    """Viterbi-decode CTC logits while requiring one enabled manifest plate rule.

    The dynamic-programming state retains rule position and the previous raw CTC
    index. That permits blank-separated duplicate symbols while avoiding a large
    enumeration of all legal plate strings.
    """

    values = np.asarray(logits)
    if values.shape != (80, codec.class_count):
        raise ReaderError(
            f"recognizer logits must have shape [80, {codec.class_count}]"
        )
    try:
        values = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ReaderError("recognizer logits must be numeric") from error

    enabled_rules = tuple(rule for rule in ruleset.rules if rule.enabled)
    if not enabled_rules:
        raise ReaderError("bundle rules contain no enabled plate rule")

    allowed_indices: list[tuple[tuple[int, ...], ...]] = []
    for rule in enabled_rules:
        token_indices: list[tuple[int, ...]] = []
        for token in rule.tokens:
            symbols = (
                ruleset.character_classes[token.value]
                if token.kind == "class"
                else token.value
            )
            token_indices.append(tuple(codec.index_by_symbol[symbol] for symbol in symbols))
        allowed_indices.append(tuple(token_indices))

    # A unique greedy path is the global Viterbi optimum. If it satisfies a
    # rule, searching alternative paths cannot improve it. Tied frame maxima
    # use the DP below to preserve the canonical/rule-id tie contract.
    log_probabilities_all = np.stack([_log_softmax(row) for row in values])
    best = values.argmax(axis=1)
    greedy = codec.decode_greedy(best.tolist())
    compatible = [
        index for index, tokens in enumerate(allowed_indices)
        if len(tokens) == len(greedy) and all(
            codec.index_by_symbol[symbol] in allowed
            for symbol, allowed in zip(greedy, tokens, strict=True)
        )
    ]
    if compatible and np.all((values == values.max(axis=1, keepdims=True)).sum(axis=1) == 1):
        rule = enabled_rules[min(compatible, key=lambda index: enabled_rules[index].id)]
        return DecodedPlate(
            greedy, _format_display(greedy, rule.separator, rule.separator_after),
            rule.id, rule.plate_type,
            float(sum(log_probabilities_all[t, index] for t, index in enumerate(best))),
        )

    states: dict[tuple[int, int, int], tuple[float, str]] = {
        (rule_index, 0, codec.blank_index): (0.0, "")
        for rule_index in range(len(enabled_rules))
    }
    for time_step in range(values.shape[0]):
        log_probabilities = log_probabilities_all[time_step]
        next_states: dict[tuple[int, int, int], tuple[float, str]] = {}
        groups: dict[tuple[int, int], list[tuple[int, float, str]]] = {}
        for (rule_index, position, previous_index), (score, canonical) in states.items():
            groups.setdefault((rule_index, position), []).append((previous_index, score, canonical))
        for (rule_index, position), entries in groups.items():
            # Emitting a new symbol excludes only the identical previous raw
            # index. Thus the best two predecessors suffice, replacing the
            # previous O(classes**2) transitions without pruning any paths.
            entries.sort(key=lambda item: (-item[1], item[2]))
            _, score, canonical = entries[0]
            _replace_if_better(
                next_states,
                (rule_index, position, codec.blank_index),
                (score + float(log_probabilities[codec.blank_index]), canonical),
            )
            for previous_index, previous_score, previous_text in entries:
                if previous_index != codec.blank_index:
                    _replace_if_better(
                        next_states,
                        (rule_index, position, previous_index),
                        (previous_score + float(log_probabilities[previous_index]), previous_text),
                    )
            if position >= len(allowed_indices[rule_index]):
                continue
            for class_index in allowed_indices[rule_index][position]:
                predecessor = 1 if entries[0][0] == class_index else 0
                if predecessor == len(entries):
                    continue
                _, score, canonical = entries[predecessor]
                _replace_if_better(
                    next_states,
                    (rule_index, position + 1, class_index),
                    (score + float(log_probabilities[class_index]), canonical + codec.symbols[class_index - 1]),
                )
        states = next_states

    accepted: list[tuple[float, str, int]] = []
    for (rule_index, position, _), (score, canonical) in states.items():
        if position == len(enabled_rules[rule_index].tokens):
            accepted.append((score, canonical, rule_index))
    if not accepted:
        raise ReaderError("recognizer produced no legal CTC plate sequence")
    score, canonical, rule_index = min(
        accepted,
        key=lambda item: (-item[0], enabled_rules[item[2]].id, item[1]),
    )
    rule = enabled_rules[rule_index]
    return DecodedPlate(
        canonical=canonical,
        display=_format_display(canonical, rule.separator, rule.separator_after),
        rule_id=rule.id,
        plate_type=rule.plate_type,
        log_probability=score,
    )


def _single_output(
    session: OrtSession,
    output_name: str,
    input_name: str,
    tensor: NDArray[np.float32],
    expected_shape: tuple[int, ...],
) -> NDArray[np.float32]:
    outputs = session.run([output_name], {input_name: tensor})
    if len(outputs) != 1:
        raise ReaderError(f"ONNX session did not return {output_name}")
    try:
        values = np.asarray(outputs[0], dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ReaderError(f"ONNX {output_name} output must be numeric") from error
    if values.shape != expected_shape:
        raise ReaderError(
            f"ONNX {output_name} output has shape {list(values.shape)}, expected {list(expected_shape)}"
        )
    if not np.isfinite(values).all():
        raise ReaderError(f"ONNX {output_name} output must be finite")
    return values


class PlateReader:
    """Persistent ONNX sessions for one hash-validated local full bundle."""

    def __init__(
        self,
        bundle_dir: Path,
        *,
        providers: Sequence[str] | None = None,
        session_factory: SessionFactory | None = None,
        schema_path: Path | None = None,
    ) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        manifest = validate_model_bundle(
            self.bundle_dir, schema_path or _default_schema_path()
        )
        if manifest["capabilities"] != ["crop-recognition", "plate-detection"]:
            raise ReaderError("bundle must provide crop-recognition and plate-detection")

        self.manifest = manifest
        self.charset = load_character_set(self.bundle_dir / manifest["charset"]["file"])
        self.codec = CTCCodec.from_charset(self.charset)
        self.ruleset = load_ruleset(
            self.bundle_dir / manifest["rules"]["file"], self.charset
        )
        if providers is None:
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise ReaderError(
                    "ONNX Runtime is required; install the reader extra or the training environment"
                ) from error
            self.providers = select_ort_providers(ort.get_available_providers())
        else:
            self.providers = tuple(providers)
            if not self.providers:
                raise ReaderError("providers must not be empty")

        factory = session_factory or _default_session_factory
        components = manifest["components"]
        self._detector_session = factory(
            self.bundle_dir / components["detector"]["file"], self.providers
        )
        self._recognizer_session = factory(
            self.bundle_dir / components["recognizer"]["file"], self.providers
        )
        self._recognizer_max_batch = components["recognizer"]["batch"]["max"]

    def read(self, image_rgb: object) -> ReaderResult:
        """Read all retained detections from one uint8 RGB image without rebuilding sessions."""

        detector_started = perf_counter()
        detector_tensor, transform = letterbox_rgb_v1(image_rgb)
        candidates = _single_output(
            self._detector_session,
            "candidates",
            "images",
            detector_tensor[np.newaxis, ...],
            (1, 8400, 13),
        )[0]
        postprocess = self.manifest["components"]["detector"]["postprocess"]
        detections = postprocess_candidates(candidates, transform, postprocess)
        detector_ms = (perf_counter() - detector_started) * 1000.0

        rectifier_started = perf_counter()
        accepted: list[tuple[PlateDetection, NDArray[np.uint8]]] = []
        rejections: list[ReaderRejection] = []
        image = np.asarray(image_rgb)
        for detection in detections:
            try:
                accepted.append((detection, rectify_plate(image, detection.corners_xy).image_rgb))
            except InvalidCornersError as error:
                rejections.append(
                    ReaderRejection(detection, "rectifier", error.reason)
                )
        rectifier_ms = (perf_counter() - rectifier_started) * 1000.0

        recognizer_started = perf_counter()
        plates: list[PlateRead] = []
        batch_sizes: list[int] = []
        onnx_ms = decoding_ms = preprocess_ms = 0.0
        for start in range(0, len(accepted), self._recognizer_max_batch):
            chunk = accepted[start : start + self._recognizer_max_batch]
            preprocessing_started = perf_counter()
            batch = np.stack(
                [preprocess_v1_rgb(crop) for _, crop in chunk], axis=0
            )
            preprocess_ms += (perf_counter() - preprocessing_started) * 1000.0
            inference_started = perf_counter()
            logits = _single_output(
                self._recognizer_session,
                "logits",
                "input",
                batch,
                (len(chunk), 80, self.codec.class_count),
            )
            onnx_ms += (perf_counter() - inference_started) * 1000.0
            for (detection, crop), one_plate_logits in zip(chunk, logits, strict=True):
                decoding_started = perf_counter()
                try:
                    raw_greedy = self.codec.decode_greedy(one_plate_logits.argmax(axis=1).tolist())
                    decoded = decode_constrained_ctc_v1(one_plate_logits, self.codec, self.ruleset)
                    plates.append(
                        PlateRead(
                            detection=detection,
                            decoded=decoded,
                            raw_greedy_text=raw_greedy,
                            crop_rgb=crop,
                            ctc_decoding_ms=(perf_counter() - decoding_started) * 1000.0,
                            recognition_batch_index=len(batch_sizes),
                        )
                    )
                except ReaderError as error:
                    rejections.append(
                        ReaderRejection(detection, "recognizer", str(error))
                    )
                finally:
                    decoding_ms += (perf_counter() - decoding_started) * 1000.0
            batch_sizes.append(len(chunk))
        recognizer_ms = (perf_counter() - recognizer_started) * 1000.0

        return ReaderResult(
            plates=tuple(plates),
            rejections=tuple(rejections),
            providers=self.providers,
            timing=ReaderTiming(
                detector_ms=detector_ms,
                rectifier_ms=rectifier_ms,
                recognizer_ms=recognizer_ms,
                retained_detection_count=len(detections),
                rectified_plate_count=len(accepted),
                recognition_batch_sizes=tuple(batch_sizes),
                onnx_inference_ms=onnx_ms,
                ctc_decoding_ms=decoding_ms,
                preprocess_ms=preprocess_ms,
            ),
        )
