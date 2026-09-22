"""Immutable contracts shared by the strict evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias

from plateai_shared.contracts import JsonValue
from plateai_shared.schema_validation import DocumentValidationError, validate_document


LaneName = Literal["oracle_ocr", "detector", "end_to_end", "parity"]
RightsStatus = Literal["pass", "restricted", "unreviewed"]
OverallStatus = Literal["REJECTED", "PASS_LOCAL_ONLY", "PROMOTION_ELIGIBLE"]
Corner: TypeAlias = tuple[float, float]
Corners: TypeAlias = tuple[Corner, Corner, Corner, Corner]


class EvaluationInputError(ValueError):
    """An immutable evaluation input failed validation."""


class EvaluationRuntimeError(RuntimeError):
    """A requested evaluation runtime could not complete exactly."""


M4_5_STRICT_V1: dict[str, JsonValue] = {
    "schema_version": 1,
    "profile_id": "m4.5-strict-v1",
    "minimums": {
        "oracle_plate_instances": 200,
        "detector_scenes": 200,
        "detector_plate_instances": 250,
        "e2e_scenes": 100,
        "parity_scenes": 2,
    },
    "thresholds": {
        "oracle_exact_match_min": 0.95,
        "oracle_micro_cer_max": 0.02,
        "detector_bbox_precision_min": 0.90,
        "detector_bbox_recall_min": 0.90,
        "detector_complete_quad_precision_min": 0.80,
        "detector_complete_quad_recall_min": 0.80,
        "e2e_exact_match_min": 0.85,
        "e2e_micro_cer_max": 0.05,
        "e2e_localization_recall_min": 0.90,
    },
    "metrics": {
        "bbox_iou_min": 0.5,
        "complete_quad_max_corner_error_640px": 8.0,
    },
    "parity": {"batch_sizes": [1, 2], "rtol": 0.0001, "atol": 0.00001},
}


def canonical_profile_bytes() -> bytes:
    """Return the only byte representation valid for the official profile ID."""

    return (
        json.dumps(M4_5_STRICT_V1, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class StrictProfile:
    path: Path
    raw_bytes: bytes
    sha256: str
    profile_id: str
    document: Mapping[str, JsonValue]
    minimums: Mapping[str, int]
    thresholds: Mapping[str, float]
    metrics: Mapping[str, float]
    parity: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SourceRights:
    review_status: Literal["reviewed", "restricted", "unreviewed"]
    local_only: bool
    allowed_uses: frozenset[str]
    notice: str


@dataclass(frozen=True, slots=True)
class EvaluationSource:
    source_id: str
    root: Path
    revision: str
    rights: SourceRights
    split: str | None = None
    snapshot_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    manifest_id: str
    path: Path
    sha256: str
    lanes: tuple[LaneName, ...]


@dataclass(frozen=True, slots=True)
class PlateTruth:
    instance_id: str
    corners: Corners
    canonical: str | None
    display: str | None
    geometry_audited: bool
    text_audited: bool


@dataclass(frozen=True, slots=True)
class Scene:
    scene_id: str
    source_id: str
    image_path: Path
    image_sha256: str
    width: int
    height: int
    plates: tuple[PlateTruth, ...]


@dataclass(frozen=True, slots=True)
class SuiteSnapshot:
    path: Path
    raw_bytes: bytes
    sha256: str
    snapshot_sha256: str
    suite_id: str
    revision: str
    sources: tuple[EvaluationSource, ...]
    manifests: tuple[EvaluationManifest, ...]
    scenes: tuple[Scene, ...]
    lane_scene_ids: Mapping[LaneName, tuple[str, ...]]
    file_hashes: Mapping[Path, str] = field(repr=False)

    def verify_unchanged(self) -> None:
        """Re-read every bound file and reject any post-load mutation."""

        manifest_paths = {manifest.path for manifest in self.manifests}
        image_paths = {scene.image_path for scene in self.scenes}
        for path, expected in sorted(
            self.file_hashes.items(), key=lambda item: item[0].as_posix()
        ):
            if path == self.path:
                label = "suite"
            elif path in manifest_paths:
                label = "manifest"
            elif path in image_paths:
                label = "image"
            else:
                label = "snapshot file"
            if path.is_symlink() or not path.is_file():
                raise EvaluationInputError(f"{label} unavailable: {path.name}")
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise EvaluationInputError(f"{label} unavailable: {path.name}") from exc
            if actual != expected:
                raise EvaluationInputError(f"{label} hash mismatch: {path.name}")


@dataclass(frozen=True, slots=True)
class LaneResult:
    lane: LaneName
    scene_count: int
    instance_count: int
    processed_scene_count: int
    processed_instance_count: int
    metrics: Mapping[str, JsonValue]
    samples: tuple[Mapping[str, JsonValue], ...]


@dataclass(frozen=True, slots=True)
class RecognitionPrediction:
    canonical: str
    display: str
    greedy: str
    rule_id: str


@dataclass(frozen=True, slots=True)
class DetectionPrediction:
    prediction_id: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    corners: Corners
    recognition: RecognitionPrediction | None = None
    rejection_stage: str | None = None


@dataclass(frozen=True, slots=True)
class ParityResult:
    passed: bool
    failures: tuple[str, ...]
    batch_sizes: tuple[int, ...]
    diagnostic_raw_difference_count: int
    recognizer_checkpoint_sha256: str
    detector_checkpoint_sha256: str
    provider: str | None = None
    pytorch_device: str | None = None


@dataclass(frozen=True, slots=True)
class PopulationCounts:
    oracle_plate_instances: int
    detector_scenes: int
    detector_plate_instances: int
    e2e_scenes: int
    parity_scenes: int

    def as_mapping(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "oracle_plate_instances": self.oracle_plate_instances,
                "detector_scenes": self.detector_scenes,
                "detector_plate_instances": self.detector_plate_instances,
                "e2e_scenes": self.e2e_scenes,
                "parity_scenes": self.parity_scenes,
            }
        )


@dataclass(frozen=True, slots=True)
class GateDecision:
    technical_gate: Literal["pass", "fail"]
    rights_gate: RightsStatus
    overall_status: OverallStatus
    checks: tuple[Mapping[str, JsonValue], ...]


def _mapping(value: object, field_name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise EvaluationInputError(f"profile.{field_name}: expected object")
    return value


def load_profile(path: Path, schema_path: Path) -> StrictProfile:
    """Load the canonical, frozen official M4.5 strict profile."""

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise EvaluationInputError(f"profile: cannot read {path}") from exc
    try:
        document = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputError("profile: invalid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise EvaluationInputError("profile: document must be an object")
    try:
        validate_document(document, schema_path)
    except DocumentValidationError as exc:
        raise EvaluationInputError(f"profile: {exc}") from exc
    if document != M4_5_STRICT_V1:
        raise EvaluationInputError("profile: official profile values differ")
    if raw_bytes != canonical_profile_bytes():
        raise EvaluationInputError(
            "profile: official profile bytes differ from canonical bytes"
        )

    minimums = _mapping(document["minimums"], "minimums")
    thresholds = _mapping(document["thresholds"], "thresholds")
    metrics = _mapping(document["metrics"], "metrics")
    parity = _mapping(document["parity"], "parity")
    return StrictProfile(
        path=path.resolve(),
        raw_bytes=raw_bytes,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        profile_id="m4.5-strict-v1",
        document=MappingProxyType(document),
        minimums=MappingProxyType({key: int(value) for key, value in minimums.items()}),
        thresholds=MappingProxyType(
            {key: float(value) for key, value in thresholds.items()}
        ),
        metrics=MappingProxyType({key: float(value) for key, value in metrics.items()}),
        parity=MappingProxyType(
            {
                "batch_sizes": tuple(int(value) for value in parity["batch_sizes"]),
                "rtol": float(parity["rtol"]),
                "atol": float(parity["atol"]),
            }
        ),
    )


__all__ = [
    "Corner",
    "Corners",
    "DetectionPrediction",
    "EvaluationInputError",
    "EvaluationManifest",
    "EvaluationRuntimeError",
    "EvaluationSource",
    "GateDecision",
    "LaneName",
    "LaneResult",
    "M4_5_STRICT_V1",
    "OverallStatus",
    "ParityResult",
    "PlateTruth",
    "PopulationCounts",
    "RecognitionPrediction",
    "RightsStatus",
    "Scene",
    "SourceRights",
    "StrictProfile",
    "SuiteSnapshot",
    "canonical_profile_bytes",
    "load_profile",
]
