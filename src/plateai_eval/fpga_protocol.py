"""Leakage-aware, provenance-preserving evaluation of the attributed OCR pair.

An exact match here is a string comparison, not a license or deployment verdict.
Only manually audited unseen holdout rows can support independent accuracy.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


class EvaluationInputError(ValueError):
    """The scorecard input is incomplete, corrupted, or leakage-prone."""


@dataclass(frozen=True, slots=True)
class AuditedPlate:
    image: Path
    sha256: str
    canonical: str
    vehicle_class: str
    split: Literal["dev", "holdout"]
    crop_xyxy: tuple[int, int, int, int]
    source_kind: Literal["audited_unseen", "training_source_replay"]


@dataclass(frozen=True, slots=True)
class Counts:
    total: int
    exact: int
    empty: int


@dataclass(frozen=True, slots=True)
class ImageResult:
    image: str
    canonical: str
    predicted: tuple[str, ...]
    exact: bool
    empty: bool
    vehicle_class: str
    split: str
    source_kind: str
    timings_ms: dict[str, float]
    error: str | None


@dataclass(frozen=True, slots=True)
class EvalReport:
    mode: str
    overall: Counts
    v1_passenger: Counts
    results: tuple[ImageResult, ...]
    providers: tuple[str, ...]
    hardware: str
    source_kinds: tuple[str, ...]
    independent_accuracy: str

    def to_dict(self) -> dict:
        return asdict(self)


class CropReader(Protocol):
    def recognize(self, image: NDArray[np.uint8], *, corner_policy: str = "compat") -> object: ...


class SceneReader(Protocol):
    def read(self, image: NDArray[np.uint8]) -> object: ...


def _canonical(value: object) -> str:
    if not isinstance(value, str):
        raise EvaluationInputError("invalid_canonical")
    normalized = "".join(char for char in value.upper() if char.isascii() and char.isalnum())
    if not normalized:
        raise EvaluationInputError("invalid_canonical")
    return normalized


def load_audited_manifest(path: Path) -> tuple[AuditedPlate, ...]:
    path = Path(path)
    rows: list[AuditedPlate] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("expected object")
            if not item.get("vehicle_class") or not isinstance(item["vehicle_class"], str):
                raise EvaluationInputError("missing_vehicle_class")
            image = Path(item["image"])
            if not image.is_absolute():
                image = path.parent / image
            sha = item["sha256"]
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise EvaluationInputError("invalid_sha256")
            crop = item["crop_xyxy"]
            if not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int for v in crop):
                raise EvaluationInputError("invalid_crop")
            split = item["split"]
            source_kind = item["source_kind"]
            if split not in ("dev", "holdout") or source_kind not in ("audited_unseen", "training_source_replay"):
                raise EvaluationInputError("invalid_provenance")
            rows.append(AuditedPlate(
                image=image.resolve(), sha256=sha, canonical=_canonical(item["canonical"]),
                vehicle_class=item["vehicle_class"], split=split,
                crop_xyxy=tuple(crop), source_kind=source_kind,
            ))
        except EvaluationInputError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EvaluationInputError(f"invalid_manifest_row:{line_number}") from exc
    if not rows:
        raise EvaluationInputError("empty_manifest")
    for key in ("canonical", "sha256"):
        dev = {getattr(row, key) for row in rows if row.split == "dev"}
        held = {getattr(row, key) for row in rows if row.split == "holdout"}
        if dev & held:
            raise EvaluationInputError("cross_split_overlap")
    return tuple(rows)


def _read_verified_rgb(row: AuditedPlate) -> NDArray[np.uint8]:
    try:
        data = row.image.read_bytes()
    except OSError as exc:
        raise EvaluationInputError("unreadable_image") from exc
    if hashlib.sha256(data).hexdigest() != row.sha256:
        raise EvaluationInputError("image_hash_mismatch")
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise EvaluationInputError("unreadable_image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _counts(results: Sequence[ImageResult]) -> Counts:
    return Counts(len(results), sum(result.exact for result in results), sum(result.empty for result in results))


def evaluate_entries(
    reader: CropReader | SceneReader,
    entries: Sequence[AuditedPlate],
    mode: Literal["crop", "scene"],
    *,
    corner_policy: Literal["compat", "safe"] = "compat",
) -> EvalReport:
    if mode not in ("crop", "scene") or corner_policy not in ("compat", "safe"):
        raise EvaluationInputError("invalid_evaluation_mode")
    results: list[ImageResult] = []
    for row in entries:
        rgb = _read_verified_rgb(row)
        started = time.perf_counter()
        error: str | None = None
        predicted: tuple[str, ...] = ()
        timings: dict[str, float] = {}
        if mode == "crop":
            left, top, right, bottom = row.crop_xyxy
            if row.source_kind == "training_source_replay" and (left, top, right, bottom) == (0, 0, -1, -1):
                right, bottom = rgb.shape[1], rgb.shape[0]
            if not (0 <= left < right <= rgb.shape[1] and 0 <= top < bottom <= rgb.shape[0]):
                raise EvaluationInputError("invalid_crop")
            try:
                read = reader.recognize(rgb[top:bottom, left:right], corner_policy=corner_policy)
                predicted = (str(read.normalized_text),) if read.normalized_text else ()
                timings = dict(read.timings_ms)
            except Exception as exc:
                error = f"recognizer:{type(exc).__name__}"
        else:
            try:
                scene = reader.read(rgb)
                predicted = tuple(str(plate.read.normalized_text) for plate in scene.plates if plate.read.normalized_text)
                timings = dict(scene.timings_ms)
            except Exception as exc:
                error = f"scene:{type(exc).__name__}"
        timings["evaluation_total"] = (time.perf_counter() - started) * 1000
        results.append(ImageResult(
            image=str(row.image), canonical=row.canonical, predicted=predicted,
            exact=row.canonical in predicted, empty=not predicted,
            vehicle_class=row.vehicle_class, split=row.split,
            source_kind=row.source_kind, timings_ms=timings, error=error,
        ))
    results_tuple = tuple(results)
    # A truly independent score needs a disjoint, manually checked holdout;
    # training-source replay is never eligible regardless of its split label.
    held = [r for r in results if r.split == "holdout" and r.source_kind == "audited_unseen"]
    independent = f"{sum(r.exact for r in held)}/{len(held)}" if held else "pending"
    return EvalReport(
        mode=mode, overall=_counts(results_tuple),
        v1_passenger=_counts([r for r in results if r.vehicle_class == "v1_passenger"]),
        results=results_tuple,
        providers=tuple(getattr(reader, "providers", getattr(getattr(reader, "recognizer", None), "providers", ()))),
        hardware=platform.platform(),
        source_kinds=tuple(sorted({r.source_kind for r in results})),
        independent_accuracy=independent,
    )
