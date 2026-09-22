"""Byte-bound loading for strict, local evaluation suites."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from plateai_reader.rectifier import InvalidCornersError, normalize_corners
from plateai_shared.contracts import JsonValue
from plateai_shared.schema_validation import DocumentValidationError, validate_document

from .contracts import (
    EvaluationInputError,
    EvaluationManifest,
    EvaluationSource,
    LaneName,
    PlateTruth,
    Scene,
    SourceRights,
    SuiteSnapshot,
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationInputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, label: str) -> dict[str, JsonValue]:
    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except EvaluationInputError as exc:
        raise EvaluationInputError(f"{label}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"{label}: invalid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise EvaluationInputError(f"{label}: document must be an object")
    return document


def _has_symlink_component(anchor: Path, candidate: Path) -> bool:
    try:
        relative = candidate.relative_to(anchor)
    except ValueError:
        return True
    cursor = anchor
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return True
    return False


def _resolve_beneath(
    root: Path,
    relative: str,
    field: str,
    *,
    kind: str,
) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or pure.as_posix() != relative
    ):
        raise EvaluationInputError(f"{field}: unsafe relative path")
    anchor = root.resolve()
    unresolved = root.joinpath(*pure.parts)
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(anchor)
    except ValueError as exc:
        raise EvaluationInputError(f"{field}: path escapes declared root") from exc
    if _has_symlink_component(root.absolute(), unresolved.absolute()):
        raise EvaluationInputError(f"{field}: symlinks are not allowed")
    if kind == "file" and not unresolved.is_file():
        raise EvaluationInputError(f"{field}: missing regular file")
    if kind == "directory" and not unresolved.is_dir():
        raise EvaluationInputError(f"{field}: missing directory")
    return resolved


def _read_hashed(path: Path, expected: str, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise EvaluationInputError(f"{label}: missing regular file")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EvaluationInputError(f"{label}: cannot read file") from exc
    if hashlib.sha256(payload).hexdigest() != expected:
        raise EvaluationInputError(f"{label} hash mismatch")
    return payload


def _validate(
    document: Mapping[str, JsonValue], schema_path: Path, label: str
) -> None:
    try:
        validate_document(document, schema_path)
    except DocumentValidationError as exc:
        raise EvaluationInputError(f"{label}: {exc}") from exc


def _decode_image(payload: bytes, label: str) -> NDArray[np.uint8]:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    try:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise EvaluationInputError(f"{label}: image decode failed") from exc
    if decoded is None or decoded.dtype != np.uint8 or decoded.ndim != 3:
        raise EvaluationInputError(f"{label}: image decode failed")
    return decoded


def load_scene_rgb(scene: Scene) -> NDArray[np.uint8]:
    """Re-read one bound scene image and return a contiguous RGB array."""

    payload = _read_hashed(scene.image_path, scene.image_sha256, "image")
    decoded = _decode_image(payload, scene.scene_id)
    height, width = decoded.shape[:2]
    if (width, height) != (scene.width, scene.height):
        raise EvaluationInputError(f"{scene.scene_id}: image dimensions mismatch")
    return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


def _plate_truth(
    document: Mapping[str, JsonValue],
    *,
    scene_id: str,
    width: int,
    height: int,
) -> PlateTruth:
    instance_id = str(document["instance_id"])
    try:
        normalized = normalize_corners(
            np.asarray(document["corners"], dtype=np.float32), (width, height)
        )
    except (InvalidCornersError, TypeError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, InvalidCornersError) else "invalid"
        raise EvaluationInputError(
            f"{scene_id}/{instance_id}: invalid corners ({reason})"
        ) from exc
    corners = tuple(
        (float(point[0]), float(point[1])) for point in normalized.points_xy
    )
    return PlateTruth(
        instance_id=instance_id,
        corners=corners,  # type: ignore[arg-type]
        canonical=(str(document["canonical"]) if "canonical" in document else None),
        display=(str(document["display"]) if "display" in document else None),
        geometry_audited=bool(document["geometry_audited"]),
        text_audited=bool(document["text_audited"]),
    )


def _enforce_lane_truth(scene: Scene, lanes: tuple[LaneName, ...]) -> None:
    if any(lane in {"oracle_ocr", "end_to_end"} for lane in lanes):
        if not scene.plates:
            raise EvaluationInputError(f"{scene.scene_id}: text lane requires a plate")
        for plate in scene.plates:
            if not plate.text_audited or not plate.canonical:
                raise EvaluationInputError(
                    f"{scene.scene_id}/{plate.instance_id}: audited canonical text required"
                )
    if lanes and any(not plate.geometry_audited for plate in scene.plates):
        raise EvaluationInputError(f"{scene.scene_id}: audited geometry required")


def _load_scene(
    document: Mapping[str, JsonValue],
    *,
    sources: Mapping[str, EvaluationSource],
    lanes: tuple[LaneName, ...],
    scene_schema: Path,
    label: str,
) -> Scene:
    _validate(document, scene_schema, label)
    scene_id = str(document["scene_id"])
    source_id = str(document["source_id"])
    if source_id not in sources:
        raise EvaluationInputError(f"{scene_id}: unknown source_id {source_id}")
    image_document = document["image"]
    assert isinstance(image_document, Mapping)
    image_path = _resolve_beneath(
        sources[source_id].root,
        str(image_document["path"]),
        f"{label}.image.path",
        kind="file",
    )
    image_sha256 = str(image_document["sha256"])
    image_payload = _read_hashed(image_path, image_sha256, "image")
    decoded = _decode_image(image_payload, scene_id)
    height, width = decoded.shape[:2]
    declared_width = int(image_document["width"])
    declared_height = int(image_document["height"])
    if (width, height) != (declared_width, declared_height):
        raise EvaluationInputError(f"{scene_id}: image dimensions mismatch")

    raw_plates = document["plates"]
    assert isinstance(raw_plates, list)
    plates: list[PlateTruth] = []
    instance_ids: set[str] = set()
    for raw_plate in raw_plates:
        assert isinstance(raw_plate, Mapping)
        instance_id = str(raw_plate["instance_id"])
        if instance_id in instance_ids:
            raise EvaluationInputError(
                f"{scene_id}: duplicate instance_id {instance_id}"
            )
        instance_ids.add(instance_id)
        plates.append(
            _plate_truth(
                raw_plate,
                scene_id=scene_id,
                width=declared_width,
                height=declared_height,
            )
        )
    scene = Scene(
        scene_id=scene_id,
        source_id=source_id,
        image_path=image_path,
        image_sha256=image_sha256,
        width=declared_width,
        height=declared_height,
        plates=tuple(plates),
    )
    _enforce_lane_truth(scene, lanes)
    return scene


def _snapshot_sha256(
    suite_sha256: str,
    manifests: tuple[EvaluationManifest, ...],
    scenes: tuple[Scene, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"suite\0" + bytes.fromhex(suite_sha256))
    for manifest in sorted(manifests, key=lambda item: item.manifest_id):
        digest.update(b"manifest\0" + manifest.manifest_id.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(manifest.sha256))
    for scene in sorted(scenes, key=lambda item: item.scene_id):
        digest.update(
            b"image\0"
            + scene.source_id.encode("utf-8")
            + b"\0"
            + scene.scene_id.encode("utf-8")
            + b"\0"
        )
        digest.update(bytes.fromhex(scene.image_sha256))
    return digest.hexdigest()


def load_suite(
    path: Path,
    suite_schema: Path,
    scene_schema: Path,
) -> SuiteSnapshot:
    """Load and bind a complete evaluation suite without inference fallbacks."""

    if path.is_symlink() or not path.is_file():
        raise EvaluationInputError("suite: missing regular file")
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise EvaluationInputError("suite: cannot read file") from exc
    suite_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    document = _parse_json(raw_bytes, "suite")
    _validate(document, suite_schema, "suite")
    suite_root = path.parent.resolve()

    raw_sources = document["sources"]
    assert isinstance(raw_sources, list)
    sources: list[EvaluationSource] = []
    sources_by_id: dict[str, EvaluationSource] = {}
    for raw_source in raw_sources:
        assert isinstance(raw_source, Mapping)
        source_id = str(raw_source["source_id"])
        if source_id in sources_by_id:
            raise EvaluationInputError(f"duplicate source_id: {source_id}")
        source_root = _resolve_beneath(
            suite_root, str(raw_source["root"]), "source.root", kind="directory"
        )
        raw_rights = raw_source["rights"]
        assert isinstance(raw_rights, Mapping)
        allowed_uses = raw_rights["allowed_uses"]
        assert isinstance(allowed_uses, list)
        source = EvaluationSource(
            source_id=source_id,
            root=source_root,
            revision=str(raw_source["revision"]),
            split=(str(raw_source["split"]) if "split" in raw_source else None),
            snapshot_sha256=(
                str(raw_source["snapshot_sha256"])
                if "snapshot_sha256" in raw_source
                else None
            ),
            rights=SourceRights(
                review_status=str(raw_rights["review_status"]),  # type: ignore[arg-type]
                local_only=bool(raw_rights["local_only"]),
                allowed_uses=frozenset(str(value) for value in allowed_uses),
                notice=str(raw_rights["notice"]),
            ),
        )
        sources.append(source)
        sources_by_id[source_id] = source

    raw_manifests = document["manifests"]
    assert isinstance(raw_manifests, list)
    manifests: list[EvaluationManifest] = []
    manifest_ids: set[str] = set()
    scenes: list[Scene] = []
    scenes_by_id: dict[str, tuple[Scene, str]] = {}
    image_hash_owners: dict[str, str] = {}
    lane_scene_ids: dict[LaneName, list[str]] = {
        "oracle_ocr": [],
        "detector": [],
        "end_to_end": [],
        "parity": [],
    }
    file_hashes: dict[Path, str] = {path.resolve(): suite_sha256}

    for raw_manifest in raw_manifests:
        assert isinstance(raw_manifest, Mapping)
        manifest_id = str(raw_manifest["manifest_id"])
        if manifest_id in manifest_ids:
            raise EvaluationInputError(f"duplicate manifest_id: {manifest_id}")
        manifest_ids.add(manifest_id)
        manifest_path = _resolve_beneath(
            suite_root,
            str(raw_manifest["file"]),
            "manifest.file",
            kind="file",
        )
        expected_manifest_sha = str(raw_manifest["sha256"])
        manifest_payload = _read_hashed(
            manifest_path, expected_manifest_sha, "manifest"
        )
        raw_lanes = raw_manifest["lanes"]
        assert isinstance(raw_lanes, list)
        lanes = tuple(str(lane) for lane in raw_lanes)
        manifest = EvaluationManifest(
            manifest_id=manifest_id,
            path=manifest_path,
            sha256=expected_manifest_sha,
            lanes=lanes,  # type: ignore[arg-type]
        )
        manifests.append(manifest)
        file_hashes[manifest_path] = expected_manifest_sha

        lines = manifest_payload.splitlines()
        if not lines:
            raise EvaluationInputError(f"manifest {manifest_id}: empty JSONL")
        for line_number, line in enumerate(lines, start=1):
            label = f"manifest {manifest_id} line {line_number}"
            if not line.strip():
                raise EvaluationInputError(f"{label}: blank line")
            raw_scene = _parse_json(line, label)
            scene = _load_scene(
                raw_scene,
                sources=sources_by_id,
                lanes=lanes,  # type: ignore[arg-type]
                scene_schema=scene_schema,
                label=label,
            )
            previous = scenes_by_id.get(scene.scene_id)
            if previous is not None:
                previous_scene, previous_manifest_id = previous
                if previous_manifest_id == manifest_id:
                    raise EvaluationInputError(f"duplicate scene_id: {scene.scene_id}")
                if previous_scene != scene:
                    raise EvaluationInputError(
                        f"incompatible duplicate scene_id: {scene.scene_id}"
                    )
            else:
                owner = image_hash_owners.get(scene.image_sha256)
                if owner is not None:
                    raise EvaluationInputError(
                        f"duplicate image sha256: {scene.scene_id} and {owner}"
                    )
                image_hash_owners[scene.image_sha256] = scene.scene_id
                scenes_by_id[scene.scene_id] = (scene, manifest_id)
                scenes.append(scene)
                file_hashes[scene.image_path] = scene.image_sha256
            for lane in lanes:
                lane_ids = lane_scene_ids[lane]  # type: ignore[index]
                if scene.scene_id not in lane_ids:
                    lane_ids.append(scene.scene_id)

    manifest_tuple = tuple(manifests)
    scene_tuple = tuple(scenes)
    snapshot_sha256 = _snapshot_sha256(
        suite_sha256, manifest_tuple, scene_tuple
    )
    return SuiteSnapshot(
        path=path.resolve(),
        raw_bytes=raw_bytes,
        sha256=suite_sha256,
        snapshot_sha256=snapshot_sha256,
        suite_id=str(document["suite_id"]),
        revision=str(document["revision"]),
        sources=tuple(sources),
        manifests=manifest_tuple,
        scenes=scene_tuple,
        lane_scene_ids=MappingProxyType(
            {lane: tuple(scene_ids) for lane, scene_ids in lane_scene_ids.items()}
        ),
        file_hashes=MappingProxyType(file_hashes),
    )


__all__ = ["load_scene_rgb", "load_suite"]
