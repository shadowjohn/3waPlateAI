"""Transactional deterministic multi-plate composite generation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from plateai_shared.publication import (
    OutputExistsError,
    PublicationError,
    publish_directory_no_replace,
    remove_owned_staging,
)
from plateai_shared.rules import generate_plate, load_character_set, load_ruleset
from plateai_shared.schema_validation import validate_document
from plateai_trainer.synthetic.encoder import encode_png
from plateai_trainer.synthetic.fonts import resolve_font
from plateai_trainer.synthetic.renderer import render_plate
from plateai_trainer.synthetic.templates import load_template

from .contracts import (
    CompositeGenerationRequest,
    CompositeGenerationSummary,
    _default_data_path,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_BACKGROUND_SCHEMA = _REPOSITORY_ROOT / "schemas/detection_background_manifest.schema.json"
_METADATA_SCHEMA = _REPOSITORY_ROOT / "schemas/detection_metadata.schema.json"


class CompositeGenerationError(PublicationError):
    """Base class for stable composite-generation failures."""


class InvalidCompositeRequest(CompositeGenerationError, ValueError):
    """Raised when a composite request or its local inputs are invalid."""


@dataclass(frozen=True, slots=True)
class _Background:
    manifest_path: str
    sha256: str
    image_rgb: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _CompositeConfig:
    id: str
    scale_range: tuple[float, float]
    rotation_degrees: tuple[float, float]
    perspective_jitter_ratio: float
    placement_attempts: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise InvalidCompositeRequest(f"cannot read file {path}: {exc}") from exc


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_bytes())
    except OSError as exc:
        raise InvalidCompositeRequest(f"cannot read {description} {path}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCompositeRequest(f"invalid UTF-8 JSON {description}: {path}") from exc
    if not isinstance(document, dict):
        raise InvalidCompositeRequest(f"{description} must be a JSON object")
    return document


def _numeric_pair(value: Any, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) not in {int, float} or not math.isfinite(item) for item in value)
    ):
        raise InvalidCompositeRequest(f"{name} must contain two finite numbers")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise InvalidCompositeRequest(f"{name} must be ordered")
    return low, high


def _load_config(path: Path) -> _CompositeConfig:
    document = _read_json(path, "composite config")
    if document.get("schema_version") != 1:
        raise InvalidCompositeRequest("composite config schema_version must be 1")
    config_id = document.get("id")
    if not isinstance(config_id, str) or not config_id:
        raise InvalidCompositeRequest("composite config id must not be empty")
    scales = _numeric_pair(document.get("scale_range"), "scale_range")
    if scales[0] <= 0.0:
        raise InvalidCompositeRequest("scale_range must be positive")
    rotations = _numeric_pair(document.get("rotation_degrees"), "rotation_degrees")
    jitter = document.get("perspective_jitter_ratio")
    if type(jitter) not in {int, float} or not math.isfinite(jitter) or not 0 <= jitter <= 0.25:
        raise InvalidCompositeRequest(
            "perspective_jitter_ratio must be finite and between 0 and 0.25"
        )
    attempts = document.get("placement_attempts")
    if type(attempts) is not int or attempts < 1:
        raise InvalidCompositeRequest("placement_attempts must be a positive integer")
    return _CompositeConfig(
        id=config_id,
        scale_range=scales,
        rotation_degrees=rotations,
        perspective_jitter_ratio=float(jitter),
        placement_attempts=attempts,
    )


def _load_and_hash_background_manifest(path: Path) -> tuple[_Background, ...]:
    document = _read_json(path, "background manifest")
    validate_document(
        document,
        _default_data_path("schemas/detection_background_manifest.schema.json"),
    )
    manifest_root = path.resolve().parent
    backgrounds: list[_Background] = []
    for entry in document["backgrounds"]:
        raw_path = entry["image_path"]
        windows_path = PureWindowsPath(raw_path)
        posix_path = PurePosixPath(raw_path)
        if windows_path.drive or windows_path.root or posix_path.root:
            raise InvalidCompositeRequest(
                f"background image path must be relative without a root, drive, or anchor: {raw_path}"
            )
        relative = Path(raw_path)
        resolved = (manifest_root / relative).resolve()
        try:
            resolved.relative_to(manifest_root)
        except ValueError as exc:
            raise InvalidCompositeRequest(
                f"background image path escapes manifest directory: {relative.as_posix()}"
            ) from exc
        try:
            encoded = resolved.read_bytes()
        except OSError as exc:
            raise InvalidCompositeRequest(
                f"cannot read background image {relative.as_posix()}: {exc}"
            ) from exc
        actual_sha256 = _sha256_bytes(encoded)
        if actual_sha256 != entry["sha256"]:
            raise InvalidCompositeRequest(
                f"background SHA-256 mismatch: {relative.as_posix()}"
            )
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
            raise InvalidCompositeRequest(
                f"background image is not a decodable RGB image: {relative.as_posix()}"
            )
        backgrounds.append(
            _Background(
                manifest_path=relative.as_posix(),
                sha256=actual_sha256,
                image_rgb=np.ascontiguousarray(
                    cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), dtype=np.uint8
                ),
            )
        )
    return tuple(backgrounds)


def _validate_request(request: CompositeGenerationRequest) -> None:
    if type(request.count) is not int or request.count < 1:
        raise InvalidCompositeRequest("count must be a positive integer")
    if type(request.seed) is not int:
        raise InvalidCompositeRequest("seed must be an integer")
    minimum, maximum = request.instances_per_image
    if (
        type(minimum) is not int
        or type(maximum) is not int
        or not 1 <= minimum <= maximum <= 3
    ):
        raise InvalidCompositeRequest(
            "instances_per_image must be an ordered pair between one and three"
        )
    if request.output.exists():
        raise OutputExistsError(f"output already exists: {request.output}")
    ancestor = request.output.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if ancestor.exists() and not ancestor.is_dir():
        raise InvalidCompositeRequest(
            f"output parent exists but is not a directory: {ancestor}"
        )


def _create_owned_staging(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise OutputExistsError(f"output already exists: {output}")
    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    (staging / "images").mkdir()
    return staging


def _snapshot_input_hashes(request: CompositeGenerationRequest, font) -> dict[str, str | None]:
    return {
        "background_manifest": _sha256_file(request.background_manifest),
        "composite_config": _sha256_file(request.config_path),
        "charset": _sha256_file(request.charset_path),
        "rules": _sha256_file(request.rules_path),
        "template": _sha256_file(request.template_path),
        "font": _sha256_file(font.path) if font.path is not None else None,
    }


def _verify_input_hashes(
    request: CompositeGenerationRequest,
    font,
    expected: dict[str, str | None],
) -> None:
    current = _snapshot_input_hashes(request, font)
    changed = sorted(key for key in expected if current[key] != expected[key])
    if changed:
        raise InvalidCompositeRequest(
            f"input changed during generation: {', '.join(changed)}"
        )


def _bbox(corners: NDArray[np.float32]) -> tuple[float, float, float, float]:
    return (
        float(np.min(corners[:, 0])),
        float(np.min(corners[:, 1])),
        float(np.max(corners[:, 0])),
        float(np.max(corners[:, 1])),
    )


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    if intersection == 0.0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _sample_homography(
    rng: np.random.Generator,
    source_corners: NDArray[np.float32],
    canvas_width: int,
    canvas_height: int,
    config: _CompositeConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
    scale = float(rng.uniform(*config.scale_range))
    angle = math.radians(float(rng.uniform(*config.rotation_degrees)))
    center = np.mean(source_corners, axis=0)
    centered = (source_corners - center) * scale
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    destination = centered @ rotation.T
    jitter = min(
        float(np.ptp(destination[:, 0])), float(np.ptp(destination[:, 1]))
    ) * config.perspective_jitter_ratio
    destination += rng.uniform(-jitter, jitter, size=(4, 2)).astype(np.float32)
    minimum = np.min(destination, axis=0)
    maximum = np.max(destination, axis=0)
    available_x = canvas_width - 1.0 - float(maximum[0] - minimum[0])
    available_y = canvas_height - 1.0 - float(maximum[1] - minimum[1])
    if available_x < 0.0 or available_y < 0.0:
        return None
    destination += np.asarray(
        [
            float(rng.uniform(0.0, available_x)) - float(minimum[0]),
            float(rng.uniform(0.0, available_y)) - float(minimum[1]),
        ],
        dtype=np.float32,
    )
    cross_products = []
    for index in range(4):
        first = destination[(index + 1) % 4] - destination[index]
        second = destination[(index + 2) % 4] - destination[(index + 1) % 4]
        cross_products.append(float(first[0] * second[1] - first[1] * second[0]))
    if any(value <= 0.0 for value in cross_products):
        return None
    homography = cv2.getPerspectiveTransform(source_corners, destination)
    transformed = cv2.perspectiveTransform(
        source_corners[None, :, :], homography
    )[0].astype(np.float32)
    if not np.isfinite(transformed).all():
        return None
    return homography.astype(np.float32), transformed


def _composite_plate(
    canvas: NDArray[np.uint8],
    plate_rgb: NDArray[np.uint8],
    homography: NDArray[np.float32],
) -> None:
    height, width = canvas.shape[:2]
    warped = cv2.warpPerspective(
        plate_rgb,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    source_mask = np.full(plate_rgb.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        source_mask,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    alpha = mask.astype(np.float32)[:, :, None] / 255.0
    canvas[:] = np.rint(warped * alpha + canvas * (1.0 - alpha)).astype(np.uint8)


def _compose_one_image(
    backgrounds: tuple[_Background, ...],
    rng: np.random.Generator,
    request: CompositeGenerationRequest,
    images_dir: Path,
    index: int,
    config: _CompositeConfig,
    ruleset,
    template,
    font,
) -> dict[str, Any]:
    background = backgrounds[int(rng.integers(0, len(backgrounds)))]
    canvas = background.image_rgb.copy()
    canvas_height, canvas_width = canvas.shape[:2]
    instance_count = int(
        rng.integers(request.instances_per_image[0], request.instances_per_image[1] + 1)
    )
    instances: list[dict[str, Any]] = []
    accepted_boxes: list[tuple[float, float, float, float]] = []
    for _ in range(instance_count):
        plate_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        sample = generate_plate(ruleset, random.Random(plate_seed))
        rendered = render_plate(sample, template, font)
        placement = None
        for _attempt in range(config.placement_attempts):
            candidate = _sample_homography(
                rng,
                rendered.corners,
                canvas_width,
                canvas_height,
                config,
            )
            if candidate is None:
                continue
            homography, transformed = candidate
            candidate_box = _bbox(transformed)
            if all(_bbox_iou(candidate_box, existing) <= 0.0 for existing in accepted_boxes):
                placement = homography, transformed, candidate_box
                break
        if placement is None:
            raise CompositeGenerationError(
                f"could not place {instance_count} non-overlapping plates on "
                f"{background.manifest_path} after {config.placement_attempts} attempts"
            )
        homography, transformed, candidate_box = placement
        _composite_plate(canvas, rendered.image_rgb, homography)
        accepted_boxes.append(candidate_box)
        instances.append(
            {
                "bbox_xyxy": list(candidate_box),
                "corners": transformed.tolist(),
                "source_plate": {
                    "canonical": sample.canonical,
                    "display": sample.display,
                    "rule_id": sample.rule_id,
                    "plate_type": sample.plate_type,
                    "template_id": template.id,
                    "plate_seed": plate_seed,
                    "corners": rendered.corners.tolist(),
                    "renderer": {
                        "font_kind": rendered.metadata["font_kind"],
                        "font_name": rendered.metadata["font_name"],
                        "font_variation_axes": rendered.metadata[
                            "font_variation_axes"
                        ],
                    },
                },
                "transform": {"homography": homography.tolist()},
            }
        )
    encoded = encode_png(canvas)
    image_path = f"images/{index:06d}.png"
    (images_dir.parent / image_path).write_bytes(encoded)
    record = {
        "schema_version": 1,
        "index": index,
        "image_path": image_path,
        "image_sha256": _sha256_bytes(encoded),
        "background": {
            "image_path": background.manifest_path,
            "sha256": background.sha256,
            "width": canvas_width,
            "height": canvas_height,
        },
        "instances": instances,
    }
    validate_document(
        record,
        _default_data_path("schemas/detection_metadata.schema.json"),
    )
    for instance in instances:
        corners = np.asarray(instance["corners"], dtype=np.float32)
        if (
            np.any(corners[:, 0] > canvas_width - 1)
            or np.any(corners[:, 1] > canvas_height - 1)
        ):
            raise CompositeGenerationError("generated corner is outside its background")
    return record


def _json_text(value: Any, *, compact: bool = False) -> str:
    if compact:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(_json_text(record, compact=True) + "\n")


def generate_composite_dataset(
    request: CompositeGenerationRequest,
) -> CompositeGenerationSummary:
    """Generate a complete composite dataset and publish it without replacement."""

    _validate_request(request)
    font = resolve_font(request.font)
    hashes = _snapshot_input_hashes(request, font)
    backgrounds = _load_and_hash_background_manifest(request.background_manifest)
    config = _load_config(request.config_path)
    charset = load_character_set(request.charset_path)
    ruleset = load_ruleset(request.rules_path, charset)
    template = load_template(request.template_path)
    if (
        template.id != "new-style-private-passenger-white-v1"
        or template.width != 380
        or template.height != 160
    ):
        raise InvalidCompositeRequest(
            "M3b composites require the M1 v1 template "
            "new-style-private-passenger-white-v1 at 380x160"
        )
    _verify_input_hashes(request, font, hashes)
    rng = np.random.default_rng(request.seed)
    staging = _create_owned_staging(request.output)
    try:
        records = [
            _compose_one_image(
                backgrounds,
                rng,
                request,
                staging / "images",
                index,
                config,
                ruleset,
                template,
                font,
            )
            for index in range(request.count)
        ]
        _write_jsonl(staging / "metadata.jsonl", records)
        _verify_input_hashes(request, font, hashes)
        generation_config = {
            "schema_version": 1,
            "seed": request.seed,
            "count": request.count,
            "instances_per_image": list(request.instances_per_image),
            "composite_profile_id": config.id,
            "config_sha256": hashes,
            "backgrounds": [
                {"image_path": item.manifest_path, "sha256": item.sha256}
                for item in backgrounds
            ],
        }
        (staging / "generation_config.json").write_text(
            _json_text(generation_config), encoding="utf-8", newline="\n"
        )
        summary_document = {
            "schema_version": 1,
            "generated": request.count,
            "seed": request.seed,
            "background_sha256s": sorted({item.sha256 for item in backgrounds}),
        }
        (staging / "summary.json").write_text(
            _json_text(summary_document), encoding="utf-8", newline="\n"
        )
        publish_directory_no_replace(staging, request.output)
    except BaseException:
        remove_owned_staging(staging, request.output)
        raise
    return CompositeGenerationSummary(
        generated=request.count,
        output=request.output,
        seed=request.seed,
        background_sha256s=tuple(summary_document["background_sha256s"]),
    )
