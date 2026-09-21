"""Transactional orchestration for deterministic synthetic datasets."""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Any

from plateai_shared.publication import (
    OutputExistsError,
    PublicationError,
    publish_directory_no_replace,
    remove_owned_staging,
)
from plateai_shared.rules import generate_plate, load_character_set, load_ruleset

from .augment import apply_augmentations, derive_sample_seed, load_augment_profile
from .encoder import encode_png
from .fonts import resolve_font
from .models import (
    GenerationRecord,
    GenerationRequest,
    GenerationSummary,
    ImageEncoder,
)
from .renderer import render_plate
from .templates import load_template


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


GenerationError = PublicationError


class InvalidGenerationRequest(GenerationError, ValueError):
    """Raised before generation when a request cannot be honored safely."""


# Compatibility seams for M1 callers and its race-condition test.
_publish_no_replace = publish_directory_no_replace
_safe_remove_staging = remove_owned_staging


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise InvalidGenerationRequest(f"cannot read configuration file {path}: {exc}") from exc


def _portable_config_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.name


def _json_text(value: Any, *, compact: bool = False) -> str:
    if compact:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _record_document(record: GenerationRecord) -> dict[str, Any]:
    return {
        "schema_version": record.schema_version,
        "index": record.index,
        "sample_seed": record.sample_seed,
        "image_path": record.image_path,
        "canonical": record.canonical,
        "display": record.display,
        "rule_id": record.rule_id,
        "plate_type": record.plate_type,
        "corners": [list(point) for point in record.corners],
        "renderer": dict(record.renderer),
        "augmentation": dict(record.augmentation),
        "image_sha256": record.image_sha256,
    }


def _summary_document(summary: GenerationSummary) -> dict[str, Any]:
    return {
        "schema_version": summary.schema_version,
        "generated": summary.generated,
        "seed": summary.seed,
        "rule_counts": dict(summary.rule_counts),
        "plate_type_counts": dict(summary.plate_type_counts),
        "charset_sha256": summary.charset_sha256,
        "rules_sha256": summary.rules_sha256,
        "template_sha256": summary.template_sha256,
        "augmentation_sha256": summary.augmentation_sha256,
    }


def _validate_output_request(request: GenerationRequest) -> None:
    if type(request.count) is not int or request.count < 1:
        raise InvalidGenerationRequest("count must be an integer greater than zero")
    if type(request.seed) is not int:
        raise InvalidGenerationRequest("seed must be an integer")
    if request.output.exists():
        raise OutputExistsError(f"output already exists: {request.output}")

    ancestor = request.output.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if ancestor.exists() and not ancestor.is_dir():
        raise InvalidGenerationRequest(
            f"output parent exists but is not a directory: {ancestor}"
        )


def generate_dataset(
    request: GenerationRequest,
    *,
    encoder: ImageEncoder = encode_png,
) -> GenerationSummary:
    """Generate into an owned sibling directory, then publish once complete."""

    _validate_output_request(request)
    charset = load_character_set(request.charset_path)
    ruleset = load_ruleset(request.rules_path, charset)
    template = load_template(request.template_path)
    profile = load_augment_profile(request.augmentation_path)
    font = resolve_font(request.font)

    hashes = {
        "charset": charset.sha256,
        "rules": _sha256_file(request.rules_path),
        "template": _sha256_file(request.template_path),
        "augmentation": _sha256_file(request.augmentation_path),
    }
    font_sha256 = _sha256_file(font.path) if font.path is not None else None
    request.output.parent.mkdir(parents=True, exist_ok=True)
    if request.output.exists():
        raise OutputExistsError(f"output already exists: {request.output}")

    staging = request.output.parent / (
        f".{request.output.name}.partial-{uuid.uuid4().hex}"
    )
    staging.mkdir(exist_ok=False)
    try:
        images = staging / "images"
        images.mkdir()
        config_document = {
            "schema_version": 1,
            "seed": request.seed,
            "count": request.count,
            "font": font.name,
            "font_variation_axes": list(font.variation_axes),
            "font_sha256": font_sha256,
            "config_paths": {
                "charset": _portable_config_path(request.charset_path),
                "rules": _portable_config_path(request.rules_path),
                "template": _portable_config_path(request.template_path),
                "augmentation": _portable_config_path(request.augmentation_path),
            },
            "config_sha256": hashes,
        }
        (staging / "generation_config.json").write_text(
            _json_text(config_document), encoding="utf-8"
        )

        rule_counts: Counter[str] = Counter()
        plate_type_counts: Counter[str] = Counter()
        labels_path = staging / "labels.txt"
        metadata_path = staging / "metadata.jsonl"
        with labels_path.open("w", encoding="utf-8", newline="\n") as labels_file, metadata_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as metadata_file:
            for index in range(request.count):
                sample_seed = derive_sample_seed(request.seed, index)
                sample = generate_plate(ruleset, random.Random(sample_seed))
                rendered = render_plate(sample, template, font)
                augmented = apply_augmentations(rendered, profile, sample_seed)
                encoded = encoder(augmented.image_rgb)
                if not isinstance(encoded, bytes) or not encoded:
                    raise GenerationError("image encoder must return non-empty bytes")

                image_path = f"images/{index:06d}.png"
                (staging / image_path).write_bytes(encoded)
                labels_file.write(f"{image_path}\t{sample.canonical}\n")

                augmentation_keys = (
                    "augmentation_profile_id",
                    "sample_seed",
                    "geometry",
                    "photometric",
                )
                record = GenerationRecord(
                    schema_version=1,
                    index=index,
                    sample_seed=sample_seed,
                    image_path=image_path,
                    canonical=sample.canonical,
                    display=sample.display,
                    rule_id=sample.rule_id,
                    plate_type=sample.plate_type,
                    corners=tuple(
                        (float(point[0]), float(point[1]))
                        for point in augmented.corners
                    ),
                    renderer=MappingProxyType(dict(rendered.metadata)),
                    augmentation=MappingProxyType(
                        {
                            key: augmented.metadata[key]
                            for key in augmentation_keys
                        }
                    ),
                    image_sha256=hashlib.sha256(encoded).hexdigest(),
                )
                metadata_file.write(_json_text(_record_document(record), compact=True) + "\n")
                rule_counts[sample.rule_id] += 1
                plate_type_counts[sample.plate_type] += 1

        summary = GenerationSummary(
            schema_version=1,
            generated=request.count,
            seed=request.seed,
            rule_counts=MappingProxyType(dict(sorted(rule_counts.items()))),
            plate_type_counts=MappingProxyType(
                dict(sorted(plate_type_counts.items()))
            ),
            charset_sha256=hashes["charset"],
            rules_sha256=hashes["rules"],
            template_sha256=hashes["template"],
            augmentation_sha256=hashes["augmentation"],
        )
        (staging / "summary.json").write_text(
            _json_text(_summary_document(summary)), encoding="utf-8"
        )

        if request.output.exists():
            raise OutputExistsError(f"output already exists: {request.output}")
        _publish_no_replace(staging, request.output)
        return summary
    except BaseException:
        _safe_remove_staging(staging, request.output)
        raise
