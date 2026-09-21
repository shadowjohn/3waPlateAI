"""Eager validation and collation for M1 v1 synthetic plate crops."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from plateai_shared.recognition import CTCCodec, preprocess_v1_rgb
from plateai_shared.rules import load_character_set, load_ruleset


_EXPECTED_SIZE = (380, 160)


class TrainingDataError(ValueError):
    """Raised before an epoch when M1 source records are not v1-compatible."""


@dataclass(frozen=True, slots=True)
class _ValidatedCrop:
    path: Path
    target: tuple[int, ...]


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise TrainingDataError(f"cannot read {path}: {exc}") from exc


def _load_json_object(path: Path, document_name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"invalid {document_name}: {path}") from exc
    if not isinstance(document, dict):
        raise TrainingDataError(f"{document_name}: must be a JSON object")
    return document


def _required_mapping(document: Mapping[str, Any], key: str, document_name: str) -> Mapping[str, Any]:
    value = document.get(key)
    if not isinstance(value, Mapping):
        raise TrainingDataError(f"{document_name}.{key}: must be an object")
    return value


def _validate_label_path(root: Path, raw_path: str) -> Path:
    relative = PurePosixPath(raw_path)
    if (
        not raw_path
        or "\\" in raw_path
        or relative.is_absolute()
        or PureWindowsPath(raw_path).is_absolute()
        or ".." in relative.parts
        or relative.parts[:1] != ("images",)
    ):
        raise TrainingDataError(f"labels.txt: unsafe image path {raw_path!r}")
    path = root.joinpath(*relative.parts)
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise TrainingDataError(f"labels.txt: missing or escaping image path {raw_path!r}") from exc
    if not path.is_file():
        raise TrainingDataError(f"labels.txt: image path is not a file {raw_path!r}")
    return path


class M1CropDataset:
    """A fully checked source directory with framework-neutral sample access."""

    def __init__(self, root: Path, charset_path: Path, rules_path: Path) -> None:
        self._root = Path(root).resolve()
        if not self._root.is_dir():
            raise TrainingDataError(f"dataset root is not a directory: {root}")
        try:
            charset = load_character_set(Path(charset_path))
        except ValueError as exc:
            raise TrainingDataError(f"invalid requested charset: {charset_path}") from exc
        self._codec = CTCCodec.from_charset(charset)
        try:
            ruleset = load_ruleset(Path(rules_path), charset)
        except ValueError as exc:
            raise TrainingDataError(f"invalid requested rules: {rules_path}") from exc
        self._allowed_plate_types = frozenset(rule.plate_type for rule in ruleset.rules)
        charset_hash = charset.sha256
        rules_hash = _sha256_file(Path(rules_path))

        configuration = _load_json_object(
            self._root / "generation_config.json", "generation_config.json"
        )
        summary = _load_json_object(self._root / "summary.json", "summary.json")
        config_hashes = _required_mapping(
            configuration, "config_sha256", "generation_config.json"
        )
        for key, expected in (("charset", charset_hash), ("rules", rules_hash)):
            if config_hashes.get(key) != expected:
                raise TrainingDataError(
                    f"generation_config.json.config_sha256.{key}: does not match requested input"
                )
            if summary.get(f"{key}_sha256") != expected:
                raise TrainingDataError(
                    f"summary.json.{key}_sha256: does not match requested input"
                )

        seed = configuration.get("seed")
        if type(seed) is not int:
            raise TrainingDataError("generation_config.json.seed: must be an integer")
        self._seed = seed
        self._records = self._validate_records()
        if configuration.get("count") != len(self._records):
            raise TrainingDataError("generation_config.json.count: does not match labels")
        if summary.get("generated") != len(self._records):
            raise TrainingDataError("summary.json.generated: does not match labels")

    @property
    def seed(self) -> int:
        """The M1 generation seed, after eager source validation."""

        return self._seed

    def _validate_records(self) -> tuple[_ValidatedCrop, ...]:
        labels_path = self._root / "labels.txt"
        metadata_path = self._root / "metadata.jsonl"
        try:
            label_lines = labels_path.read_text(encoding="utf-8").splitlines()
            metadata_lines = metadata_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise TrainingDataError("cannot read labels.txt or metadata.jsonl") from exc
        if not label_lines:
            raise TrainingDataError("labels.txt: must contain at least one record")
        if len(metadata_lines) != len(label_lines):
            raise TrainingDataError("metadata.jsonl: must contain one record per label")

        labels: list[tuple[str, str, Path, tuple[int, ...]]] = []
        seen_paths: set[str] = set()
        for line_number, line in enumerate(label_lines, start=1):
            if line.count("\t") != 1:
                raise TrainingDataError(f"labels.txt:{line_number}: requires exactly one tab")
            raw_path, canonical = line.split("\t")
            if raw_path in seen_paths:
                raise TrainingDataError(f"labels.txt:{line_number}: duplicate image path")
            if not canonical:
                raise TrainingDataError(f"labels.txt:{line_number}: canonical text is empty")
            try:
                target = self._codec.encode(canonical)
            except ValueError as exc:
                raise TrainingDataError(f"labels.txt:{line_number}: {exc}") from exc
            seen_paths.add(raw_path)
            labels.append((raw_path, canonical, _validate_label_path(self._root, raw_path), target))

        metadata_by_path: dict[str, Mapping[str, Any]] = {}
        for line_number, line in enumerate(metadata_lines, start=1):
            try:
                metadata = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataError(f"metadata.jsonl:{line_number}: invalid JSON") from exc
            if not isinstance(metadata, Mapping):
                raise TrainingDataError(f"metadata.jsonl:{line_number}: must be an object")
            image_path = metadata.get("image_path")
            if not isinstance(image_path, str) or image_path in metadata_by_path:
                raise TrainingDataError(f"metadata.jsonl:{line_number}: duplicate or invalid image_path")
            metadata_by_path[image_path] = metadata

        if set(metadata_by_path) != seen_paths:
            raise TrainingDataError("metadata.jsonl: image paths do not match labels.txt")

        records: list[_ValidatedCrop] = []
        labeled_paths: set[Path] = set()
        for raw_path, canonical, image_path, target in labels:
            metadata = metadata_by_path[raw_path]
            if metadata.get("canonical") != canonical:
                raise TrainingDataError(f"metadata.jsonl:{raw_path}: canonical text does not match")
            if metadata.get("plate_type") not in self._allowed_plate_types:
                raise TrainingDataError(f"metadata.jsonl:{raw_path}: incompatible plate_type")
            if metadata.get("image_sha256") != _sha256_file(image_path):
                raise TrainingDataError(f"metadata.jsonl:{raw_path}: PNG SHA-256 does not match")
            try:
                with Image.open(image_path) as image:
                    image.load()
                    if image.format != "PNG" or image.mode != "RGB" or image.size != _EXPECTED_SIZE:
                        raise TrainingDataError(
                            f"metadata.jsonl:{raw_path}: requires an RGB 380x160 PNG"
                        )
            except (OSError, ValueError) as exc:
                if isinstance(exc, TrainingDataError):
                    raise
                raise TrainingDataError(f"metadata.jsonl:{raw_path}: cannot read PNG") from exc
            labeled_paths.add(image_path.resolve())
            records.append(_ValidatedCrop(path=image_path, target=target))

        images_directory = self._root / "images"
        if not images_directory.is_dir():
            raise TrainingDataError("images: directory is missing")
        discovered_paths = {path.resolve() for path in images_directory.rglob("*") if path.is_file()}
        if discovered_paths != labeled_paths:
            raise TrainingDataError("images: contains unlabeled or missing image paths")
        return tuple(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[NDArray[np.float32], tuple[int, ...]]:
        record = self._records[index]
        try:
            with Image.open(record.path) as image:
                image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except OSError as exc:
            raise TrainingDataError(f"cannot reopen validated image {record.path}") from exc
        return preprocess_v1_rgb(image_rgb), record.target


def collate_crop_samples(
    samples: Sequence[tuple[NDArray[np.float32], tuple[int, ...]]],
):
    """Build Torch tensors while keeping the dataset module Torch-import-free."""

    if not samples:
        raise TrainingDataError("cannot collate an empty batch")
    import torch

    images, targets = zip(*samples, strict=True)
    image_batch = np.stack(images, axis=0)
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.int64)
    flattened_targets = torch.tensor(
        [index for target in targets for index in target], dtype=torch.int64
    )
    return torch.from_numpy(image_batch).to(dtype=torch.float32), flattened_targets, target_lengths
