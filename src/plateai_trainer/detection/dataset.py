"""Eagerly validated Task 2 composites with Task 1 affine and Task 3 targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from plateai_shared.detection import LetterboxTransform, letterbox_rgb_v1, map_points_to_letterbox
from plateai_shared.schema_validation import DocumentValidationError, validate_document
from .contracts import CompositeInstance, _default_data_path
from .targets import DetectionTargets, assign_detection_targets


class DetectionDataError(ValueError):
    """Malformed or inconsistent local detector input."""


@dataclass(frozen=True, slots=True)
class DetectionSample:
    image: np.ndarray
    instances: tuple[CompositeInstance, ...]
    targets: DetectionTargets
    transform: LetterboxTransform


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DetectionDataError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(raw: bytes):
    def invalid(value):
        raise DetectionDataError(f"non-finite JSON value: {value}")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_object_pairs, parse_constant=invalid)
    except (UnicodeError, ValueError) as exc:
        raise DetectionDataError(f"invalid UTF-8 JSON: {exc}") from exc


def _hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _fields(document, fields, name):
    if not isinstance(document, dict) or set(document) != set(fields):
        raise DetectionDataError(f"{name}: invalid fields")


class DetectionDataset:
    """Validate all metadata and image bytes before any sample tensor conversion.

    Provenance files are snapshotted; image bytes are rehashed on each read so
    in-place changes cannot silently become training inputs. Original background
    files need not be available: their recorded identities define split isolation.
    """

    def __init__(self, directory: Path):
        self.root = Path(directory).resolve()
        self._files = {}
        try:
            raw_config = self._read("generation_config.json")
            raw_summary = self._read("summary.json")
            raw_metadata = self._read("metadata.jsonl")
            config, summary = _json(raw_config), _json(raw_summary)
            self._validate_provenance(config, summary)
            self._records = tuple(_json(line) for line in raw_metadata.splitlines())
            if not self._records or len(self._records) != config["count"]:
                raise DetectionDataError("metadata count does not match generation_config")
            backgrounds = {(item["image_path"], item["sha256"]) for item in config["backgrounds"]}
            for index, record in enumerate(self._records):
                validate_document(record, _default_data_path("schemas/detection_metadata.schema.json"))
                if type(record["schema_version"]) is not int or type(record["index"]) is not int:
                    raise DetectionDataError("metadata version/index must be integers")
                if record["index"] != index or record["image_path"] != f"images/{index:06d}.png":
                    raise DetectionDataError("metadata index/image path sequence mismatch")
                bg = record["background"]
                if (bg["image_path"], bg["sha256"]) not in backgrounds:
                    raise DetectionDataError("metadata background SHA-256 not in generation config")
                if any(type(bg[key]) is not int for key in ("width", "height")):
                    raise DetectionDataError("background dimensions must be integers")
                if not config["instances_per_image"][0] <= len(record["instances"]) <= config["instances_per_image"][1]:
                    raise DetectionDataError("instance count does not match generation config")
                self._image(record)
                for item in record["instances"]:
                    self._validate_instance(item, bg)
                boxes = np.asarray([item["bbox_xyxy"] for item in record["instances"]])
                for first in range(len(boxes)):
                    for second in range(first + 1, len(boxes)):
                        overlap = np.minimum(boxes[first, 2:], boxes[second, 2:]) - np.maximum(boxes[first, :2], boxes[second, :2])
                        if np.all(overlap > 0):
                            raise DetectionDataError("v1 instances must not overlap")
            expected = {self._safe_path(record["image_path"]) for record in self._records}
            found = {path.resolve() for path in (self.root / "images").rglob("*") if path.is_file()}
            if expected != found:
                raise DetectionDataError("images contains unlisted or missing files")
        except (OSError, DocumentValidationError) as exc:
            raise DetectionDataError(str(exc)) from exc
        self.background_sha256s = frozenset(item["sha256"] for item in config["backgrounds"])
        self.input_hashes = {"files": dict(self._files), "generation_inputs": dict(config["config_sha256"]),
                             "background_sha256s": sorted(self.background_sha256s)}

    def _safe_path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise DetectionDataError(f"input path escapes dataset: {name}")
        return path

    def _read(self, name):
        raw = self._safe_path(name).read_bytes()
        digest = _digest(raw)
        if name in self._files and digest != self._files[name]:
            raise DetectionDataError(f"input SHA-256 changed: {name}")
        self._files[name] = digest
        return raw

    def verify_unchanged(self):
        for name in tuple(self._files):
            try:
                self._read(name)
            except OSError as exc:
                raise DetectionDataError(f"input disappeared: {name}") from exc

    @staticmethod
    def _validate_provenance(config, summary):
        _fields(config, ("schema_version", "seed", "count", "instances_per_image", "composite_profile_id", "config_sha256", "backgrounds"), "generation_config")
        _fields(summary, ("schema_version", "generated", "seed", "background_sha256s"), "summary")
        for doc in (config, summary):
            if type(doc["schema_version"]) is not int or doc["schema_version"] != 1 or type(doc["seed"]) is not int or doc["seed"] < 0:
                raise DetectionDataError("invalid generation version/seed")
        if type(config["count"]) is not int or config["count"] < 1 or type(summary["generated"]) is not int or summary["generated"] != config["count"] or summary["seed"] != config["seed"]:
            raise DetectionDataError("generation config and summary count/seed mismatch")
        limits = config["instances_per_image"]
        if not isinstance(limits, list) or len(limits) != 2 or any(type(x) is not int for x in limits) or not 1 <= limits[0] <= limits[1] <= 3:
            raise DetectionDataError("instances_per_image must be 1-3")
        if not isinstance(config["composite_profile_id"], str) or not config["composite_profile_id"]:
            raise DetectionDataError("missing composite profile")
        hashes = config["config_sha256"]
        _fields(hashes, ("background_manifest", "composite_config", "charset", "rules", "template", "font"), "config_sha256")
        if any(not _hash(value) for key, value in hashes.items() if key != "font" or value is not None):
            raise DetectionDataError("invalid generation input SHA-256")
        backgrounds = config["backgrounds"]
        if not isinstance(backgrounds, list) or not backgrounds:
            raise DetectionDataError("backgrounds must not be empty")
        paths = set()
        for item in backgrounds:
            _fields(item, ("image_path", "sha256"), "background")
            if not isinstance(item["image_path"], str) or not item["image_path"] or item["image_path"] in paths or not _hash(item["sha256"]):
                raise DetectionDataError("invalid background identity/SHA-256")
            paths.add(item["image_path"])
        if summary["background_sha256s"] != sorted({item["sha256"] for item in backgrounds}):
            raise DetectionDataError("summary background SHA-256 mismatch")

    def _image(self, record):
        try:
            raw = self._read(record["image_path"])
            if _digest(raw) != record["image_sha256"]:
                raise DetectionDataError("image SHA-256 mismatch")
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                bg = record["background"]
                if image.format != "PNG" or image.mode != "RGB" or image.size != (bg["width"], bg["height"]):
                    raise DetectionDataError("image must be RGB PNG with recorded dimensions")
                return np.asarray(image, dtype=np.uint8).copy()
        except (OSError, ValueError) as exc:
            raise DetectionDataError(str(exc)) from exc

    @staticmethod
    def _validate_instance(item, bg):
        corners = np.asarray(item["corners"], dtype=np.float64)
        if np.any(corners < 0) or np.any(corners > [bg["width"] - 1, bg["height"] - 1]):
            raise DetectionDataError("corners are outside image")
        edges = np.roll(corners, -1, axis=0) - corners
        following = np.roll(edges, -1, axis=0)
        if np.any(edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0] <= 0):
            raise DetectionDataError("corners must be strictly convex clockwise semantic points")
        enclosure = np.r_[corners.min(axis=0), corners.max(axis=0)]
        if not np.array_equal(enclosure, item["bbox_xyxy"]):
            raise DetectionDataError("bbox must exactly enclose corners")
        source = item["source_plate"]
        source_quad = [[0., 0.], [379., 0.], [379., 159.], [0., 159.]]
        if source["template_id"] != "new-style-private-passenger-white-v1" or source["corners"] != source_quad or source["plate_type"] != "new-style-private-passenger":
            raise DetectionDataError("requires M1 v1 380x160 semantic source corners")
        if re.fullmatch(r"[A-HJ-NP-Z]{3}[0-35-9]{4}", source["canonical"]) is None or source["display"] != source["canonical"][:3] + "-" + source["canonical"][3:]:
            raise DetectionDataError("requires v1 canonical/display plate text")
        h = np.asarray(item["transform"]["homography"], dtype=np.float64)
        projected = np.c_[source_quad, np.ones(4)] @ h.T
        if abs(np.linalg.det(h)) < 1e-12 or np.any(np.abs(projected[:, 2]) < 1e-12):
            raise DetectionDataError("degenerate semantic homography")
        projected = projected[:, :2] / projected[:, 2:]
        # Task 2 stores a float32 homography after projecting with float64.
        if not np.allclose(projected, corners, rtol=0, atol=1e-3):
            raise DetectionDataError("semantic corner order disagrees with source homography")

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        record = self._records[index]
        image, transform = letterbox_rgb_v1(self._image(record))
        instances = tuple(CompositeInstance(
            tuple(float(x) for x in map_points_to_letterbox(np.asarray(item["bbox_xyxy"]).reshape(2, 2), transform).ravel()),
            map_points_to_letterbox(item["corners"], transform), item["source_plate"],
        ) for item in record["instances"])
        targets = assign_detection_targets(instances)
        return DetectionSample(image, instances, targets, transform)


def validate_train_validation_pair(train: DetectionDataset, validation: DetectionDataset):
    if train.background_sha256s & validation.background_sha256s:
        raise DetectionDataError("train/validation background SHA-256 sets must be disjoint")
