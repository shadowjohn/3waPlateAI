"""Prepare a deterministic, local-only geometry validation review set."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import uuid

import numpy as np

from plateai_shared.publication import publish_directory_no_replace, remove_owned_staging
from plateai_trainer.detection.dataset import DetectionDataError
from plateai_trainer.detection.real_dataset import canonical_quad


_BUCKETS = ("128-191", "192-255", "ge256")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _label_bytes(corners) -> bytes:
    return json.dumps(
        {"corners": corners},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _label_sha256(corners) -> str:
    return hashlib.sha256(_label_bytes(corners)).hexdigest()


def _size_bucket(short_side: float) -> str:
    if short_side < 128:
        raise ValueError("geometry-clean candidates require bbox short side >=128px")
    return "128-191" if short_side < 192 else "192-255" if short_side < 256 else "ge256"


def _geometry(corners, width: int, height: int):
    points = canonical_quad(corners, width, height).astype(np.float64)
    scale = min(640.0 / width, 640.0 / height)
    projected = points * scale
    size = projected.max(axis=0) - projected.min(axis=0)
    short_side = float(size.min())
    top_edge = projected[1] - projected[0]
    rotation = abs(math.degrees(math.atan2(float(top_edge[1]), float(top_edge[0]))))
    return short_side, rotation


def _candidate(index: int, record: dict):
    width, height = record["width"], record["height"]
    large = []
    for instance_index, corners in enumerate(record["corners"]):
        short_side, rotation = _geometry(corners, width, height)
        if short_side >= 128:
            large.append((instance_index, short_side, rotation))
    if not large:
        return None
    if len(large) != 1:
        raise DetectionDataError(
            f"validation index {index} has {len(large)} large instances; selection policy requires one"
        )
    instance_index, short_side, rotation = large[0]
    return {
        "validation_index": index,
        "record": record,
        "selection_instance_index": instance_index,
        "bbox_short_side_640px": short_side,
        "rotation_deg": rotation,
        "size_bucket": _size_bucket(short_side),
    }


def _sample(candidates, seed: int, target: int, expected_tilted: int, ge256_minimum: int):
    tilted = sorted((item for item in candidates if item["rotation_deg"] >= 15),
                    key=lambda item: item["validation_index"])
    if len(tilted) != expected_tilted:
        raise DetectionDataError(
            f"expected exactly {expected_tilted} fixed tilted images, found {len(tilted)}"
        )
    remaining_target = target - len(tilted)
    if remaining_target <= ge256_minimum:
        raise DetectionDataError("target is too small for fixed tilted images and ge256 minimum")
    tilted_indices = {item["validation_index"] for item in tilted}
    pools = {
        bucket: sorted(
            (item for item in candidates
             if item["validation_index"] not in tilted_indices and item["size_bucket"] == bucket),
            key=lambda item: item["validation_index"],
        )
        for bucket in _BUCKETS
    }
    if len(pools["ge256"]) < ge256_minimum:
        raise DetectionDataError("not enough non-tilted ge256 candidates for minimum guarantee")
    other_target = remaining_target - ge256_minimum
    other_count = len(pools["128-191"]) + len(pools["192-255"])
    if other_count < other_target:
        raise DetectionDataError("not enough non-tilted candidates")
    first = round(other_target * len(pools["128-191"]) / other_count)
    allocation = {
        "128-191": first,
        "192-255": other_target - first,
        "ge256": ge256_minimum,
    }
    rng = random.Random(seed)
    selected = []
    for item in tilted:
        selected.append({**item, "selection_group": "fixed_rotation_ge15"})
    for bucket in _BUCKETS:
        count = allocation[bucket]
        if len(pools[bucket]) < count:
            raise DetectionDataError(f"not enough {bucket} candidates for allocation {count}")
        for item in rng.sample(pools[bucket], count):
            selected.append({**item, "selection_group": "stratified_seeded"})
    selected.sort(key=lambda item: item["validation_index"])
    if len(selected) != target or len({item["validation_index"] for item in selected}) != target:
        raise AssertionError("selection count/uniqueness invariant failed")
    return selected, allocation


def prepare(source: Path, output: Path, seed: int, target: int,
            expected_tilted: int, ge256_minimum: int):
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    metadata = _read_json(source / "dataset.json")
    if metadata.get("schema_version") != "real-detection-v1" or metadata.get("split") != "validation":
        raise DetectionDataError("source must be a real-detection-v1 validation split")
    raw_lines = (source / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in raw_lines]
    if len(records) != metadata.get("count"):
        raise DetectionDataError("source metadata count mismatch")
    candidates = [item for index, record in enumerate(records)
                  if (item := _candidate(index, record)) is not None]
    selected, allocation = _sample(candidates, seed, target, expected_tilted, ge256_minimum)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    (staging / "images").mkdir(parents=True)
    (staging / "labels").mkdir()
    manifest = []
    try:
        for item in selected:
            index = item["validation_index"]
            record = item["record"]
            source_image = (source / record["image_path"]).resolve()
            source_image.relative_to(source)
            payload = source_image.read_bytes()
            image_sha = hashlib.sha256(payload).hexdigest()
            if image_sha != record["image_sha256"]:
                raise DetectionDataError(f"source image hash mismatch at validation index {index}")
            image_name = f"{index:06d}.img"
            label_name = f"{index:06d}.json"
            shutil.copyfile(source_image, staging / "images" / image_name)
            original_label_sha = _label_sha256(record["corners"])
            label = {
                "schema_version": "geometry-review-label-v1",
                "validation_index": index,
                "source_index": record["source_index"],
                "image_sha256": image_sha,
                "original_label_sha256": original_label_sha,
                "image_size_wh": [record["width"], record["height"]],
                "corner_order": ["left_top", "right_top", "right_bottom", "left_bottom"],
                "original_corners": record["corners"],
                "reviewed_corners": record["corners"],
            }
            _write_json(staging / "labels" / label_name, label)
            manifest.append({
                "validation_index": index,
                "source_index": record["source_index"],
                "image_path": f"images/{image_name}",
                "label_path": f"labels/{label_name}",
                "image_sha256": image_sha,
                "original_label_sha256": original_label_sha,
                "selection_instance_index": item["selection_instance_index"],
                "bbox_short_side_640px": item["bbox_short_side_640px"],
                "size_bucket": item["size_bucket"],
                "rotation_deg": item["rotation_deg"],
                "selection_group": item["selection_group"],
                "review_status": "pending",
                "exclude_reason": None,
                "reviewer_note": "",
                "reviewed_at": None,
            })
        (staging / "manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in manifest),
            encoding="utf-8",
            newline="\n",
        )
        _write_json(staging / "review_set.json", {
            "schema_version": "geometry-validation-review-v1",
            "status": "pending_manual_review",
            "source": {
                "directory": str(source),
                "dataset": metadata["source"],
                "split": "validation",
                "metadata_sha256": hashlib.sha256((source / "metadata.jsonl").read_bytes()).hexdigest(),
            },
            "selection": {
                "seed": seed,
                "target_images": target,
                "bbox_short_side_min_640px": 128,
                "fixed_rotation_min_degrees": 15,
                "fixed_rotation_count": expected_tilted,
                "remaining_seeded_count": target - expected_tilted,
                "remaining_allocation": allocation,
                "ge256_minimum_in_remaining": ge256_minimum,
            },
            "label_hash": "sha256(canonical compact sorted-key UTF-8 JSON object containing only corners)",
            "local_only": True,
            "license_reviewed": False,
            "boundary": "Pending image-level manual review; not a training or benchmark dataset until separately finalized.",
        })
        publish_directory_no_replace(staging, output)
    except BaseException:
        remove_owned_staging(staging, output)
        raise
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--expected-tilted", type=int, default=30)
    parser.add_argument("--ge256-minimum", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.validation, args.output, args.seed, args.target,
                           args.expected_tilted, args.ge256_minimum)
    except (DetectionDataError, FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "records": len(manifest),
        "pending": sum(row["review_status"] == "pending" for row in manifest),
        "fixed_tilted": sum(row["selection_group"] == "fixed_rotation_ge15" for row in manifest),
        "size_buckets": {bucket: sum(row["size_bucket"] == bucket for row in manifest) for bucket in _BUCKETS},
        "local_only": True,
    }, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
