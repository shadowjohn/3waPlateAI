"""Finalize a reviewed geometry queue as an immutable real-detection benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

from plateai_shared.publication import publish_directory_no_replace, remove_owned_staging
from plateai_trainer.detection.dataset import DetectionDataError
from plateai_trainer.detection.real_dataset import RealDetectionDataset, canonical_quad


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise DetectionDataError("invalid review workspace path")
    path = (root / relative).resolve()
    path.relative_to(root)
    if path.is_symlink():
        raise DetectionDataError("review workspace files must not be symlinks")
    return path


def _validate_quad(points, width: int, height: int):
    raw = np.asarray(points, dtype=np.float64)
    canonical = canonical_quad(points, width, height).astype(np.float64)
    if raw.shape != (4, 2) or not np.allclose(raw, canonical, rtol=0, atol=1e-4):
        raise DetectionDataError("reviewed corners must be semantic LT, RT, RB, LB")
    return [[float(value) for value in point] for point in canonical]


def finalize(workspace: Path, output: Path):
    workspace = workspace.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")

    review_set = _read_json(workspace / "review_set.json")
    if review_set.get("schema_version") != "geometry-validation-review-v1":
        raise DetectionDataError("invalid geometry review metadata")
    lines = (workspace / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows or len({row.get("validation_index") for row in rows}) != len(rows):
        raise DetectionDataError("empty or duplicate geometry review manifest")

    accepted = []
    snapshot_rows = []
    for row in rows:
        status = row.get("review_status")
        if status not in {"accepted", "excluded"}:
            raise DetectionDataError(
                f"validation index {row.get('validation_index')} is not reviewed"
            )
        if not isinstance(row.get("reviewed_at"), str) or not row["reviewed_at"]:
            raise DetectionDataError("reviewed records require reviewed_at")
        if status == "excluded" and not row.get("exclude_reason"):
            raise DetectionDataError("excluded records require exclude_reason")
        if status == "accepted" and row.get("exclude_reason") is not None:
            raise DetectionDataError("accepted records cannot have exclude_reason")

        image_path = _safe_path(workspace, row.get("image_path"))
        label_path = _safe_path(workspace, row.get("label_path"))
        payload = image_path.read_bytes()
        if _sha256(payload) != row.get("image_sha256"):
            raise DetectionDataError("review image SHA-256 mismatch")
        label = _read_json(label_path)
        identity_fields = ("validation_index", "source_index", "image_sha256", "original_label_sha256")
        if any(label.get(field) != row.get(field) for field in identity_fields):
            raise DetectionDataError("review label identity mismatch")
        original_hash = _sha256(_canonical_bytes({"corners": label.get("original_corners")}))
        if original_hash != row.get("original_label_sha256"):
            raise DetectionDataError("original review label SHA-256 mismatch")

        size = label.get("image_size_wh")
        if (not isinstance(size, list) or len(size) != 2
                or any(type(value) is not int or value < 1 for value in size)):
            raise DetectionDataError("invalid review image dimensions")
        width, height = size
        reviewed = label.get("reviewed_corners")
        if not isinstance(reviewed, list):
            raise DetectionDataError("reviewed_corners must be a list")
        canonical = [_validate_quad(points, width, height) for points in reviewed]
        if status == "accepted" and not canonical:
            raise DetectionDataError("accepted images require at least one reviewed plate")

        snapshot_rows.append({
            "manifest": row,
            "reviewed_corners": canonical,
        })
        if status == "accepted":
            accepted.append((row, payload, width, height, canonical))

    if not accepted:
        raise DetectionDataError("review contains no accepted images")

    snapshot_sha256 = _sha256(_canonical_bytes({
        "review_set": review_set,
        "records": snapshot_rows,
    }))
    source = review_set.get("source")
    upstream = source.get("dataset") if isinstance(source, dict) else None
    if (not isinstance(upstream, dict) or not isinstance(upstream.get("dataset_id"), str)
            or not isinstance(upstream.get("revision"), str)):
        raise DetectionDataError("review set is missing upstream dataset identity")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    (staging / "images").mkdir(parents=True)
    records = []
    try:
        for row, payload, width, height, corners in accepted:
            image_name = f"images/{row['validation_index']:06d}.img"
            (staging / image_name).write_bytes(payload)
            records.append({
                "source_index": row["source_index"],
                "validation_index": row["validation_index"],
                "image_path": image_name,
                "image_sha256": row["image_sha256"],
                "width": width,
                "height": height,
                "corners": corners,
            })
        (staging / "metadata.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records),
            encoding="utf-8",
            newline="\n",
        )
        _write_json(staging / "dataset.json", {
            "schema_version": "real-detection-v1",
            "source": {
                "dataset_id": f"{upstream['dataset_id']}/validation_geometry_clean",
                "revision": snapshot_sha256,
                "upstream": upstream,
            },
            "split": "validation",
            "count": len(records),
            "license_reviewed": bool(review_set.get("license_reviewed", False)),
            "local_only": True,
            "review": {
                "schema_version": review_set["schema_version"],
                "snapshot_sha256": snapshot_sha256,
                "total_images": len(rows),
                "accepted_images": len(accepted),
                "excluded_images": len(rows) - len(accepted),
                "plate_instances": sum(len(record[4]) for record in accepted),
            },
        })
        shutil.copyfile(workspace / "manifest.jsonl", staging / "review_manifest.jsonl")
        RealDetectionDataset(staging)
        publish_directory_no_replace(staging, output)
    except BaseException:
        remove_owned_staging(staging, output)
        raise
    return {
        "output": str(output),
        "snapshot_sha256": snapshot_sha256,
        "reviewed_images": len(rows),
        "accepted_images": len(accepted),
        "excluded_images": len(rows) - len(accepted),
        "plate_instances": sum(len(record[4]) for record in accepted),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = finalize(args.workspace, args.output)
    except (DetectionDataError, FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
