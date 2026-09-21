#!/usr/bin/env python3
"""Fetch the pinned EZCon Taiwan plate-recognition test split into an ignored folder.

The upstream Dataset Card exposes image, rotated-box, plate-text, and EV fields,
but it does not publish a dataset license.  This tool therefore requires an
explicit acknowledgement and records that unresolved status beside the files.
It never writes into the source tree's tracked fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any
from urllib.request import Request, urlopen
import uuid


DATASET_ID = "EZCon/taiwan-license-plate-recognition"
DATASET_REVISION = "15fd0d583d88c4a87a837cc2ef31b15c6a1b3719"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
API_URL = f"https://huggingface.co/api/datasets/{DATASET_ID}"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
SPLIT = "test"
PAGE_SIZE = 100
MANIFEST_SCHEMA_VERSION = "external-plate-evaluation-manifest-v1"

# The current Reader v1 accepts only LLL-DDDD and excludes I, O, and 4.
V1_DISPLAY_PATTERN = re.compile(r"^[A-HJ-NP-Z]{3}-[0-35-9]{4}$")


def request_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "3waPlateAI-evaluation-fetch/1"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def request_json(url: str) -> dict[str, Any]:
    try:
        document = json.loads(request_bytes(url).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Expected JSON from {url}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return document


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def assert_source_revision(metadata: dict[str, Any]) -> None:
    revision = metadata.get("sha")
    if revision != DATASET_REVISION:
        raise RuntimeError(
            "Upstream revision changed from the pinned value. Review its license, "
            "schema, and contents before updating this tool: "
            f"expected {DATASET_REVISION}, received {revision!r}."
        )
    if metadata.get("id") != DATASET_ID:
        raise RuntimeError(f"Unexpected dataset identity: {metadata.get('id')!r}")
    if metadata.get("gated") or metadata.get("private"):
        raise RuntimeError("The pinned source is no longer a public, ungated dataset.")


def fetch_rows() -> list[dict[str, Any]]:
    first_page = request_json(
        f"{ROWS_URL}?dataset={DATASET_ID.replace('/', '%2F')}&config=default"
        f"&split={SPLIT}&offset=0&length={PAGE_SIZE}"
    )
    total = first_page.get("num_rows_total")
    rows = first_page.get("rows")
    if not isinstance(total, int) or total <= 0 or not isinstance(rows, list):
        raise RuntimeError("The source did not provide a usable test-row page.")

    collected = list(rows)
    for offset in range(PAGE_SIZE, total, PAGE_SIZE):
        page = request_json(
            f"{ROWS_URL}?dataset={DATASET_ID.replace('/', '%2F')}&config=default"
            f"&split={SPLIT}&offset={offset}&length={PAGE_SIZE}"
        )
        if page.get("num_rows_total") != total or not isinstance(page.get("rows"), list):
            raise RuntimeError("The source row count changed during retrieval.")
        collected.extend(page["rows"])

    if len(collected) != total:
        raise RuntimeError(f"Expected {total} rows but retrieved {len(collected)}.")
    return collected


def image_url_and_ground_truth(item: dict[str, Any], expected_index: int) -> tuple[str, dict[str, Any]]:
    if item.get("row_idx") != expected_index:
        raise RuntimeError(f"Expected source row {expected_index}, received {item.get('row_idx')!r}.")
    row = item.get("row")
    if not isinstance(row, dict):
        raise RuntimeError(f"Source row {expected_index} has no object payload.")
    image = row.get("image")
    if not isinstance(image, dict) or not isinstance(image.get("src"), str):
        raise RuntimeError(f"Source row {expected_index} has no image URL.")
    label = row.get("license_number")
    rotated_box = row.get("xywhr")
    electric = row.get("is_electric_car")
    if not isinstance(label, str) or not label or not isinstance(rotated_box, list) or len(rotated_box) != 5:
        raise RuntimeError(f"Source row {expected_index} has an invalid label or rotated box.")
    if not isinstance(electric, bool):
        raise RuntimeError(f"Source row {expected_index} has an invalid EV flag.")
    if image.get("width") != 640 or image.get("height") != 640:
        raise RuntimeError(f"Source row {expected_index} no longer has the expected 640x640 image.")
    if f"/{DATASET_REVISION}/" not in image["src"]:
        raise RuntimeError(f"Source row {expected_index} image is not pinned to the reviewed revision.")
    if not all(isinstance(value, (int, float)) for value in rotated_box):
        raise RuntimeError(f"Source row {expected_index} has a non-numeric rotated box.")
    return image["src"], {
        "canonical": label.replace("-", ""),
        "display": label,
        "is_electric_car": electric,
        "xywhr": rotated_box,
    }


def fetch_dataset(output: Path) -> dict[str, int]:
    metadata = request_json(API_URL)
    assert_source_revision(metadata)
    rows = fetch_rows()

    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    images = staging / "images"
    images.mkdir(parents=True)
    all_records: list[dict[str, Any]] = []
    v1_records: list[dict[str, Any]] = []

    try:
        source_card = metadata.get("cardData")
        declared_license = source_card.get("license") if isinstance(source_card, dict) else None
        write_json(
            staging / "source.json",
            {
                "dataset_id": DATASET_ID,
                "dataset_url": DATASET_URL,
                "revision": DATASET_REVISION,
                "split": SPLIT,
                "retrieval_contract": "Hugging Face rows API image URLs pinned to revision",
                "upstream_declared_license": declared_license,
                "license_status": "unreviewed-no-declared-license" if not declared_license else "declared-by-upstream",
                "restriction": "Keep local. Do not redistribute or use commercially until the provider confirms rights and terms.",
            },
        )

        for row_index, item in enumerate(rows):
            source_url, ground_truth = image_url_and_ground_truth(item, row_index)
            payload = request_bytes(source_url)
            if not payload.startswith(b"\xff\xd8\xff"):
                raise RuntimeError(f"Source row {row_index} did not return a JPEG image.")
            image_name = f"{row_index:06d}.jpg"
            (images / image_name).write_bytes(payload)
            record = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "source": {"dataset_id": DATASET_ID, "revision": DATASET_REVISION, "split": SPLIT, "row_index": row_index},
                "image": {"path": f"images/{image_name}", "sha256": sha256_bytes(payload), "width": 640, "height": 640},
                "ground_truth": ground_truth,
                "reader_v1_eligible": bool(V1_DISPLAY_PATTERN.fullmatch(ground_truth["display"])),
            }
            all_records.append(record)
            if record["reader_v1_eligible"]:
                v1_records.append(record)

        write_jsonl(staging / "all_test.jsonl", all_records)
        write_jsonl(staging / "reader_v1_eligible_test.jsonl", v1_records)
        write_json(
            staging / "summary.json",
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "source_dataset": DATASET_ID,
                "source_revision": DATASET_REVISION,
                "split": SPLIT,
                "total_samples": len(all_records),
                "reader_v1_eligible_samples": len(v1_records),
                "reader_v1_rule": "LLL-DDDD; excludes I, O, and 4",
            },
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {"total": len(all_records), "v1_eligible": len(v1_records)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New ignored destination directory.")
    parser.add_argument(
        "--acknowledge-unreviewed-license",
        action="store_true",
        help="Confirm local-only acquisition despite the upstream Dataset Card having no declared license.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if not args.acknowledge_unreviewed_license:
        print(
            "Refusing download: EZCon's Dataset Card/API has no declared license. "
            "Review the source and rerun with --acknowledge-unreviewed-license for local-only acquisition.",
            file=sys.stderr,
        )
        return 2
    if output.exists():
        print(f"Refusing to replace existing destination: {output}", file=sys.stderr)
        return 3
    if output.parent.exists() and not output.parent.is_dir():
        print(f"Output parent is not a directory: {output.parent}", file=sys.stderr)
        return 4
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching an unreviewed-license, local-only dataset; it must not be redistributed.", file=sys.stderr)
    counts = fetch_dataset(output)
    print(
        f"Fetched {counts['total']} test images; {counts['v1_eligible']} match the current Reader v1 label contract.\n"
        f"Output: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
