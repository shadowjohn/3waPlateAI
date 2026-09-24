"""Reproducible, leak-free split for TLPD dataset.

Groups images strictly by normalized plate string so no plate number
ever crosses train, validation, or test partitions.
Saves the frozen test set for unbiased final re-test evaluation.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "tlpd-taiwan-detector"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
SPLITS_DIR = DATASET_DIR / "splits"

V1_PATTERN = re.compile(r"^[A-Z]{3}[0-9]{4}$")


def normalize_plate(stem: str) -> str:
    return re.sub(r"(?:\(\d*\))+$", "", stem.upper())


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images found in {IMAGES_DIR}")

    # Group by plate string
    groups: dict[str, list[dict]] = {}
    for img_path in images:
        plate = normalize_plate(img_path.stem)
        gt_json = LABELS_DIR / img_path.with_suffix(".json").name
        has_gt = gt_json.is_file()
        is_v1 = bool(V1_PATTERN.fullmatch(plate))
        item = {
            "file": img_path.name,
            "plate": plate,
            "is_v1": is_v1,
            "has_gt": has_gt,
        }
        groups.setdefault(plate, []).append(item)

    # Sort plates for deterministic ordering before shuffle
    sorted_plates = sorted(groups.keys())
    rng = random.Random(42)
    rng.shuffle(sorted_plates)

    total_plates = len(sorted_plates)
    train_end = int(total_plates * 0.70)
    val_end = int(total_plates * 0.85)

    train_plates = set(sorted_plates[:train_end])
    val_plates = set(sorted_plates[train_end:val_end])
    test_plates = set(sorted_plates[val_end:])

    # Verify 0 overlap between plate strings
    assert not (train_plates & val_plates)
    assert not (train_plates & test_plates)
    assert not (val_plates & test_plates)

    splits = {
        "train": [item for p in sorted_plates[:train_end] for item in groups[p]],
        "val": [item for p in sorted_plates[train_end:val_end] for item in groups[p]],
        "test_frozen": [item for p in sorted_plates[val_end:] for item in groups[p]],
    }

    summary = {
        "seed": 42,
        "total_images": len(images),
        "total_unique_plates": total_plates,
        "partitions": {},
    }

    for name, items in splits.items():
        out_path = SPLITS_DIR / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        plate_set = {it["plate"] for it in items}
        v1_count = sum(1 for it in items if it["is_v1"])
        gt_count = sum(1 for it in items if it["has_gt"])
        lengths = Counter(len(it["plate"]) for it in items)

        summary["partitions"][name] = {
            "image_count": len(items),
            "plate_count": len(plate_set),
            "v1_count": v1_count,
            "gt_count": gt_count,
            "lengths": dict(lengths),
        }

    summary_path = SPLITS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== TLPD Split Complete ===")
    print(f"Total Images: {len(images)}, Unique Plates: {total_plates}")
    for name, stats in summary["partitions"].items():
        print(f"[{name.upper()}] Images: {stats['image_count']}, Plates: {stats['plate_count']}, "
              f"v1 (LLLDDDD): {stats['v1_count']}, with GT corners: {stats['gt_count']}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
