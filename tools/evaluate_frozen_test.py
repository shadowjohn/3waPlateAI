"""Evaluate a model bundle on the frozen TLPD test set.

Reports exact match, greedy match, empty output rate, and latency breakdown.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plateai_bench.runner import BenchmarkRunner
from plateai_reader.rectifier import rectify_plate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=ROOT / "models" / "bundles" / "active-v1")
    parser.add_argument("--split", type=Path, default=ROOT / "datasets" / "tlpd-taiwan-detector" / "splits" / "test_frozen.jsonl")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "datasets" / "tlpd-taiwan-detector" / "images")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "datasets" / "tlpd-taiwan-detector" / "labels")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_gt_corners(labels_dir: Path, filename: str) -> np.ndarray | None:
    json_path = labels_dir / Path(filename).with_suffix(".json").name
    if not json_path.is_file():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for shape in data.get("shapes", []):
            if shape.get("label") == "carplate":
                return np.asarray(shape["points"], dtype=np.float32)
    except Exception:
        return None
    return None


def main():
    args = parse_args()
    if not args.split.is_file():
        raise SystemExit(f"Split file not found: {args.split}")

    records = [json.loads(line) for line in args.split.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(records)} test samples from {args.split.name}")
    print(f"Evaluating bundle: {args.bundle}")

    runner = BenchmarkRunner(args.bundle, device=args.device)

    counts = Counter()
    v1_counts = Counter()
    results = []

    for idx, rec in enumerate(records, 1):
        img_name = rec["file"]
        expected = rec["plate"]
        is_v1 = rec["is_v1"]

        img_path = args.images_dir / img_name
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            counts["unreadable"] += 1
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Direct resize to 380x160
        crop = cv2.resize(rgb, (380, 160), interpolation=cv2.INTER_LINEAR)

        constrained, display, greedy, score, timing = runner._eval_single_crop(crop)

        exact_c = (constrained == expected)
        exact_g = (greedy == expected)
        empty_g = (len(greedy) == 0)

        counts["total"] += 1
        if exact_c:
            counts["constrained_exact"] += 1
        if exact_g:
            counts["greedy_exact"] += 1
        if empty_g:
            counts["greedy_empty"] += 1

        if is_v1:
            v1_counts["total"] += 1
            if exact_c:
                v1_counts["constrained_exact"] += 1
            if exact_g:
                v1_counts["greedy_exact"] += 1
            if empty_g:
                v1_counts["greedy_empty"] += 1

        results.append({
            "file": img_name,
            "expected": expected,
            "is_v1": is_v1,
            "constrained": constrained,
            "greedy": greedy,
            "exact_constrained": exact_c,
            "exact_greedy": exact_g,
            "empty_greedy": empty_g,
            "score": score,
        })

    total = counts["total"]
    c_acc = counts["constrained_exact"] / total if total else 0.0
    g_acc = counts["greedy_exact"] / total if total else 0.0
    empty_rate = counts["greedy_empty"] / total if total else 0.0

    v1_total = v1_counts["total"]
    v1_c_acc = v1_counts["constrained_exact"] / v1_total if v1_total else 0.0
    v1_g_acc = v1_counts["greedy_exact"] / v1_total if v1_total else 0.0

    report = {
        "bundle": str(args.bundle),
        "split": str(args.split),
        "overall": {
            "total": total,
            "constrained_exact": counts["constrained_exact"],
            "constrained_acc": round(c_acc * 100, 2),
            "greedy_exact": counts["greedy_exact"],
            "greedy_acc": round(g_acc * 100, 2),
            "greedy_empty": counts["greedy_empty"],
            "greedy_empty_rate": round(empty_rate * 100, 2),
        },
        "v1_only": {
            "total": v1_total,
            "constrained_exact": v1_counts["constrained_exact"],
            "constrained_acc": round(v1_c_acc * 100, 2),
            "greedy_exact": v1_counts["greedy_exact"],
            "greedy_acc": round(v1_g_acc * 100, 2),
        },
    }

    print("\n=== Frozen Test Results ===")
    print(f"Overall Total: {total}")
    print(f"  Constrained Exact: {counts['constrained_exact']}/{total} ({report['overall']['constrained_acc']}%)")
    print(f"  Greedy Exact:      {counts['greedy_exact']}/{total} ({report['overall']['greedy_acc']}%)")
    print(f"  Greedy Empty:      {counts['greedy_empty']}/{total} ({report['overall']['greedy_empty_rate']}%)")
    print(f"v1 Format Only: {v1_total}")
    print(f"  v1 Constrained:    {v1_counts['constrained_exact']}/{v1_total} ({report['v1_only']['constrained_acc']}%)")
    print(f"  v1 Greedy:         {v1_counts['greedy_exact']}/{v1_total} ({report['v1_only']['greedy_acc']}%)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"report": report, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Details saved to {args.output}")


if __name__ == "__main__":
    main()
