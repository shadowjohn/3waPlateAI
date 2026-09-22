"""Evaluate 3waPlateAI models on the local EZCon Taiwan recognition benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import onnxruntime as ort

from plateai_reader import decode_constrained_ctc_v1
from plateai_reader.rectifier import rectify_plate
from plateai_reader.runtime import PlateReader
from plateai_shared.recognition import CTCCodec, preprocess_v1_rgb
from plateai_shared.rules import load_character_set, load_ruleset


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate character edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


def evaluate_dataset(
    records_file: Path,
    dataset_root: Path,
    recognizer_session: ort.InferenceSession,
    codec: CTCCodec,
    ruleset: Any,
    reader: PlateReader | None = None,
    tag: str = "eval",
) -> dict[str, Any]:
    with open(records_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    total = len(records)
    exact_canonical = 0
    exact_display = 0
    total_char_dist = 0
    total_chars = 0
    failures = []
    successes = []

    print(f"\n==================================================")
    print(f"[{tag}] Evaluating {total} plates from {records_file.name}")
    print(f"==================================================")

    for idx, r in enumerate(records):
        gt = r["ground_truth"]
        gt_canon = gt["canonical"]
        gt_disp = gt["display"]
        cx, cy, w, h, rad = gt["xywhr"]

        img_path = dataset_root / r["image"]["path"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Oracle crop using annotated rotated bounding box
        rect = ((cx, cy), (w, h), math.degrees(rad))
        box = cv2.boxPoints(rect)

        try:
            rectified = rectify_plate(img_rgb, box)
            crop_rgb = rectified.image_rgb
            tensor = preprocess_v1_rgb(crop_rgb)
            logits = recognizer_session.run(["logits"], {"input": tensor[np.newaxis, ...]})[0]
            greedy = codec.decode_greedy(logits[0].argmax(axis=1).tolist())
            decoded = decode_constrained_ctc_v1(logits[0], codec, ruleset)
            pred_canon = decoded.canonical
            pred_disp = decoded.display
            rule_id = decoded.rule_id
            logp = decoded.log_probability
        except Exception as e:
            pred_canon = ""
            pred_disp = ""
            rule_id = "error"
            logp = -999.0

        is_exact = (pred_canon == gt_canon)
        if is_exact:
            exact_canonical += 1
            exact_display += (pred_disp == gt_disp)
            successes.append({
                "idx": idx,
                "gt": gt_disp,
                "pred": pred_disp,
                "rule": rule_id,
                "logp": logp,
            })
        else:
            failures.append({
                "idx": idx,
                "gt": gt_disp,
                "pred": pred_disp,
                "greedy": greedy if 'greedy' in locals() else "",
                "rule": rule_id,
                "logp": logp,
            })

        dist = levenshtein_distance(pred_canon, gt_canon)
        total_char_dist += dist
        total_chars += len(gt_canon)

    acc = exact_canonical / total if total > 0 else 0
    cer = total_char_dist / total_chars if total_chars > 0 else 0
    print(f"\n--- [{tag}] Results Summary ---")
    print(f"Total Samples:          {total}")
    print(f"Exact Plate Accuracy:   {exact_canonical}/{total} ({acc * 100:.2f}%)")
    print(f"Character Error Rate:   {cer * 100:.2f}%")
    print(f"Character Accuracy:     {(1 - cer) * 100:.2f}%")

    print(f"\n--- Sample Successful Recognitions (First 8) ---")
    for item in successes[:8]:
        print(f"Row {item['idx']:3d} | GT: {item['gt']:10s} -> Pred: {item['pred']:10s} ({item['rule']}) [logp={item['logp']:.2f}]")

    print(f"\n--- Sample Failure Cases (First 8) ---")
    for item in failures[:8]:
        print(f"Row {item['idx']:3d} | GT: {item['gt']:10s} -> Pred: {item['pred']:10s} | Greedy: {item['greedy']:10s} ({item['rule']}) [logp={item['logp']:.2f}]")

    return {
        "tag": tag,
        "total": total,
        "exact_canonical": exact_canonical,
        "accuracy": acc,
        "cer": cer,
        "success_count": len(successes),
        "failure_count": len(failures),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    bundle_dir = root / "models" / "bundles" / "tw-std-v1-recognizer"
    full_bundle_dir = root / "models" / "bundles" / "tw-std-v1-full"
    ezcon_dir = root / "datasets" / "restricted" / "ezcon-taiwan-recognition-test"

    if not ezcon_dir.exists():
        print(f"Dataset directory not found: {ezcon_dir}")
        return

    session = ort.InferenceSession(str(bundle_dir / "recognizer.onnx"), providers=["CPUExecutionProvider"])
    charset = load_character_set(bundle_dir / "charset.txt")
    codec = CTCCodec.from_charset(charset)
    ruleset = load_ruleset(bundle_dir / "plate_rules.json", charset)

    # 1. Evaluate on Reader v1 eligible subset (157 car plates)
    v1_results = evaluate_dataset(
        ezcon_dir / "reader_v1_eligible_test.jsonl",
        ezcon_dir,
        session,
        codec,
        ruleset,
        tag="Car Plates (v1 Eligible: LLL-DDDD)",
    )

    # 2. Evaluate on All 259 plates (including motorcycle, 6-digit, 5-digit)
    all_results = evaluate_dataset(
        ezcon_dir / "all_test.jsonl",
        ezcon_dir,
        session,
        codec,
        ruleset,
        tag="All 259 Test Plates",
    )

    out_report = root / "out" / "ezcon_evaluation_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        json.dumps({"car_v1_eligible": v1_results, "all_test": all_results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nReport written to: {out_report}")


if __name__ == "__main__":
    main()
