"""Offline scorecards for the maintained MIT OCR pair.

Examples:
  python tools/evaluate_fpga_lpr.py --tlpd-replay-root datasets/tlpd-taiwan-detector/images --output-dir runs/fpga-lpr-eval/tlpd-replay
  python tools/evaluate_fpga_lpr.py --manifest runs/fpga-lpr-eval/audited.jsonl --split dev --output-dir runs/fpga-lpr-eval/dev-compat

TLPD filenames are provisional transcriptions from the author's training source.
They MUST NOT be reported as independent generalization accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plateai_eval.fpga_protocol import (  # noqa: E402
    AuditedPlate, EvaluationInputError, evaluate_entries, load_audited_manifest,
)
from plateai_reader.fpga_lpr import FpgaLprRecognizer  # noqa: E402
from plateai_reader.fpga_pipeline import FpgaSceneReader  # noqa: E402
from plateai_reader.runtime import PlateReader  # noqa: E402


def tlpd_replay_entries(image_dir: Path) -> tuple[AuditedPlate, ...]:
    """Filename-derived compatibility replay; no independent labels implied."""

    rows: list[AuditedPlate] = []
    for image in sorted(Path(image_dir).glob("*.jpg")):
        label = re.sub(r"(?:\(\d+\))*$", "", image.stem).upper()
        canonical = re.sub(r"[^A-Z0-9]", "", label)
        if not canonical:
            raise EvaluationInputError("invalid_tlpd_filename")
        # The replay path uses whole, plate-centric images; dimensions are read
        # at score time after the SHA check, so this sentinel means full image.
        rows.append(AuditedPlate(
            image=image.resolve(), sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            canonical=canonical, vehicle_class="unclassified", split="dev",
            crop_xyxy=(0, 0, -1, -1), source_kind="training_source_replay",
        ))
    if not rows:
        raise EvaluationInputError("empty_tlpd_replay")
    return tuple(rows)


def _atomic_json(path: Path, payload: object) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            name = Path(handle.name)
        except Exception:
            Path(handle.name).unlink(missing_ok=True)
            raise
    os.replace(name, path)


def _atomic_jsonl(path: Path, rows: tuple) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        try:
            for row in rows:
                handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            name = Path(handle.name)
        except Exception:
            Path(handle.name).unlink(missing_ok=True)
            raise
    os.replace(name, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--tlpd-replay-root", type=Path)
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--mode", choices=("crop", "scene"), default="crop")
    parser.add_argument("--corner-policy", choices=("compat", "safe"), default="compat")
    parser.add_argument("--roi-margin", type=float, choices=(0.06, 0.10), default=0.06)
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "third_party" / "fpga_lpr")
    parser.add_argument("--detector-bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.tlpd_replay_root and args.split != "dev":
        parser.error("TLPD training-source replay cannot be labelled holdout")
    if args.mode == "scene" and not args.detector_bundle:
        parser.error("scene mode requires a local detector bundle")
    rows = (
        tlpd_replay_entries(args.tlpd_replay_root)
        if args.tlpd_replay_root else load_audited_manifest(args.manifest)
    )
    selected = tuple(row for row in rows if row.split == args.split)
    if not selected:
        raise EvaluationInputError("empty_selected_split")
    output = args.output_dir.resolve()
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise EvaluationInputError("output_dir_not_empty")
    output.mkdir(parents=True, exist_ok=True)
    recognizer = FpgaLprRecognizer(args.assets_dir)
    reader = (
        recognizer if args.mode == "crop"
        else FpgaSceneReader(PlateReader(args.detector_bundle), recognizer, roi_margin=args.roi_margin)
    )
    report = evaluate_entries(reader, selected, mode=args.mode, corner_policy=args.corner_policy)
    _atomic_jsonl(output / "per_image.jsonl", report.results)
    summary = report.to_dict()
    summary.pop("results")
    summary["settings"] = {
        "split": args.split, "corner_policy": args.corner_policy,
        "roi_margin": args.roi_margin, "model_id": recognizer.manifest.model_id,
    }
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
