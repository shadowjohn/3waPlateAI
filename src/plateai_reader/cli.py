"""Local full-bundle Reader command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .runtime import PlateReader, ReaderError


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _read_rgb(path: Path) -> np.ndarray:
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise ReaderError(f"cannot read image: {path}") from error
    image_bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ReaderError(f"image is not a supported decodable file: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _result_json(result) -> dict[str, object]:
    return {
        "providers": list(result.providers),
        "plates": [
            {
                "canonical": item.decoded.canonical,
                "display": item.decoded.display,
                "rule_id": item.decoded.rule_id,
                "plate_type": item.decoded.plate_type,
                "ctc_log_probability": item.decoded.log_probability,
                "confidence": item.detection.confidence,
                "bbox_xyxy": item.detection.bbox_xyxy.tolist(),
                "corners_xy": item.detection.corners_xy.tolist(),
            }
            for item in result.plates
        ],
        "rejections": [
            {
                "stage": item.stage,
                "reason": item.reason,
                "confidence": item.detection.confidence,
                "bbox_xyxy": item.detection.bbox_xyxy.tolist(),
                "corners_xy": item.detection.corners_xy.tolist(),
            }
            for item in result.rejections
        ],
        "timing": {
            "detector_ms": result.timing.detector_ms,
            "rectifier_ms": result.timing.rectifier_ms,
            "recognizer_ms": result.timing.recognizer_ms,
            "retained_detection_count": result.timing.retained_detection_count,
            "rectified_plate_count": result.timing.rectified_plate_count,
            "recognition_batch_sizes": list(result.timing.recognition_batch_sizes),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plateai-read",
        description="Read Taiwan-style plates with a validated local full Model Bundle.",
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help="explicit ONNX Runtime provider; repeat to specify fallback order",
    )
    parser.add_argument(
        "--warmup",
        type=_nonnegative_integer,
        default=0,
        help="run this many unreported warm-up inferences before emitting JSON",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        image_rgb = _read_rgb(args.image)
        reader = PlateReader(args.bundle, providers=args.providers)
        for _ in range(args.warmup):
            reader.read(image_rgb)
        print(json.dumps(_result_json(reader.read(image_rgb)), ensure_ascii=False))
        return 0
    except ReaderError as error:
        if args.debug:
            raise
        print(f"plateai-read: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
