"""Command-line interface for local v1 PyTorch CTC training."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from plateai_shared.publication import OutputExistsError

from .dataset import TrainingDataError
from .engine import TrainingConfig, train_recognizer


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plateai-train", description="Train the local v1 CTC plate recognizer."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=_positive_integer, default=10)
    parser.add_argument("--batch-size", type=_positive_integer, default=32)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--charset", type=Path, default=None)
    parser.add_argument("--rules", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        train_directory=args.train,
        validation_directory=args.validation,
        output_directory=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        charset_path=args.charset,
        rules_path=args.rules,
    )
    try:
        train_recognizer(config)
    except (OutputExistsError, TrainingDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
