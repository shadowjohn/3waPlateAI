"""Local CPU/CUDA detector training entry point."""

import argparse
from pathlib import Path
import sys

from plateai_shared.publication import OutputExistsError, PublicationError
from .dataset import DetectionDataError


def main(argv=None):
    parser = argparse.ArgumentParser(prog="plateai-detect-train", description="Train the native multi-plate pose detector locally on CPU or CUDA.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    from .engine import DetectorTrainingConfig, train_detector
    try:
        train_detector(DetectorTrainingConfig(args.train, args.validation, args.output,
                       args.epochs, args.batch_size, args.learning_rate, args.seed, args.device))
    except OutputExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (DetectionDataError, PublicationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
