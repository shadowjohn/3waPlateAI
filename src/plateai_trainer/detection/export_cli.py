"""Local full-bundle export command."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from plateai_shared.publication import PublicationError
from .export import DetectionExportRequest, export_full_bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plateai-detect-export")
    parser.add_argument("--recognizer-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        export_full_bundle(DetectionExportRequest(
            args.recognizer_bundle, args.checkpoint, args.report, args.output,
        ))
    except (PublicationError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
