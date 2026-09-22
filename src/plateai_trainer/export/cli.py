from __future__ import annotations
import argparse, sys
from collections.abc import Sequence
from pathlib import Path
from .bundle import ExportRequest, ExportParityError, export_crop_bundle
from plateai_shared.publication import OutputExistsError
def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser(prog="plateai-export")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--charset", type=Path, default=None)
    parser.add_argument("--rules", type=Path, default=None)
    args=parser.parse_args(argv)
    try: export_crop_bundle(ExportRequest(args.checkpoint, args.report, args.output, charset_path=args.charset, rules_path=args.rules))
    except (OutputExistsError, ExportParityError, ValueError, OSError) as exc: print(f"error: {exc}",file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
