"""Command-line interface for deterministic synthetic dataset generation."""

from __future__ import annotations

import argparse
import json
import sys
import sysconfig
from collections.abc import Sequence
from pathlib import Path

from .dataset import (
    GenerationError,
    InvalidGenerationRequest,
    OutputExistsError,
    generate_dataset,
)
from .models import GenerationRequest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_INSTALL_DATA_ROOT = (
    Path(sysconfig.get_path("data")) / "share" / "3wa-plate-ai"
)


def _default_config_path(relative_path: str) -> Path:
    checkout_path = _REPOSITORY_ROOT / relative_path
    if checkout_path.is_file():
        return checkout_path
    return _INSTALL_DATA_ROOT / relative_path


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plateai-generate",
        description="Generate deterministic synthetic Taiwan-style plate crops.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate a recognition dataset")
    generate.add_argument("--count", type=_positive_integer, default=100)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--charset",
        type=Path,
        default=_default_config_path(
            "configs/charsets/tw_new_style_private_passenger_v1.txt"
        ),
    )
    generate.add_argument(
        "--rules",
        type=Path,
        default=_default_config_path(
            "configs/plate_rules/tw_new_style_private_passenger_v1.json"
        ),
    )
    generate.add_argument(
        "--template",
        type=Path,
        default=_default_config_path(
            "configs/plate_templates/new_style_private_passenger_white_v1.json"
        ),
    )
    generate.add_argument(
        "--augmentation",
        type=Path,
        default=_default_config_path("configs/augmentation/standard_v1.json"),
    )
    generate.add_argument(
        "--font",
        type=Path,
        help="override the bundled OFL font with a local .ttf or .otf file",
    )
    generate.add_argument(
        "--debug",
        action="store_true",
        help="show a traceback for generation failures",
    )
    return parser


def _report_error(exc: BaseException, *, debug: bool, exit_code: int) -> int:
    if debug:
        raise exc
    print(f"error: {exc}", file=sys.stderr)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "generate":
        return 2

    request = GenerationRequest(
        count=args.count,
        seed=args.seed,
        output=args.output,
        charset_path=args.charset,
        rules_path=args.rules,
        template_path=args.template,
        augmentation_path=args.augmentation,
        font=args.font,
    )
    try:
        summary = generate_dataset(request)
    except OutputExistsError as exc:
        return _report_error(exc, debug=args.debug, exit_code=3)
    except (InvalidGenerationRequest, FileNotFoundError, ValueError) as exc:
        return _report_error(exc, debug=args.debug, exit_code=2)
    except (GenerationError, OSError, RuntimeError) as exc:
        return _report_error(exc, debug=args.debug, exit_code=4)

    payload = {
        "output": str(args.output),
        "generated": summary.generated,
        "seed": summary.seed,
        "rule_counts": dict(summary.rule_counts),
        "plate_type_counts": dict(summary.plate_type_counts),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0
