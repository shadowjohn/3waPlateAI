"""CLI for deterministic multi-instance detection composites."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from plateai_shared.publication import OutputExistsError

from .composite import CompositeGenerationError, InvalidCompositeRequest, generate_composite_dataset
from .contracts import CompositeGenerationRequest


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
        prog="plateai-compose",
        description="Generate deterministic multi-plate detection composites.",
    )
    parser.add_argument("--background-manifest", type=Path, required=True)
    parser.add_argument("--count", type=_positive_integer, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-instances", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--max-instances", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--charset", type=Path, default=None)
    parser.add_argument("--rules", type=Path, default=None)
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser


def _error(exc: BaseException, *, debug: bool, exit_code: int) -> int:
    if debug:
        raise exc
    print(f"error: {exc}", file=sys.stderr)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kwargs = {}
    if args.charset is not None:
        kwargs["charset_path"] = args.charset
    if args.rules is not None:
        kwargs["rules_path"] = args.rules
    if args.template is not None:
        kwargs["template_path"] = args.template

    request = CompositeGenerationRequest(
        output=args.output,
        count=args.count,
        seed=args.seed,
        background_manifest=args.background_manifest,
        instances_per_image=(args.min_instances, args.max_instances),
        **kwargs,
    )
    try:
        summary = generate_composite_dataset(request)
    except OutputExistsError as exc:
        return _error(exc, debug=args.debug, exit_code=3)
    except (InvalidCompositeRequest, FileNotFoundError, ValueError) as exc:
        return _error(exc, debug=args.debug, exit_code=2)
    except (CompositeGenerationError, OSError, RuntimeError) as exc:
        return _error(exc, debug=args.debug, exit_code=4)
    print(
        json.dumps(
            {
                "generated": summary.generated,
                "output": str(summary.output),
                "seed": summary.seed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0
