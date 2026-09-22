"""Fail-closed command-line orchestration for the M4.5 evaluation gate."""

from __future__ import annotations

import argparse
import json
import sys
import sysconfig
import uuid
from collections.abc import Sequence
from pathlib import Path

from plateai_shared.publication import (
    OutputExistsError,
    PublicationError,
)

from .contracts import EvaluationInputError, EvaluationRuntimeError, load_profile
from .engine import (
    PROVIDERS,
    OnnxEvaluationEngine,
    run_detector_lane,
    run_e2e_lane,
    run_oracle_lane,
)
from .gates import decide_gates, preflight
from .parity import run_parity
from .reporting import build_report, publish_run, verify_run
from .suite import load_suite


_STATUS_EXIT_CODES = {
    "PROMOTION_ELIGIBLE": 0,
    "PASS_LOCAL_ONLY": 10,
    "REJECTED": 20,
}


def _default_schema_root() -> Path:
    checkout = Path(__file__).resolve().parents[2] / "schemas"
    if checkout.is_dir():
        return checkout
    return Path(sysconfig.get_path("data")) / "share/3wa-plate-ai/schemas"


def _repository_root() -> Path:
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / ".git").exists():
        return checkout
    return Path.cwd()


def _new_run_id() -> str:
    return f"m45-{uuid.uuid4().hex}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plateai-eval",
        description="Run or verify the fail-closed M4.5 evaluation gate.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run all strict evaluation lanes")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--profile", type=Path, required=True)
    run.add_argument("--provider", choices=tuple(PROVIDERS), required=True)
    run.add_argument("--recognizer-checkpoint", type=Path, required=True)
    run.add_argument("--detector-checkpoint", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify", help="verify immutable run evidence")
    verify.add_argument("--run", dest="run_dir", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace, schema_root: Path) -> int:
    provider = PROVIDERS[args.provider]
    model_schema = schema_root / "model_manifest.schema.json"
    profile = load_profile(
        args.profile,
        schema_root / "evaluation_profile.schema.json",
    )
    snapshot = load_suite(
        args.suite,
        schema_root / "evaluation_suite.schema.json",
        schema_root / "evaluation_scene.schema.json",
    )
    counts = preflight(snapshot, profile)
    engine = OnnxEvaluationEngine(args.bundle, provider, model_schema)
    oracle = run_oracle_lane(snapshot, engine)
    detector = run_detector_lane(snapshot, engine)
    e2e = run_e2e_lane(snapshot, engine)
    parity = run_parity(
        snapshot,
        profile,
        args.bundle,
        args.recognizer_checkpoint,
        args.detector_checkpoint,
        provider,
        schema_path=model_schema,
    )
    snapshot.verify_unchanged()
    lane_results = (oracle, detector, e2e)
    decision = decide_gates(
        profile,
        engine.manifest,
        snapshot,
        lane_results,
        parity,
    )
    run_id = _new_run_id()
    report = build_report(
        run_id=run_id,
        profile=profile,
        bundle_manifest=engine.manifest,
        bundle_sha256=engine.bundle_sha256,
        manifest_sha256=engine.manifest_sha256,
        artifact_hashes=engine.artifact_hashes,
        snapshot=snapshot,
        counts=counts,
        lane_results=lane_results,
        parity_result=parity,
        decision=decision,
        requested_provider=engine.requested_provider,
        actual_provider=engine.actual_provider,
        repository_root=_repository_root(),
    )
    samples = tuple(
        sample for result in lane_results for sample in result.samples
    )
    publish_run(args.output, report, samples, schema_root)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "overall_status": decision.overall_status,
                "technical_gate": decision.technical_gate,
                "rights_gate": decision.rights_gate,
                "counts": dict(counts.as_mapping()),
                "requested_provider": engine.requested_provider,
                "actual_provider": engine.actual_provider,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return _STATUS_EXIT_CODES[decision.overall_status]


def _verify(args: argparse.Namespace, schema_root: Path) -> int:
    receipt = verify_run(args.run_dir, schema_root)
    print(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schema_root = _default_schema_root()
    try:
        if args.command == "run":
            return _run(args, schema_root)
        return _verify(args, schema_root)
    except OutputExistsError as exc:
        print(f"plateai-eval: {exc}", file=sys.stderr)
        return 4
    except EvaluationInputError as exc:
        print(f"plateai-eval: {exc}", file=sys.stderr)
        return 2
    except (EvaluationRuntimeError, PublicationError, OSError) as exc:
        print(f"plateai-eval: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
