"""Canonical, transactional M4.5 reports and self-verifying receipts."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from plateai_shared.contracts import JsonValue
from plateai_shared.publication import (
    OutputExistsError,
    publish_directory_no_replace,
    remove_owned_staging,
)
from plateai_shared.schema_validation import DocumentValidationError, validate_document

from .contracts import (
    EvaluationInputError,
    GateDecision,
    LaneResult,
    ParityResult,
    PopulationCounts,
    StrictProfile,
    SuiteSnapshot,
)


_EVIDENCE_FILES = {"receipt.json", "report.json", "samples.jsonl"}
_ATTRIBUTIONS = (
    "DETECTOR_MISSED",
    "DETECTOR_FALSE_POSITIVE",
    "DETECTOR_BAD_BBOX",
    "DETECTOR_BAD_QUAD",
    "RECTIFIER_REJECTED",
    "RECOGNIZER_MISREAD",
    "RULE_FILTERED",
    "SUCCESS",
)


def _thaw(value: Any) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise EvaluationInputError(
        f"evidence contains non-JSON value of type {type(value).__name__}"
    )


def _canonical_json(document: Mapping[str, JsonValue]) -> bytes:
    try:
        return (
            json.dumps(
                _thaw(document),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError(f"evidence is not canonical JSON: {exc}") from exc


def _reject_duplicate_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise EvaluationInputError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _parse_object(payload: bytes, label: str) -> dict[str, JsonValue]:
    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except EvaluationInputError as exc:
        raise EvaluationInputError(f"{label}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"{label}: invalid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise EvaluationInputError(f"{label}: must contain one JSON object")
    return document


def _canonical_samples(samples: Sequence[Mapping[str, JsonValue]]) -> bytes:
    ordered = sorted(
        samples,
        key=lambda sample: tuple(
            str(sample.get(key, ""))
            for key in (
                "lane",
                "source_id",
                "scene_id",
                "instance_id",
                "prediction_id",
            )
        ),
    )
    return b"".join(_canonical_json(sample) for sample in ordered)


def _validate_schema(document, path: Path, label: str) -> None:
    try:
        validate_document(document, path)
    except DocumentValidationError as exc:
        raise EvaluationInputError(f"{label} schema: {exc}") from exc


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "unavailable"


def _source_revision(repository_root: Path) -> Mapping[str, JsonValue]:
    commands = (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--porcelain"]),
    )
    values: dict[str, str] = {}
    for name, command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "commit": None,
                "dirty": None,
                "source_revision_error": f"git metadata unavailable: {type(exc).__name__}",
            }
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            message = detail[0][:200] if detail else f"git exited {result.returncode}"
            return {
                "commit": None,
                "dirty": None,
                "source_revision_error": message,
            }
        values[name] = result.stdout.strip()
    return {
        "commit": values["commit"],
        "dirty": bool(values["status"]),
        "source_revision_error": None,
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _latency(result: LaneResult) -> Mapping[str, JsonValue]:
    by_scene: dict[str, float] = {}
    for sample in result.samples:
        scene_id = sample.get("scene_id")
        timings = sample.get("timings_ms")
        if not isinstance(scene_id, str) or not isinstance(timings, Mapping):
            raise EvaluationInputError(
                f"{result.lane}: sample latency identity is missing"
            )
        key = "detector_ms" if result.lane == "detector" else "total_ms"
        value = timings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvaluationInputError(f"{result.lane}: {key} is not numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise EvaluationInputError(f"{result.lane}: {key} must be finite")
        if result.lane == "oracle_ocr":
            by_scene[scene_id] = by_scene.get(scene_id, 0.0) + numeric
        else:
            by_scene.setdefault(scene_id, numeric)
    observed = tuple(by_scene.values())
    return {
        "observations": len(observed),
        "p50": _percentile(observed, 0.50),
        "p95": _percentile(observed, 0.95),
    }


def _reported_lane(
    result: LaneResult, decision: GateDecision
) -> Mapping[str, JsonValue]:
    prefix = {
        "oracle_ocr": "oracle_",
        "detector": "detector_",
        "end_to_end": "e2e_",
    }[result.lane]
    checks = tuple(
        check
        for check in decision.checks
        if str(check.get("metric", "")).startswith(prefix)
    )
    attribution_counts = Counter(
        str(sample.get("attribution"))
        for sample in result.samples
        if sample.get("attribution") in _ATTRIBUTIONS
    )
    return {
        "lane": result.lane,
        "scene_count": result.scene_count,
        "instance_count": result.instance_count,
        "processed_scene_count": result.processed_scene_count,
        "processed_instance_count": result.processed_instance_count,
        "error_count": 0,
        "metrics": _thaw(result.metrics),
        "threshold_checks": [_thaw(check) for check in checks],
        "passed": all(bool(check["passed"]) for check in checks),
        "attribution_counts": {
            code: attribution_counts[code]
            for code in _ATTRIBUTIONS
            if attribution_counts[code]
        },
        "latency_ms": _latency(result),
    }


def build_report(
    *,
    run_id: str,
    profile: StrictProfile,
    bundle_manifest: Mapping[str, JsonValue],
    bundle_sha256: str,
    manifest_sha256: str,
    artifact_hashes: Mapping[str, str],
    snapshot: SuiteSnapshot,
    counts: PopulationCounts,
    lane_results: Sequence[LaneResult],
    parity_result: ParityResult,
    decision: GateDecision,
    requested_provider: str,
    actual_provider: str,
    repository_root: Path,
    created_at: str | None = None,
) -> Mapping[str, JsonValue]:
    """Build a complete report input; publication binds its samples hash."""

    if parity_result.provider is None or parity_result.pytorch_device is None:
        raise EvaluationInputError("parity provider/device evidence is missing")
    if tuple(result.lane for result in lane_results) != (
        "oracle_ocr",
        "detector",
        "end_to_end",
    ):
        raise EvaluationInputError("report lanes must be ordered oracle, detector, E2E")
    timestamp = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    capabilities = bundle_manifest.get("capabilities")
    if not isinstance(capabilities, list):
        raise EvaluationInputError("bundle capabilities are missing")
    hardware = f"machine={platform.machine() or 'unknown'}; processor={platform.processor() or 'unknown'}"
    sources = [
        {
            "source_id": source.source_id,
            "revision": source.revision,
            "review_status": source.rights.review_status,
            "local_only": source.rights.local_only,
            "allowed_uses": sorted(source.rights.allowed_uses),
            "notice": source.rights.notice,
        }
        for source in snapshot.sources
    ]
    manifests = [
        {"manifest_id": manifest.manifest_id, "sha256": manifest.sha256}
        for manifest in snapshot.manifests
    ]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": timestamp,
        "evaluator_version": _package_version("3wa-plate-ai"),
        "source_revision": _source_revision(Path(repository_root)),
        "profile": {
            "profile_id": profile.profile_id,
            "path": "configs/evaluation/m4_5_strict_v1.json",
            "profile_sha256": profile.sha256,
            "document": _thaw(profile.document),
        },
        "bundle": {
            "bundle_sha256": bundle_sha256,
            "manifest_sha256": manifest_sha256,
            "capabilities": list(capabilities),
            "artifacts": [
                {"file": name, "sha256": sha256}
                for name, sha256 in sorted(artifact_hashes.items())
            ],
        },
        "suite": {
            "suite_id": snapshot.suite_id,
            "revision": snapshot.revision,
            "suite_sha256": snapshot.sha256,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "sources": sources,
            "manifests": manifests,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "opencv": _package_version("opencv-python-headless", "opencv-python"),
            "onnx": _package_version("onnx"),
            "onnxruntime": _package_version("onnxruntime", "onnxruntime-gpu"),
            "pytorch": _package_version("torch"),
            "hardware": hardware,
            "requested_provider": requested_provider,
            "actual_provider": actual_provider,
            "warmup_runs": 0,
            "measured_rounds": 1,
        },
        "counts": _thaw(counts.as_mapping()),
        "lane_results": [
            _reported_lane(result, decision) for result in lane_results
        ],
        "parity": {
            "passed": parity_result.passed,
            "failures": list(parity_result.failures),
            "batch_sizes": list(parity_result.batch_sizes),
            "diagnostic_raw_difference_count": parity_result.diagnostic_raw_difference_count,
            "recognizer_checkpoint_sha256": parity_result.recognizer_checkpoint_sha256,
            "detector_checkpoint_sha256": parity_result.detector_checkpoint_sha256,
            "provider": parity_result.provider,
            "pytorch_device": parity_result.pytorch_device,
        },
        "decisions": {
            "technical_gate": decision.technical_gate,
            "rights_gate": decision.rights_gate,
            "overall_status": decision.overall_status,
            "checks": [_thaw(check) for check in decision.checks],
        },
    }


def publish_run(
    output: Path,
    report: Mapping[str, JsonValue],
    samples: Sequence[Mapping[str, JsonValue]],
    schema_root: Path,
) -> Path:
    """Atomically publish canonical samples, report, then linked receipt."""

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise OutputExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    try:
        samples_bytes = _canonical_samples(samples)
        samples_sha256 = hashlib.sha256(samples_bytes).hexdigest()
        (staging / "samples.jsonl").write_bytes(samples_bytes)

        report_document = dict(_thaw(report))
        report_document["samples_sha256"] = samples_sha256
        _validate_schema(
            report_document,
            Path(schema_root) / "evaluation_report.schema.json",
            "report",
        )
        report_bytes = _canonical_json(report_document)
        (staging / "report.json").write_bytes(report_bytes)
        report_sha256 = hashlib.sha256(report_bytes).hexdigest()

        receipt = {
            "schema_version": 1,
            "run_id": report_document["run_id"],
            "report_sha256": report_sha256,
            "samples_sha256": samples_sha256,
            "bundle_sha256": report_document["bundle"]["bundle_sha256"],
            "suite_sha256": report_document["suite"]["suite_sha256"],
            "profile_sha256": report_document["profile"]["profile_sha256"],
            "overall_status": report_document["decisions"]["overall_status"],
        }
        _validate_schema(
            receipt,
            Path(schema_root) / "evaluation_receipt.schema.json",
            "receipt",
        )
        (staging / "receipt.json").write_bytes(_canonical_json(receipt))
        verify_run(staging, schema_root)
        publish_directory_no_replace(staging, output)
    except BaseException:
        remove_owned_staging(staging, output)
        raise
    return output


def _read_evidence_file(output: Path, name: str) -> bytes:
    path = output / name
    if path.is_symlink():
        raise EvaluationInputError(f"{name}: symlink is not allowed")
    if not path.is_file():
        raise EvaluationInputError(f"{name}: missing regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise EvaluationInputError(f"{name}: cannot read file") from exc


def _verify_samples(payload: bytes) -> tuple[dict[str, JsonValue], ...]:
    if payload and not payload.endswith(b"\n"):
        raise EvaluationInputError("samples.jsonl: canonical trailing newline missing")
    samples: list[dict[str, JsonValue]] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if line == b"\n":
            raise EvaluationInputError(
                f"samples.jsonl line {line_number}: blank line is not canonical"
            )
        sample = _parse_object(line, f"samples.jsonl line {line_number}")
        if line != _canonical_json(sample):
            raise EvaluationInputError(
                f"samples.jsonl line {line_number}: noncanonical JSON"
            )
        samples.append(sample)
    if payload != _canonical_samples(samples):
        raise EvaluationInputError("samples.jsonl: noncanonical sample order")
    return tuple(samples)


def verify_run(
    output: Path, schema_root: Path
) -> Mapping[str, JsonValue]:
    """Verify canonical bytes, schemas, hashes, and all receipt identities."""

    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise EvaluationInputError("evaluation run must be a non-symlink directory")
    try:
        names = {path.name for path in output.iterdir()}
    except OSError as exc:
        raise EvaluationInputError("evaluation run directory is unreadable") from exc
    if names != _EVIDENCE_FILES:
        raise EvaluationInputError(
            "evaluation run must contain exactly receipt.json, report.json, samples.jsonl"
        )

    samples_bytes = _read_evidence_file(output, "samples.jsonl")
    report_bytes = _read_evidence_file(output, "report.json")
    receipt_bytes = _read_evidence_file(output, "receipt.json")
    _verify_samples(samples_bytes)
    report = _parse_object(report_bytes, "report.json")
    receipt = _parse_object(receipt_bytes, "receipt.json")
    if report_bytes != _canonical_json(report):
        raise EvaluationInputError("report.json: noncanonical JSON")
    if receipt_bytes != _canonical_json(receipt):
        raise EvaluationInputError("receipt.json: noncanonical JSON")
    _validate_schema(
        report,
        Path(schema_root) / "evaluation_report.schema.json",
        "report",
    )
    _validate_schema(
        receipt,
        Path(schema_root) / "evaluation_receipt.schema.json",
        "receipt",
    )

    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    samples_sha256 = hashlib.sha256(samples_bytes).hexdigest()
    expected = {
        "run_id": report["run_id"],
        "report_sha256": report_sha256,
        "samples_sha256": samples_sha256,
        "bundle_sha256": report["bundle"]["bundle_sha256"],
        "suite_sha256": report["suite"]["suite_sha256"],
        "profile_sha256": report["profile"]["profile_sha256"],
        "overall_status": report["decisions"]["overall_status"],
    }
    if report["samples_sha256"] != samples_sha256:
        raise EvaluationInputError("samples hash differs from report")
    for key, value in expected.items():
        if receipt[key] != value:
            raise EvaluationInputError(f"receipt {key} hash/identity mismatch")
    return receipt


__all__ = ["build_report", "publish_run", "verify_run"]
