"""Frozen population, technical, and data-rights gates for M4.5."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

from plateai_shared.contracts import JsonValue

from .contracts import (
    EvaluationInputError,
    GateDecision,
    LaneName,
    LaneResult,
    ParityResult,
    PopulationCounts,
    RightsStatus,
    Scene,
    StrictProfile,
    SuiteSnapshot,
)


def _lane_scenes(snapshot: SuiteSnapshot, lane: LaneName) -> tuple[Scene, ...]:
    try:
        scene_ids = snapshot.lane_scene_ids[lane]
    except KeyError as exc:
        raise EvaluationInputError(f"suite missing lane: {lane}") from exc
    if len(set(scene_ids)) != len(scene_ids):
        raise EvaluationInputError(f"{lane}: duplicate scene_id in lane routing")
    by_id = {scene.scene_id: scene for scene in snapshot.scenes}
    if len(by_id) != len(snapshot.scenes):
        raise EvaluationInputError("suite contains duplicate scene_id")
    missing = sorted(set(scene_ids) - set(by_id))
    if missing:
        raise EvaluationInputError(
            f"{lane}: unknown scene_id {missing[0]} in lane routing"
        )
    return tuple(by_id[scene_id] for scene_id in scene_ids)


def _validate_audits(lane: LaneName, scenes: Sequence[Scene]) -> None:
    for scene in scenes:
        if lane in {"oracle_ocr", "end_to_end"}:
            if not scene.plates:
                raise EvaluationInputError(
                    f"{scene.scene_id}: text lane requires a plate"
                )
            for plate in scene.plates:
                if not plate.text_audited or not plate.canonical:
                    raise EvaluationInputError(
                        f"{scene.scene_id}/{plate.instance_id}: audited canonical text required"
                    )
        if any(not plate.geometry_audited for plate in scene.plates):
            raise EvaluationInputError(
                f"{scene.scene_id}: audited geometry required"
            )


def preflight(snapshot: SuiteSnapshot, profile: StrictProfile) -> PopulationCounts:
    """Reject every population shortage before any model runtime is constructed."""

    oracle = _lane_scenes(snapshot, "oracle_ocr")
    detector = _lane_scenes(snapshot, "detector")
    e2e = _lane_scenes(snapshot, "end_to_end")
    parity = _lane_scenes(snapshot, "parity")
    for lane, scenes in (
        ("oracle_ocr", oracle),
        ("detector", detector),
        ("end_to_end", e2e),
        ("parity", parity),
    ):
        _validate_audits(lane, scenes)  # type: ignore[arg-type]

    counts = PopulationCounts(
        oracle_plate_instances=sum(len(scene.plates) for scene in oracle),
        detector_scenes=len(detector),
        detector_plate_instances=sum(len(scene.plates) for scene in detector),
        e2e_scenes=len(e2e),
        parity_scenes=len(parity),
    )
    observed = counts.as_mapping()
    shortages = [
        (key, int(observed[key]), int(profile.minimums[key]))
        for key in sorted(profile.minimums)
        if int(observed[key]) < int(profile.minimums[key])
    ]
    if shortages:
        rendered = "; ".join(
            f"{key}={value}<{required}" for key, value, required in shortages
        )
        raise EvaluationInputError(f"evaluation population below minimum: {rendered}")
    return counts


def _number(value: JsonValue, metric: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EvaluationInputError(f"{metric}: metric must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationInputError(f"{metric}: metric must be finite")
    return result


def _check(
    metric: str,
    observed: int | float | bool,
    operator: str,
    required: int | float | bool,
) -> Mapping[str, JsonValue]:
    if operator == ">=":
        passed = observed >= required
    elif operator == "<=":
        passed = observed <= required
    elif operator == "==":
        passed = observed == required
    else:  # pragma: no cover - all callers use the frozen operators above.
        raise AssertionError(operator)
    return {
        "metric": metric,
        "observed": observed,
        "operator": operator,
        "required": required,
        "passed": bool(passed),
    }


def _result_map(results: Sequence[LaneResult]) -> Mapping[LaneName, LaneResult]:
    mapped: dict[LaneName, LaneResult] = {}
    for result in results:
        if result.lane == "parity":
            raise EvaluationInputError("parity must use the ParityResult contract")
        if result.lane in mapped:
            raise EvaluationInputError(f"duplicate lane result: {result.lane}")
        mapped[result.lane] = result
    required = {"oracle_ocr", "detector", "end_to_end"}
    if set(mapped) != required:
        missing = sorted(required - set(mapped))
        extra = sorted(set(mapped) - required)
        detail = f"missing={missing}, extra={extra}"
        raise EvaluationInputError(f"lane results incomplete: {detail}")
    return mapped


def _metric(result: LaneResult, key: str, reported_name: str) -> float:
    if key not in result.metrics:
        raise EvaluationInputError(f"{reported_name}: metric is missing")
    return _number(result.metrics[key], reported_name)


def _rights_gate(
    bundle_manifest: Mapping[str, JsonValue], snapshot: SuiteSnapshot
) -> RightsStatus:
    provenance = bundle_manifest.get("provenance")
    license_reviewed = (
        provenance.get("license_reviewed")
        if isinstance(provenance, Mapping)
        else None
    )
    if license_reviewed is not True or any(
        source.rights.review_status == "unreviewed" for source in snapshot.sources
    ):
        return "unreviewed"
    if any(
        source.rights.review_status == "restricted"
        or source.rights.local_only
        or "evaluation" not in source.rights.allowed_uses
        for source in snapshot.sources
    ):
        return "restricted"
    return "pass"


def decide_gates(
    profile: StrictProfile,
    bundle_manifest: Mapping[str, JsonValue],
    snapshot: SuiteSnapshot,
    lane_results: Sequence[LaneResult],
    parity_result: ParityResult,
) -> GateDecision:
    """Apply technical and rights decisions independently, then map final status."""

    preflight(snapshot, profile)
    results = _result_map(lane_results)
    oracle = results["oracle_ocr"]
    detector = results["detector"]
    e2e = results["end_to_end"]
    expected_scenes = {
        lane: len(_lane_scenes(snapshot, lane))
        for lane in ("oracle_ocr", "detector", "end_to_end")
    }
    expected_instances = {
        lane: sum(len(scene.plates) for scene in _lane_scenes(snapshot, lane))
        for lane in ("oracle_ocr", "detector", "end_to_end")
    }

    checks: list[Mapping[str, JsonValue]] = []
    for lane in ("oracle_ocr", "detector", "end_to_end"):
        result = results[lane]
        prefix = "e2e" if lane == "end_to_end" else lane.replace("_ocr", "")
        checks.extend(
            (
                _check(
                    f"{prefix}_declared_scene_count",
                    result.scene_count,
                    "==",
                    expected_scenes[lane],
                ),
                _check(
                    f"{prefix}_declared_instance_count",
                    result.instance_count,
                    "==",
                    expected_instances[lane],
                ),
                _check(
                    f"{prefix}_processed_scene_count",
                    result.processed_scene_count,
                    "==",
                    result.scene_count,
                ),
                _check(
                    f"{prefix}_processed_instance_count",
                    result.processed_instance_count,
                    "==",
                    result.instance_count,
                ),
            )
        )

    threshold_specs = (
        (
            "oracle_exact_match",
            oracle,
            "exact_match",
            ">=",
            "oracle_exact_match_min",
        ),
        (
            "oracle_micro_cer",
            oracle,
            "micro_cer",
            "<=",
            "oracle_micro_cer_max",
        ),
        (
            "detector_bbox_precision",
            detector,
            "bbox_precision",
            ">=",
            "detector_bbox_precision_min",
        ),
        (
            "detector_bbox_recall",
            detector,
            "bbox_recall",
            ">=",
            "detector_bbox_recall_min",
        ),
        (
            "detector_complete_quad_precision",
            detector,
            "complete_quad_precision",
            ">=",
            "detector_complete_quad_precision_min",
        ),
        (
            "detector_complete_quad_recall",
            detector,
            "complete_quad_recall",
            ">=",
            "detector_complete_quad_recall_min",
        ),
        (
            "e2e_exact_match",
            e2e,
            "exact_match",
            ">=",
            "e2e_exact_match_min",
        ),
        (
            "e2e_micro_cer",
            e2e,
            "micro_cer",
            "<=",
            "e2e_micro_cer_max",
        ),
        (
            "e2e_localization_recall",
            e2e,
            "localization_recall",
            ">=",
            "e2e_localization_recall_min",
        ),
    )
    for metric, result, key, operator, threshold_key in threshold_specs:
        checks.append(
            _check(
                metric,
                _metric(result, key, metric),
                operator,
                float(profile.thresholds[threshold_key]),
            )
        )
    checks.extend(
        (
            _check("native_onnx_parity", parity_result.passed, "==", True),
            _check(
                "parity_batch_sizes",
                parity_result.batch_sizes
                == tuple(profile.parity["batch_sizes"]),
                "==",
                True,
            ),
        )
    )

    technical_gate = (
        "pass" if all(bool(check["passed"]) for check in checks) else "fail"
    )
    rights_gate = _rights_gate(bundle_manifest, snapshot)
    if technical_gate == "fail":
        overall_status = "REJECTED"
    elif rights_gate == "pass":
        overall_status = "PROMOTION_ELIGIBLE"
    else:
        overall_status = "PASS_LOCAL_ONLY"
    return GateDecision(
        technical_gate=technical_gate,
        rights_gate=rights_gate,
        overall_status=overall_status,
        checks=tuple(checks),
    )


__all__ = ["decide_gates", "preflight"]
