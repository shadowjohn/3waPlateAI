from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import MappingProxyType

import pytest

from plateai_eval.cli import build_parser
from plateai_eval.contracts import (
    EvaluationInputError,
    PlateTruth,
    Scene,
    SuiteSnapshot,
    load_profile,
)
from plateai_eval.gates import preflight


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_strict_profile_is_canonical_and_rejects_relaxed_copy(tmp_path: Path) -> None:
    profile_path = PROJECT_ROOT / "configs" / "evaluation" / "m4_5_strict_v1.json"
    schema_path = PROJECT_ROOT / "schemas" / "evaluation_profile.schema.json"
    profile = load_profile(profile_path, schema_path)

    assert profile.profile_id == "m4.5-strict-v1"
    assert profile.minimums["oracle_plate_instances"] == 200

    relaxed = json.loads(profile_path.read_text(encoding="utf-8"))
    relaxed["thresholds"]["oracle_exact_match_min"] = 0.0
    changed_path = tmp_path / "relaxed-profile.json"
    changed_path.write_text(json.dumps(relaxed, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(EvaluationInputError, match="oracle_exact_match_min"):
        load_profile(changed_path, schema_path)


def test_preflight_fails_closed_for_an_undersized_suite() -> None:
    plate = PlateTruth(
        instance_id="plate-001",
        corners=((0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)),
        canonical="ABC1234",
        display="ABC-1234",
        geometry_audited=True,
        text_audited=True,
    )
    scene = Scene(
        scene_id="scene-001",
        source_id="source-001",
        image_path=Path("image.png"),
        image_sha256="0" * 64,
        width=10,
        height=4,
        plates=(plate,),
    )
    snapshot = SuiteSnapshot(
        path=Path("suite.json"),
        raw_bytes=b"{}",
        sha256="0" * 64,
        snapshot_sha256="0" * 64,
        suite_id="test-suite",
        revision="test",
        sources=(),
        manifests=(),
        scenes=(scene,),
        lane_scene_ids=MappingProxyType(
            {
                "oracle_ocr": (scene.scene_id,),
                "detector": (scene.scene_id,),
                "end_to_end": (scene.scene_id,),
                "parity": (scene.scene_id,),
            }
        ),
        file_hashes=MappingProxyType({}),
    )
    profile = load_profile(
        PROJECT_ROOT / "configs" / "evaluation" / "m4_5_strict_v1.json",
        PROJECT_ROOT / "schemas" / "evaluation_profile.schema.json",
    )

    with pytest.raises(EvaluationInputError, match="evaluation population below minimum"):
        preflight(snapshot, profile)


def test_source_package_declares_the_evaluation_entry_point_and_data_files() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["plateai-eval"] == "plateai_eval.cli:main"
    data_files = project["tool"]["setuptools"]["data-files"]
    assert data_files["share/3wa-plate-ai/configs/evaluation"] == [
        "configs/evaluation/*.json"
    ]
    assert data_files["share/3wa-plate-ai/web"] == ["web/*.html"]
    assert data_files["share/3wa-plate-ai/web/assets"] == [
        "web/assets/*.jpg",
        "web/assets/*.png",
    ]
    assert "verify" in build_parser().format_help()
