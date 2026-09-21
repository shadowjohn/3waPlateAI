from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from plateai_shared.bundle import validate_crop_bundle
from plateai_trainer.export.bundle import (
    ExportParityError,
    ExportRequest,
    create_cpu_session,
    export_crop_bundle,
)
from plateai_trainer.training.engine import TrainingConfig, train_recognizer
from tests.conftest import ROOT


SCHEMA = ROOT / "schemas/model_manifest.schema.json"


@pytest.fixture
def trained_run(v1_train_dir, v1_validation_dir, tmp_path):
    return train_recognizer(
        TrainingConfig(
            v1_train_dir, v1_validation_dir, tmp_path / "trained", epochs=1, batch_size=2, seed=9
        )
    )


def _request(run, output: Path) -> ExportRequest:
    return ExportRequest(run.best_checkpoint, run.report_path, output)


def _failing_session(failure):
    class Session:
        def __init__(self, path):
            self._session = create_cpu_session(path)

        def get_inputs(self):
            return self._session.get_inputs()

        def get_outputs(self):
            return self._session.get_outputs()

        def run(self, output_names, input_feed):
            result = self._session.run(output_names, input_feed)
            logits = result[0].copy()
            if failure == "logit_divergence":
                logits[0, 0, 0] += 0.01
            else:
                logits[:] = -100.0
                logits[:, :, 10] = 100.0
            return [logits]

    return Session


def test_exported_bundle_has_v1_manifest_and_native_onnx_parity(trained_run, tmp_path):
    bundle = export_crop_bundle(_request(trained_run, tmp_path / "bundle"))
    manifest = validate_crop_bundle(bundle, SCHEMA)

    assert manifest["decoder"]["blank_index"] == 0
    assert manifest["components"]["recognizer"]["outputs"][0]["shape"] == ["batch", 80, 34]


@pytest.mark.parametrize("failure", ["logit_divergence", "decode_divergence"])
def test_export_refuses_parity_failure_without_publishing_output(trained_run, tmp_path, failure):
    output = tmp_path / "bad"
    with pytest.raises(ExportParityError):
        export_crop_bundle(_request(trained_run, output), session_factory=_failing_session(failure))
    assert not output.exists()


def test_exported_bundle_supports_35_class_unified_charset(tmp_path):
    from plateai_trainer.synthetic.dataset import generate_dataset
    from plateai_trainer.synthetic.models import GenerationRequest
    from tests.conftest import V1_TEMPLATE, V1_NONE_AUGMENTATION

    charset_path = ROOT / "configs/charsets/tw_standard_v1.txt"
    rules_path = ROOT / "configs/plate_rules/tw_standard_v1.json"

    train_dir = tmp_path / "std_train"
    val_dir = tmp_path / "std_val"

    generate_dataset(
        GenerationRequest(
            count=2,
            seed=11,
            output=train_dir,
            charset_path=charset_path,
            rules_path=rules_path,
            template_path=V1_TEMPLATE,
            augmentation_path=V1_NONE_AUGMENTATION,
        )
    )
    generate_dataset(
        GenerationRequest(
            count=2,
            seed=22,
            output=val_dir,
            charset_path=charset_path,
            rules_path=rules_path,
            template_path=V1_TEMPLATE,
            augmentation_path=V1_NONE_AUGMENTATION,
        )
    )

    run = train_recognizer(
        TrainingConfig(
            train_directory=train_dir,
            validation_directory=val_dir,
            output_directory=tmp_path / "std_run",
            epochs=1,
            batch_size=2,
            seed=9,
            charset_path=charset_path,
            rules_path=rules_path,
        )
    )

    bundle_dir = tmp_path / "std_bundle"
    export_crop_bundle(
        ExportRequest(
            checkpoint=run.best_checkpoint,
            report=run.report_path,
            output=bundle_dir,
            charset_path=charset_path,
            rules_path=rules_path,
        )
    )

    manifest = validate_crop_bundle(bundle_dir, SCHEMA)
    assert manifest["charset"]["visible_symbols"] == 34
    assert manifest["decoder"]["class_count"] == 35
    assert manifest["components"]["recognizer"]["outputs"][0]["shape"] == ["batch", 80, 35]


def test_installed_plateai_export_command_shows_help():
    executable = Path(sys.executable).with_name("plateai-export.exe")
    result = subprocess.run(
        [str(executable), "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0
    assert "--checkpoint" in result.stdout
