from __future__ import annotations

import shutil
from pathlib import Path

import pytest


def _copy_dataset(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


@pytest.mark.parametrize("run_name", ["../escape", "nested/run", "CON", "run name"])
def test_validate_training_request_rejects_unsafe_run_names(
    tmp_path: Path, run_name: str
) -> None:
    from plateai_web.trainer import validate_training_request

    with pytest.raises(ValueError, match="run_name"):
        validate_training_request(
            tmp_path,
            "a1b2c3d4e5f6",
            {"run_name": run_name, "train_dataset": str(tmp_path / "missing")},
        )


def test_validate_training_request_does_not_overwrite_existing_run(
    tmp_path: Path, v1_train_dir: Path
) -> None:
    from plateai_web.trainer import validate_training_request

    train_dataset = _copy_dataset(v1_train_dir, tmp_path / "out" / "train")
    existing_run = tmp_path / "runs" / "already-there"
    existing_run.mkdir(parents=True)
    marker = existing_run / "keep.txt"
    marker.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        validate_training_request(
            tmp_path,
            "a1b2c3d4e5f6",
            {"run_name": "already-there", "train_dataset": str(train_dataset)},
        )

    assert marker.read_text(encoding="utf-8") == "do not replace"


def test_validate_training_request_rejects_matching_validation_seed(
    tmp_path: Path, v1_train_dir: Path
) -> None:
    from plateai_web.trainer import validate_training_request

    train_dataset = _copy_dataset(v1_train_dir, tmp_path / "out" / "train")
    validation_dataset = _copy_dataset(v1_train_dir, tmp_path / "out" / "validation")

    with pytest.raises(ValueError, match="seed"):
        validate_training_request(
            tmp_path,
            "a1b2c3d4e5f6",
            {
                "run_name": "safe-run",
                "train_dataset": str(train_dataset),
                "validation_dataset": str(validation_dataset),
            },
        )


def test_pipeline_keeps_completed_run_when_export_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    v1_train_dir: Path,
    v1_validation_dir: Path,
) -> None:
    from plateai_trainer.export import bundle
    from plateai_web.trainer import run_training_pipeline

    def fail_export(*_args, **_kwargs):
        raise RuntimeError("export failed after training")

    monkeypatch.setattr(bundle, "export_crop_bundle", fail_export)
    results: list[dict] = []

    with pytest.raises(RuntimeError, match="export failed"):
        run_training_pipeline(
            tmp_path,
            "a1b2c3d4e5f6",
            {
                "run_name": "keep-run",
                "train_dataset": str(v1_train_dir),
                "validation_dataset": str(v1_validation_dir),
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
            },
            on_progress=lambda _event: None,
            on_result=results.append,
            check_cancelled=lambda: None,
        )

    assert (tmp_path / "runs" / "keep-run" / "best.pt").is_file()
    assert not (tmp_path / "models" / "bundles" / "train-a1b2c3d4e5f6").exists()
    assert results == []
