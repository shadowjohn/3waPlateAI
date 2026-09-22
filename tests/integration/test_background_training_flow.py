from __future__ import annotations

import json
import shutil
from pathlib import Path


def test_real_cpu_worker_training_publishes_a_task_local_bundle(
    tmp_path: Path, v1_train_dir: Path, v1_validation_dir: Path
) -> None:
    from plateai_shared.bundle import validate_crop_bundle
    from plateai_web.training_store import TrainingStore
    from plateai_web.training_worker import run_worker

    root = tmp_path / "isolated-project"
    train_directory = root / "out" / "train-fixture"
    validation_directory = root / "out" / "val-fixture"
    shutil.copytree(v1_train_dir, train_directory)
    shutil.copytree(v1_validation_dir, validation_directory)
    sentinel = root / "models" / "bundles" / "active-v1" / "sentinel.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("must not be replaced", encoding="utf-8")

    store = TrainingStore(root / "runs" / ".web-training")
    task_id = "0a1b2c3d4e5f"
    store.create(
        task_id,
        "模型訓練",
        {
            "run_name": "cpu-integration",
            "train_dataset": str(train_directory),
            "validation_dataset": str(validation_directory),
            "epochs": 1,
            "batch_size": 2,
            "device": "cpu",
        },
    )

    assert run_worker(root, task_id) == 0
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == "completed"
    assert task["progress"] == 100
    result = task["result"]
    assert Path(result["checkpoint"]).is_file()
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert result["final_accuracy"] == round(report["history"][-1]["val_acc"] * 100, 1)
    bundle = Path(result["bundle_dir"])
    assert bundle.parent == root / "models" / "bundles"
    validate_crop_bundle(bundle, Path(__file__).resolve().parents[2] / "schemas" / "model_manifest.schema.json")
    assert sentinel.read_text(encoding="utf-8") == "must not be replaced"
