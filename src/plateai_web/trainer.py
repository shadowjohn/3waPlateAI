"""Model trainer and exporter wrapper."""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path
from .tasks import TaskManager


def train_model_task(
    task_id: str,
    tm: TaskManager,
    epochs: int = 5,
    train_dir: str | None = None,
    val_dir: str | None = None,
    run_name: str | None = None,
):
    root = Path(__file__).resolve().parent.parent.parent
    if not run_name:
        run_name = f"run-{uuid.uuid4().hex[:6]}"
        
    runs_dir = root / "runs" / run_name
    model_bundle_dir = root / "models" / "bundles" / "active-v1"
    
    tm.update_progress(task_id, 5, "正在準備訓練環境...")
    tm.append_log(task_id, f"=== 3waPlateAI PyTorch CTC 辨識模型訓練 ===")
    tm.append_log(task_id, f"目標 Epochs: {epochs}")
    tm.append_log(task_id, f"Run 輸出目錄: {runs_dir}")
    
    # Check or generate minimal train/val set if not specified
    if not train_dir or not Path(train_dir).exists():
        # Look for existing out directories
        existing_out = list((root / "out").glob("*train*"))
        if existing_out and existing_out[0].is_dir():
            train_dir = str(existing_out[0])
            tm.append_log(task_id, f"使用既有訓練集: {train_dir}")
        else:
            default_train = root / "out" / "train-default"
            if not default_train.exists():
                tm.append_log(task_id, "尚未指定訓練資料集，自動快速合成 2,000 張訓練樣本...")
                from .generator import generate_dataset_task
                generate_dataset_task(task_id, tm, count=2000, font="taiwan_plate", output_dir_name="train-default")
            train_dir = str(default_train)

    if not val_dir or not Path(val_dir).exists():
        default_val = root / "out" / "val-default"
        if not default_val.exists():
            tm.append_log(task_id, "自動快速合成 500 張驗證樣本...")
            from .generator import generate_dataset_task
            generate_dataset_task(task_id, tm, count=500, font="taiwan_plate", output_dir_name="val-default")
        val_dir = str(default_val)

    tm.update_progress(task_id, 20, "開始啟動 PyTorch CTC 訓練進程...")
    py_exec = sys.executable
    charset = root / "configs" / "charsets" / "tw_standard_v1.txt"
    rules = root / "configs" / "plate_rules" / "tw_standard_v1.json"
    
    cmd_train = [
        py_exec,
        "-m", "plateai_trainer.training.cli",
        "--train", str(train_dir),
        "--validation", str(val_dir),
        "--output", str(runs_dir),
        "--charset", str(charset),
        "--rules", str(rules),
        "--epochs", str(epochs),
        "--device", "auto",
    ]

    def on_train_line(line: str):
        m = re.search(r"epoch\s*(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            pct = int(20 + (cur / total) * 60)
            tm.update_progress(task_id, pct, f"訓練中 Epoch {cur}/{total} ({pct}%)")

    rc = tm.run_subprocess_command(task_id, cmd_train, cwd=str(root), on_line=on_train_line)
    if rc != 0:
        tm.fail_task(task_id, f"模型訓練中斷，Exit Code: {rc}")
        return

    tm.update_progress(task_id, 85, "訓練完成！正在匯出 ONNX Model Bundle...")
    checkpoint_pt = runs_dir / "best.pt"
    report_json = runs_dir / "report.json"
    
    cmd_export = [
        py_exec,
        "-m", "plateai_trainer.export.cli",
        "--checkpoint", str(checkpoint_pt),
        "--report", str(report_json),
        "--output", str(model_bundle_dir),
        "--charset", str(charset),
        "--rules", str(rules),
    ]

    rc_exp = tm.run_subprocess_command(task_id, cmd_export, cwd=str(root))
    if rc_exp == 0:
        tm.update_progress(task_id, 100, "模型訓練與 ONNX 匯出全部就緒！")
        tm.complete_task(
            task_id,
            result={
                "status": "success",
                "bundle_dir": str(model_bundle_dir),
                "run_dir": str(runs_dir),
            },
            message="模型訓練與 ONNX Bundle 匯出成功！",
        )
    else:
        tm.fail_task(task_id, f"ONNX 匯出失敗，Exit Code: {rc_exp}")
