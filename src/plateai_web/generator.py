"""Synthetic dataset generator wrapper with real-time 100-sample progress updates."""
from __future__ import annotations

import time
import uuid
import re
from pathlib import Path
from plateai_trainer.synthetic.dataset import generate_dataset
from plateai_trainer.synthetic.models import GenerationRequest
from .paths import workspace_root
from .tasks import TaskManager


def generate_dataset_task(
    task_id: str,
    tm: TaskManager,
    count: int = 10000,
    font: str = "noto_mono",
    seed: int = 42,
    output_dir_name: str | None = None,
):
    root = workspace_root()
    name = output_dir_name or f"generated-{count}-{uuid.uuid4().hex[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("output_name must be a simple directory name")
    out_dir = root / "out" / name

    tm.update_progress(task_id, 1, f"準備生成 {count} 張車牌...")
    tm.append_log(task_id, "=== 3waPlateAI 車牌合成引擎 ===")
    tm.append_log(task_id, f"生成目標數量: {count} 張")
    tm.append_log(task_id, f"使用車牌字型: {font}")
    tm.append_log(task_id, f"隨機種子: {seed}")
    tm.append_log(task_id, f"輸出路徑: {out_dir}")

    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"輸出目錄已存在，保留原資料：{out_dir}")

    font_request = None if font in {"", "noto_mono"} else font

    # Standard configuration paths
    charset_path = root / "configs" / "charsets" / "tw_standard_v1.txt"
    if not charset_path.exists():
        charset_path = root / "configs" / "charsets" / "tw_new_style_private_passenger_v1.txt"

    rules_path = root / "configs" / "plate_rules" / "tw_standard_v1.json"
    if not rules_path.exists():
        rules_path = root / "configs" / "plate_rules" / "tw_new_style_private_passenger_v1.json"

    template_path = root / "configs" / "plate_templates" / "tw_unified_v1.json"
    if not template_path.exists():
        template_path = root / "configs" / "plate_templates" / "new_style_private_passenger_white_v1.json"
    augmentation_path = root / "configs" / "augmentation" / "standard_v1.json"

    request = GenerationRequest(
        count=count,
        seed=seed,
        output=out_dir,
        charset_path=charset_path,
        rules_path=rules_path,
        template_path=template_path,
        augmentation_path=augmentation_path,
        font=font_request,
    )

    t0 = time.perf_counter()
    last_log_time = t0

    def progress_callback(cur: int, total: int, sample_display: str):
        nonlocal last_log_time
        pct = max(1, min(99, int(cur / total * 98)))
        now = time.perf_counter()
        speed = cur / max(0.001, now - t0)
        
        tm.update_progress(task_id, pct, f"已生成 {cur}/{total} 張車牌 ({pct}%)")
        tm.append_log(
            task_id,
            f"[生成進度] {cur:6d}/{total} ({cur/total*100:5.1f}%) | 速度: {speed:5.1f}張/秒 | 最新樣板: {sample_display}"
        )
        last_log_time = now

    try:
        summary = generate_dataset(request, progress_callback=progress_callback)
        total_time = round(time.perf_counter() - t0, 1)
        
        tm.append_log(task_id, f"=== 合成完畢！總耗時: {total_time} 秒, 平均每秒 {count/max(0.1, total_time):.1f} 張 ===")
        tm.append_log(task_id, f"[OK] 資料已成功儲存至: {out_dir}")
        tm.update_progress(task_id, 100, f"成功生成 {count} 張車牌！")
        tm.complete_task(
            task_id,
            result={
                "status": "success",
                "count": summary.generated,
                "output": str(out_dir),
                "duration_seconds": total_time,
            },
            message=f"已成功生成 {count} 張範例車牌至 {out_dir.name} (耗時 {total_time}s)",
        )
    except Exception as exc:
        tm.append_log(task_id, f"[ERROR] 生成失敗: {exc}")
        tm.fail_task(task_id, str(exc))
