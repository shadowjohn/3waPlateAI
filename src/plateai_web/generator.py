"""Synthetic dataset generator wrapper with real-time 100-sample progress updates."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from plateai_trainer.synthetic.dataset import generate_dataset
from plateai_trainer.synthetic.models import GenerationRequest
from .tasks import TaskManager


def generate_dataset_task(
    task_id: str,
    tm: TaskManager,
    count: int = 10000,
    font: str = "taiwan_plate",
    seed: int = 42,
    output_dir_name: str = "demo-10000",
):
    root = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "out" / output_dir_name

    tm.update_progress(task_id, 1, f"準備生成 {count} 張車牌...")
    tm.append_log(task_id, "=== 3waPlateAI 車牌合成引擎 ===")
    tm.append_log(task_id, f"生成目標數量: {count} 張")
    tm.append_log(task_id, f"使用車牌字型: {font}")
    tm.append_log(task_id, f"隨機種子: {seed}")
    tm.append_log(task_id, f"輸出路徑: {out_dir}")

    if out_dir.exists():
        import shutil
        tm.append_log(task_id, f"[更新] 清理舊有的 {out_dir.name} 目錄，重新生成標準訓練集...")
        shutil.rmtree(out_dir)

    # Determine font path
    font_path: Path | None = None
    if font == "taiwan_plate":
        candidate = root / "assets" / "fonts" / "TaiwanPlate-Regular.ttf"
        if candidate.exists():
            font_path = candidate
        else:
            candidate2 = root / "assets" / "fonts" / "NotoSansMono-Regular.ttf"
            if candidate2.exists():
                font_path = candidate2

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
        font=font_path,
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
