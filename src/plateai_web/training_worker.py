"""Detached training worker which owns one persisted Web training task."""
from __future__ import annotations

import argparse
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from plateai_trainer.training.control import TrainingCancelled, TrainingProgress

from .training_process import TrainingBusy, current_process_identity, training_lock
from .training_store import TrainingStore


HEARTBEAT_INTERVAL_SECONDS = 3.0


def _state_dir(root: Path) -> Path:
    return root / "runs" / ".web-training"


def _progress_for(event: TrainingProgress, previous: int) -> tuple[str, int, str]:
    phase = event.phase
    store_phase = "preparing" if phase.startswith("generating_") else phase
    if store_phase not in {"preparing", "training", "validating", "exporting"}:
        store_phase = "preparing"
    progress = previous
    if phase.startswith("generating_"):
        progress = max(previous, 3)
        message = "正在準備訓練資料..." if phase == "generating_train" else "正在準備驗證資料..."
    elif phase == "preparing":
        progress, message = max(previous, 3), "正在準備訓練..."
    elif phase == "exporting":
        progress, message = max(previous, 88), "訓練完成，正在匯出 ONNX bundle..."
    elif phase == "training":
        if event.total_epochs > 0 and event.total_batches > 0:
            ratio = (event.epoch - 1 + 0.8 * event.batch / event.total_batches) / event.total_epochs
            progress = max(previous, int(10 + 75 * ratio))
        else:
            progress = max(previous, 10)
        message = f"訓練中 Epoch {event.epoch}/{event.total_epochs}"
    elif phase == "validating":
        if event.total_epochs > 0 and event.total_batches > 0:
            ratio = (event.epoch - 1 + 0.8 + 0.2 * event.batch / event.total_batches) / event.total_epochs
            progress = max(previous, int(10 + 75 * ratio))
        elif event.kind == "epoch" and event.total_epochs > 0:
            progress = max(previous, int(10 + 75 * event.epoch / event.total_epochs))
        else:
            progress = max(previous, 10)
        message = f"驗證中 Epoch {event.epoch}/{event.total_epochs}"
    else:
        message = "正在處理訓練任務..."
    return store_phase, min(progress, 99), message


def run_worker(root: Path, task_id: str, *, pipeline: Callable[..., dict] | None = None) -> int:
    """Claim and execute a task, retaining outputs even when a late cancel wins."""

    root = Path(root).resolve()
    store = TrainingStore(_state_dir(root))
    try:
        with training_lock(store.state_dir / "worker.lock"):
            task = store.get(task_id)
            if task is None:
                print(f"unknown training task: {task_id}", flush=True)
                return 1
            if task["status"] in {"completed", "failed", "cancelled"}:
                return 0
            if task["cancel_requested"]:
                store.finish(task_id, "cancelled", message="已在啟動前取消", result=task["result"])
                return 0
            process_id, start_token = current_process_identity()
            if not store.claim(task_id, process_id, start_token):
                refreshed = store.get(task_id)
                if refreshed is not None and refreshed["cancel_requested"]:
                    store.finish(
                        task_id, "cancelled", message="已在啟動前取消", result=refreshed["result"]
                    )
                return 0

            stop_heartbeat = threading.Event()
            cache_lock = threading.Lock()
            storage_error: list[BaseException] = []
            result_cache = dict(task["result"])
            last_progress = int(task["progress"])
            last_snapshot = 0.0

            def heartbeat_loop() -> None:
                while not stop_heartbeat.wait(HEARTBEAT_INTERVAL_SECONDS):
                    try:
                        if not store.heartbeat(task_id):
                            raise OSError("training task is no longer running")
                    except BaseException as exc:
                        with cache_lock:
                            if not storage_error:
                                storage_error.append(exc)
                        return

            heartbeat = threading.Thread(target=heartbeat_loop, name="training-heartbeat", daemon=True)
            heartbeat.start()

            def check_cancelled() -> None:
                with cache_lock:
                    failure = storage_error[0] if storage_error else None
                if failure is not None:
                    raise OSError(f"training status storage failed: {failure}") from failure
                if store.cancel_requested(task_id):
                    raise TrainingCancelled("training cancelled by user")

            def on_progress(event: TrainingProgress) -> None:
                nonlocal last_progress, last_snapshot
                check_cancelled()
                now = time.monotonic()
                with cache_lock:
                    phase, progress, message = _progress_for(event, last_progress)
                    if task['request'].get('kind') == 'pose':
                        message = {'preparing': 'Pose 圖片、hash 與分組校驗中',
                                   'validating': 'Pose 36 張 holdout 評估中（不挑選模型）',
                                   'exporting': '打包 Pose .pt 與 manifest；不自動啟用'}.get(phase, message)
                    if event.kind == "batch":
                        result_cache["current_batch"] = event.batch
                        result_cache["total_batches"] = event.total_batches
                        result_cache["current_phase"] = event.phase
                    if event.kind == "epoch":
                        history = result_cache.setdefault("history", [])
                        if isinstance(history, list):
                            record = {
                                "epoch": event.epoch,
                                "train_loss": event.train_loss,
                                "val_loss": event.val_loss,
                                "val_acc": round(float(event.val_acc) * 100, 1) if event.val_acc is not None else None,
                            }
                            if not history or history[-1].get("epoch") != event.epoch:
                                history.append(record)
                            else:
                                history[-1] = record
                        result_cache["current_metrics"] = result_cache.get("history", [None])[-1]
                        result_cache["total_epochs"] = event.total_epochs
                    forced = event.kind in {"stage", "epoch"}
                    if not forced and now - last_snapshot < 1.0:
                        return
                    last_progress = progress
                    last_snapshot = now
                    snapshot_result = dict(result_cache)
                if not store.snapshot(
                    task_id,
                    phase=phase,
                    progress=progress,
                    message=message,
                    result=snapshot_result,
                    advanced=event.kind != "stage",
                ):
                    raise OSError("training task snapshot was not accepted")

            def on_result(result: dict) -> None:
                if not isinstance(result, dict):
                    raise ValueError("training pipeline result must be an object")
                with cache_lock:
                    result_cache.update(result)

            try:
                if pipeline is None:
                    if task['request'].get('kind') == 'pose':
                        from .pose_training import run_pose_pipeline
                        selected_pipeline = run_pose_pipeline
                    else:
                        from .trainer import run_training_pipeline
                        selected_pipeline = run_training_pipeline
                else:
                    selected_pipeline = pipeline
                result = selected_pipeline(
                    root,
                    task_id,
                    task["request"],
                    on_progress=on_progress,
                    on_result=on_result,
                    check_cancelled=check_cancelled,
                )
                on_result(result)
                check_cancelled()
                with cache_lock:
                    completed_result = dict(result_cache)
                if not store.finish(
                    task_id,
                    "completed",
                    message="Pose 模型包已完成，尚未啟用" if task['request'].get('kind') == 'pose' else "訓練與匯出完成，尚未啟用",
                    result=completed_result,
                ) and store.cancel_requested(task_id):
                    store.finish(
                        task_id,
                        "cancelled",
                        message="已停止；已發布產物保留",
                        result=completed_result,
                    )
                return 0
            except TrainingCancelled:
                with cache_lock:
                    cancelled_result = dict(result_cache)
                store.finish(
                    task_id,
                    "cancelled",
                    message="訓練已停止；已發布產物保留" if cancelled_result else "訓練已停止",
                    result=cancelled_result,
                )
                return 0
            except BaseException as exc:
                traceback.print_exc()
                with cache_lock:
                    failed_result = dict(result_cache)
                changed = store.finish(
                    task_id,
                    "failed",
                    message="訓練失敗，請查看 log",
                    error=f"{type(exc).__name__}: {exc}",
                    result=failed_result,
                )
                if not changed and store.cancel_requested(task_id):
                    store.finish(
                        task_id,
                        "cancelled",
                        message="已停止；已發布產物保留",
                        result=failed_result,
                    )
                    return 0
                return 1
            finally:
                stop_heartbeat.set()
                heartbeat.join(timeout=5.5)
    except TrainingBusy:
        print("another training worker already owns the machine lock", flush=True)
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="3waPlateAI detached training worker")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args(argv)
    return run_worker(args.root, args.task_id)


if __name__ == "__main__":
    raise SystemExit(main())
