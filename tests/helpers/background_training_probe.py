from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _hold_lock(path: Path, *, crash: bool) -> int:
    from plateai_web.training_process import training_lock

    with training_lock(path):
        print("LOCKED", flush=True)
        if crash:
            os._exit(9)
        time.sleep(0.2)
    return 0


def _probe_pipeline(root, task_id, request, *, on_progress, on_result, check_cancelled):
    from plateai_trainer.training.control import TrainingProgress

    print("probe worker started", flush=True)
    for batch in range(1, 301):
        check_cancelled()
        on_progress(
            TrainingProgress(
                kind="batch",
                phase="training",
                epoch=1,
                total_epochs=1,
                batch=batch,
                total_batches=300,
                train_loss=1.0 / batch,
            )
        )
        time.sleep(0.1)
    result = {"probe": True, "run_dir": str(root / "runs" / task_id)}
    on_result(result)
    return result


def _worker(root: Path, task_id: str) -> int:
    from plateai_web.training_worker import run_worker

    return run_worker(root, task_id, pipeline=_probe_pipeline)


def _parent(root: Path, task_id: str, log_path: Path) -> int:
    from plateai_web.training_process import _spawn_detached

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "worker",
        "--root",
        str(root),
        "--task-id",
        task_id,
    ]
    _spawn_detached(command, cwd=root, log_path=log_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("lock", "lock-crash", "worker", "parent"))
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args(argv)
    if args.mode in {"lock", "lock-crash"}:
        if args.lock is None:
            parser.error("--lock is required")
        return _hold_lock(args.lock, crash=args.mode == "lock-crash")
    if args.mode == "worker":
        if args.root is None or args.task_id is None:
            parser.error("--root and --task-id are required")
        return _worker(args.root, args.task_id)
    if args.root is None or args.task_id is None or args.log is None:
        parser.error("--root, --task-id, and --log are required")
    return _parent(args.root, args.task_id, args.log)


if __name__ == "__main__":
    raise SystemExit(main())
