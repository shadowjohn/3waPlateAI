"""Task management for background worker jobs with real-time log capturing."""
from __future__ import annotations

import collections
import dataclasses
import enum
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclasses.dataclass
class TaskInfo:
    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    message: str = "準備中..."
    logs: list[str] = dataclasses.field(default_factory=list)
    result: dict[str, Any] = dataclasses.field(default_factory=dict)
    error: str | None = None
    created_at: float = dataclasses.field(default_factory=time.time)
    updated_at: float = dataclasses.field(default_factory=time.time)


class TaskManager:
    """In-memory thread-safe background task manager."""

    def __init__(self, max_history: int = 50, max_logs_per_task: int = 1500):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()
        self._max_history = max_history
        self._max_logs = max_logs_per_task

    def create_task(self, name: str) -> str:
        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            info = TaskInfo(id=task_id, name=name)
            self._tasks[task_id] = info
        return task_id

    def get_task(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[TaskInfo]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)

    def update_progress(self, task_id: str, progress: int, message: str | None = None):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.progress = min(100, max(0, progress))
                if message is not None:
                    task.message = message
                task.updated_at = time.time()

    def append_log(self, task_id: str, line: str):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                # strip trailing newline but keep format
                clean_line = line.rstrip("\r\n")
                if clean_line:
                    task.logs.append(clean_line)
                    if len(task.logs) > self._max_logs:
                        task.logs = task.logs[-self._max_logs:]
                    task.updated_at = time.time()

    def complete_task(self, task_id: str, result: dict[str, Any] | None = None, message: str = "執行完成"):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.COMPLETED
                task.progress = 100
                task.message = message
                if result:
                    task.result = result
                task.updated_at = time.time()

    def fail_task(self, task_id: str, error: str):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.FAILED
                task.error = error
                task.message = f"執行失敗: {error}"
                task.updated_at = time.time()

    def run_in_background(
        self,
        name: str,
        target: Callable[[str, TaskManager], None],
    ) -> str:
        task_id = self.create_task(name)

        def runner():
            with self._lock:
                task = self._tasks[task_id]
                task.status = TaskStatus.RUNNING
                task.message = "正在執行..."
            try:
                target(task_id, self)
            except Exception as e:
                self.append_log(task_id, f"[ERROR] {type(e).__name__}: {e}")
                self.fail_task(task_id, str(e))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return task_id

    def run_subprocess_command(
        self,
        task_id: str,
        cmd: list[str],
        cwd: str | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> int:
        """Execute a CLI process and stream logs to task."""
        self.append_log(task_id, f"[CMD] {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                self.append_log(task_id, line)
                if on_line:
                    on_line(line)
            process.stdout.close()
        rc = process.wait()
        return rc


# Global singleton manager
task_manager = TaskManager()
