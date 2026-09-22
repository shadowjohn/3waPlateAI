"""Autocommit persistence for the occasional Web Studio training task."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import deque
from contextlib import closing
from pathlib import Path
from typing import Any, Callable


_TASK_ID = re.compile(r"[0-9a-f]{12}\Z")
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_ACTIVE_STATUSES = ("pending", "running")


class TrainingStore:
    """Persist training state without retaining a database connection."""

    def __init__(
        self,
        state_dir: Path,
        *,
        clock: Callable[[], float] = time.time,
        trace: Callable[[str], None] | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self._clock = clock
        self._trace = trace

    @property
    def _database_path(self) -> Path:
        return self.state_dir / "tasks.sqlite3"

    def _validate_task_id(self, task_id: str) -> str:
        if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
            raise ValueError("task_id must be 12 lowercase hexadecimal characters")
        return task_id

    @staticmethod
    def _json(value: dict[str, Any]) -> str:
        if not isinstance(value, dict):
            raise ValueError("task document fields must be dictionaries")
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)

    @staticmethod
    def _document(row: sqlite3.Row) -> dict[str, Any]:
        document = dict(row)
        document["request"] = json.loads(document.pop("request_json"))
        document["result"] = json.loads(document.pop("result_json"))
        document["cancel_requested"] = bool(document["cancel_requested"])
        return document

    def _connect(self, *, create: bool) -> sqlite3.Connection | None:
        if create:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        elif not self._database_path.is_file():
            return None

        connection = sqlite3.connect(
            self._database_path,
            isolation_level=None,
            timeout=5.0,
        )
        connection.row_factory = sqlite3.Row
        if self._trace is not None:
            connection.set_trace_callback(self._trace)
        if create:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS training_tasks ("
                "id TEXT PRIMARY KEY, name TEXT NOT NULL, request_json TEXT NOT NULL, "
                "status TEXT NOT NULL, phase TEXT NOT NULL, progress INTEGER NOT NULL, "
                "message TEXT NOT NULL, result_json TEXT NOT NULL, error TEXT, "
                "worker_pid INTEGER, worker_start_token TEXT, cancel_requested INTEGER NOT NULL, "
                "created_at REAL NOT NULL, updated_at REAL NOT NULL, heartbeat_at REAL, "
                "last_progress_at REAL)"
            )
        return connection

    def create(self, task_id: str, name: str, request: dict[str, Any]) -> dict[str, Any]:
        task_id = self._validate_task_id(task_id)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("task name must be a non-empty string")
        request_json = self._json(request)
        now = self._clock()
        with closing(self._connect(create=True)) as connection:
            connection.execute(
                "INSERT INTO training_tasks ("
                "id, name, request_json, status, phase, progress, message, result_json, "
                "error, worker_pid, worker_start_token, cancel_requested, created_at, "
                "updated_at, heartbeat_at, last_progress_at) "
                "VALUES (?, ?, ?, 'pending', 'preparing', 0, ?, '{}', NULL, NULL, NULL, "
                "0, ?, ?, NULL, NULL)",
                (task_id, name, request_json, "準備中...", now, now),
            )
            row = connection.execute(
                "SELECT * FROM training_tasks WHERE id=?", (task_id,)
            ).fetchone()
        assert row is not None
        return self._document(row)

    def get(self, task_id: str) -> dict[str, Any] | None:
        task_id = self._validate_task_id(task_id)
        connection = self._connect(create=False)
        if connection is None:
            return None
        with closing(connection):
            row = connection.execute(
                "SELECT * FROM training_tasks WHERE id=?", (task_id,)
            ).fetchone()
        return self._document(row) if row is not None else None

    def active(self) -> list[dict[str, Any]]:
        connection = self._connect(create=False)
        if connection is None:
            return []
        with closing(connection):
            rows = connection.execute(
                "SELECT * FROM training_tasks WHERE status IN ('pending', 'running') "
                "ORDER BY created_at ASC"
            ).fetchall()
        return [self._document(row) for row in rows]

    def latest(self) -> dict[str, Any] | None:
        connection = self._connect(create=False)
        if connection is None:
            return None
        with closing(connection):
            row = connection.execute(
                "SELECT * FROM training_tasks ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return self._document(row) if row is not None else None

    def claim(self, task_id: str, pid: int, start_token: str) -> bool:
        task_id = self._validate_task_id(task_id)
        if not isinstance(pid, int) or pid < 1 or not isinstance(start_token, str) or not start_token:
            raise ValueError("worker identity is invalid")
        connection = self._connect(create=False)
        if connection is None:
            return False
        now = self._clock()
        with closing(connection):
            changed = connection.execute(
                "UPDATE training_tasks SET status='running', worker_pid=?, "
                "worker_start_token=?, updated_at=?, heartbeat_at=? "
                "WHERE id=? AND status='pending' AND cancel_requested=0",
                (pid, start_token, now, now, task_id),
            ).rowcount
        return changed == 1

    def snapshot(
        self,
        task_id: str,
        *,
        phase: str,
        progress: int,
        message: str,
        result: dict[str, Any],
        advanced: bool,
    ) -> bool:
        task_id = self._validate_task_id(task_id)
        if phase not in {"preparing", "training", "validating", "exporting"}:
            raise ValueError("unknown training phase")
        if not isinstance(progress, int) or not 0 <= progress <= 99:
            raise ValueError("progress must be an integer from 0 to 99")
        if not isinstance(message, str):
            raise ValueError("message must be a string")
        result_json = self._json(result)
        connection = self._connect(create=False)
        if connection is None:
            return False
        now = self._clock()
        fields = (
            "phase=?, progress=?, message=?, result_json=?, updated_at=?, "
            "last_progress_at=?"
            if advanced
            else "phase=?, progress=?, message=?, result_json=?, updated_at=?"
        )
        values: tuple[Any, ...] = (
            (phase, progress, message, result_json, now, now, task_id)
            if advanced
            else (phase, progress, message, result_json, now, task_id)
        )
        with closing(connection):
            changed = connection.execute(
                f"UPDATE training_tasks SET {fields} WHERE id=? AND status='running'",
                values,
            ).rowcount
        return changed == 1

    def heartbeat(self, task_id: str) -> bool:
        task_id = self._validate_task_id(task_id)
        connection = self._connect(create=False)
        if connection is None:
            return False
        with closing(connection):
            changed = connection.execute(
                "UPDATE training_tasks SET heartbeat_at=?, updated_at=? "
                "WHERE id=? AND status='running'",
                (self._clock(), self._clock(), task_id),
            ).rowcount
        return changed == 1

    def request_cancel(self, task_id: str) -> bool:
        task_id = self._validate_task_id(task_id)
        connection = self._connect(create=False)
        if connection is None:
            return False
        with closing(connection):
            changed = connection.execute(
                "UPDATE training_tasks SET cancel_requested=1, updated_at=? "
                "WHERE id=? AND status IN ('pending', 'running')",
                (self._clock(), task_id),
            ).rowcount
        return changed == 1

    def cancel_requested(self, task_id: str) -> bool:
        task_id = self._validate_task_id(task_id)
        document = self.get(task_id)
        if document is None:
            raise KeyError(f"unknown training task: {task_id}")
        return bool(document["cancel_requested"])

    def finish(
        self,
        task_id: str,
        status: str,
        *,
        message: str,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> bool:
        task_id = self._validate_task_id(task_id)
        if status not in _TERMINAL_STATUSES:
            raise ValueError("status must be completed, failed, or cancelled")
        if not isinstance(message, str):
            raise ValueError("message must be a string")
        if error is not None and not isinstance(error, str):
            raise ValueError("error must be a string or None")
        connection = self._connect(create=False)
        if connection is None:
            return False
        now = self._clock()
        result_assignment = "result_json=?" if result is not None else "result_json=result_json"
        result_values: tuple[Any, ...] = (self._json(result),) if result is not None else ()
        if status == "completed":
            condition = "status='running' AND cancel_requested=0"
            progress = 100
        else:
            condition = "status IN ('pending', 'running')"
            progress = 0
        with closing(connection):
            changed = connection.execute(
                f"UPDATE training_tasks SET status=?, progress=?, message=?, error=?, "
                f"{result_assignment}, updated_at=? WHERE id=? AND {condition}",
                (status, progress, message, error, *result_values, now, task_id),
            ).rowcount
        return changed == 1

    def fail_if_unchanged(self, observed: dict[str, Any], error: str) -> bool:
        task_id = self._validate_task_id(str(observed.get("id", "")))
        if not isinstance(error, str) or not error:
            raise ValueError("error must be a non-empty string")
        required = ("status", "updated_at", "worker_pid", "worker_start_token")
        if any(key not in observed for key in required):
            raise ValueError("observed task is incomplete")
        if observed["status"] not in _ACTIVE_STATUSES:
            return False
        connection = self._connect(create=False)
        if connection is None:
            return False
        now = self._clock()
        with closing(connection):
            changed = connection.execute(
                "UPDATE training_tasks SET status='failed', progress=0, message=?, error=?, "
                "updated_at=? WHERE id=? AND status=? AND updated_at=? "
                "AND worker_pid IS ? AND worker_start_token IS ?",
                (
                    f"訓練中斷: {error}",
                    error,
                    now,
                    task_id,
                    observed["status"],
                    observed["updated_at"],
                    observed["worker_pid"],
                    observed["worker_start_token"],
                ),
            ).rowcount
        return changed == 1

    def log_path(self, task_id: str) -> Path:
        task_id = self._validate_task_id(task_id)
        return self.state_dir / "logs" / f"{task_id}.log"

    def read_logs(self, task_id: str, limit: int = 100) -> tuple[list[str], int]:
        task_id = self._validate_task_id(task_id)
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        path = self.log_path(task_id)
        if not path.is_file():
            return [], 0
        recent: deque[str] = deque(maxlen=limit)
        total = 0
        with path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    continue
                clean = line.rstrip(b"\r\n")
                if clean:
                    total += 1
                    recent.append(clean.decode("utf-8", errors="replace"))
        return list(recent), total
