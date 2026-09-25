"""Windows process primitives for an independently running training worker."""
from __future__ import annotations

import ctypes
import errno
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager

from .training_store import TrainingStore


_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 258
_WAIT_FAILED = 0xFFFFFFFF
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87


class TrainingBusy(RuntimeError):
    """Another training worker currently owns the single-machine lock."""


def _require_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError("background training workers are supported on Windows only")


def _kernel32():
    _require_windows()
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32, wintypes


def _creation_token(handle, *, kernel32, wintypes) -> str | None:
    created = wintypes.FILETIME()
    exited = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    if not kernel32.GetProcessTimes(
        handle,
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        return None
    return f"{created.dwHighDateTime:08x}{created.dwLowDateTime:08x}"


@contextmanager
def training_lock(path: Path) -> ContextManager[None]:
    """Acquire the one-byte Windows advisory lock without deleting its file."""

    _require_windows()
    import msvcrt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    locked = False
    try:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            locked = True
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EDEADLK} or exc.winerror in {33, 36}:
                raise TrainingBusy(f"another worker owns {path}") from exc
            raise
        yield
    finally:
        if locked:
            try:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                stream.close()
        else:
            stream.close()


def current_process_identity() -> tuple[int, str]:
    """Return this process PID plus a Windows creation-time token."""

    kernel32, wintypes = _kernel32()
    process_id = os.getpid()
    handle = kernel32.OpenProcess(
        _PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE, False, process_id
    )
    if not handle:
        error = ctypes.get_last_error()
        raise OSError(error, "OpenProcess failed for current process")
    try:
        token = _creation_token(handle, kernel32=kernel32, wintypes=wintypes)
        if token is None:
            error = ctypes.get_last_error()
            raise OSError(error, "GetProcessTimes failed for current process")
        return process_id, token
    finally:
        kernel32.CloseHandle(handle)


def process_state(pid: int, start_token: str) -> str:
    """Return alive, dead, or unknown without signalling the target process."""

    if not isinstance(pid, int) or pid < 1 or not isinstance(start_token, str) or not start_token:
        return "dead"
    kernel32, wintypes = _kernel32()
    handle = kernel32.OpenProcess(
        _PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE, False, pid
    )
    if not handle:
        error = ctypes.get_last_error()
        return "dead" if error == _ERROR_INVALID_PARAMETER else "unknown"
    try:
        token = _creation_token(handle, kernel32=kernel32, wintypes=wintypes)
        if token is None:
            return "unknown"
        if token != start_token:
            return "dead"
        status = kernel32.WaitForSingleObject(handle, 0)
        if status == _WAIT_TIMEOUT:
            return "alive"
        if status == _WAIT_OBJECT_0:
            return "dead"
        if status == _WAIT_FAILED:
            return "unknown"
        return "unknown"
    finally:
        kernel32.CloseHandle(handle)


def _child_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    source = root / "src"
    if source.is_dir():
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(source) if not existing else str(source) + os.pathsep + existing
    return environment


def _spawn_detached(command: list[str], *, cwd: Path, log_path: Path) -> subprocess.Popen:
    """Start a worker without an inherited stdio pipe or shell wrapper."""

    _require_windows()
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("worker command must be a non-empty string list")
    cwd = Path(cwd).resolve()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log_stream:
        return subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
            env=_child_environment(cwd),
        )


def launch_training_worker(root: Path, task_id: str) -> None:
    """Detach the task worker; all authoritative state remains in SQLite."""

    root = Path(root).resolve()
    store = TrainingStore(root / "runs" / ".web-training")
    task = store.get(task_id)
    executable = sys.executable
    if task and task['request'].get('kind') == 'pose':
        from .pose_training import python_path
        executable = str(python_path(root))
    try:
        _spawn_detached(
            [
                executable,
                "-u",
                "-m",
                "plateai_web.training_worker",
                "--root",
                str(root),
                "--task-id",
                task_id,
            ],
            cwd=root,
            log_path=store.log_path(task_id),
        )
    except BaseException as exc:
        store.finish(
            task_id,
            "failed",
            message="背景訓練程序無法啟動",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def reconcile_training_tasks(store: TrainingStore) -> None:
    """Mark only conclusively stale active records as failed after a Web restart."""

    now = time.time()
    for task in store.active():
        if task["status"] == "pending":
            if now - float(task["updated_at"]) > 30:
                store.fail_if_unchanged(task, "背景 worker 未在 30 秒內接手")
            continue
        state = process_state(task["worker_pid"], task["worker_start_token"])
        if state == "dead":
            store.fail_if_unchanged(task, "背景 worker 已結束")
