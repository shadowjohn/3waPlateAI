"""Validate an isolated bundle before a reversible local activation."""

from __future__ import annotations

import re
import shutil
import stat
import threading
import uuid
from pathlib import Path

from plateai_shared.publication import publish_directory_no_replace, remove_owned_staging
from .predictor import PredictorEngine


_activation_lock = threading.Lock()


def _check_tree(path: Path) -> None:
    for item in (path, *path.rglob("*")):
        info = item.lstat()
        if item.is_symlink() or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError("Bundle 不可包含符號連結或 junction")


def activate_bundle(root: Path, name: str, predictor: PredictorEngine) -> dict[str, str | None]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name) or name in {"active-v1", ".", ".."}:
        raise ValueError("請指定單一候選 Bundle 名稱")
    bundles = root / "models" / "bundles"
    source = bundles / name
    if not source.is_dir():
        raise FileNotFoundError("找不到可啟用的模型 Bundle")
    if bundles.resolve() != bundles.absolute():
        raise ValueError("Bundle 根目錄不可使用重新導向路徑")
    active = bundles / "active-v1"
    token = uuid.uuid4().hex
    staging = bundles / f".active-v1.partial-{token}"
    backup = bundles / f"active-v1-backup-{token}"
    failed = bundles / f".active-v1.failed-{token}"
    with _activation_lock, predictor._lock:
        _check_tree(source)
        if active.exists() or active.is_symlink():
            _check_tree(active)
        try:
            shutil.copytree(source, staging)
            _check_tree(staging)
            probe = PredictorEngine(root, bundle_name=staging.name)
            if probe.load_error or probe.recognizer_session is None:
                raise ValueError(f"候選 Bundle 驗證／載入失敗：{probe.load_error}")
            del probe
            had_active = active.exists()
            if had_active:
                publish_directory_no_replace(active, backup)
            published = False
            try:
                publish_directory_no_replace(staging, active)
                published = True
                predictor._bundle_stamp = None
                predictor._load_active_model()
                if predictor.load_error or predictor.recognizer_session is None:
                    raise RuntimeError(f"現役模型載入失敗：{predictor.load_error}")
            except Exception:
                # Retain both the failed candidate and the previous active bytes.
                if published and active.exists():
                    publish_directory_no_replace(active, failed)
                if had_active:
                    publish_directory_no_replace(backup, active)
                predictor._bundle_stamp = None
                predictor._load_active_model()
                raise
            return {"bundle": name, "backup_path": str(backup) if had_active else None}
        finally:
            remove_owned_staging(staging, active)
