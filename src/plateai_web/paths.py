"""Resolve writable workspace state separately from packaged Web resources."""
from __future__ import annotations

import sysconfig
from pathlib import Path


def package_data_root() -> Path:
    """Return the wheel's shared data directory without assuming site-packages."""
    return Path(sysconfig.get_path("data")) / "share" / "3wa-plate-ai"


def workspace_root(cwd: Path | None = None) -> Path:
    """Use the checkout while developing, otherwise keep mutable state in CWD."""
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return (Path.cwd() if cwd is None else Path(cwd)).resolve()


def web_root(
    workspace: Path | None = None, packaged_data: Path | None = None
) -> Path:
    """Prefer a checkout's editable Web files, then use wheel-provided assets."""
    resolved_workspace = workspace_root() if workspace is None else Path(workspace)
    checkout_web = resolved_workspace / "web"
    if checkout_web.is_dir():
        return checkout_web
    data_root = package_data_root() if packaged_data is None else Path(packaged_data)
    packaged_web = data_root / "web"
    if packaged_web.is_dir():
        return packaged_web
    return checkout_web
