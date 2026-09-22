from __future__ import annotations

from pathlib import Path

from plateai_web.paths import web_root, workspace_root


def test_workspace_root_prefers_the_source_checkout() -> None:
    root = workspace_root()

    assert (root / "pyproject.toml").is_file()
    assert (root / "web").is_dir()


def test_web_root_uses_packaged_assets_when_workspace_has_no_web(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    packaged_data = tmp_path / "package-data"
    (packaged_data / "web" / "assets").mkdir(parents=True)
    (packaged_data / "web" / "index.html").write_text("ok", encoding="utf-8")

    assert web_root(workspace, packaged_data) == packaged_data / "web"
