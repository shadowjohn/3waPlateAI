from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_run_server_module():
    path = Path(__file__).resolve().parents[2] / "tools" / "run_server.py"
    spec = importlib.util.spec_from_file_location("plateai_run_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("argv, expected", [([], False), (["--dev-reload"], True)])
def test_package_launcher_passes_explicit_reload_setting(
    argv: list[str], expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plateai_web import cli

    captured: dict[str, object] = {}
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))
    assert cli.main(argv) is None
    assert captured["reload"] is expected


@pytest.mark.parametrize("argv, expected", [([], False), (["--dev-reload"], True)])
def test_tools_launcher_passes_explicit_reload_setting(
    argv: list[str], expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_run_server_module()
    captured: dict[str, object] = {}
    monkeypatch.setattr(module.webbrowser, "open", lambda _url: None)
    monkeypatch.setattr(module.threading, "Thread", lambda *args, **kwargs: type("Thread", (), {"start": lambda self: None})())
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))
    assert module.main(argv) is None
    assert captured["reload"] is expected
