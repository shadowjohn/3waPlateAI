"""Isolated Uvicorn host for browser and lifecycle acceptance checks."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import uvicorn


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def create_app():
    web_module = importlib.import_module("plateai_web.app")
    web_module.ROOT = Path(os.environ["PLATEAI_TEST_ROOT"])
    return web_module.app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dev-reload", action="store_true")
    args = parser.parse_args(argv)
    os.environ["PLATEAI_TEST_ROOT"] = str(args.root.resolve())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    uvicorn.run(
        "tests.helpers.background_training_server:create_app",
        factory=True,
        host="127.0.0.1",
        port=args.port,
        reload=args.dev_reload,
        app_dir=str(_REPOSITORY_ROOT),
    )


if __name__ == "__main__":
    main()
