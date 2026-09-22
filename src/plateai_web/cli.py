"""CLI entry point for starting 3waPlateAI Web Studio."""
from __future__ import annotations

import argparse

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="start 3waPlateAI Web Studio")
    parser.add_argument(
        "--dev-reload",
        action="store_true",
        help="enable Uvicorn reload for development only",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print("[3waPlateAI] 正在啟動 Web 工作台 (http://localhost:1688)...")
    print("[提示] 關閉工作台不會停止已啟動訓練；請從訓練頁停止。")
    uvicorn.run("plateai_web.app:app", host="0.0.0.0", port=1688, reload=args.dev_reload)

if __name__ == "__main__":
    main()
