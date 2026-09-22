"""3waPlateAI Web Studio Launcher with robust encoding support."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows to prevent cp950 emoji crash
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

def open_browser_later():
    time.sleep(1.5)
    try:
        webbrowser.open("http://localhost:1688")
    except Exception:
        pass


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
    print("=" * 66)
    print("   3waPlateAI Studio - 台灣車牌 AI 視覺化工作台")
    print("   服務網址: http://localhost:1688")
    print("   3wa 老司機看板娘已就緒！[🚗💨]")
    print("=" * 66)
    print()
    print("[提示] 關閉工作台不會停止已啟動訓練；請從訓練頁停止。")

    # Check dependencies
    try:
        import fastapi
        import uvicorn
        import multipart
    except ImportError:
        print("[提示] 正在自動安裝必要依賴套件 (fastapi, uvicorn, python-multipart)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "python-multipart"])

    threading.Thread(target=open_browser_later, daemon=True).start()

    print("[提示] 正在啟動 Uvicorn 服務於 http://0.0.0.0:1688 (按 Ctrl+C 可停止)...")
    import uvicorn
    uvicorn.run(
        "plateai_web.app:app",
        host="0.0.0.0",
        port=1688,
        reload=args.dev_reload,
        app_dir=str(ROOT / "src"),
    )

if __name__ == "__main__":
    main()
