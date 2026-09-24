"""Release packager for standalone FastAPI service on port 1788."""
from __future__ import annotations

import shutil
from pathlib import Path
from .paths import workspace_root
from .tasks import TaskManager

STANDALONE_SERVER_CODE = '''"""3waPlateAI Standalone API Scaffold (Port 1788)."""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI(title="3waPlateAI Standalone Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>3waPlateAI Standalone API (Port 1788)</title>
        <style>
            body { font-family: system-ui, sans-serif; margin: 40px; background: #f8fafc; color: #1e293b; }
            .card { background: #fff; padding: 24px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); max-width: 600px; }
            h1 { color: #1e3a8a; }
            code { background: #e2e8f0; padding: 2px 6px; border-radius: 4px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🚗 3waPlateAI 獨立 API 雛形</h1>
            <p>服務已啟動，但尚未接上真實推論；此 source-only 發行包不附模型。</p>
            <h3>API 端點：</h3>
            <ul>
                <li><code>POST /api/predict</code> (Multipart 檔案上傳 <code>file</code>)</li>
                <li><code>GET /api/health</code> (健康檢查)</li>
            </ul>
        </div>
    </body>
    </html>
    """

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "3waPlateAI Standalone Scaffold", "port": 1788, "inference_ready": False}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    raise HTTPException(status_code=503, detail="Inference is not configured in this source-only scaffold")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1788)
'''

STANDALONE_BAT = '''@echo off
chcp 65001 >nul
echo ========================================================
echo   3waPlateAI 獨立 API 雛形 (尚無真實推論)
echo   即將在 http://localhost:1788 啟動 (一起發發)
echo ========================================================
python -m uvicorn app:app --host 0.0.0.0 --port 1788 --reload
pause
'''

STANDALONE_REQUIREMENTS = '''fastapi>=0.110.0
uvicorn>=0.28.0
numpy>=1.24.0
opencv-python-headless>=4.8.0
pillow>=10.0.0
python-multipart>=0.0.9
'''


def build_release_task(task_id: str, tm: TaskManager):
    root = workspace_root()
    release_dir = root / "release" / "3wa_plate_api"
    models_dst = release_dir / "models"

    # Never overwrite or silently ship a model left by an earlier local build.
    if models_dst.is_symlink() or (models_dst.exists() and (not models_dst.is_dir() or any(models_dst.iterdir()))):
        raise RuntimeError("release/3wa_plate_api/models contains local files; move them out before a source-only build")
    
    tm.update_progress(task_id, 10, "準備打包 Release 目錄...")
    tm.append_log(task_id, "=== 開始打包 3waPlateAI source-only API 雛形 ===")
    tm.append_log(task_id, f"目標目錄: {release_dir}")
    
    release_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Write server app.py
    (release_dir / "app.py").write_text(STANDALONE_SERVER_CODE, encoding="utf-8")
    tm.append_log(task_id, "寫入 app.py (FastAPI 輕量伺服器)...")
    tm.update_progress(task_id, 35, "寫入伺服器代碼...")
    
    # 2. Write run_api_1788.bat
    (release_dir / "run_api_1788.bat").write_text(STANDALONE_BAT, encoding="utf-8")
    tm.append_log(task_id, "寫入 run_api_1788.bat (Port 1788 一鍵啟動腳本)...")
    tm.update_progress(task_id, 55, "寫入啟動批次檔...")
    
    # 3. Write requirements.txt
    (release_dir / "requirements.txt").write_text(STANDALONE_REQUIREMENTS, encoding="utf-8")
    tm.append_log(task_id, "寫入 requirements.txt (極簡相依清單)...")
    tm.update_progress(task_id, 70, "打包依賴檔...")
    
    # 4. Publish only code and notices; locally trained or external weights stay local.
    models_dst.mkdir(exist_ok=True)
    shutil.copy2(root / "LICENSE", release_dir / "LICENSE")
    shutil.copy2(root / "THIRD_PARTY_NOTICES.md", release_dir / "THIRD_PARTY_NOTICES.md")
    tm.append_log(task_id, "未附任何模型權重；已附 MIT 授權與第三方告知。")
        
    # 5. README
    readme_content = """# 3waPlateAI Standalone Release API (Port 1788)

此目錄目前是 source-only API 雛形，**不是可用的車牌辨識發行版**。它不附模型權重或 Bundle；`POST /api/predict` 對有效圖片會回傳 HTTP 503，直到接上真實 Reader。請勿將它的回應當成辨識結果。

原始碼採 MIT 授權；第三方模型、資料、字型與執行環境不因此取得 MIT 授權。詳見 `LICENSE` 與 `THIRD_PARTY_NOTICES.md`。

## 快速啟動
1. 雙擊執行 `run_api_1788.bat`。
2. 服務將於 `http://localhost:1788` 啟動。
3. 真實推論服務尚待整合與驗證；模型須由使用者自行合法取得或訓練，不會由打包器複製。

## API 呼叫範例 (Python)
```python
import requests

url = "http://localhost:1788/api/predict"
files = {"file": open("car.jpg", "rb")}
resp = requests.post(url, files=files)
print(resp.json())
```
"""
    (release_dir / "README.md").write_text(readme_content, encoding="utf-8")
    
    tm.update_progress(task_id, 100, "API source-only 雛形打包完成；尚無真實推論。")
    tm.complete_task(
        task_id,
        result={
            "status": "scaffold",
            "release_dir": str(release_dir),
            "port": 1788,
        },
        message="3waPlateAI API 雛形已生成於 release/3wa_plate_api；尚無真實推論。",
    )
