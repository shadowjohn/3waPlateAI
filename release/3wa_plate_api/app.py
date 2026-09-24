"""3waPlateAI Standalone API Scaffold (Port 1788)."""
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
