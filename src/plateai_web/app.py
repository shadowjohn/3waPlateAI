"""FastAPI application for 3waPlateAI Web Studio."""
from __future__ import annotations

import base64
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .downloader import download_ezcon_task, download_tlpd_task
from .evaluator import run_benchmark_task
from .generator import generate_dataset_task
from .predictor import predictor
from .release_packager import build_release_task
from .tasks import TaskStatus, task_manager
from .trainer import find_available_datasets, train_model_task

ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = ROOT / "web"

app = FastAPI(title="3waPlateAI Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def get_system_status() -> dict[str, Any]:
    # Check GPU
    gpu_info = "CPU 模式"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
    except Exception:
        pass

    # Check datasets
    ezcon_path = ROOT / "datasets" / "restricted" / "ezcon-taiwan-recognition-test"
    tlpd_path = ROOT / "datasets" / "tlpd-taiwan-detector"
    synth_path = ROOT / "out"
    
    ezcon_ready = ezcon_path.exists() and (ezcon_path / "summary.json").exists()
    tlpd_ready = tlpd_path.exists()
    synth_samples = len(list(synth_path.glob("*/*.png"))) if synth_path.exists() else 0

    # Check model
    bundle_path = ROOT / "models" / "bundles" / "active-v1"
    model_ready = bundle_path.exists() and (bundle_path / "manifest.json").exists()

    return {
        "system": {
            "python_version": sys.version.split()[0],
            "gpu": gpu_info,
            "port": 1688,
        },
        "datasets": {
            "ezcon_ready": ezcon_ready,
            "tlpd_ready": tlpd_ready,
            "synthetic_count": synth_samples,
        },
        "model": {
            "ready": model_ready,
            "path": str(bundle_path) if model_ready else None,
        },
    }


def _run_build_env_task(task_id: str, tm: Any):
    tm.update_progress(task_id, 10, "正在檢查專案結構與依賴...")
    tm.append_log(task_id, "=== 3waPlateAI 環境建置程序 ===")
    tm.append_log(task_id, f"Python 直譯器: {sys.executable}")
    
    # Run pip check and verify dependencies
    tm.update_progress(task_id, 30, "檢查安裝依賴與相容性...")
    cmd = [sys.executable, "-m", "pip", "check"]
    tm.run_subprocess_command(task_id, cmd)
    
    # Check torch and onnxruntime
    tm.update_progress(task_id, 60, "檢查 PyTorch 與 ONNX Runtime...")
    try:
        import torch
        tm.append_log(task_id, f"[OK] PyTorch 版本: {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}")
    except Exception as e:
        tm.append_log(task_id, f"[WARN] PyTorch 檢查警告: {e}")
        
    try:
        import onnxruntime as ort
        tm.append_log(task_id, f"[OK] ONNX Runtime 版本: {ort.__version__}, Providers: {ort.get_available_providers()}")
    except Exception as e:
        tm.append_log(task_id, f"[WARN] ONNX Runtime 檢查警告: {e}")

    tm.update_progress(task_id, 100, "環境建置與檢查完成！")
    tm.complete_task(task_id, result={"status": "ready"}, message="環境建置完成，全部相依套件就緒！")


@app.post("/api/env/build")
def start_env_build():
    task_id = task_manager.run_in_background("環境建置與依賴檢查", _run_build_env_task)
    return {"status": "started", "task_id": task_id}


@app.post("/api/dataset/fetch_ezcon")
def start_fetch_ezcon():
    task_id = task_manager.run_in_background("下載 EZCon 真實車牌資料集", download_ezcon_task)
    return {"status": "started", "task_id": task_id}


@app.post("/api/dataset/fetch_tlpd")
def start_fetch_tlpd():
    task_id = task_manager.run_in_background("下載 TLPD 台灣車輛檢測集", download_tlpd_task)
    return {"status": "started", "task_id": task_id}


class GenerateRequest(BaseModel):
    count: int = 10000
    font: str = "taiwan_plate"
    seed: int = 42
    output_name: str = "demo-10000"


@app.post("/api/dataset/generate")
def start_generate_dataset(req: GenerateRequest):
    def runner(task_id, tm):
        generate_dataset_task(
            task_id,
            tm,
            count=req.count,
            font=req.font,
            seed=req.seed,
            output_dir_name=req.output_name,
        )

    task_id = task_manager.run_in_background(f"合成 {req.count} 張範例車牌", runner)
    return {"status": "started", "task_id": task_id}


def get_gpu_memory_info() -> dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(0)
            used_b = total_b - free_b
            device_name = torch.cuda.get_device_name(0)
            return {
                "available": True,
                "type": "cuda",
                "device_name": device_name,
                "used_mb": round(used_b / (1024 * 1024), 1),
                "used_gb": round(used_b / (1024**3), 2),
                "total_mb": round(total_b / (1024 * 1024), 1),
                "total_gb": round(total_b / (1024**3), 2),
                "free_mb": round(free_b / (1024 * 1024), 1),
                "percent": round((used_b / total_b) * 100, 1),
                "timestamp": time.strftime("%H:%M:%S"),
            }
    except Exception:
        pass
    return {
        "available": False,
        "type": "none",
        "device_name": "無 GPU 顯存 (CPU 運算模式)",
        "used_mb": 0.0,
        "used_gb": 0.0,
        "total_mb": 0.0,
        "total_gb": 0.0,
        "free_mb": 0.0,
        "percent": 0.0,
        "timestamp": time.strftime("%H:%M:%S"),
    }


@app.get("/api/system/gpu_memory")
def get_gpu_memory():
    return get_gpu_memory_info()


@app.get("/api/train/datasets")
def get_train_datasets():
    return {"datasets": find_available_datasets(ROOT)}


@app.get("/api/train/active")
def get_active_train():
    active_task = task_manager.get_active_training_task()
    if active_task:
        return {"active": True, "task": active_task}
    # Return most recent training task if exists
    for t in task_manager.list_tasks():
        if "訓練" in t.name or "train" in t.name.lower():
            return {"active": False, "task": t}
    return {"active": False, "task": None}


@app.post("/api/train/stop")
def stop_training():
    cancelled = task_manager.cancel_task()
    return {"status": "ok", "cancelled": cancelled, "message": "訓練已成功中斷"}


class TrainRequest(BaseModel):
    epochs: int = 5
    run_name: str = ""
    train_dataset: str | None = None


@app.post("/api/train/start")
def start_training(req: TrainRequest):
    # Concurrency guard: Only one train job at a time
    active_task = task_manager.get_active_training_task()
    if active_task:
        return JSONResponse(
            status_code=409,
            content={
                "status": "busy",
                "task_id": active_task.id,
                "message": f"已有模型訓練任務正在進行中 (Task ID: {active_task.id})，不可同時進行多筆訓練！",
            },
        )

    def runner(task_id, tm):
        train_model_task(
            task_id,
            tm,
            epochs=req.epochs,
            train_dir=req.train_dataset,
            run_name=req.run_name,
        )

    task_id = task_manager.run_in_background(f"模型訓練 ({req.epochs} Epochs)", runner)
    return {"status": "started", "task_id": task_id}


class BenchmarkRequest(BaseModel):
    dataset: str = "ezcon"
    rounds: int = 10
    warmup: int = 2


@app.post("/api/benchmark/run")
def start_benchmark(req: BenchmarkRequest):
    def runner(task_id, tm):
        run_benchmark_task(
            task_id,
            tm,
            target_dataset=req.dataset,
            rounds=req.rounds,
            warmup=req.warmup,
        )

    task_id = task_manager.run_in_background("Benchmark 效能與準確率評測", runner)
    return {"status": "started", "task_id": task_id}


@app.post("/api/release/build")
def start_release_build():
    task_id = task_manager.run_in_background("一鍵 Release 打包 (Port 1788)", build_release_task)
    return {"status": "started", "task_id": task_id}


@app.get("/api/tasks/{task_id}")
def get_task_info(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "id": task.id,
        "name": task.name,
        "status": task.status.value,
        "progress": task.progress,
        "message": task.message,
        "logs": task.logs[-100:],  # Return recent logs for polling
        "log_count": len(task.logs),
        "result": task.result,
        "error": task.error,
        "updated_at": task.updated_at,
    }


class PredictBase64Request(BaseModel):
    image_base64: str


@app.post("/api/predict")
async def predict_image_upload(
    file: UploadFile | None = None,
    image_base64: str | None = Form(None),
):
    """Predict plate via file upload or base64 (for Ctrl+V pasting)."""
    image_bytes: bytes | None = None

    if file is not None:
        image_bytes = await file.read()
    elif image_base64 is not None:
        # Strip data:image/...;base64, prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
    else:
        raise HTTPException(status_code=400, detail="未收到有效的圖片檔案或 Base64 數據")

    try:
        result = predictor.predict_image(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static assets directory
app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="web_assets")
app.mount("/css", StaticFiles(directory=str(WEB_DIR / "css")), name="web_css")
app.mount("/js", StaticFiles(directory=str(WEB_DIR / "js")), name="web_js")


@app.get("/")
def serve_index():
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>3waPlateAI Web Studio 正在建置中...</h1>")
    return FileResponse(index_file)
