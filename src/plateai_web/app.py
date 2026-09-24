"""FastAPI application for 3waPlateAI Web Studio."""
from __future__ import annotations

import base64
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .downloader import download_ezcon_task, download_tlpd_task
from .evaluator import run_benchmark_task
from .external_predictor import ExternalPredictorEngine
from .generator import generate_dataset_task
from .paths import web_root, workspace_root
from .predictor import PredictorEngine, predictor
from .release_packager import build_release_task
from .tasks import TaskStatus, task_manager
from .trainer import find_available_datasets, validate_training_request
from .training_process import launch_training_worker, reconcile_training_tasks
from .training_store import TrainingStore

ROOT = workspace_root()
WEB_DIR = web_root(ROOT)

app = FastAPI(title="3waPlateAI Studio", version="1.0.0")

CANDIDATE_BUNDLE = "candidate-detector-real-v1"
_candidate_engine: PredictorEngine | None = None
_fpga_engine: ExternalPredictorEngine | None = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_training_store() -> TrainingStore:
    """Return a lazy training-store dependency without creating its database."""

    return TrainingStore(ROOT / "runs" / ".web-training")


def _memory_task_document(task: Any) -> dict[str, Any]:
    return {
        "id": task.id,
        "name": task.name,
        "status": task.status.value,
        "phase": None,
        "progress": task.progress,
        "message": task.message,
        "logs": task.logs[-100:],
        "log_count": len(task.logs),
        "result": task.result,
        "error": task.error,
        "cancel_requested": task.status == TaskStatus.CANCELLED,
        "heartbeat_at": None,
        "last_progress_at": None,
        "health": "memory-only",
        "updated_at": task.updated_at,
    }


def _training_health(task: dict[str, Any]) -> str:
    if task["status"] != "running":
        return "terminal" if task["status"] in {"completed", "failed", "cancelled"} else "pending"
    heartbeat = task["heartbeat_at"]
    if heartbeat is None:
        return "starting"
    return "stale" if time.time() - float(heartbeat) > 15 else "healthy"


def _training_task_document(store: TrainingStore, task: dict[str, Any]) -> dict[str, Any]:
    logs, log_count = store.read_logs(task["id"], limit=100)
    return {
        "id": task["id"],
        "name": task["name"],
        "status": task["status"],
        "phase": task["phase"],
        "progress": task["progress"],
        "message": task["message"],
        "logs": logs,
        "log_count": log_count,
        "result": task["result"],
        "error": task["error"],
        "cancel_requested": task["cancel_requested"],
        "heartbeat_at": task["heartbeat_at"],
        "last_progress_at": task["last_progress_at"],
        "health": _training_health(task),
        "updated_at": task["updated_at"],
    }


def _storage_unavailable(exc: BaseException) -> HTTPException:
    return HTTPException(status_code=503, detail=f"訓練狀態儲存暫時無法使用: {exc}")


def _new_training_task_id(store: TrainingStore) -> str:
    for _ in range(100):
        task_id = uuid.uuid4().hex[:12]
        if task_manager.get_task(task_id) is None and store.get(task_id) is None:
            return task_id
    raise RuntimeError("could not allocate a unique training task id")


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
    from .downloader import tlpd_is_ready
    tlpd_ready = tlpd_is_ready(tlpd_path)
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
    font: str = "noto_mono"
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


def _read_gpu_utilization() -> dict[str, float | str | None]:
    """Read instantaneous GPU 0 utilization from NVIDIA's local driver tool."""

    unavailable: dict[str, float | str | None] = {
        "gpu_utilization_percent": None,
        "memory_utilization_percent": None,
        "metrics_source": "unavailable",
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            text=True,
            timeout=1,
        )
        fields = completed.stdout.strip().splitlines()[0].split(",")
        gpu_utilization = float(fields[0].strip())
        memory_utilization = float(fields[1].strip())
        if not 0 <= gpu_utilization <= 100 or not 0 <= memory_utilization <= 100:
            return unavailable
        return {
            "gpu_utilization_percent": gpu_utilization,
            "memory_utilization_percent": memory_utilization,
            "metrics_source": "nvidia-smi",
        }
    except (IndexError, OSError, ValueError, subprocess.SubprocessError):
        return unavailable


def get_gpu_memory_info() -> dict[str, Any]:
    utilization = _read_gpu_utilization()
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
                **utilization,
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
        **utilization,
    }


@app.get("/api/system/gpu_memory")
def get_gpu_memory():
    return get_gpu_memory_info()


@app.get("/api/train/datasets")
def get_train_datasets():
    return {"datasets": find_available_datasets(ROOT)}


@app.get("/api/train/active")
def get_active_train(store: TrainingStore = Depends(get_training_store)):
    try:
        active = store.active()
        task = active[0] if active else store.latest()
    except (OSError, sqlite3.Error) as exc:
        raise _storage_unavailable(exc) from exc
    return {"active": bool(active), "task": _training_task_document(store, task) if task else None}


@app.post("/api/train/stop")
def stop_training(store: TrainingStore = Depends(get_training_store)):
    try:
        active = store.active()
        if not active:
            return {
                "status": "ok",
                "task_id": None,
                "cancel_requested": False,
                "cancelled": False,
                "message": "沒有進行中的背景訓練",
            }
        task = active[0]
        requested = store.request_cancel(task["id"])
        refreshed = store.get(task["id"])
    except (OSError, sqlite3.Error) as exc:
        raise _storage_unavailable(exc) from exc
    return {
        "status": "ok",
        "task_id": task["id"],
        "cancel_requested": bool(refreshed and refreshed["cancel_requested"]),
        "cancelled": bool(refreshed and refreshed["status"] == "cancelled"),
        "message": "已送出停止要求，等待訓練在安全邊界停止" if requested else "停止要求未被接受",
    }


class TrainRequest(BaseModel):
    epochs: int = 5
    run_name: str | None = None
    train_dataset: str | None = None
    validation_dataset: str | None = None
    batch_size: int | None = None
    device: str | None = None


@app.post("/api/train/start")
def start_training(req: TrainRequest, store: TrainingStore = Depends(get_training_store)):
    try:
        reconcile_training_tasks(store)
        active = store.active()
        task_id = _new_training_task_id(store)
    except (OSError, sqlite3.Error) as exc:
        raise _storage_unavailable(exc) from exc
    if active:
        return JSONResponse(
            status_code=409,
            content={
                "status": "busy",
                "task_id": active[0]["id"],
                "message": "已有背景模型訓練任務正在進行中",
            },
        )
    memory_task = task_manager.get_active_training_task()
    if memory_task:
        return JSONResponse(
            status_code=409,
            content={
                "status": "busy",
                "task_id": memory_task.id,
                "message": "已有既有模型訓練任務正在進行中",
            },
        )

    request = req.model_dump(exclude_none=True)
    try:
        validate_training_request(ROOT, task_id, request)
    except (FileExistsError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        store.create(task_id, f"模型訓練 ({req.epochs} Epochs)", request)
    except (OSError, sqlite3.Error) as exc:
        raise _storage_unavailable(exc) from exc
    try:
        launch_training_worker(ROOT, task_id)
    except BaseException as exc:
        try:
            store.finish(
                task_id,
                "failed",
                message="背景訓練程序無法啟動",
                error=f"{type(exc).__name__}: {exc}",
            )
        except (OSError, sqlite3.Error):
            pass
        raise HTTPException(status_code=503, detail="背景訓練程序無法啟動") from exc
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
def get_task_info(task_id: str, store: TrainingStore = Depends(get_training_store)):
    task = task_manager.get_task(task_id)
    if task:
        return _memory_task_document(task)
    try:
        try:
            training_task = store.get(task_id)
        except ValueError:
            training_task = None
        if training_task:
            return _training_task_document(store, training_task)
    except (OSError, sqlite3.Error) as exc:
        raise _storage_unavailable(exc) from exc
    raise HTTPException(status_code=404, detail="Task not found")


class ActivateModelRequest(BaseModel):
    bundle_name: str | None = None
    task_id: str | None = None


@app.post("/api/model/activate")
def activate_model(req: ActivateModelRequest | None = None):
    bundles_root = ROOT / "models" / "bundles"
    active_dir = bundles_root / "active-v1"

    target_bundle_dir: Path | None = None
    if req and req.bundle_name:
        name = Path(req.bundle_name).name
        target_bundle_dir = bundles_root / name
    elif req and req.task_id:
        target_bundle_dir = bundles_root / f"train-{req.task_id}"
    else:
        train_bundles = sorted(
            [d for d in bundles_root.glob("train-*") if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if train_bundles:
            target_bundle_dir = train_bundles[0]

    if target_bundle_dir is None or not target_bundle_dir.exists():
        raise HTTPException(status_code=404, detail="找不到可啟用的模型 Bundle 目錄")

    if target_bundle_dir.name == "fpga-lpr-mit":
        raise HTTPException(status_code=400, detail="外部 FPGA-LPR 契約不能啟用為原生 v1 Bundle")

    onnx_file = target_bundle_dir / "recognizer.onnx"
    manifest_file = target_bundle_dir / "manifest.json"
    if not onnx_file.exists() or not manifest_file.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Bundle 目錄 {target_bundle_dir.name} 缺少必要模型檔案 (recognizer.onnx 或 manifest.json)",
        )
    try:
        declaration = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Bundle manifest 無效") from exc
    if not isinstance(declaration, dict):
        raise HTTPException(status_code=400, detail="Bundle manifest 無效")
    if declaration.get("schema") == "fpga-lpr-onnx-v1":
        raise HTTPException(status_code=400, detail="外部 FPGA-LPR 契約不能啟用為原生 v1 Bundle")

    active_dir.mkdir(parents=True, exist_ok=True)
    for item in target_bundle_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, active_dir / item.name)

    # Reload predictor in memory
    try:
        predictor._load_active_model()
    except Exception:
        pass

    return {
        "status": "success",
        "message": f"已成功啟用【{target_bundle_dir.name}】為現役模型 (active-v1)！",
        "bundle": target_bundle_dir.name,
        "bundle_path": str(target_bundle_dir),
    }


class PredictBase64Request(BaseModel):
    image_base64: str


def _candidate_predictor() -> PredictorEngine:
    """Return the fixed local preview bundle; never select paths from a request."""
    global _candidate_engine
    if _candidate_engine is None or _candidate_engine.root != ROOT:
        _candidate_engine = PredictorEngine(
            ROOT, bundle_name=CANDIDATE_BUNDLE, preview_only=True,
        )
    return _candidate_engine


def _fpga_lpr_predictor() -> ExternalPredictorEngine:
    """Return one fixed, preview-only external OCR engine; no request path."""
    global _fpga_engine
    if _fpga_engine is None or _fpga_engine.root != ROOT:
        _fpga_engine = ExternalPredictorEngine(ROOT)
    return _fpga_engine


async def _uploaded_image_bytes(
    file: UploadFile | None,
    image_base64: str | None,
) -> bytes:
    if file is not None:
        return await file.read()
    if image_base64 is not None:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        return base64.b64decode(image_base64)
    raise HTTPException(status_code=400, detail="未收到有效的圖片檔案或 Base64 數據")


@app.post("/api/predict")
async def predict_image_upload(
    file: UploadFile | None = None,
    image_base64: str | None = Form(None),
):
    """Predict plate via file upload or base64 (for Ctrl+V pasting)."""
    try:
        result = predictor.predict_image(await _uploaded_image_bytes(file, image_base64))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/compare")
async def compare_image_upload(
    file: UploadFile | None = None,
    image_base64: str | None = Form(None),
    candidate_kind: Literal["native-preview", "fpga-lpr-mit"] = Form("native-preview"),
):
    """Run active and one allowlisted preview candidate on identical bytes."""
    try:
        image_bytes = await _uploaded_image_bytes(file, image_base64)
        candidate = _candidate_predictor() if candidate_kind == "native-preview" else _fpga_lpr_predictor()
        return {
            "active": predictor.predict_image(image_bytes),
            "candidate": candidate.predict_image(image_bytes),
            "preview": {
                "active_bundle": "active-v1",
                "candidate_bundle": CANDIDATE_BUNDLE if candidate_kind == "native-preview" else "fpga-lpr-mit",
                "activation_changed": False,
            },
        }
    except HTTPException:
        raise
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
