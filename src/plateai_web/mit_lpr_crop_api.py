"""Minimal plate-crop OCR API; this service does not detect plates in scenes."""

from __future__ import annotations

import io
from typing import Protocol

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MODEL_ID = "fpga-lpr-mit-v1"


class CropRecognizer(Protocol):
    def recognize(self, image: NDArray[np.uint8]) -> object: ...


def decode_limited_rgb(data: bytes) -> NDArray[np.uint8]:
    """Check encoded bytes and header dimensions before OpenCV allocates pixels."""

    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("image_too_large" if data else "invalid_image")
    try:
        with Image.open(io.BytesIO(data)) as header:
            width, height = header.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("image_too_large")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("invalid_image") from exc
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("invalid_image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def create_crop_api(recognizer: CropRecognizer | None) -> FastAPI:
    app = FastAPI(title="3waPlateAI MIT crop OCR", version="1.0.0")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok" if recognizer is not None else "degraded",
            "inference_ready": recognizer is not None,
            "model_id": MODEL_ID,
            "scope": "crop-only",
        }

    @app.post("/api/recognize/crop")
    async def recognize_crop(file: UploadFile = File(...)) -> dict:
        if recognizer is None:
            raise HTTPException(status_code=503, detail="OCR assets unavailable; no fallback model")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            rgb = decode_limited_rgb(data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            result = recognizer.recognize(rgb)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"OCR inference failed: {type(exc).__name__}") from exc
        return {
            "raw_text": str(result.raw_text),
            "normalized_text": str(result.normalized_text),
            "model_id": MODEL_ID,
            "score_kind": "uncalibrated",
            "timings_ms": {str(key): float(value) for key, value in result.timings_ms.items()},
        }

    @app.post("/api/predict")
    async def no_full_scene_claim() -> dict:
        raise HTTPException(status_code=503, detail="crop-only service; full-scene plate detection is not included")

    return app
