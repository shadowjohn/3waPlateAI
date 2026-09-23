"""Run one native detector checkpoint and save a bbox-only debug image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from plateai_reader.detector import numpy_nms_v1, postprocess_candidates
from plateai_shared.detection import letterbox_rgb_v1
from plateai_trainer.detection.export import _POSTPROCESS, _load_detector_checkpoint


def _probability(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return result


def _read_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read input image: {path}") from error
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"unsupported or invalid input image: {path}")
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _draw_bbox(image_bgr: np.ndarray, detections) -> np.ndarray:
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    thickness = max(2, round(min(width, height) / 320))
    font_scale = max(0.5, min(width, height) / 900)
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = np.round(detection.bbox_xyxy).astype(int)
        x1, x2 = np.clip((x1, x2), 0, width - 1)
        y1, y2 = np.clip((y1, y2), 0, height - 1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), thickness)
        label = f"#{index} {detection.confidence:.3f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(
            canvas,
            (x1, label_top),
            (min(width - 1, x1 + text_width + 8), min(height - 1, label_top + text_height + baseline + 6)),
            (0, 165, 255),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 4, label_top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return canvas


def _save_image(path: Path, image_bgr: np.ndarray) -> None:
    suffix = path.suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
        raise ValueError("output extension must be .jpg, .jpeg, .png, or .webp")
    options = [cv2.IMWRITE_JPEG_QUALITY, 94] if suffix in (".jpg", ".jpeg") else []
    ok, encoded = cv2.imencode(suffix, image_bgr, options)
    if not ok:
        raise ValueError(f"cannot encode debug image as {suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded.tobytes())
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite output: {path}") from error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--score-threshold", type=_probability, default=float(_POSTPROCESS["score_threshold"]))
    parser.add_argument("--nms-iou", type=_probability, default=float(_POSTPROCESS["iou_threshold"]))
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")

    image_bgr, image_rgb = _read_image(args.image)
    tensor, transform = letterbox_rgb_v1(image_rgb)
    model = _load_detector_checkpoint(args.checkpoint.read_bytes()).to(args.device).eval()
    with torch.no_grad():
        candidates = model(torch.from_numpy(tensor[None]).to(args.device))[0].cpu().numpy()
    postprocess = {
        "score_threshold": args.score_threshold,
        "iou_threshold": args.nms_iou,
        "max_detections": _POSTPROCESS["max_detections"],
    }
    retained = numpy_nms_v1(
        candidates,
        args.score_threshold,
        args.nms_iou,
        int(_POSTPROCESS["max_detections"]),
    )
    detections = postprocess_candidates(candidates, transform, postprocess)
    _save_image(args.output, _draw_bbox(image_bgr, detections))
    print(json.dumps({
        "input": str(args.image.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "output": str(args.output.resolve()),
        "device": args.device,
        "score_threshold": args.score_threshold,
        "nms_iou": args.nms_iou,
        "image_pipeline": {
            "source_size_wh": list(transform.source_size_wh),
            "letterbox_resized_size_wh": list(transform.resized_size_wh),
            "letterbox_padding_ltrb": list(transform.padding_ltrb),
            "letterbox_scale": transform.scale,
            "model_input_shape_nchw": [1, *tensor.shape],
            "model_input_size_wh": [tensor.shape[2], tensor.shape[1]],
        },
        "detections": [
            {
                "confidence": item.confidence,
                "bbox_source_xyxy": item.bbox_xyxy.tolist(),
                "bbox_source_wh": [
                    float(item.bbox_xyxy[2] - item.bbox_xyxy[0]),
                    float(item.bbox_xyxy[3] - item.bbox_xyxy[1]),
                ],
                "bbox_model_input_xyxy": [
                    float(candidates[row, 0] - candidates[row, 2] / 2),
                    float(candidates[row, 1] - candidates[row, 3] / 2),
                    float(candidates[row, 0] + candidates[row, 2] / 2),
                    float(candidates[row, 1] + candidates[row, 3] / 2),
                ],
                "bbox_model_input_wh": [float(candidates[row, 2]), float(candidates[row, 3])],
                "bbox_model_input_short_side_px": float(min(candidates[row, 2], candidates[row, 3])),
                "bbox_model_input_area_px2": float(candidates[row, 2] * candidates[row, 3]),
            }
            for row, item in zip(retained, detections, strict=True)
        ],
    }, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
