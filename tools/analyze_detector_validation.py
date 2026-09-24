"""Validation-only detector failure atlas; never consumes the frozen test split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import uuid

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from plateai_reader.rectifier import InvalidCornersError, rectify_plate
from plateai_shared.publication import publish_directory_no_replace, remove_owned_staging
from plateai_trainer.detection.engine import (
    _METRIC_CONFIG,
    _collate,
    _edge_stratum,
    _iou,
    _nms,
    _prediction_metrics,
)
from plateai_trainer.detection.export import _load_detector_checkpoint
from plateai_trainer.detection.real_dataset import RealDetectionDataset


_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
_CORNER_NAMES = ("left_top", "right_top", "right_bottom", "left_bottom")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> None:
    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"cannot serialize {type(item).__name__}")

    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=encode) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _predict(model, loader):
    rows, truths = [], []
    model.eval()
    with torch.no_grad():
        for batch, samples in loader:
            output = model(batch.to(next(model.parameters()).device))
            if not torch.isfinite(output).all():
                raise ValueError("non-finite validation predictions")
            rows.extend(output.cpu().numpy())
            truths.extend(sample.instances for sample in samples)
    return rows, truths


def _match(rows, instances, threshold):
    survivors, boxes = _nms(rows)
    keep = survivors[:, 4] >= threshold
    survivors, boxes = survivors[keep], boxes[keep]
    gt_boxes = np.asarray([item.bbox_xyxy for item in instances], dtype=np.float32).reshape(-1, 4)
    remaining = set(range(len(instances)))
    detections = []
    for order, (row, box) in enumerate(zip(survivors, boxes, strict=True)):
        overlaps = _iou(box, gt_boxes)
        best = max(sorted(remaining), key=lambda index: float(overlaps[index]), default=None)
        matched = best is not None and overlaps[best] >= _METRIC_CONFIG["match_iou"]
        errors = None
        if matched:
            remaining.remove(best)
            errors = np.linalg.norm(row[5:].reshape(4, 2) - instances[best].corners_xy, axis=1)
        detections.append({
            "order": order,
            "score": float(row[4]),
            "box": box,
            "corners": row[5:].reshape(4, 2),
            "matched": matched,
            "gt_index": int(best) if matched else None,
            "iou": float(overlaps[best]) if matched else (float(overlaps.max()) if len(overlaps) else 0.0),
            "errors": errors,
        })
    return detections, remaining


def _threshold_sweep(predictions, truths):
    result = []
    for threshold in _THRESHOLDS:
        detections = bbox_tp = quad_tp = total_gt = 0
        for rows, instances in zip(predictions, truths, strict=True):
            matched, _ = _match(rows, instances, threshold)
            detections += len(matched)
            total_gt += len(instances)
            bbox_tp += sum(item["matched"] for item in matched)
            quad_tp += sum(
                item["matched"] and bool(np.all(item["errors"] <= _METRIC_CONFIG["complete_quad_max_corner_error_640px"]))
                for item in matched
            )
        result.append({
            "confidence": threshold,
            "detections": detections,
            "bbox_true_positives": bbox_tp,
            "bbox_precision": bbox_tp / detections if detections else 0.0,
            "bbox_recall": bbox_tp / total_gt if total_gt else 0.0,
            "complete_quad_true_positives": quad_tp,
            "complete_quad_precision": quad_tp / detections if detections else 0.0,
            "complete_quad_recall": quad_tp / total_gt if total_gt else 0.0,
        })
    return result


def _quantiles(values):
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _quad_iou(first, second):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if (first.shape != (4, 2) or second.shape != (4, 2)
            or not np.isfinite(first).all() or not np.isfinite(second).all()
            or not cv2.isContourConvex(first) or not cv2.isContourConvex(second)):
        return 0.0
    first_area = float(abs(cv2.contourArea(first)))
    second_area = float(abs(cv2.contourArea(second)))
    if first_area <= 0 or second_area <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = first_area + second_area - float(intersection)
    return float(intersection) / union if union > 0 else 0.0


def _rotation_stratum(corners):
    edge = corners[1] - corners[0]
    angle = abs(math.degrees(math.atan2(float(edge[1]), float(edge[0]))))
    return "lt15" if angle < 15 else "15to30" if angle < 30 else "ge30"


def _group_summary(groups):
    result = {}
    for name, values in sorted(groups.items()):
        result[name] = {
            "instances": values["instances"],
            "bbox_true_positives": values["bbox_tp"],
            "bbox_recall": values["bbox_tp"] / values["instances"] if values["instances"] else 0.0,
            "complete_quad_true_positives": values["quad_tp"],
            "complete_quad_recall": values["quad_tp"] / values["instances"] if values["instances"] else 0.0,
            "corner_error_640px": _quantiles(values["errors"]),
        }
    return result


def _failure_analysis(predictions, truths, dataset):
    threshold = _METRIC_CONFIG["operating_confidence"]
    groups = {"size": defaultdict(lambda: {"instances": 0, "bbox_tp": 0, "quad_tp": 0, "errors": []}),
              "rotation": defaultdict(lambda: {"instances": 0, "bbox_tp": 0, "quad_tp": 0, "errors": []}),
              "scene_instances": defaultdict(lambda: {"instances": 0, "bbox_tp": 0, "quad_tp": 0, "errors": []})}
    scores = {"bbox_matches": [], "false_positives": []}
    corner_errors = [[] for _ in range(4)]
    all_corner_errors = []
    quad_ious = []
    quad_le_4 = 0
    failures = {"false_positives": [], "missed_ground_truth": [], "quad_failures": []}
    rectifier_rejections = Counter()
    total_detections = bbox_tp = quad_tp = 0
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for image_index, (rows, instances) in enumerate(zip(predictions, truths, strict=True)):
        detections, remaining = _match(rows, instances, threshold)
        total_detections += len(detections)
        by_gt = {item["gt_index"]: item for item in detections if item["matched"]}
        for item in detections:
            try:
                rectify_plate(dummy, item["corners"])
            except InvalidCornersError as error:
                rectifier_rejections[error.reason] += 1
            if item["matched"]:
                bbox_tp += 1
                scores["bbox_matches"].append(item["score"])
                is_quad = bool(np.all(item["errors"] <= _METRIC_CONFIG["complete_quad_max_corner_error_640px"]))
                quad_tp += int(is_quad)
                for index, value in enumerate(item["errors"]):
                    corner_errors[index].append(float(value))
                    all_corner_errors.append(float(value))
                quad_ious.append(_quad_iou(
                    item["corners"], instances[item["gt_index"]].corners_xy
                ))
                quad_le_4 += int(np.all(item["errors"] <= 4.0))
                if not is_quad:
                    failures["quad_failures"].append({
                        "image_index": image_index, "gt_index": item["gt_index"],
                        "score": item["score"], "iou": item["iou"],
                        "max_corner_error": float(item["errors"].max()),
                        "corner_errors": [float(value) for value in item["errors"]],
                        "corners": item["corners"],
                    })
            else:
                scores["false_positives"].append(item["score"])
                failures["false_positives"].append({
                    "image_index": image_index, "score": item["score"], "best_iou": item["iou"],
                    "corners": item["corners"],
                })
        for gt_index, instance in enumerate(instances):
            matched = by_gt.get(gt_index)
            is_quad = matched is not None and bool(np.all(matched["errors"] <= _METRIC_CONFIG["complete_quad_max_corner_error_640px"]))
            for group_name, key in (
                ("size", _edge_stratum(instance.corners_xy)),
                ("rotation", _rotation_stratum(instance.corners_xy)),
                ("scene_instances", str(len(instances))),
            ):
                group = groups[group_name][key]
                group["instances"] += 1
                group["bbox_tp"] += int(matched is not None)
                group["quad_tp"] += int(is_quad)
                if matched is not None:
                    group["errors"].extend(float(value) for value in matched["errors"])
            if matched is None:
                failures["missed_ground_truth"].append({
                    "image_index": image_index, "gt_index": gt_index,
                    "size": _edge_stratum(instance.corners_xy),
                    "rotation": _rotation_stratum(instance.corners_xy),
                    "corners": instance.corners_xy,
                })
    failures["false_positives"].sort(key=lambda item: (-item["score"], item["image_index"]))
    failures["quad_failures"].sort(key=lambda item: (-item["max_corner_error"], -item["score"]))
    failures["missed_ground_truth"].sort(key=lambda item: (item["size"], item["image_index"]))
    return {
        "operating_confidence": threshold,
        "detections": total_detections,
        "bbox_true_positives": bbox_tp,
        "false_positives": total_detections - bbox_tp,
        "missed_ground_truth": sum(len(items) for items in truths) - bbox_tp,
        "complete_quad_true_positives": quad_tp,
        "geometry_summary": {
            "corner_error_640px": _quantiles(all_corner_errors),
            "quad_iou": _quantiles(quad_ious),
            "complete_quad_le_4px_true_positives": quad_le_4,
            "complete_quad_le_4px_recall": quad_le_4 / sum(len(items) for items in truths),
            "complete_quad_le_8px_true_positives": quad_tp,
            "complete_quad_le_8px_recall": quad_tp / sum(len(items) for items in truths),
        },
        "scores": {key: _quantiles(value) for key, value in scores.items()},
        "semantic_corner_error_640px": {name: _quantiles(values) for name, values in zip(_CORNER_NAMES, corner_errors, strict=True)},
        "rectifier_rejections": dict(sorted(rectifier_rejections.items())),
        "strata": {name: _group_summary(value) for name, value in groups.items()},
    }, failures


def _canvas(dataset, image_index):
    sample = dataset[image_index]
    return np.round(sample.image.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), sample.instances


def _atlas(dataset, failures, output):
    colors = {"gt": (30, 144, 255), "bad": (255, 64, 64)}
    ordering = {
        "false_positives": (lambda item: f"FP score={item['score']:.3f} iou={item['best_iou']:.3f}"),
        "missed_ground_truth": (lambda item: f"MISS size={item['size']} rot={item['rotation']}"),
        "quad_failures": (lambda item: f"QUAD score={item['score']:.3f} err={item['max_corner_error']:.1f}px"),
    }
    index = {}
    for kind, labeler in ordering.items():
        panels = []
        records = []
        for item in failures[kind][:24]:
            rgb, instances = _canvas(dataset, item["image_index"])
            panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for instance in instances:
                cv2.polylines(panel, [np.round(instance.corners_xy).astype(np.int32)], True, colors["gt"], 2)
            if kind != "missed_ground_truth":
                cv2.polylines(panel, [np.round(item["corners"]).astype(np.int32)], True, colors["bad"], 2)
            label = f"#{item['image_index']} {labeler(item)}"
            cv2.rectangle(panel, (0, 0), (640, 34), (0, 0, 0), -1)
            cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            panels.append(cv2.resize(panel, (320, 320), interpolation=cv2.INTER_AREA))
            records.append({key: value for key, value in item.items() if key != "corners"})
        if panels:
            blank = np.zeros_like(panels[0])
            panels.extend(blank.copy() for _ in range((-len(panels)) % 4))
            sheet = np.vstack([np.hstack(panels[start:start + 4]) for start in range(0, len(panels), 4)])
            filename = f"{kind}.jpg"
            cv2.imwrite(str(output / filename), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
            index[kind] = {"image": filename, "records": records}
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    torch.set_num_threads(1)
    dataset = RealDetectionDataset(args.validation)
    if dataset.provenance["split"] != "validation":
        parser.error("validation split required; frozen test input is forbidden")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=_collate)
    models = {}
    candidate_failures = None
    for name, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        payload = path.read_bytes()
        model = _load_detector_checkpoint(payload).to(args.device).eval()
        predictions, truths = _predict(model, loader)
        metrics = _prediction_metrics(predictions, truths)
        analysis, failures = _failure_analysis(predictions, truths, dataset)
        models[name] = {
            "checkpoint": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
            "metrics": metrics, "threshold_sweep": _threshold_sweep(predictions, truths),
            "failure_analysis": analysis,
        }
        if name == "candidate":
            candidate_failures = failures
        print(json.dumps({name: {
            "bbox_ap50": metrics["bbox_ap50"],
            "bbox_precision": analysis["bbox_true_positives"] / analysis["detections"],
            "bbox_recall": analysis["bbox_true_positives"] / metrics["instances"],
            "complete_quad_recall": metrics["complete_quad_recall"],
            "corner_error_640px": metrics["corner_error_640px"],
        }}, allow_nan=False), flush=True)
        del model, predictions
        if args.device == "cuda":
            torch.cuda.empty_cache()
    dataset.verify_unchanged()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.parent / f".{args.output.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        atlas = _atlas(dataset, candidate_failures, staging)
        _json(staging / "report.json", {
            "schema_version": "detector-validation-atlas-v1",
            "models": models,
            "atlas": atlas,
            "validation_input_hashes": dataset.input_hashes,
            "metric_config": _METRIC_CONFIG,
            "local_only": True,
            "license_reviewed": dataset.provenance["license_reviewed"],
            "boundary": "Validation-only diagnostic; frozen test, OCR, export, activation, and production are not evaluated.",
        })
        publish_directory_no_replace(staging, args.output)
    except BaseException:
        remove_owned_staging(staging, args.output)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
