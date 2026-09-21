"""Deterministic CPU training and atomic, local-only detector runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from plateai_reader.rectifier import InvalidCornersError, rectify_plate
from plateai_shared.publication import OutputExistsError, publish_directory_no_replace, remove_owned_staging
from .dataset import DetectionDataError, DetectionDataset, validate_train_validation_pair
from .loss import detection_loss
from .model import PlatePoseNet


@dataclass(frozen=True, slots=True)
class DetectorTrainingConfig:
    train_directory: Path
    validation_directory: Path
    output_directory: Path
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class DetectorTrainingRun:
    output_directory: Path
    best_checkpoint: Path
    report_path: Path
    report: dict


_METRIC_CONFIG = {
    "ap_confidence_floor": 0.05, "operating_confidence": 0.25,
    "nms_iou": 0.45, "max_detections": 100, "match_iou": 0.5,
    "complete_quad_max_corner_error_640px": 8.0,
    "ap_method": "all-points interpolated precision envelope",
    "corner_error_population": "semantic corners of unique bbox AP50 matches at operating confidence; null if none",
    "rectifier_population": "all operating-confidence NMS survivors, M3a at 640x640 without clipping/reordering before handoff",
}


def _iou(box, boxes):
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    union = np.maximum(0, box[2:] - box[:2]).prod() + np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1) - intersection
    return intersection / np.maximum(union, 1e-12)


def _nms(rows):
    rows = np.asarray(rows, dtype=np.float32)
    valid = np.isfinite(rows).all(axis=1) & (rows[:, 2:4] > 0).all(axis=1) & (rows[:, 4] >= _METRIC_CONFIG["ap_confidence_floor"])
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(-rows[indices, 4], kind="stable")]
    boxes = np.c_[rows[:, :2] - rows[:, 2:4] / 2, rows[:, :2] + rows[:, 2:4] / 2]
    selected = []
    while len(indices) and len(selected) < _METRIC_CONFIG["max_detections"]:
        index = indices[0]
        selected.append(index)
        indices = indices[1:]
        indices = indices[_iou(boxes[index], boxes[indices]) <= _METRIC_CONFIG["nms_iou"]]
    return rows[selected], boxes[selected]


def _prediction_metrics(predictions, truths):
    """Single-class, score-ranked one-to-one bbox AP and semantic quad metrics.

    A complete quad needs bbox IoU >= .5 and *each* semantic corner <= 8px.
    Missing matches yield null corner error, never a misleading perfect zero.
    """
    detections, total_gt, operating_count, quad_tp, accepted = [], 0, 0, 0, 0
    corner_errors = []
    strata = {"instance_count": {"1": 0, "2": 0, "3": 0},
              "short_side_640px": {"lt64": 0, "64to128": 0, "ge128": 0},
              "projected_shortest_edge_640px": {"lt16": 0, "16to31": 0, "32to63": 0, "ge64": 0},
              "rotation_abs_degrees": {"lt15": 0, "15to30": 0, "ge30": 0}}
    dummy_rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    for image_index, (rows, instances) in enumerate(zip(predictions, truths, strict=True)):
        total_gt += len(instances)
        strata["instance_count"][str(len(instances))] += 1
        for instance in instances:
            box = instance.bbox_xyxy
            short = min(box[2] - box[0], box[3] - box[1])
            strata["short_side_640px"]["lt64" if short < 64 else "64to128" if short < 128 else "ge128"] += 1
            shortest_edge = float(np.linalg.norm(np.roll(instance.corners_xy, -1, axis=0) - instance.corners_xy, axis=1).min())
            strata["projected_shortest_edge_640px"]["lt16" if shortest_edge < 16 else "16to31" if shortest_edge < 32 else "32to63" if shortest_edge < 64 else "ge64"] += 1
            edge = instance.corners_xy[1] - instance.corners_xy[0]
            angle = abs(math.degrees(math.atan2(float(edge[1]), float(edge[0]))))
            strata["rotation_abs_degrees"]["lt15" if angle < 15 else "15to30" if angle < 30 else "ge30"] += 1
        gt_boxes = np.asarray([item.bbox_xyxy for item in instances], dtype=np.float32).reshape(-1, 4)
        remaining = set(range(len(instances)))
        survivors, boxes = _nms(rows)
        for order, (row, box) in enumerate(zip(survivors, boxes, strict=True)):
            operating = row[4] >= _METRIC_CONFIG["operating_confidence"]
            corners = row[5:].reshape(4, 2)
            if operating:
                operating_count += 1
                try:
                    rectify_plate(dummy_rgb, corners)
                    accepted += 1
                except InvalidCornersError:
                    pass
            overlaps = _iou(box, gt_boxes)
            best = max(sorted(remaining), key=lambda index: float(overlaps[index]), default=None)
            matched = best is not None and overlaps[best] >= _METRIC_CONFIG["match_iou"]
            detections.append((float(row[4]), image_index, order, int(matched)))
            if matched:
                remaining.remove(best)
                if operating:
                    errors = np.linalg.norm(corners - instances[best].corners_xy, axis=1)
                    corner_errors.extend(float(value) for value in errors)
                    quad_tp += int(np.all(errors <= _METRIC_CONFIG["complete_quad_max_corner_error_640px"]))
    detections.sort(key=lambda item: (-item[0], item[1], item[2]))
    tp = np.cumsum([item[3] for item in detections], dtype=np.float64)
    recall = tp / max(total_gt, 1)
    precision = tp / np.arange(1, len(tp) + 1)
    recall = np.r_[0., recall, 1.]
    precision = np.r_[0., precision, 0.]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = float(np.sum(np.diff(recall) * precision[1:]))
    return {"samples": len(truths), "instances": total_gt, "bbox_ap50": ap,
            "corner_error_640px": float(np.mean(corner_errors)) if corner_errors else None,
            "corner_matched_instances": len(corner_errors) // 4,
            "complete_quad_precision": quad_tp / operating_count if operating_count else 0.,
            "complete_quad_recall": quad_tp / total_gt if total_gt else 0.,
            "complete_quad_true_positives": quad_tp, "nms_predictions": operating_count,
            "rectifier_acceptance": accepted / operating_count if operating_count else 0.,
            "rectifier_accepted": accepted, "strata": strata}


def _collate(samples):
    return torch.from_numpy(np.stack([sample.image for sample in samples])), samples


def _finite_loss(loss):
    value = float(loss.detach())
    if not math.isfinite(value):
        raise DetectionDataError("non-finite detector loss; run not published")
    return value


def _step(model, optimizer, images, samples):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = detection_loss(model(images), [sample.targets for sample in samples]).total
    value = _finite_loss(loss)
    loss.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise DetectionDataError("non-finite detector gradients; run not published")
    optimizer.step()
    return value


def _evaluate(model, loader):
    model.eval()
    predictions, truths = [], []
    loss_sum = count = 0
    with torch.no_grad():
        for images, samples in loader:
            rows = model(images)
            if not torch.isfinite(rows).all():
                raise DetectionDataError("non-finite validation predictions")
            loss = _finite_loss(detection_loss(rows, [sample.targets for sample in samples]).total)
            loss_sum += loss * len(samples)
            count += len(samples)
            predictions.extend(rows.numpy())
            truths.extend(sample.instances for sample in samples)
    return {**_prediction_metrics(predictions, truths), "loss": loss_sum / count}


def _overfit_smoke(dataset):
    # Separate fixed model/optimizer/seed; does not alter the trained checkpoint.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        model = PlatePoseNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        images, samples = _collate([dataset[0]])
        losses = [_step(model, optimizer, images, samples) for _ in range(8)]
        with torch.no_grad():
            last = _finite_loss(detection_loss(model(images), samples[0].targets).total)
    return {"seed": 1729, "sample_indices": [0], "steps": 8, "learning_rate": 1e-3,
            "evaluation_mode": "train; same fixed batch before and after updates",
            "first_loss": losses[0], "last_loss": last, "losses": losses + [last],
            "decreased": last < losses[0]}


def _seed_detector_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _validate_config(config):
    if config.output_directory.exists():
        raise OutputExistsError(f"output already exists: {config.output_directory}")
    for name in ("epochs", "batch_size"):
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise DetectionDataError(f"{name} must be a positive integer")
    if type(config.learning_rate) not in (int, float) or not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise DetectionDataError("learning_rate must be positive and finite")
    if type(config.seed) is not int or not 0 <= config.seed < 2**32:
        raise DetectionDataError("seed must be an integer in [0, 2**32)")
    if config.device != "cpu":
        raise DetectionDataError("this deterministic trainer supports device=cpu only")


def _runtime_provenance():
    modules = {name: importlib.import_module(f"plateai_trainer.detection.{name}")
               for name in ("model", "loss", "targets", "dataset", "engine")}
    modules["letterbox"] = importlib.import_module("plateai_shared.detection")
    modules["rectifier"] = importlib.import_module("plateai_reader.rectifier")
    source_hashes = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                     for name, module in modules.items()}
    checkout = Path(__file__).resolve().parents[3]
    locks = {}
    for name in ("py311.lock.txt", "py311.training.lock.txt"):
        path = checkout / "requirements" / name
        locks[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {"source_sha256": source_hashes, "dependency_locks": locks,
            "dependency_lock_boundary": "checkout lock hashes; null when not shipped in an installed distribution",
            "installed_versions": {name: importlib.metadata.version(name) for name in
                                   ("torch", "numpy", "Pillow", "opencv-python-headless", "jsonschema")}}


def train_detector(config: DetectorTrainingConfig) -> DetectorTrainingRun:
    """Publish best.pt and report.json together into a new directory only."""
    _validate_config(config)
    train = DetectionDataset(config.train_directory)
    validation = DetectionDataset(config.validation_directory)
    validate_train_validation_pair(train, validation)
    _seed_detector_training(config.seed)
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(config.seed), num_workers=0, collate_fn=_collate)
    validation_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False,
                                   num_workers=0, collate_fn=_collate)
    model = PlatePoseNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    config_values = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    config_hash = hashlib.sha256(json.dumps(config_values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    inputs = {"train": train.input_hashes, "validation": validation.input_hashes}
    runtime = _runtime_provenance()
    config.output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = config.output_directory.parent / f".{config.output_directory.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    try:
        best_key = None
        history = []
        best_metrics = None
        best_epoch = 0
        for epoch in range(1, config.epochs + 1):
            total_loss = count = 0
            for images, samples in train_loader:
                total_loss += _step(model, optimizer, images, samples) * len(samples)
                count += len(samples)
            train_loss = total_loss / count
            metrics = _evaluate(model, validation_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": metrics["loss"], "bbox_ap50": metrics["bbox_ap50"]})
            key = (metrics["bbox_ap50"], -metrics["loss"])
            if best_key is None or key > best_key:
                best_key, best_metrics, best_epoch = key, metrics, epoch
                torch.save({"schema_version": 1, "architecture": "PlatePoseNet", "model_state_dict": model.state_dict(),
                            "epoch": epoch, "input_shape": [3, 640, 640], "output_shape": [8400, 13],
                            "preprocess": "letterbox_rgb_v1", "corner_order": ["left_top", "right_top", "right_bottom", "left_bottom"],
                            "input_hashes": inputs, "runtime_provenance": runtime,
                            "config": config_values, "config_sha256": config_hash}, staging / "best.pt")
        overfit = _overfit_smoke(train)
        train.verify_unchanged()
        validation.verify_unchanged()
        report = {"schema_version": 1, "train": {"loss": train_loss, "samples": len(train)},
                  "validation": best_metrics, "best_epoch": best_epoch,
                  "checkpoint_selection": "maximum validation bbox_ap50; minimum validation loss breaks ties; earliest exact tie",
                  "epochs": history, "overfit": overfit, "metric_config": dict(_METRIC_CONFIG),
                  "input_hashes": inputs, "config": config_values, "config_sha256": config_hash,
                  "runtime_provenance": runtime,
                  "checkpoint_sha256": hashlib.sha256((staging / "best.pt").read_bytes()).hexdigest(),
                  "pytorch_version": str(torch.__version__), "numpy_version": np.__version__,
                  "seed": config.seed, "cpu_threads": 1, "num_workers": 0,
                  "validation_boundary": "local synthetic composites only; no real-image or production accuracy claim"}
        (staging / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        publish_directory_no_replace(staging, config.output_directory)
    except BaseException:
        remove_owned_staging(staging, config.output_directory)
        raise
    return DetectorTrainingRun(config.output_directory, config.output_directory / "best.pt", config.output_directory / "report.json", report)
