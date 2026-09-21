"""Deterministic, CPU-first PyTorch CTC training for the fixed v1 contract."""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from plateai_shared.publication import (
    OutputExistsError,
    publish_directory_no_replace,
    remove_owned_staging,
)
from plateai_shared.recognition import CTCCodec, V1_PREPROCESS
from plateai_shared.rules import load_character_set
from plateai_trainer.synthetic.cli import _default_config_path

from .dataset import M1CropDataset, TrainingDataError, collate_crop_samples
from .model import PlateCTCNet


_TIME_STEPS = 80


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    train_directory: Path
    validation_directory: Path
    output_directory: Path
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    images: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    required_timesteps: list[int]


@dataclass(frozen=True, slots=True)
class BatchResult:
    loss: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    samples: int
    exact_plate_accuracy: float
    character_accuracy: float
    rejected: int


@dataclass(frozen=True, slots=True)
class TrainingRun:
    output_directory: Path
    best_checkpoint: Path
    report_path: Path
    report: dict[str, Any]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _targets_for_batch(targets: torch.Tensor, target_lengths: torch.Tensor) -> list[list[int]]:
    values = targets.detach().cpu().tolist()
    result: list[list[int]] = []
    offset = 0
    for length in target_lengths.detach().cpu().tolist():
        result.append(values[offset : offset + length])
        offset += length
    return result


def _make_batch(samples) -> TrainingBatch:
    images, targets, target_lengths = collate_crop_samples(samples)
    target_sequences = _targets_for_batch(targets, target_lengths)
    required = [
        len(target) + sum(current == previous for previous, current in zip(target, target[1:]))
        for target in target_sequences
    ]
    return TrainingBatch(images, targets, target_lengths, required)


def _ctc_loss(model: PlateCTCNet, batch: TrainingBatch, device: torch.device) -> torch.Tensor:
    if any(required > _TIME_STEPS for required in batch.required_timesteps):
        raise TrainingDataError("CTC target requires more than 80 timesteps")
    images = batch.images.to(device=device, dtype=torch.float32)
    targets = batch.targets.to(device=device, dtype=torch.int64)
    target_lengths = batch.target_lengths.to(device=device, dtype=torch.int64)
    logits = model(images)
    input_lengths = torch.full(
        (images.shape[0],), _TIME_STEPS, dtype=torch.int64, device=device
    )
    return nn.CTCLoss(blank=0, zero_infinity=False)(
        functional.log_softmax(logits, dim=2).transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
    )


def train_one_batch(
    model: PlateCTCNet, batch: TrainingBatch, optimizer: torch.optim.Optimizer
) -> BatchResult:
    """Perform one complete CTC update and return its scalar loss."""

    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = _ctc_loss(model, batch, device)
    loss.backward()
    optimizer.step()
    return BatchResult(loss=float(loss.detach().cpu()))


def _target_text(codec: CTCCodec, target: list[int]) -> str:
    return "".join(codec.symbols[index - 1] for index in target)


def evaluate_recognizer(
    model: PlateCTCNet, loader: DataLoader, codec: CTCCodec, device: torch.device
) -> EvaluationReport:
    """Evaluate greedy CTC decoding without allowing malformed inputs through."""

    model.eval()
    samples = exact = matching_characters = character_total = rejected = 0
    with torch.no_grad():
        for batch in loader:
            if any(required > _TIME_STEPS for required in batch.required_timesteps):
                rejected += len(batch.required_timesteps)
                continue
            logits = model(batch.images.to(device=device, dtype=torch.float32))
            predictions = logits.argmax(dim=2).detach().cpu().tolist()
            targets = _targets_for_batch(batch.targets, batch.target_lengths)
            for indices, target in zip(predictions, targets, strict=True):
                expected = _target_text(codec, target)
                actual = codec.decode_greedy(indices)
                samples += 1
                exact += actual == expected
                matching_characters += sum(
                    actual_char == expected_char
                    for actual_char, expected_char in zip(actual, expected)
                )
                character_total += len(expected)
    return EvaluationReport(
        samples=samples,
        exact_plate_accuracy=exact / samples if samples else 0.0,
        character_accuracy=matching_characters / character_total if character_total else 0.0,
        rejected=rejected,
    )


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def _validate_config(config: TrainingConfig) -> torch.device:
    if config.output_directory.exists():
        raise OutputExistsError(f"output already exists: {config.output_directory}")
    if type(config.epochs) is not int or config.epochs < 1:
        raise TrainingDataError("epochs must be a positive integer")
    if type(config.batch_size) is not int or config.batch_size < 1:
        raise TrainingDataError("batch size must be a positive integer")
    if type(config.learning_rate) is not float or config.learning_rate <= 0:
        raise TrainingDataError("learning rate must be positive")
    if type(config.seed) is not int:
        raise TrainingDataError("seed must be an integer")
    try:
        device = torch.device(config.device)
    except (TypeError, RuntimeError) as exc:
        raise TrainingDataError(f"invalid device {config.device!r}") from exc
    if device.type != "cpu" and not torch.cuda.is_available():
        raise TrainingDataError(f"device is unavailable: {config.device}")
    return device


def train_recognizer(config: TrainingConfig) -> TrainingRun:
    """Train and publish a local v1 recognition run without replacing output."""

    device = _validate_config(config)
    charset_path = _default_config_path(
        "configs/charsets/tw_new_style_private_passenger_v1.txt"
    )
    rules_path = _default_config_path(
        "configs/plate_rules/tw_new_style_private_passenger_v1.json"
    )
    train_dataset = M1CropDataset(config.train_directory, charset_path, rules_path)
    validation_dataset = M1CropDataset(
        config.validation_directory, charset_path, rules_path
    )
    if train_dataset.seed == validation_dataset.seed:
        raise TrainingDataError("train and validation generation seeds must differ")

    _seed_everything(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=_make_batch,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_make_batch,
    )
    charset = load_character_set(charset_path)
    codec = CTCCodec.from_charset(charset)
    model = PlateCTCNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    config.output_directory.parent.mkdir(parents=True, exist_ok=True)
    if config.output_directory.exists():
        raise OutputExistsError(f"output already exists: {config.output_directory}")
    staging = config.output_directory.parent / (
        f".{config.output_directory.name}.partial-{uuid.uuid4().hex}"
    )
    staging.mkdir(exist_ok=False)
    try:
        best_accuracy = -1.0
        train_loss = 0.0
        validation = EvaluationReport(0, 0.0, 0.0, 0)
        for _ in range(config.epochs):
            losses = [train_one_batch(model, batch, optimizer).loss for batch in train_loader]
            train_loss = sum(losses) / len(losses)
            validation = evaluate_recognizer(model, validation_loader, codec, device)
            if validation.exact_plate_accuracy > best_accuracy:
                best_accuracy = validation.exact_plate_accuracy
                _atomic_save_checkpoint(
                    staging / "best.pt",
                    {
                        "model_state_dict": model.state_dict(),
                        "class_count": model.class_count,
                        "charset_sha256": charset.sha256,
                        "rules_sha256": _sha256_file(rules_path),
                        "preprocess": asdict(V1_PREPROCESS),
                    },
                )

        report: dict[str, Any] = {
            "train": {"loss": train_loss},
            "validation": {
                "samples": validation.samples,
                "exact_plate_accuracy": validation.exact_plate_accuracy,
                "character_accuracy": validation.character_accuracy,
                "rejected": validation.rejected,
            },
            "seed": config.seed,
            "input_hashes": {
                "charset": charset.sha256,
                "rules": _sha256_file(rules_path),
            },
            "pytorch_version": torch.__version__,
            "config": {
                "train_directory": str(config.train_directory),
                "validation_directory": str(config.validation_directory),
                "output_directory": str(config.output_directory),
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "seed": config.seed,
                "device": str(device),
            },
        }
        _atomic_write_json(staging / "report.json", report)
        publish_directory_no_replace(staging, config.output_directory)
        return TrainingRun(
            output_directory=config.output_directory,
            best_checkpoint=config.output_directory / "best.pt",
            report_path=config.output_directory / "report.json",
            report=report,
        )
    except BaseException:
        remove_owned_staging(staging, config.output_directory)
        raise
