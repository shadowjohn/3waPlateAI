"""Deterministic, CPU-first PyTorch CTC training for the fixed v1 contract."""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

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
from .control import TrainingProgress


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
    device: str = "auto"
    charset_path: Path | None = None
    rules_path: Path | None = None


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
    loss: float = 0.0


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
    # ponytail: ctc_loss_backward_gpu lacks deterministic kernel, warn_only allows CUDA execution
    torch.use_deterministic_algorithms(True, warn_only=True)


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
    model: PlateCTCNet,
    loader: DataLoader,
    codec: CTCCodec,
    device: torch.device,
    *,
    on_batch: Callable[[int, int], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> EvaluationReport:
    """Evaluate greedy CTC decoding without allowing malformed inputs through."""

    model.eval()
    samples = exact = matching_characters = character_total = rejected = 0
    val_losses = []
    with torch.no_grad():
        total_batches = len(loader)
        for batch_index, batch in enumerate(loader, 1):
            if check_cancelled is not None:
                check_cancelled()
            if any(required > _TIME_STEPS for required in batch.required_timesteps):
                rejected += len(batch.required_timesteps)
                if on_batch is not None:
                    on_batch(batch_index, total_batches)
                if check_cancelled is not None:
                    check_cancelled()
                continue
            batch_loss = _ctc_loss(model, batch, device)
            val_losses.append(float(batch_loss.detach().cpu()))
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
            if on_batch is not None:
                on_batch(batch_index, total_batches)
            if check_cancelled is not None:
                check_cancelled()
    mean_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
    return EvaluationReport(
        samples=samples,
        exact_plate_accuracy=exact / samples if samples else 0.0,
        character_accuracy=matching_characters / character_total if character_total else 0.0,
        rejected=rejected,
        loss=mean_val_loss,
    )


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path = path.resolve()
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path = path.resolve()
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def resolve_training_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve training device: default to CUDA if available, falling back to CPU."""
    if device is None or (isinstance(device, str) and device.strip().lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        dev = device
    else:
        try:
            dev = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise TrainingDataError(f"invalid device {device!r}") from exc
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            return torch.device("cpu")
        return dev
    if dev.type != "cpu" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


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
    return resolve_training_device(config.device)


def train_recognizer(
    config: TrainingConfig,
    *,
    on_progress: Callable[[TrainingProgress], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> TrainingRun:
    """Train and publish a local v1 recognition run without replacing output."""

    def emit(progress: TrainingProgress) -> None:
        if on_progress is not None:
            on_progress(progress)

    def check() -> None:
        if check_cancelled is not None:
            check_cancelled()

    check()
    emit(TrainingProgress(kind="stage", phase="preparing", total_epochs=config.epochs))
    device = _validate_config(config)
    if device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(device)
            print(f"Training device: {device} ({gpu_name})")
        except Exception:
            print(f"Training device: {device}")
    else:
        print("Training device: cpu")

    charset_path = config.charset_path or _default_config_path(
        "configs/charsets/tw_new_style_private_passenger_v1.txt"
    )
    rules_path = config.rules_path or _default_config_path(
        "configs/plate_rules/tw_new_style_private_passenger_v1.json"
    )
    train_dataset = M1CropDataset(config.train_directory, charset_path, rules_path)
    validation_dataset = M1CropDataset(
        config.validation_directory, charset_path, rules_path
    )
    if train_dataset.seed == validation_dataset.seed:
        raise TrainingDataError("train and validation generation seeds must differ")
    check()

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
    model = PlateCTCNet(class_count=codec.class_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    check()

    config.output_directory.parent.mkdir(parents=True, exist_ok=True)
    if config.output_directory.exists():
        raise OutputExistsError(f"output already exists: {config.output_directory}")
    staging = (
        config.output_directory.parent
        / f".{config.output_directory.name}.partial-{uuid.uuid4().hex}"
    ).resolve()
    staging.mkdir(parents=True, exist_ok=False)
    try:
        best_accuracy = -1.0
        train_loss = 0.0
        history: list[dict[str, Any]] = []
        for epoch in range(1, config.epochs + 1):
            emit(
                TrainingProgress(
                    kind="stage",
                    phase="training",
                    epoch=epoch,
                    total_epochs=config.epochs,
                )
            )
            losses: list[float] = []
            total_train_batches = len(train_loader)
            for batch_index, batch in enumerate(train_loader, 1):
                check()
                losses.append(train_one_batch(model, batch, optimizer).loss)
                emit(
                    TrainingProgress(
                        kind="batch",
                        phase="training",
                        epoch=epoch,
                        total_epochs=config.epochs,
                        batch=batch_index,
                        total_batches=total_train_batches,
                        train_loss=sum(losses) / len(losses),
                    )
                )
                check()
            train_loss = sum(losses) / len(losses)
            emit(
                TrainingProgress(
                    kind="stage",
                    phase="validating",
                    epoch=epoch,
                    total_epochs=config.epochs,
                    train_loss=train_loss,
                )
            )

            def on_validation_batch(batch_index: int, total_batches: int) -> None:
                emit(
                    TrainingProgress(
                        kind="batch",
                        phase="validating",
                        epoch=epoch,
                        total_epochs=config.epochs,
                        batch=batch_index,
                        total_batches=total_batches,
                        train_loss=train_loss,
                    )
                )

            validation = evaluate_recognizer(
                model,
                validation_loader,
                codec,
                device,
                on_batch=on_validation_batch,
                check_cancelled=check,
            )
            print(
                f"Epoch {epoch}/{config.epochs} - train_loss: {train_loss:.4f} "
                f"- val_loss: {validation.loss:.4f} - val_acc: {validation.exact_plate_accuracy:.4f}",
                flush=True,
            )
            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(validation.loss, 4),
                "val_acc": round(validation.exact_plate_accuracy, 4),
                "char_acc": round(validation.character_accuracy, 4),
            }
            history.append(epoch_record)
            emit(
                TrainingProgress(
                    kind="epoch",
                    phase="validating",
                    epoch=epoch,
                    total_epochs=config.epochs,
                    train_loss=epoch_record["train_loss"],
                    val_loss=epoch_record["val_loss"],
                    val_acc=epoch_record["val_acc"],
                )
            )
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
            "schema_version": 1,
            "checkpoint_sha256": _sha256_file(staging / "best.pt"),
            "train": {"loss": train_loss},
            "history": history,
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
        check()
        _atomic_write_json(staging / "report.json", report)
        check()
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
