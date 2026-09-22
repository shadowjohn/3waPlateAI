"""Small training-control values shared by local and Web execution paths."""
from __future__ import annotations

from dataclasses import dataclass


class TrainingCancelled(RuntimeError):
    """The user requested cancellation at a safe training boundary."""


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """A real training event emitted at a stage, batch, or completed epoch."""

    kind: str
    phase: str
    epoch: int = 0
    total_epochs: int = 0
    batch: int = 0
    total_batches: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    val_acc: float | None = None
