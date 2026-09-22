from __future__ import annotations

from pathlib import Path

import pytest

from plateai_trainer.training.control import TrainingCancelled
from plateai_trainer.training.engine import TrainingConfig, train_recognizer


def _config(v1_train_dir: Path, v1_validation_dir: Path, output: Path) -> TrainingConfig:
    return TrainingConfig(
        v1_train_dir,
        v1_validation_dir,
        output,
        epochs=1,
        batch_size=2,
        device="cpu",
    )


def test_first_train_batch_precedes_epoch_metrics(v1_train_dir, v1_validation_dir, tmp_path):
    """Removing the batch callback would leave the UI stuck until validation finishes."""
    events = []

    run = train_recognizer(
        _config(v1_train_dir, v1_validation_dir, tmp_path / "run"),
        on_progress=events.append,
    )

    first_batch = next(
        event
        for event in events
        if event.kind == "batch" and event.phase == "training"
    )
    completed_epoch = next(event for event in events if event.kind == "epoch")
    assert events.index(first_batch) < events.index(completed_epoch)
    assert (first_batch.batch, first_batch.total_batches) == (1, 2)
    assert first_batch.train_loss is not None
    assert first_batch.val_loss is None
    assert first_batch.val_acc is None
    assert completed_epoch.val_acc == run.report["history"][0]["val_acc"]


def test_cancellation_after_train_batch_removes_unpublished_output(
    v1_train_dir, v1_validation_dir, tmp_path
):
    """Ignoring cancellation after a batch would publish a run the user stopped."""
    cancel_requested = False

    def on_progress(event):
        nonlocal cancel_requested
        if event.kind == "batch" and event.phase == "training":
            cancel_requested = True

    def check_cancelled():
        if cancel_requested:
            raise TrainingCancelled("requested by test")

    output = tmp_path / "cancelled-training"
    with pytest.raises(TrainingCancelled, match="requested by test"):
        train_recognizer(
            _config(v1_train_dir, v1_validation_dir, output),
            on_progress=on_progress,
            check_cancelled=check_cancelled,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".cancelled-training.partial-*"))


def test_cancellation_before_validation_batch_removes_unpublished_output(
    v1_train_dir, v1_validation_dir, tmp_path
):
    """Ignoring the validation boundary would make Stop look accepted while work continued."""
    cancel_requested = False

    def on_progress(event):
        nonlocal cancel_requested
        if event.kind == "stage" and event.phase == "validating":
            cancel_requested = True

    def check_cancelled():
        if cancel_requested:
            raise TrainingCancelled("requested before validation")

    output = tmp_path / "cancelled-validation"
    with pytest.raises(TrainingCancelled, match="requested before validation"):
        train_recognizer(
            _config(v1_train_dir, v1_validation_dir, output),
            on_progress=on_progress,
            check_cancelled=check_cancelled,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".cancelled-validation.partial-*"))
