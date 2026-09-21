from __future__ import annotations

import math

import torch

from plateai_shared.recognition import CTCCodec
from plateai_shared.rules import load_character_set
from plateai_trainer.training.dataset import M1CropDataset, collate_crop_samples
from plateai_trainer.training.engine import TrainingBatch, train_one_batch
from plateai_trainer.training.model import PlateCTCNet
from tests.conftest import V1_CHARSET, V1_RULES


torch.set_num_threads(1)


def _batch_from_dataset(dataset_root, count: int) -> TrainingBatch:
    dataset = M1CropDataset(dataset_root, V1_CHARSET, V1_RULES)
    samples = [dataset[index] for index in range(count)]
    images, targets, target_lengths = collate_crop_samples(samples)
    codec = CTCCodec.from_charset(load_character_set(V1_CHARSET))
    required_timesteps = []
    target_offset = 0
    for target_length in target_lengths.tolist():
        target = targets[target_offset : target_offset + target_length].tolist()
        required_timesteps.append(codec.required_timesteps(target))
        target_offset += target_length
    return TrainingBatch(images, targets, target_lengths, required_timesteps)


def test_plate_ctc_net_produces_80_timestep_34_class_logits():
    logits = PlateCTCNet()(torch.zeros((2, 1, 64, 160), dtype=torch.float32))
    assert tuple(logits.shape) == (2, 80, 34)


def test_plate_ctc_net_supports_configurable_class_count():
    logits_35 = PlateCTCNet(class_count=35)(torch.zeros((2, 1, 64, 160), dtype=torch.float32))
    assert tuple(logits_35.shape) == (2, 80, 35)
    import pytest
    with pytest.raises(ValueError, match="at least 2 classes"):
        PlateCTCNet(class_count=1)


def test_ctc_training_step_is_finite_and_updates_a_parameter(v1_train_dir):
    torch.manual_seed(7)
    model = PlateCTCNet()
    before = next(model.parameters()).detach().clone()
    batch = _batch_from_dataset(v1_train_dir, 2)
    result = train_one_batch(
        model, batch, torch.optim.AdamW(model.parameters(), lr=1e-3)
    )

    assert math.isfinite(result.loss)
    assert not torch.equal(before, next(model.parameters()).detach())


def test_repeat_heavy_ctc_batch_has_80_steps_for_a_12_step_alignment(v1_repeat_dataset):
    model = PlateCTCNet()
    batch = _batch_from_dataset(v1_repeat_dataset, 1)
    logits = model(batch.images)

    assert logits.shape[1] == 80
    assert batch.required_timesteps == [12]
