from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image

from plateai_shared.recognition import CTCCodec
from plateai_shared.rules import load_character_set
from plateai_trainer.training.dataset import (
    M1CropDataset,
    TrainingDataError,
    collate_crop_samples,
)
from tests.conftest import V1_CHARSET, V1_RULES


def _first_metadata_path(root):
    return root / "metadata.jsonl"


def _replace_first_metadata(root, **changes):
    path = _first_metadata_path(root)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(changes)
    path.write_text(json.dumps(metadata) + "\n", encoding="utf-8", newline="\n")


def mutate_dataset(root, mutation):
    image_path = root / "images/000000.png"
    if mutation == "wrong_image_size":
        image = Image.new("RGB", (320, 96), color=(255, 255, 255))
        image.save(image_path, format="PNG")
        _replace_first_metadata(
            root, image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest()
        )
    elif mutation == "changed_png":
        image_path.write_bytes(image_path.read_bytes() + b"changed")
    elif mutation == "wrong_plate_type":
        _replace_first_metadata(root, plate_type="motorcycle")
    else:
        raise AssertionError(f"unknown mutation {mutation}")


@pytest.mark.parametrize("mutation", ["wrong_image_size", "changed_png", "wrong_plate_type"])
def test_m1_v1_dataset_rejects_tampered_or_incompatible_inputs(v1_dataset, mutation):
    mutate_dataset(v1_dataset, mutation)
    with pytest.raises(TrainingDataError):
        M1CropDataset(v1_dataset, V1_CHARSET, V1_RULES)


def test_repeat_heavy_synthetic_record_is_a_valid_12_step_ctc_target(v1_repeat_dataset):
    dataset = M1CropDataset(v1_repeat_dataset, V1_CHARSET, V1_RULES)
    _, target = dataset[0]

    assert CTCCodec.from_charset(load_character_set(V1_CHARSET)).required_timesteps(target) == 12


def test_dataset_exposes_seed_and_collates_v1_preprocessed_tensors(v1_dataset):
    dataset = M1CropDataset(v1_dataset, V1_CHARSET, V1_RULES)
    images, targets, target_lengths = collate_crop_samples([dataset[0]])

    assert dataset.seed == 42
    assert tuple(images.shape) == (1, 1, 64, 160)
    assert images.dtype == torch.float32
    assert targets.dtype == torch.int64
    assert target_lengths.tolist() == [len(dataset[0][1])]
    assert np.all(images.numpy()[:, :, :, :4] == 1.0)
