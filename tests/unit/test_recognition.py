from __future__ import annotations

import hashlib

import numpy as np
import pytest

from plateai_shared.recognition import CTCCodec, V1_PREPROCESS, preprocess_v1_rgb


def _rgb_conformance_vector() -> np.ndarray:
    y, x = np.indices((160, 380), dtype=np.uint16)
    return np.stack(
        (
            (17 * x + 29 * y) % 256,
            (43 * x + 11 * y) % 256,
            (7 * x + 53 * y) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def test_v1_ctc_indices_are_blank_zero_and_charset_order_skips_blank(v1_charset):
    codec = CTCCodec.from_charset(v1_charset)

    assert codec.blank_index == 0
    assert codec.class_count == 34
    assert codec.encode("035AZ") == (1, 4, 5, 10, 33)
    assert codec.encode("8") == (8,)
    with pytest.raises(ValueError, match="not in charset"):
        codec.encode("A4I")


def test_greedy_ctc_decoder_preserves_repeat_heavy_plate_with_blank_runs(v1_charset):
    codec = CTCCodec.from_charset(v1_charset)
    indices = [10, 0, 10, 0, 10, 8, 0, 8, 0, 8, 0, 8]

    assert codec.required_timesteps(codec.encode("AAA8888")) == 12
    assert codec.decode_greedy(indices) == "AAA8888"


def test_preprocess_v1_matches_pillow_12_3_conformance_digest():
    tensor = preprocess_v1_rgb(_rgb_conformance_vector())

    assert V1_PREPROCESS.input_size_hw == (64, 160)
    assert tensor.shape == (1, 64, 160)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert (
        hashlib.sha256(tensor.tobytes()).hexdigest()
        == "c076714634e482aa40d18b763447b1897e255b29351b0fcc05262c7dc64fd091"
    )
    np.testing.assert_array_equal(tensor[:, :, :4], np.ones((1, 64, 4)))
    np.testing.assert_array_equal(tensor[:, :, -4:], np.ones((1, 64, 4)))


def test_preprocess_v1_rejects_non_v1_source_shape():
    with pytest.raises(ValueError, match="uint8 RGB"):
        preprocess_v1_rgb(np.zeros((64, 160, 3), dtype=np.uint8))
