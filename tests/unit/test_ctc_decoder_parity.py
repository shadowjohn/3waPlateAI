"""Independent exhaustive CTC paths guard optimized decoding and tie order."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from plateai_reader.runtime import decode_constrained_ctc_v1
from plateai_shared.recognition import CTCCodec


@pytest.mark.parametrize('seed', range(8))
def test_decoder_matches_exhaustive_paths_including_repeats_and_ties(seed):
    # Deliberately non-alphabetic charset order catches argmax tie shortcuts.
    codec = CTCCodec(('B', 'A'), {'B': 1, 'A': 2})
    token = SimpleNamespace(kind='class', value='letter')
    rules = [SimpleNamespace(id=name, enabled=True, tokens=(token, token),
                             separator='-', separator_after=(1,), plate_type='test')
             for name in ('z', 'a')]
    ruleset = SimpleNamespace(rules=rules, character_classes={'letter': 'AB'})
    logits = np.full((80, 3), -100.0, dtype=np.float32)
    logits[:, 0] = 0
    logits[:5] = np.random.default_rng(seed).integers(-2, 3, (5, 3))
    values = logits.astype(np.float64)
    logp = values - np.log(np.exp(values).sum(axis=1, keepdims=True))
    paths = []
    for path in itertools.product(range(3), repeat=5):
        text = codec.decode_greedy(path)
        if len(text) == 2:
            score = sum(logp[t, c] for t, c in enumerate(path)) + logp[5:, 0].sum()
            paths.append((score, text))
    expected_score, expected_text = min(paths, key=lambda x: (-x[0], x[1]))
    actual = decode_constrained_ctc_v1(logits, codec, ruleset)
    assert actual.canonical == expected_text
    assert actual.log_probability == pytest.approx(expected_score, abs=1e-12)
    assert actual.rule_id == 'a'
