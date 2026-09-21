from __future__ import annotations

import random
from pathlib import Path
import pytest

from plateai_shared.recognition import CTCCodec
from plateai_shared.rules import generate_plate, load_character_set, load_ruleset


ROOT = Path(__file__).resolve().parents[2]
STD_CHARSET_PATH = ROOT / "configs/charsets/tw_standard_v1.txt"
STD_RULES_PATH = ROOT / "configs/plate_rules/tw_standard_v1.json"


def test_standard_charset_contains_34_symbols_including_4_without_io():
    charset = load_character_set(STD_CHARSET_PATH)
    assert len(charset.symbols) == 34
    assert "4" in charset.symbols
    assert "I" not in charset.symbols
    assert "O" not in charset.symbols
    codec = CTCCodec.from_charset(charset)
    assert codec.class_count == 35


def test_standard_rules_load_and_sample_deterministically():
    charset = load_character_set(STD_CHARSET_PATH)
    ruleset = load_ruleset(STD_RULES_PATH, charset)

    assert ruleset.id == "tw-standard-v1"
    assert len(ruleset.rules) == 11
    assert all(rule.enabled for rule in ruleset.rules)

    rule_ids = {rule.id for rule in ruleset.rules}
    expected_ids = {
        "new-style-lll-dddd",
        "legacy-ll-dddd",
        "legacy-dddd-ll",
        "moto-lll-ddd",
        "moto-ddd-lll",
        "moto-lld-ddd",
        "moto-dll-ddd",
        "legacy-ll-ddd",
        "legacy-ddd-ll",
        "legacy-ll-dd",
        "legacy-dd-ll",
    }
    assert rule_ids == expected_ids

    seen_rules = set()
    for seed in range(50):
        sample = generate_plate(ruleset, random.Random(seed))
        seen_rules.add(sample.rule_id)
        assert "-" in sample.display
        assert sample.display.replace("-", "") == sample.canonical
        assert all(c in charset.symbols for c in sample.canonical)

    # Across 50 samples, multiple distinct rules must be hit
    assert len(seen_rules) >= 5


def test_target_plate_formats_match_standard_rules():
    charset = load_character_set(STD_CHARSET_PATH)
    ruleset = load_ruleset(STD_RULES_PATH, charset)

    plates_to_test = [
        ("AQ-560", "legacy-ll-ddd"),
        ("XHU-013", "moto-lll-ddd"),
        ("HJ9-037", "moto-lld-ddd"),
        ("LAB-6531", "new-style-lll-dddd"),
    ]

    for display, expected_rule_id in plates_to_test:
        canonical = display.replace("-", "")
        # Verify characters are in charset
        assert all(c in charset.symbols for c in canonical)

        # Verify a matching rule exists and validates the token structure
        matching = [r for r in ruleset.rules if r.id == expected_rule_id]
        assert len(matching) == 1
        rule = matching[0]
        assert len(rule.tokens) == len(canonical)
        for token, char in zip(rule.tokens, canonical):
            if token.kind == "class":
                assert char in ruleset.character_classes[token.value]
            else:
                assert char == token.value
