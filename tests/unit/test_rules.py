from __future__ import annotations

import random

import pytest
from hypothesis import given, strategies as st

from plateai_shared.rules import generate_plate, load_character_set, load_ruleset

from tests.conftest import DEFAULT_CHARSET, DEFAULT_RULES, write_json


def test_default_rules_generate_canonical_and_display_text(default_ruleset):
    result = generate_plate(default_ruleset, random.Random(7))
    assert result.rule_id in {"standard-lll-dddd", "legacy-dddd-ll"}
    assert "-" not in result.canonical
    assert result.display.count("-") == 1
    assert result.display.replace("-", "") == result.canonical


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["rules"][0].update(tokens=[{"class": "UNKNOWN"}]), "unknown character class"),
        (lambda d: d["character_classes"].update(L=""), "must not be empty"),
        (lambda d: d["rules"][0].update(separator_after=[99]), "separator position"),
        (lambda d: [r.update(enabled=False) for r in d["rules"]], "no enabled rules"),
    ],
)
def test_invalid_rules_fail_before_generation(
    tmp_path, valid_rule_document, mutate, message
):
    mutate(valid_rule_document)
    path = write_json(tmp_path / "rules.json", valid_rule_document)
    with pytest.raises(ValueError, match=message):
        load_ruleset(path, load_character_set(DEFAULT_CHARSET))


def test_charset_rejects_blank_or_duplicate_symbols(tmp_path):
    path = tmp_path / "charset.txt"
    path.write_text("A\nB\nA\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate|blank"):
        load_character_set(path)


def test_charset_rejects_non_printable_symbols(tmp_path):
    path = tmp_path / "charset.txt"
    path.write_bytes(b"A\n\x00\n")
    with pytest.raises(ValueError, match="visible"):
        load_character_set(path)


@given(st.integers(min_value=-(2**31), max_value=2**31 - 1))
def test_generated_plate_round_trips_through_display(seed):
    charset = load_character_set(DEFAULT_CHARSET)
    ruleset = load_ruleset(DEFAULT_RULES, charset)
    result = generate_plate(ruleset, random.Random(seed))
    assert result.display.replace("-", "") == result.canonical
    assert all(symbol in charset.symbols for symbol in result.canonical)
