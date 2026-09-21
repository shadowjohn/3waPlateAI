"""Strict loading and deterministic sampling for data-driven plate rules."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

from .contracts import (
    CharacterSet,
    GeneratedPlate,
    PlateRule,
    PlateRuleset,
    RuleToken,
)


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def load_character_set(path: Path) -> CharacterSet:
    """Load visible, unique, single-code-point symbols in file order."""

    raw = _read_bytes(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"character set must be UTF-8: {path}") from exc

    lines = text.splitlines()
    if not lines:
        _invalid("character set must not be empty")

    symbols: list[str] = []
    seen: set[str] = set()
    for line_number, symbol in enumerate(lines, start=1):
        if not symbol or symbol.isspace():
            _invalid(f"blank character-set line at {line_number}")
        if len(symbol) != 1:
            _invalid(
                f"character-set line {line_number} must contain one Unicode code point"
            )
        if not symbol.isprintable():
            _invalid(
                f"character-set line {line_number} must contain a visible symbol"
            )
        if symbol in seen:
            _invalid(f"duplicate character-set symbol {symbol!r}")
        seen.add(symbol)
        symbols.append(symbol)

    return CharacterSet(
        id=path.stem,
        symbols=tuple(symbols),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _invalid(f"{context} must be an object")
    return value


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        _invalid(f"{context} must not be empty and must be a string")
    return value


def _parse_token(
    value: Any,
    *,
    context: str,
    character_classes: dict[str, str],
    charset_symbols: frozenset[str],
) -> RuleToken:
    token = _require_object(value, context)
    if set(token) == {"class"}:
        class_name = _require_nonempty_string(token["class"], f"{context}.class")
        if class_name not in character_classes:
            _invalid(f"{context} references unknown character class {class_name!r}")
        return RuleToken(kind="class", value=class_name)
    if set(token) == {"literal"}:
        literal = _require_nonempty_string(token["literal"], f"{context}.literal")
        if len(literal) != 1 or literal not in charset_symbols:
            _invalid(f"{context} literal must be one character-set symbol")
        return RuleToken(kind="literal", value=literal)
    _invalid(f"{context} must contain exactly one of 'class' or 'literal'")


def load_ruleset(path: Path, charset: CharacterSet) -> PlateRuleset:
    """Load and fully validate a version-one plate rules document."""

    raw = _read_bytes(path)
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON rules file: {path}") from exc

    root = _require_object(document, "plate rules")
    schema_version = root.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        _invalid("plate-rules schema_version must be 1")
    ruleset_id = _require_nonempty_string(root.get("id"), "plate-rules id")

    raw_classes = _require_object(
        root.get("character_classes"), "character_classes"
    )
    if not raw_classes:
        _invalid("character_classes must not be empty")
    charset_symbols = frozenset(charset.symbols)
    character_classes: dict[str, str] = {}
    for class_name, members_value in raw_classes.items():
        _require_nonempty_string(class_name, "character-class name")
        members = _require_nonempty_string(
            members_value, f"character class {class_name!r}"
        )
        if len(set(members)) != len(members):
            _invalid(f"character class {class_name!r} contains duplicate symbols")
        unknown = sorted(set(members) - charset_symbols)
        if unknown:
            _invalid(
                f"character class {class_name!r} contains symbols outside charset: {unknown}"
            )
        character_classes[class_name] = members

    raw_rules = root.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        _invalid("rules must be a non-empty array")

    rule_ids: set[str] = set()
    rules: list[PlateRule] = []
    for rule_index, raw_rule in enumerate(raw_rules):
        context = f"rules[{rule_index}]"
        rule = _require_object(raw_rule, context)
        rule_id = _require_nonempty_string(rule.get("id"), f"{context}.id")
        if rule_id in rule_ids:
            _invalid(f"duplicate rule id {rule_id!r}")
        rule_ids.add(rule_id)

        raw_tokens = rule.get("tokens")
        if not isinstance(raw_tokens, list) or not raw_tokens:
            _invalid(f"{context}.tokens must be a non-empty array")
        tokens = tuple(
            _parse_token(
                token,
                context=f"{context}.tokens[{token_index}]",
                character_classes=character_classes,
                charset_symbols=charset_symbols,
            )
            for token_index, token in enumerate(raw_tokens)
        )

        separator = _require_nonempty_string(
            rule.get("separator"), f"{context}.separator"
        )
        if len(separator) != 1:
            _invalid(f"{context}.separator must contain one Unicode code point")
        raw_positions = rule.get("separator_after")
        if not isinstance(raw_positions, list):
            _invalid(f"{context}.separator_after must be an array")
        if any(type(position) is not int for position in raw_positions):
            _invalid(f"{context} separator position must be an integer")
        positions = tuple(raw_positions)
        if len(set(positions)) != len(positions):
            _invalid(f"{context} separator positions must be unique")
        if any(position < 1 or position >= len(tokens) for position in positions):
            _invalid(
                f"{context} separator position must be between 1 and {len(tokens) - 1}"
            )

        plate_type = _require_nonempty_string(
            rule.get("plate_type"), f"{context}.plate_type"
        )
        enabled = rule.get("enabled")
        if type(enabled) is not bool:
            _invalid(f"{context}.enabled must be a boolean")
        weight_value = rule.get("weight")
        if isinstance(weight_value, bool) or not isinstance(weight_value, (int, float)):
            _invalid(f"{context}.weight must be a number")
        weight = float(weight_value)
        if not math.isfinite(weight) or (enabled and weight <= 0):
            _invalid(f"{context} enabled weight must be finite and positive")

        rules.append(
            PlateRule(
                id=rule_id,
                tokens=tokens,
                separator=separator,
                separator_after=positions,
                plate_type=plate_type,
                weight=weight,
                enabled=enabled,
            )
        )

    if not any(rule.enabled for rule in rules):
        _invalid("plate rules contain no enabled rules")

    return PlateRuleset(
        schema_version=schema_version,
        id=ruleset_id,
        character_classes=MappingProxyType(character_classes),
        rules=tuple(rules),
    )


def format_display(
    canonical: str, separator: str, separator_after: Sequence[int]
) -> str:
    """Format canonical characters with rule separators at configured positions."""
    pieces: list[str] = []
    positions = set(separator_after)
    for position, symbol in enumerate(canonical, start=1):
        pieces.append(symbol)
        if position in positions:
            pieces.append(separator)
    return "".join(pieces)


def matches_rule(
    canonical: str, rule: PlateRule, character_classes: Mapping[str, str]
) -> bool:
    """Check if a canonical string strictly matches one plate rule."""
    if len(canonical) != len(rule.tokens):
        return False
    for symbol, token in zip(canonical, rule.tokens, strict=True):
        if token.kind == "literal":
            if symbol != token.value:
                return False
        elif symbol not in character_classes[token.value]:
            return False
    return True


def generate_plate(ruleset: PlateRuleset, rng: random.Random) -> GeneratedPlate:
    """Sample one plate without reading global random state."""

    enabled = tuple(rule for rule in ruleset.rules if rule.enabled)
    rule = rng.choices(enabled, weights=[rule.weight for rule in enabled], k=1)[0]
    chars: list[str] = []
    for token in rule.tokens:
        if token.kind == "literal":
            chars.append(token.value)
        else:
            chars.append(rng.choice(ruleset.character_classes[token.value]))
    canonical = "".join(chars)
    return GeneratedPlate(
        canonical=canonical,
        display=format_display(canonical, rule.separator, rule.separator_after),
        rule_id=rule.id,
        plate_type=rule.plate_type,
    )
