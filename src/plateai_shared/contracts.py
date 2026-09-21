"""Immutable value objects shared by training and inference packages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class CharacterSet:
    id: str
    symbols: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class RuleToken:
    kind: Literal["class", "literal"]
    value: str


@dataclass(frozen=True, slots=True)
class PlateRule:
    id: str
    tokens: tuple[RuleToken, ...]
    separator: str
    separator_after: tuple[int, ...]
    plate_type: str
    weight: float
    enabled: bool


@dataclass(frozen=True, slots=True)
class PlateRuleset:
    schema_version: int
    id: str
    character_classes: Mapping[str, str]
    rules: tuple[PlateRule, ...]


@dataclass(frozen=True, slots=True)
class GeneratedPlate:
    canonical: str
    display: str
    rule_id: str
    plate_type: str


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
