"""Shared contracts for 3waPlateAI."""

from .contracts import CharacterSet, GeneratedPlate, PlateRule, PlateRuleset, RuleToken
from .rules import generate_plate, load_character_set, load_ruleset
from .schema_validation import (
    DocumentValidationError,
    load_schema,
    validate_document,
    validate_model_manifest,
)

__all__ = [
    "CharacterSet",
    "GeneratedPlate",
    "DocumentValidationError",
    "PlateRule",
    "PlateRuleset",
    "RuleToken",
    "generate_plate",
    "load_character_set",
    "load_ruleset",
    "load_schema",
    "validate_document",
    "validate_model_manifest",
]
