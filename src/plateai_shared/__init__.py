"""Shared contracts for 3waPlateAI."""

from .contracts import CharacterSet, GeneratedPlate, PlateRule, PlateRuleset, RuleToken
from .recognition import CTCCodec, PreprocessSpec, V1_PREPROCESS, preprocess_v1_rgb
from .rules import generate_plate, load_character_set, load_ruleset
from .schema_validation import (
    DocumentValidationError,
    load_schema,
    validate_document,
    validate_model_manifest,
)

__all__ = [
    "CharacterSet",
    "CTCCodec",
    "GeneratedPlate",
    "DocumentValidationError",
    "PlateRule",
    "PlateRuleset",
    "PreprocessSpec",
    "RuleToken",
    "V1_PREPROCESS",
    "generate_plate",
    "load_character_set",
    "load_ruleset",
    "preprocess_v1_rgb",
    "load_schema",
    "validate_document",
    "validate_model_manifest",
]
