"""Fail-closed M4.5 evaluation contracts and runtime."""

from .contracts import (
    DetectionPrediction,
    EvaluationInputError,
    EvaluationRuntimeError,
    GateDecision,
    LaneResult,
    ParityResult,
    PlateTruth,
    PopulationCounts,
    RecognitionPrediction,
    Scene,
    SourceRights,
    StrictProfile,
    SuiteSnapshot,
    load_profile,
)

__all__ = [
    "DetectionPrediction",
    "EvaluationInputError",
    "EvaluationRuntimeError",
    "GateDecision",
    "LaneResult",
    "ParityResult",
    "PlateTruth",
    "PopulationCounts",
    "RecognitionPrediction",
    "Scene",
    "SourceRights",
    "StrictProfile",
    "SuiteSnapshot",
    "load_profile",
]
