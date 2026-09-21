"""Synthetic multi-instance detection data generation."""

from .composite import generate_composite_dataset
from .contracts import (
    CompositeGenerationRequest,
    CompositeGenerationSummary,
    CompositeInstance,
)

__all__ = [
    "CompositeGenerationRequest",
    "CompositeGenerationSummary",
    "CompositeInstance",
    "generate_composite_dataset",
]
