"""Deterministic reader components for Taiwan license-plate crops."""

from .rectifier import (
    InvalidCornersError,
    NormalizedCorners,
    normalize_corners,
)

__all__ = [
    "InvalidCornersError",
    "NormalizedCorners",
    "normalize_corners",
]
