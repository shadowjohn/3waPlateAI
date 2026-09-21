"""Synthetic recognition-dataset generation."""

from .augment import apply_augmentations, derive_sample_seed, load_augment_profile
from .cli import build_parser, main
from .dataset import generate_dataset
from .encoder import encode_png
from .fonts import resolve_font
from .renderer import render_plate
from .templates import load_template

__all__ = [
    "apply_augmentations",
    "build_parser",
    "derive_sample_seed",
    "encode_png",
    "generate_dataset",
    "load_augment_profile",
    "load_template",
    "render_plate",
    "resolve_font",
    "main",
]
