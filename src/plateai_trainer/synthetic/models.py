"""Immutable models used by the synthetic data pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from plateai_shared.contracts import JsonValue


@dataclass(frozen=True, slots=True)
class PlateTemplate:
    id: str
    width: int
    height: int
    background_rgb: tuple[int, int, int]
    foreground_rgb: tuple[int, int, int]
    border_rgb: tuple[int, int, int]
    border_width: int
    corner_radius: int
    text_box: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class FontSpec:
    kind: Literal["hershey", "truetype"]
    name: str
    path: Path | None
    variation_axes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RenderedPlate:
    image_rgb: NDArray[np.uint8]
    corners: NDArray[np.float32]
    metadata: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class AugmentProfile:
    id: str
    perspective_probability: float
    max_corner_jitter_ratio: float
    rotation_probability: float
    rotation_degrees: tuple[float, float]
    brightness_probability: float
    brightness_range: tuple[float, float]
    gamma_probability: float
    gamma_range: tuple[float, float]
    noise_probability: float
    noise_std_range: tuple[float, float]
    motion_blur_probability: float
    motion_blur_kernels: tuple[int, ...]
    glare_probability: float
    glare_opacity_range: tuple[float, float]
    glare_radius_ratio_range: tuple[float, float]
    jpeg_probability: float
    jpeg_quality_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class AugmentedPlate:
    image_rgb: NDArray[np.uint8]
    corners: NDArray[np.float32]
    metadata: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    count: int
    seed: int
    output: Path
    charset_path: Path
    rules_path: Path
    template_path: Path
    augmentation_path: Path
    font: str | Path | None = None


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    schema_version: int
    generated: int
    seed: int
    rule_counts: Mapping[str, int]
    plate_type_counts: Mapping[str, int]
    charset_sha256: str
    rules_sha256: str
    template_sha256: str
    augmentation_sha256: str


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    schema_version: int
    index: int
    sample_seed: int
    image_path: str
    canonical: str
    display: str
    rule_id: str
    plate_type: str
    corners: tuple[tuple[float, float], ...]
    renderer: Mapping[str, JsonValue]
    augmentation: Mapping[str, JsonValue]
    image_sha256: str


ImageEncoder: TypeAlias = Callable[[NDArray[np.uint8]], bytes]
