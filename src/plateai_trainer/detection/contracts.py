"""Immutable public contracts for detection composite generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import sysconfig

import numpy as np
from numpy.typing import NDArray

from plateai_shared.contracts import JsonValue


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_INSTALL_DATA_ROOT = Path(sysconfig.get_path("data")) / "share" / "3wa-plate-ai"


def _default_data_path(relative_path: str) -> Path:
    checkout_path = _REPOSITORY_ROOT / relative_path
    if checkout_path.is_file():
        return checkout_path
    return _INSTALL_DATA_ROOT / relative_path


@dataclass(frozen=True, slots=True)
class CompositeGenerationRequest:
    output: Path
    count: int
    seed: int
    background_manifest: Path
    instances_per_image: tuple[int, int] = (1, 3)
    config_path: Path = field(
        default_factory=lambda: _default_data_path(
            "configs/detection/composite_v1.json"
        )
    )
    charset_path: Path = field(
        default_factory=lambda: _default_data_path(
            "configs/charsets/tw_new_style_private_passenger_v1.txt"
        )
    )
    rules_path: Path = field(
        default_factory=lambda: _default_data_path(
            "configs/plate_rules/tw_new_style_private_passenger_v1.json"
        )
    )
    template_path: Path = field(
        default_factory=lambda: _default_data_path(
            "configs/plate_templates/new_style_private_passenger_white_v1.json"
        )
    )
    font: str | Path | None = None


@dataclass(frozen=True, slots=True)
class CompositeInstance:
    bbox_xyxy: tuple[float, float, float, float]
    corners_xy: NDArray[np.float32]
    source_plate: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class CompositeGenerationSummary:
    generated: int
    output: Path
    seed: int
    background_sha256s: tuple[str, ...]
