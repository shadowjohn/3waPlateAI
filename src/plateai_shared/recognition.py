"""Framework-neutral contracts for v1 license-plate recognition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .contracts import CharacterSet


@dataclass(frozen=True, slots=True)
class PreprocessSpec:
    """The byte-level preprocessing contract used by the v1 Model Bundle."""

    source_size_wh: tuple[int, int]
    input_size_hw: tuple[int, int]
    resized_size_hw: tuple[int, int]
    padding_ltrb: tuple[int, int, int, int]
    padding_raw_value: int
    grayscale_reference: str
    resize_reference: str
    pillow_version: str


V1_PREPROCESS = PreprocessSpec(
    source_size_wh=(380, 160),
    input_size_hw=(64, 160),
    resized_size_hw=(64, 152),
    padding_ltrb=(4, 0, 4, 0),
    padding_raw_value=255,
    grayscale_reference="pillow-image-convert-l",
    resize_reference="pillow-image-resize-bilinear",
    pillow_version="12.3.0",
)


@dataclass(frozen=True, slots=True)
class CTCCodec:
    """One-based visible-symbol mapping with PyTorch-compatible blank zero."""

    symbols: tuple[str, ...]
    index_by_symbol: Mapping[str, int]
    blank_index: int = 0

    @classmethod
    def from_charset(cls, charset: CharacterSet) -> CTCCodec:
        symbols = tuple(charset.symbols)
        if any(not symbol or symbol.isspace() for symbol in symbols):
            raise ValueError("charset must not contain a blank symbol")
        if len(set(symbols)) != len(symbols):
            raise ValueError("charset must not contain duplicate symbols")
        return cls(
            symbols=symbols,
            index_by_symbol=MappingProxyType(
                {symbol: index for index, symbol in enumerate(symbols, start=1)}
            ),
        )

    @property
    def class_count(self) -> int:
        return len(self.symbols) + 1

    def encode(self, text: str) -> tuple[int, ...]:
        encoded: list[int] = []
        for symbol in text:
            try:
                encoded.append(self.index_by_symbol[symbol])
            except KeyError as error:
                raise ValueError(f"character {symbol!r} is not in charset") from error
        return tuple(encoded)

    def decode_greedy(self, indices: Sequence[int]) -> str:
        collapsed: list[int] = []
        for index in indices:
            if not isinstance(index, int):
                raise ValueError("CTC index must be an integer")
            if not collapsed or collapsed[-1] != index:
                collapsed.append(index)

        decoded: list[str] = []
        for index in collapsed:
            if index == self.blank_index:
                continue
            if not 0 <= index < self.class_count:
                raise ValueError(f"CTC index {index} is outside 0..{self.class_count - 1}")
            decoded.append(self.symbols[index - 1])
        return "".join(decoded)

    def required_timesteps(self, target: Sequence[int]) -> int:
        if any(index == self.blank_index for index in target):
            raise ValueError("CTC targets must not contain the blank index")
        return len(target) + sum(
            current == previous for previous, current in zip(target, target[1:])
        )


def preprocess_v1_rgb(image_rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Apply the pinned Pillow 12.3.0 v1 preprocessing reference exactly."""

    source_width, source_height = V1_PREPROCESS.source_size_wh
    if image_rgb.dtype != np.uint8 or image_rgb.shape != (source_height, source_width, 3):
        raise ValueError(
            "v1 preprocessing requires a uint8 RGB array shaped [160, 380, 3]"
        )

    grayscale = Image.fromarray(image_rgb, mode="RGB").convert("L")
    resized_width = V1_PREPROCESS.resized_size_hw[1]
    resized_height = V1_PREPROCESS.resized_size_hw[0]
    resized = grayscale.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )

    input_height, input_width = V1_PREPROCESS.input_size_hw
    left, _, _, _ = V1_PREPROCESS.padding_ltrb
    canvas = np.full(
        (input_height, input_width), V1_PREPROCESS.padding_raw_value, dtype=np.uint8
    )
    canvas[:, left : left + resized_width] = np.asarray(resized, dtype=np.uint8)
    return np.ascontiguousarray(canvas[np.newaxis, ...], dtype=np.float32) / np.float32(
        255.0
    )
