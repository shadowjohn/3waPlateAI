"""Lossless image encoding at the OpenCV RGB/BGR boundary."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def encode_png(image_rgb: NDArray[np.uint8]) -> bytes:
    """Encode an RGB uint8 image as PNG bytes."""

    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.dtype != np.uint8
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.shape[0] < 1
        or image_rgb.shape[1] < 1
    ):
        raise ValueError("image_rgb must be a non-empty HxWx3 uint8 array")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("PNG encoding failed")
    return encoded.tobytes()
