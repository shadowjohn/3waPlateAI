"""Bounded decoding shared by file and base64 requests; returns oriented BGR."""
from __future__ import annotations

import base64
import binascii
import io
import re
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ServiceError

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_BASE64_CHARS = 16 * 1024 * 1024
MAX_PIXELS = 20_000_000
MIMES = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp'}


def decode_base64(value: str) -> tuple[bytes, str | None]:
    if not isinstance(value, str) or not value:
        raise ServiceError('invalid_base64')
    if len(value) > MAX_BASE64_CHARS + 64:
        raise ServiceError('image_too_large', 413)
    mime = None
    if value.startswith('data:'):
        header, separator, value = value.partition(',')
        if not separator or not re.fullmatch(r'data:image/(jpeg|png|webp);base64', header):
            raise ServiceError('invalid_base64')
        mime = header[5:-7]
    if len(value) > MAX_BASE64_CHARS:
        raise ServiceError('image_too_large', 413)
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ServiceError('invalid_base64') from exc
    if not data:
        raise ServiceError('invalid_image')
    if len(data) > MAX_IMAGE_BYTES:
        raise ServiceError('image_too_large', 413)
    return data, mime


def decode_image(data: bytes, declared_mime: str | None = None) -> np.ndarray:
    if len(data) > MAX_IMAGE_BYTES:
        raise ServiceError('image_too_large', 413)
    if not data:
        raise ServiceError('invalid_image')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                w, h = source.size
                if w < 1 or h < 1 or w * h > MAX_PIXELS:
                    raise ServiceError('image_too_large', 413)
                mime = MIMES.get(source.format)
                if not mime or getattr(source, 'n_frames', 1) != 1:
                    raise ServiceError('unsupported_image', 415)
                if declared_mime and declared_mime != mime:
                    raise ServiceError('unsupported_image', 415)
                source.load()
                oriented = ImageOps.exif_transpose(source)
                # Transparent pixels use a deterministic white background.
                if 'A' in oriented.getbands() or 'transparency' in oriented.info:
                    rgba = oriented.convert('RGBA')
                    background = Image.new('RGBA', rgba.size, 'white')
                    oriented = Image.alpha_composite(background, rgba)
                return np.ascontiguousarray(np.asarray(oriented.convert('RGB'))[:, :, ::-1])
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ServiceError('image_too_large', 413) from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ServiceError('invalid_image') from exc
