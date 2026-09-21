"""Deterministic geometric and photometric plate augmentation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

import cv2
import numpy as np

from .models import AugmentedPlate, AugmentProfile, RenderedPlate


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        _invalid(f"{context} must be finite")
    return result


def _probability(document: dict[str, Any], name: str) -> float:
    result = _number(document.get(name), name)
    if not 0.0 <= result <= 1.0:
        _invalid(f"{name} must be between 0 and 1")
    return result


def _float_range(
    document: dict[str, Any],
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float]:
    raw = document.get(name)
    if not isinstance(raw, list) or len(raw) != 2:
        _invalid(f"{name} must contain two numbers")
    low = _number(raw[0], f"{name}[0]")
    high = _number(raw[1], f"{name}[1]")
    if low > high:
        _invalid(f"{name} must be ordered from low to high")
    if minimum is not None and low < minimum:
        _invalid(f"{name} must be >= {minimum}")
    if maximum is not None and high > maximum:
        _invalid(f"{name} must be <= {maximum}")
    return (low, high)


def _jpeg_range(document: dict[str, Any]) -> tuple[int, int]:
    raw = document.get("jpeg_quality_range")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(type(value) is not int for value in raw)
    ):
        _invalid("jpeg_quality_range must contain two integers")
    low, high = raw
    if not (1 <= low <= high <= 100):
        _invalid("jpeg_quality_range must be ordered within 1..100")
    return (low, high)


def load_augment_profile(path: Path) -> AugmentProfile:
    """Read a bounded version-one augmentation profile."""

    try:
        document = json.loads(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"cannot read augmentation profile {path}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON augmentation profile: {path}") from exc

    if not isinstance(document, dict):
        _invalid("augmentation profile must be an object")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        _invalid("augmentation-profile schema_version must be 1")
    profile_id = document.get("id")
    if not isinstance(profile_id, str) or not profile_id:
        _invalid("augmentation-profile id must not be empty")

    jitter = _number(
        document.get("max_corner_jitter_ratio"), "max_corner_jitter_ratio"
    )
    if not 0.0 <= jitter <= 0.2:
        _invalid("max_corner_jitter_ratio must be between 0 and 0.20")

    raw_kernels = document.get("motion_blur_kernels")
    if (
        not isinstance(raw_kernels, list)
        or not raw_kernels
        or any(
            type(kernel) is not int or kernel <= 0 or kernel % 2 == 0
            for kernel in raw_kernels
        )
    ):
        _invalid("motion_blur_kernels must contain odd positive integers")

    return AugmentProfile(
        id=profile_id,
        perspective_probability=_probability(document, "perspective_probability"),
        max_corner_jitter_ratio=jitter,
        rotation_probability=_probability(document, "rotation_probability"),
        rotation_degrees=_float_range(document, "rotation_degrees"),
        brightness_probability=_probability(document, "brightness_probability"),
        brightness_range=_float_range(
            document, "brightness_range", minimum=0.01
        ),
        gamma_probability=_probability(document, "gamma_probability"),
        gamma_range=_float_range(document, "gamma_range", minimum=0.01),
        noise_probability=_probability(document, "noise_probability"),
        noise_std_range=_float_range(
            document, "noise_std_range", minimum=0.0
        ),
        motion_blur_probability=_probability(
            document, "motion_blur_probability"
        ),
        motion_blur_kernels=tuple(raw_kernels),
        glare_probability=_probability(document, "glare_probability"),
        glare_opacity_range=_float_range(
            document, "glare_opacity_range", minimum=0.0, maximum=1.0
        ),
        glare_radius_ratio_range=_float_range(
            document,
            "glare_radius_ratio_range",
            minimum=0.001,
            maximum=1.0,
        ),
        jpeg_probability=_probability(document, "jpeg_probability"),
        jpeg_quality_range=_jpeg_range(document),
    )


def derive_sample_seed(run_seed: int, index: int) -> int:
    payload = f"3waPlateAI:{run_seed}:{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if low == high:
        return low
    return float(rng.uniform(low, high))


def _chance(rng: np.random.Generator, probability: float) -> bool:
    return probability > 0.0 and bool(rng.random() < probability)


def _apply_geometry(
    rendered: RenderedPlate,
    profile: AugmentProfile,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = np.asarray(rendered.corners, dtype=np.float32)
    height, width = rendered.image_rgb.shape[:2]
    perspective_applied = _chance(rng, profile.perspective_probability)
    rotation_applied = _chance(rng, profile.rotation_probability)
    metadata: dict[str, Any] = {
        "perspective_applied": perspective_applied,
        "rotation_applied": rotation_applied,
        "rotation_degrees": 0.0,
        "corner_offsets": [[0.0, 0.0] for _ in range(4)],
        "geometry_fallback": False,
    }
    if not perspective_applied and not rotation_applied:
        return rendered.image_rgb.copy(), source.copy(), metadata

    original_area = abs(cv2.contourArea(source, oriented=True))
    max_jitter = profile.max_corner_jitter_ratio * min(width, height)
    for _ in range(8):
        destination = source.copy()
        offsets = np.zeros((4, 2), dtype=np.float32)
        if perspective_applied and max_jitter > 0.0:
            offsets = rng.uniform(-max_jitter, max_jitter, size=(4, 2)).astype(
                np.float32
            )
            destination += offsets

        angle = (
            _uniform(rng, profile.rotation_degrees) if rotation_applied else 0.0
        )
        if rotation_applied:
            rotation = cv2.getRotationMatrix2D(
                ((width - 1.0) / 2.0, (height - 1.0) / 2.0), angle, 1.0
            )
            destination = cv2.transform(destination[None, :, :], rotation)[0]

        destination[:, 0] = np.clip(destination[:, 0], 0.0, width - 1.0)
        destination[:, 1] = np.clip(destination[:, 1], 0.0, height - 1.0)
        signed_area = cv2.contourArea(destination.astype(np.float32), oriented=True)
        if (
            np.isfinite(destination).all()
            and signed_area > 0.0
            and signed_area >= 0.4 * original_area
        ):
            homography = cv2.getPerspectiveTransform(
                source.astype(np.float32), destination.astype(np.float32)
            )
            image = cv2.warpPerspective(
                rendered.image_rgb,
                homography,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            corners = cv2.perspectiveTransform(
                source[None, :, :], homography
            )[0].astype(np.float32)
            metadata["rotation_degrees"] = angle
            metadata["corner_offsets"] = offsets.astype(float).tolist()
            return image, corners, metadata

    metadata["geometry_fallback"] = True
    return rendered.image_rgb.copy(), source.copy(), metadata


def _clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _apply_photometric(
    image_rgb: np.ndarray,
    profile: AugmentProfile,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = image_rgb.copy()
    metadata: dict[str, Any] = {}

    applied = _chance(rng, profile.brightness_probability)
    factor = _uniform(rng, profile.brightness_range) if applied else 1.0
    if applied:
        image = _clip_uint8(image.astype(np.float32) * factor)
    metadata["brightness"] = {"applied": applied, "factor": factor}

    applied = _chance(rng, profile.gamma_probability)
    gamma = _uniform(rng, profile.gamma_range) if applied else 1.0
    if applied:
        normalized = image.astype(np.float32) / 255.0
        image = _clip_uint8(np.power(normalized, gamma) * 255.0)
    metadata["gamma"] = {"applied": applied, "gamma": gamma}

    applied = _chance(rng, profile.noise_probability)
    sigma = _uniform(rng, profile.noise_std_range) if applied else 0.0
    if applied:
        noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
        image = _clip_uint8(image.astype(np.float32) + noise)
    metadata["noise"] = {"applied": applied, "sigma": sigma}

    applied = _chance(rng, profile.motion_blur_probability)
    kernel_size = (
        int(rng.choice(profile.motion_blur_kernels)) if applied else 1
    )
    if applied and kernel_size > 1:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        image = _clip_uint8(cv2.filter2D(image, -1, kernel).astype(np.float32))
    metadata["motion_blur"] = {
        "applied": applied,
        "kernel_size": kernel_size,
    }

    applied = _chance(rng, profile.glare_probability)
    opacity = _uniform(rng, profile.glare_opacity_range) if applied else 0.0
    radius_ratio = (
        _uniform(rng, profile.glare_radius_ratio_range) if applied else 0.0
    )
    center_x = int(rng.integers(0, image.shape[1])) if applied else 0
    center_y = int(rng.integers(0, image.shape[0])) if applied else 0
    if applied:
        radius = max(1.0, radius_ratio * min(image.shape[:2]))
        yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        alpha = np.clip(1.0 - distance / radius, 0.0, 1.0) * opacity
        image_float = image.astype(np.float32)
        image = _clip_uint8(
            image_float * (1.0 - alpha[..., None]) + 255.0 * alpha[..., None]
        )
    metadata["glare"] = {
        "applied": applied,
        "opacity": opacity,
        "radius_ratio": radius_ratio,
        "center": [center_x, center_y],
    }

    applied = _chance(rng, profile.jpeg_probability)
    quality = (
        int(rng.integers(profile.jpeg_quality_range[0], profile.jpeg_quality_range[1] + 1))
        if applied
        else 100
    )
    if applied:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        encoded, buffer = cv2.imencode(
            ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not encoded:
            raise RuntimeError("JPEG augmentation encode failed")
        decoded_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            raise RuntimeError("JPEG augmentation decode failed")
        image = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    metadata["jpeg"] = {"applied": applied, "quality": quality}

    return np.ascontiguousarray(image, dtype=np.uint8), metadata


def apply_augmentations(
    rendered: RenderedPlate,
    profile: AugmentProfile,
    sample_seed: int,
) -> AugmentedPlate:
    """Apply the stable M1 transform sequence using only per-sample state."""

    rng = np.random.default_rng(sample_seed)
    image, corners, geometry = _apply_geometry(rendered, profile, rng)
    image, photometric = _apply_photometric(image, profile, rng)
    metadata = dict(rendered.metadata)
    metadata.update(
        {
            "augmentation_profile_id": profile.id,
            "sample_seed": sample_seed,
            "geometry": geometry,
            "photometric": photometric,
        }
    )
    return AugmentedPlate(
        image_rgb=image,
        corners=np.ascontiguousarray(corners, dtype=np.float32),
        metadata=MappingProxyType(metadata),
    )
