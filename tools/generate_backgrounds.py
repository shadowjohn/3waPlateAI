"""Generate deterministic synthetic street and vehicle backgrounds for detection training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def generate_scene(rng: np.random.Generator, width: int = 960, height: int = 640) -> np.ndarray:
    """Generate a single realistic background image (road, vehicle surface, parking bay, wall)."""
    scene_type = rng.integers(0, 4)
    yy, xx = np.indices((height, width), dtype=np.float32)

    if scene_type == 0:
        # Asphalt road with horizon and optional road line
        horizon = int(height * rng.uniform(0.25, 0.45))
        img = np.zeros((height, width, 3), dtype=np.float32)
        # Sky / upper background
        sky_color = rng.uniform(140, 210, size=3)
        img[:horizon] = sky_color + rng.normal(0, 5, (horizon, width, 3))
        # Asphalt road
        road_color = rng.uniform(50, 90, size=3)
        road_grad = ((yy[horizon:] - horizon) / (height - horizon))[:, :, None] * rng.uniform(10, 30)
        img[horizon:] = road_color + road_grad + rng.normal(0, 8, (height - horizon, width, 3))
        # Optional lane line
        if rng.random() > 0.4:
            line_x = int(width * rng.uniform(0.3, 0.7))
            line_w = int(rng.uniform(15, 30))
            line_color = np.array([210, 205, 70] if rng.random() > 0.5 else [230, 230, 230], dtype=np.float32)
            mask = (xx[horizon:] >= line_x - line_w / 2) & (xx[horizon:] <= line_x + line_w / 2)
            img[horizon:][mask] = line_color + rng.normal(0, 10, (int(np.sum(mask)), 3))

    elif scene_type == 1:
        # Vehicle rear bumper / trunk panel
        body_color = rng.choice([
            np.array([220, 220, 225]),  # White/silver
            np.array([35, 35, 40]),     # Black/dark grey
            np.array([160, 30, 35]),    # Red
            np.array([30, 50, 120]),    # Blue
            np.array([120, 125, 130]),  # Gunmetal
        ]).astype(np.float32)
        # Vignette / lighting gradient
        center_x, center_y = width / 2.0, height / 2.0
        dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2) / np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1.0 - 0.25 * dist[:, :, None]
        img = body_color[None, None, :] * vignette + rng.normal(0, 4, (height, width, 3))

    elif scene_type == 2:
        # Parking lot / concrete floor with wall
        wall_h = int(height * rng.uniform(0.3, 0.5))
        img = np.zeros((height, width, 3), dtype=np.float32)
        wall_color = rng.uniform(120, 170, size=3)
        floor_color = rng.uniform(80, 120, size=3)
        img[:wall_h] = wall_color + rng.normal(0, 6, (wall_h, width, 3))
        img[wall_h:] = floor_color + rng.normal(0, 7, (height - wall_h, width, 3))
        # Parking line
        line_angle = rng.uniform(0.1, 0.4)
        line_mask = np.abs((xx - width * 0.5) - line_angle * (yy - wall_h)) < rng.uniform(10, 20)
        img[line_mask] = np.array([220, 210, 80], dtype=np.float32)

    else:
        # Urban garage / shaded street texture
        base = rng.uniform(70, 150, size=3)
        grad_x = (xx / width)[:, :, None] * rng.uniform(-30, 30)
        grad_y = (yy / height)[:, :, None] * rng.uniform(-30, 30)
        img = base + grad_x + grad_y + rng.normal(0, 10, (height, width, 3))

    # Add Gaussian blur to simulate realistic depth of field / natural camera capture
    img_clipped = np.clip(img, 0, 255).astype(np.uint8)
    ksize = rng.choice([3, 5])
    return cv2.GaussianBlur(img_clipped, (ksize, ksize), 0)


def create_manifest(output_dir: Path, manifest_name: str, count: int, seed: int) -> Path:
    """Generate images and write detection_background_manifest.schema.json compliant file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(seed))

    background_entries = []
    for idx in range(count):
        img_rgb = generate_scene(rng)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        filename = f"images/bg_{idx:04d}.png"
        file_path = output_dir / filename
        ok, buf = cv2.imencode(".png", img_bgr)
        assert ok, "Failed to encode background png"
        file_bytes = buf.tobytes()
        file_path.write_bytes(file_bytes)
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        background_entries.append({
            "image_path": filename,
            "sha256": sha256,
        })

    manifest = {
        "schema_version": 1,
        "backgrounds": background_entries,
    }
    manifest_path = output_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "out" / "backgrounds"

    train_manifest = create_manifest(out_dir / "train", "train_manifest.json", count=20, seed=1001)
    val_manifest = create_manifest(out_dir / "val", "val_manifest.json", count=6, seed=2002)

    print(f"Generated train background manifest: {train_manifest}")
    print(f"Generated val background manifest: {val_manifest}")


if __name__ == "__main__":
    main()
