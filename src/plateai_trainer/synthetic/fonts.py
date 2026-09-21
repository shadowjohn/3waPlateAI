"""Font selection with a redistribution-safe built-in fallback."""

from __future__ import annotations

from pathlib import Path

from .models import FontSpec


def resolve_font(value: str | Path | None) -> FontSpec:
    """Resolve an optional TrueType/OpenType path or OpenCV's built-in face."""

    if value is None:
        return FontSpec(kind="hershey", name="opencv-hershey-simplex", path=None)

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"font file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"font path is not a regular file: {path}")
    if path.suffix.lower() not in {".ttf", ".otf"}:
        raise ValueError("font file must use a .ttf or .otf extension")
    return FontSpec(kind="truetype", name=path.name, path=path.resolve())
