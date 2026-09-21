"""Font selection with a redistributable default and local overrides."""

from __future__ import annotations

from pathlib import Path
import sysconfig

from .models import FontSpec


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_INSTALL_DATA_ROOT = Path(sysconfig.get_path("data")) / "share" / "3wa-plate-ai"
_DEFAULT_FONT_NAME = "NotoSansMono[wdth,wght].ttf"
_TAIWAN_PLATE_FONT_NAME = "TaiwanPlate-Regular.ttf"
_DEFAULT_VARIATION_AXES = (700, 62)


def _bundled_font_path(name: str) -> Path:
    """Find a packaged font in either a checkout or an installed wheel."""

    checkout_path = _REPOSITORY_ROOT / "assets" / "fonts" / name
    if checkout_path.is_file():
        return checkout_path.resolve()
    installed_path = _INSTALL_DATA_ROOT / "fonts" / name
    if installed_path.is_file():
        return installed_path.resolve()
    raise FileNotFoundError(
        f"bundled font is missing: {name}; install the package with its data files"
    )


def _default_font_path() -> Path:
    return _bundled_font_path(_DEFAULT_FONT_NAME)


def resolve_font(value: str | Path | None) -> FontSpec:
    """Resolve the bundled OFL font or a user-provided TrueType/OpenType path."""

    if value is None:
        path = _default_font_path()
        return FontSpec(
            kind="truetype",
            name=path.name,
            path=path,
            variation_axes=_DEFAULT_VARIATION_AXES,
        )

    if str(value) in {"taiwan_plate", "official", _TAIWAN_PLATE_FONT_NAME}:
        path = _bundled_font_path(_TAIWAN_PLATE_FONT_NAME)
        return FontSpec(kind="truetype", name=path.name, path=path)

    path = Path(value)
    if not path.exists():
        if path.parent == Path("."):
            try:
                bundled_candidate = _bundled_font_path(path.name)
            except FileNotFoundError:
                bundled_candidate = None
            if bundled_candidate is not None:
                return FontSpec(
                    kind="truetype",
                    name=bundled_candidate.name,
                    path=bundled_candidate,
                )
        raise FileNotFoundError(f"font file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"font path is not a regular file: {path}")
    if path.suffix.lower() not in {".ttf", ".otf"}:
        raise ValueError("font file must use a .ttf or .otf extension")
    return FontSpec(kind="truetype", name=path.name, path=path.resolve())
