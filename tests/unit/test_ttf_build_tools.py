from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_local_ttf_build_entrypoints_keep_the_generated_font_out_of_release_assets():
    prepare = REPOSITORY_ROOT / "build_prepare_ttf_env.bat"
    build = REPOSITORY_ROOT / "build_taiwanplate_ttf.bat"
    generator = REPOSITORY_ROOT / "tools" / "build_plate_font.py"

    assert prepare.is_file()
    assert build.is_file()
    assert ".venv\\Scripts\\python.exe" in prepare.read_text(encoding="utf-8")
    assert ".venv-font-build" in prepare.read_text(encoding="utf-8")
    assert ".venv-font-build" in build.read_text(encoding="utf-8")
    assert "tools\\build_plate_font.py" in build.read_text(encoding="utf-8")
    assert ".venv-font-build/" in (REPOSITORY_ROOT / ".gitignore").read_text(
        encoding="utf-8"
    )
    assert "assets/fonts/TaiwanPlate-Regular.ttf" in (
        REPOSITORY_ROOT / ".gitignore"
    ).read_text(encoding="utf-8")
    assert "TaiwanPlate-Regular.ttf" not in (
        REPOSITORY_ROOT / "MANIFEST.in"
    ).read_text(encoding="utf-8")
    assert "assets" in generator.read_text(encoding="utf-8")
    assert "local" in generator.read_text(encoding="utf-8")
    assert not (REPOSITORY_ROOT / "assets" / "fonts" / "TaiwanPlate-Regular.ttf").exists()
