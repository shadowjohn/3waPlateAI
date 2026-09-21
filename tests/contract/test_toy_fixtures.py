from __future__ import annotations

import hashlib
import json

from PIL import Image

from plateai_shared.rules import load_character_set

from tests.conftest import DEFAULT_CHARSET, ROOT


FIXTURE_ROOT = ROOT / "tests/fixtures/synthetic"


def test_committed_toy_fixtures_are_safe_complete_and_reproducible():
    manifest_path = FIXTURE_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["synthetic_only"] is True
    assert manifest["generator_version"] == "0.1.0.dev0"
    entries = manifest["fixtures"]
    assert len(entries) == 4
    assert len({entry["path"] for entry in entries}) == 4
    assert len({entry["seed"] for entry in entries}) == 4

    symbols = set(load_character_set(DEFAULT_CHARSET).symbols)
    fixture_root = FIXTURE_ROOT.resolve()
    for entry in entries:
        image_path = (FIXTURE_ROOT / entry["path"]).resolve()
        image_path.relative_to(fixture_root)
        assert image_path.parent == fixture_root
        payload = image_path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert entry["canonical"]
        assert set(entry["canonical"]) <= symbols
        assert set(entry["display"]) <= symbols | {"-"}
        assert entry["display"].replace("-", "") == entry["canonical"]
        with Image.open(image_path) as image:
            image.load()
            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.size == (320, 96)
