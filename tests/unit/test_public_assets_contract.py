from __future__ import annotations

import hashlib
import json
from pathlib import Path

from plateai_shared.schema_validation import validate_document


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_public_asset_inventory_matches_checked_in_assets() -> None:
    config_path = PROJECT_ROOT / "configs" / "publication" / "public_assets_v1.json"
    schema_path = PROJECT_ROOT / "schemas" / "public_assets.schema.json"
    document = json.loads(config_path.read_text(encoding="utf-8"))
    validate_document(document, schema_path)

    for asset in document["assets"]:
        path = PROJECT_ROOT / asset["path"]
        assert path.is_file(), asset["path"]
        assert not path.is_symlink(), asset["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
