"""Validation for self-contained local v1 Model Bundles."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Mapping
from .contracts import JsonValue
from .rules import load_character_set
from .schema_validation import DocumentValidationError, validate_model_manifest

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_crop_bundle(bundle_dir: Path, schema_path: Path) -> Mapping[str, JsonValue]:
    root = Path(bundle_dir).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DocumentValidationError("manifest.json: invalid or missing") from exc
    if not isinstance(manifest, dict):
        raise DocumentValidationError("manifest.json: must be an object")
    try:
        charset_path = root / manifest["charset"]["file"]
        rules_path = root / manifest["rules"]["file"]
        model_path = root / manifest["components"]["recognizer"]["file"]
        report_path = root / manifest["provenance"]["training_report"]["file"]
    except (KeyError, TypeError) as exc:
        raise DocumentValidationError("manifest.json: missing bundle file declaration") from exc
    for path, declared in ((charset_path, manifest["charset"]["sha256"]), (rules_path, manifest["rules"]["sha256"]), (model_path, manifest["components"]["recognizer"]["sha256"]), (report_path, manifest["provenance"]["training_report"]["sha256"])):
        if not path.is_file() or path.resolve().parent != root or _sha(path) != declared:
            raise DocumentValidationError(f"bundle file hash mismatch: {path.name}")
    charset = load_character_set(charset_path)
    validate_model_manifest(manifest, schema_path, visible_charset_symbol_count=len(charset.symbols))
    expected = {"manifest.json", charset_path.name, rules_path.name, model_path.name, report_path.name}
    if {path.name for path in root.iterdir() if path.is_file()} != expected:
        raise DocumentValidationError("bundle: contains undeclared files")
    return manifest
