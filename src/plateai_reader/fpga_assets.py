"""Fail-closed contract for the separately licensed FPGA-LPR ONNX pair."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


SCHEMA = "fpga-lpr-onnx-v1"
MODEL_ID = "fpga-lpr-mit-v1"
SOURCE_COMMIT = "574667ca7f5730d17b4b6fcda3ec568521bcbcd8"
MODEL_REVISION = "51b9606b174eedbc091aec844c288a72aa9cd25b"
FPGA_CHARS = tuple("0123456789ABCDEFGHJKLMNPQRSTUVWXYZIO-")


class FpgaAssetError(ValueError):
    """An external model asset cannot be trusted or used."""


@dataclass(frozen=True, slots=True)
class FpgaComponent:
    filename: str
    sha256: str
    inputs: Mapping[str, tuple[str | int, ...]]
    outputs: Mapping[str, tuple[str | int, ...]]


@dataclass(frozen=True, slots=True)
class FpgaLprManifest:
    schema: str
    model_id: str
    source_commit: str
    model_revision: str
    license_notice: str
    charset: tuple[str, ...]
    components: Mapping[str, FpgaComponent]

    @classmethod
    def from_document(cls, data: object) -> "FpgaLprManifest":
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise FpgaAssetError("unsupported external model schema")
        for field, expected in (
            ("model_id", MODEL_ID),
            ("source_commit", SOURCE_COMMIT),
            ("model_revision", MODEL_REVISION),
            ("license_notice", "MIT-LICENSE.txt"),
        ):
            if data.get(field) != expected:
                raise FpgaAssetError(f"invalid {field}")
        if data.get("charset") != list(FPGA_CHARS):
            raise FpgaAssetError("invalid charset")
        definitions = {
            "cpm": (
                "cpm.onnx",
                {"input": ("batch", 3, 100, 100)},
                {"stage": ("batch", 4, 50, 50), "heatmap": ("batch", 4, 50, 50)},
            ),
            "lprnet": (
                "lprnet.onnx",
                {"input": ("batch", 3, 48, 94)},
                {"logits": ("batch", 37, 18)},
            ),
        }
        raw_components = data.get("components")
        if not isinstance(raw_components, dict) or set(raw_components) != set(definitions):
            raise FpgaAssetError("invalid components")
        components: dict[str, FpgaComponent] = {}
        for name, (filename, inputs, outputs) in definitions.items():
            raw = raw_components[name]
            if not isinstance(raw, dict):
                raise FpgaAssetError(f"invalid {name} component")
            if raw.get("filename") != filename:
                raise FpgaAssetError(f"invalid {name} filename")
            digest = raw.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise FpgaAssetError(f"invalid {name} sha256")
            if raw.get("inputs") != {key: list(value) for key, value in inputs.items()}:
                raise FpgaAssetError(f"invalid {name} input metadata")
            if raw.get("outputs") != {key: list(value) for key, value in outputs.items()}:
                raise FpgaAssetError(f"invalid {name} output metadata")
            components[name] = FpgaComponent(filename, digest, inputs, outputs)
        return cls(
            schema=SCHEMA,
            model_id=MODEL_ID,
            source_commit=SOURCE_COMMIT,
            model_revision=MODEL_REVISION,
            license_notice="MIT-LICENSE.txt",
            charset=FPGA_CHARS,
            components=components,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fpga_manifest(assets_dir: Path) -> FpgaLprManifest:
    """Validate provenance, exact metadata, and both bytes before ONNX loading."""

    base = Path(assets_dir)
    try:
        data = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FpgaAssetError("missing or malformed external manifest") from exc
    manifest = FpgaLprManifest.from_document(data)
    for relative, description in (
        (manifest.license_notice, "license notice"),
        ("UPSTREAM.md", "upstream attribution"),
    ):
        path = base / relative
        if path.is_symlink() or not path.is_file() or not path.read_bytes().strip():
            raise FpgaAssetError(f"missing {description}")
    for name, component in manifest.components.items():
        path = base / component.filename
        if path.is_symlink():
            raise FpgaAssetError(f"invalid {name} model path")
        try:
            digest = sha256_file(path)
        except OSError as exc:
            raise FpgaAssetError(f"missing {name} model") from exc
        if digest != component.sha256:
            raise FpgaAssetError(f"{name} sha256 mismatch")
    return manifest
