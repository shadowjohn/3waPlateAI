"""The external OCR pair has its own immutable, attributed asset contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from plateai_reader.fpga_assets import FpgaAssetError, load_fpga_manifest


CHARS = list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZIO-")


def _fixture_assets(folder: Path) -> Path:
    (folder / "cpm.onnx").write_bytes(b"cpm")
    (folder / "lprnet.onnx").write_bytes(b"lpr")
    (folder / "MIT-LICENSE.txt").write_text("MIT permission notice", encoding="utf-8")
    (folder / "UPSTREAM.md").write_text("evan6007/FPGA-LPR", encoding="utf-8")
    document = {
        "schema": "fpga-lpr-onnx-v1",
        "model_id": "fpga-lpr-mit-v1",
        "source_commit": "574667ca7f5730d17b4b6fcda3ec568521bcbcd8",
        "model_revision": "51b9606b174eedbc091aec844c288a72aa9cd25b",
        "license_notice": "MIT-LICENSE.txt",
        "charset": CHARS,
        "components": {
            "cpm": {
                "filename": "cpm.onnx",
                "sha256": hashlib.sha256(b"cpm").hexdigest(),
                "inputs": {"input": ["batch", 3, 100, 100]},
                "outputs": {"stage": ["batch", 4, 50, 50], "heatmap": ["batch", 4, 50, 50]},
            },
            "lprnet": {
                "filename": "lprnet.onnx",
                "sha256": hashlib.sha256(b"lpr").hexdigest(),
                "inputs": {"input": ["batch", 3, 48, 94]},
                "outputs": {"logits": ["batch", 37, 18]},
            },
        },
    }
    (folder / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
    return folder


def test_valid_pinned_manifest_loads(tmp_path: Path) -> None:
    manifest = load_fpga_manifest(_fixture_assets(tmp_path))
    assert manifest.model_id == "fpga-lpr-mit-v1"
    assert manifest.charset[-1] == "-"


def test_modified_onnx_fails_closed(tmp_path: Path) -> None:
    folder = _fixture_assets(tmp_path)
    (folder / "cpm.onnx").write_bytes(b"cpx")
    with pytest.raises(FpgaAssetError, match="cpm sha256 mismatch"):
        load_fpga_manifest(folder)


def test_missing_notice_fails_closed(tmp_path: Path) -> None:
    folder = _fixture_assets(tmp_path)
    (folder / "MIT-LICENSE.txt").unlink()
    with pytest.raises(FpgaAssetError, match="license notice"):
        load_fpga_manifest(folder)


def test_unknown_schema_fails_closed(tmp_path: Path) -> None:
    folder = _fixture_assets(tmp_path)
    path = folder / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema"] = "other"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(FpgaAssetError, match="unsupported external model schema"):
        load_fpga_manifest(folder)


def test_filename_escape_and_charset_change_fail_closed(tmp_path: Path) -> None:
    folder = _fixture_assets(tmp_path)
    path = folder / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["components"]["cpm"]["filename"] = "../cpm.onnx"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(FpgaAssetError, match="filename"):
        load_fpga_manifest(folder)
    document["components"]["cpm"]["filename"] = "cpm.onnx"
    document["charset"][-1] = "_"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(FpgaAssetError, match="charset"):
        load_fpga_manifest(folder)
