import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from plateai_web.mit_lpr_crop_api import create_crop_api, decode_limited_rgb
from tools.build_mit_lpr_crop_release import build_mit_lpr_crop_release


ROOT = Path(__file__).resolve().parents[2]


class FakeRecognizer:
    def recognize(self, image):
        assert image.dtype == np.uint8 and image.shape[2] == 3
        return SimpleNamespace(
            raw_text="ABC1234", normalized_text="ABC1234", score_kind="uncalibrated",
            timings_ms={"total": 12.5},
        )


def _image_bytes() -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((20, 40, 3), 255, np.uint8))
    assert ok
    return encoded.tobytes()


def test_crop_api_returns_only_json_scalars_and_no_full_scene_claim() -> None:
    client = TestClient(create_crop_api(FakeRecognizer()))
    response = client.post("/api/recognize/crop", files={"file": ("crop.png", _image_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json() == {
        "raw_text": "ABC1234", "normalized_text": "ABC1234",
        "model_id": "fpga-lpr-mit-v1", "score_kind": "uncalibrated",
        "timings_ms": {"total": 12.5},
    }
    unavailable = client.post("/api/predict", files={"file": ("scene.png", _image_bytes(), "image/png")})
    assert unavailable.status_code == 503
    assert "crop-only" in unavailable.json()["detail"]


def test_missing_assets_and_bad_image_fail_closed() -> None:
    client = TestClient(create_crop_api(None))
    assert client.get("/api/health").json()["inference_ready"] is False
    assert client.post("/api/recognize/crop", files={"file": ("x.png", _image_bytes())}).status_code == 503
    ready = TestClient(create_crop_api(FakeRecognizer()))
    assert ready.post("/api/recognize/crop", files={"file": ("x.png", b"no image")}).status_code == 400
    with pytest.raises(ValueError, match="image_too_large"):
        decode_limited_rgb(b"x" * (12 * 1024 * 1024 + 1))


def test_release_contains_only_allowed_sources_and_verified_pair(tmp_path: Path) -> None:
    target = tmp_path / "release"
    build_mit_lpr_crop_release(ROOT, target)
    manifest = json.loads((target / "third_party/fpga_lpr/manifest.json").read_text(encoding="utf-8"))
    for name in ("cpm", "lprnet"):
        component = manifest["components"][name]
        assert hashlib.sha256((target / "third_party/fpga_lpr" / component["filename"]).read_bytes()).hexdigest() == component["sha256"]
    assert (target / "LICENSE").is_file()
    assert (target / "THIRD_PARTY_NOTICES.md").is_file()
    assert (target / "third_party/fpga_lpr/MIT-LICENSE.txt").is_file()
    assert not list(target.rglob("*.pth"))
    assert not list(target.rglob("plateai_trainer"))
    assert not list(target.rglob("plate_pose_net.onnx"))
    assert not (target / "src/plateai_reader/runtime.py").exists()
    assert {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()} == {
        "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "app.py", "requirements.txt",
        "src/plateai_reader/__init__.py", "src/plateai_reader/fpga_assets.py",
        "src/plateai_reader/fpga_lpr.py", "src/plateai_reader/rectifier.py",
        "src/plateai_web/__init__.py", "src/plateai_web/mit_lpr_crop_api.py",
        "third_party/fpga_lpr/manifest.json", "third_party/fpga_lpr/cpm.onnx",
        "third_party/fpga_lpr/lprnet.onnx", "third_party/fpga_lpr/MIT-LICENSE.txt",
        "third_party/fpga_lpr/UPSTREAM.md",
    }
    with pytest.raises(ValueError, match="destination_not_empty"):
        build_mit_lpr_crop_release(ROOT, target)
    (target / "third_party/fpga_lpr/cpm.onnx").write_bytes(b"corrupt")
    check = subprocess.run(
        [sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import app; print(app.recognizer is None)", str(target)],
        cwd=target, capture_output=True, text=True, check=True,
    )
    assert check.stdout.strip() == "True"
