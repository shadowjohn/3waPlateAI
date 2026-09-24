"""Assemble a separate, attributed MIT crop-OCR service without a detector."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plateai_reader.fpga_assets import load_fpga_manifest  # noqa: E402


SOURCE_MODULES = (
    "src/plateai_reader/fpga_assets.py",
    "src/plateai_reader/fpga_lpr.py",
    "src/plateai_reader/rectifier.py",
    "src/plateai_web/mit_lpr_crop_api.py",
)
ASSETS = (
    "manifest.json", "cpm.onnx", "lprnet.onnx", "MIT-LICENSE.txt", "UPSTREAM.md",
)
ROOT_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md")

LAUNCHER = '''"""Attributed crop OCR service; whole-scene detection is unavailable."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from plateai_reader.fpga_lpr import FpgaLprRecognizer
from plateai_web.mit_lpr_crop_api import create_crop_api

try:
    recognizer = FpgaLprRecognizer(ROOT / "third_party" / "fpga_lpr")
except Exception:
    recognizer = None

app = create_crop_api(recognizer)
'''

REQUIREMENTS = '''fastapi==0.141.1
uvicorn==0.53.0
python-multipart==0.0.32
numpy==2.4.2
opencv-python-headless==5.0.0.93
onnxruntime==1.30.0
Pillow==12.3.0
'''

README = '''# 3waPlateAI MIT crop-only OCR service

This separately packaged service recognizes an **already-cropped, plate-centric image**.
It does not find a plate in a whole vehicle image. `POST /api/predict` returns HTTP 503
instead of inventing a full-scene result. No detector weights, dataset, original `.pth`
weights, or training framework are included.

The included CPM + LPRNet ONNX pair is attributed to the FPGA-LPR author; see
`third_party/fpga_lpr/UPSTREAM.md`, `MIT-LICENSE.txt`, and `THIRD_PARTY_NOTICES.md`.
The returned OCR score kind is `uncalibrated`; real-photo accuracy has not been
independently established. This package is distinct from the source-only 1788 scaffold.

Install Python 3.11+ runtime dependencies using `pip install -r requirements.txt`.
From this directory, run `uvicorn app:app --host 127.0.0.1 --port 1788`.
Use `GET /api/health`, then multipart `POST /api/recognize/crop` with field `file`.
If the asset validation fails, health says `inference_ready=false` and OCR returns 503.
'''


def build_mit_lpr_crop_release(root: Path, destination: Path) -> Path:
    root = Path(root).resolve()
    target = Path(destination).absolute()
    if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
        raise ValueError("destination_not_empty")
    source_assets = root / "third_party" / "fpga_lpr"
    load_fpga_manifest(source_assets)
    sources = SOURCE_MODULES + ROOT_FILES + tuple(f"third_party/fpga_lpr/{name}" for name in ASSETS)
    for relative in sources:
        source = root / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"invalid_source:{relative}")
    target.mkdir(parents=True, exist_ok=True)
    for relative in sources:
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, output)
    for package in ("plateai_reader", "plateai_web"):
        (target / "src" / package / "__init__.py").write_text("", encoding="utf-8")
    (target / "app.py").write_text(LAUNCHER, encoding="utf-8")
    (target / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (target / "README.md").write_text(README, encoding="utf-8")
    load_fpga_manifest(target / "third_party" / "fpga_lpr")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "mit-lpr-crop-release")
    args = parser.parse_args()
    print(build_mit_lpr_crop_release(ROOT, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
