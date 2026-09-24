"""Convert only the pinned author CPM/LPRNet weights to attributed ONNX files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plateai_reader.fpga_assets import (  # noqa: E402
    FPGA_CHARS,
    MODEL_ID,
    MODEL_REVISION,
    SCHEMA,
    SOURCE_COMMIT,
    load_fpga_manifest,
    sha256_file,
)
from third_party.fpga_lpr.model_arch import CHARS, CPMLicensePlateNet, LPRNet  # noqa: E402


CPM_SOURCE_SHA256 = "70450f3e24570bf714e53d0c169ac52948e448c3945f6748d19909d525420fa4"
LPR_SOURCE_SHA256 = "21cda2d7f095d958eb725ac2dc05e87cbd342aed5fae7c0cf304477fe5b5a42b"
DEFAULT_FIXTURES = ROOT / "datasets" / "tlpd-taiwan-detector" / "images"
REPORT_DIR = ROOT / "runs" / "fpga-lpr-parity"


def verify_original_weights(weights_dir: Path) -> None:
    """Check both fixed byte hashes before any PyTorch deserialization."""

    for filename, expected, label in (
        ("best_val_loss.pth", CPM_SOURCE_SHA256, "CPM"),
        ("lpr_model_weight.pth", LPR_SOURCE_SHA256, "LPRNet"),
    ):
        path = Path(weights_dir) / filename
        try:
            actual = sha256_file(path)
        except OSError as exc:
            raise ValueError(f"missing {label} source") from exc
        if actual != expected:
            raise ValueError(f"{label} source sha256 mismatch")


def _load_models(weights_dir: Path) -> tuple[CPMLicensePlateNet, LPRNet]:
    verify_original_weights(weights_dir)
    if tuple(CHARS) != FPGA_CHARS:
        raise ValueError("pinned architecture charset mismatch")
    cpm = CPMLicensePlateNet(num_stages=6)
    lpr = LPRNet(lpr_max_len=7, phase=False, class_num=37, dropout_rate=0.5)
    cpm.load_state_dict(
        torch.load(Path(weights_dir) / "best_val_loss.pth", map_location="cpu", weights_only=True),
        strict=True,
    )
    lpr.load_state_dict(
        torch.load(Path(weights_dir) / "lpr_model_weight.pth", map_location="cpu", weights_only=True),
        strict=True,
    )
    return cpm.eval(), lpr.eval()


def _specification(filename: str, digest: str, model: str) -> dict[str, object]:
    if model == "cpm":
        inputs = {"input": ["batch", 3, 100, 100]}
        outputs = {
            "stage": ["batch", 4, 50, 50],
            "heatmap": ["batch", 4, 50, 50],
        }
    else:
        inputs = {"input": ["batch", 3, 48, 94]}
        outputs = {"logits": ["batch", 37, 18]}
    return {"filename": filename, "sha256": digest, "inputs": inputs, "outputs": outputs}


def _check_session(path: Path, inputs: dict[str, list[object]], outputs: dict[str, list[object]]) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual_inputs = {value.name: value.shape for value in session.get_inputs()}
    actual_outputs = {value.name: value.shape for value in session.get_outputs()}
    if actual_inputs != inputs or actual_outputs != outputs:
        raise ValueError(f"ONNX signature mismatch: {path.name}")
    for batch in (1, 2):
        tensor_shape = tuple(batch if value == "batch" else value for value in next(iter(inputs.values())))
        result = session.run(None, {"input": np.zeros(tensor_shape, np.float32)})
        expected_shapes = [
            tuple(batch if value == "batch" else value for value in values)
            for values in outputs.values()
        ]
        if [array.shape for array in result] != expected_shapes:
            raise ValueError(f"ONNX batch output mismatch: {path.name}")


def _decode(logits: np.ndarray) -> str:
    previous = -1
    letters: list[str] = []
    for value in np.argmax(logits, axis=0):
        current = int(value)
        if current != 36 and current != previous:
            letters.append(CHARS[current])
        previous = current
    return "".join(letters)


def _verify_reference(
    cpm: CPMLicensePlateNet,
    lpr: LPRNet,
    cpm_file: Path,
    lpr_file: Path,
    fixtures_dir: Path,
) -> list[dict[str, object]]:
    if not fixtures_dir.is_dir():
        raise ValueError(f"reference fixtures missing: {fixtures_dir}")
    fixtures = sorted(fixtures_dir.glob("*.jpg"))[:3]
    if len(fixtures) != 3:
        raise ValueError("reference parity requires three fixed plate crops")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    cpm_session = ort.InferenceSession(str(cpm_file), providers=["CPUExecutionProvider"])
    lpr_session = ort.InferenceSession(str(lpr_file), providers=["CPUExecutionProvider"])
    results: list[dict[str, object]] = []
    with torch.inference_mode():
        for path in fixtures:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"unreadable reference fixture: {path.name}")
            resized = cv2.resize(bgr, (100, 100)).astype(np.float32) / 255.0
            cpm_input = np.ascontiguousarray(resized.transpose(2, 0, 1)[None])
            reference_stage, reference_heatmap = cpm(torch.from_numpy(cpm_input))
            onnx_stage, onnx_heatmap = cpm_session.run(None, {"input": cpm_input})
            np.testing.assert_allclose(reference_stage.numpy(), onnx_stage, rtol=1e-3, atol=1e-4)
            np.testing.assert_allclose(reference_heatmap.numpy(), onnx_heatmap, rtol=1e-3, atol=1e-4)
            heatmap = F.interpolate(reference_heatmap, size=(100, 100), mode="bilinear", align_corners=False)
            points = np.array(
                [(xy[1], xy[0]) for xy in (np.unravel_index(np.argmax(hm), hm.shape) for hm in heatmap[0].numpy())],
                dtype=np.float32,
            )
            sums = points.sum(axis=1)
            diffs = np.diff(points, axis=1).ravel()
            corners = points[[np.argmin(sums), np.argmin(diffs), np.argmax(sums), np.argmax(diffs)]]
            destination = np.float32([[0, 0], [93, 0], [93, 47], [0, 47]])
            transform = cv2.getPerspectiveTransform(corners, destination)
            warped = cv2.warpPerspective((resized * 255).astype(np.uint8), transform, (94, 48))
            lpr_input = np.ascontiguousarray(
                (((warped.astype(np.float32) - 127.5) * 0.0078125).transpose(2, 0, 1))[None]
            )
            reference_logits = lpr(torch.from_numpy(lpr_input)).numpy()
            onnx_logits = lpr_session.run(None, {"input": lpr_input})[0]
            np.testing.assert_allclose(reference_logits, onnx_logits, rtol=1e-3, atol=1e-4)
            reference_text = _decode(reference_logits[0])
            onnx_text = _decode(onnx_logits[0])
            if reference_text != onnx_text:
                raise ValueError(f"greedy parity mismatch: {path.name}")
            results.append(
                {
                    "fixture": path.name,
                    "fixture_sha256": sha256_file(path),
                    "reference_text": reference_text,
                    "onnx_text": onnx_text,
                    "max_cpm_abs_delta": float(np.max(np.abs(reference_heatmap.numpy() - onnx_heatmap))),
                    "max_lpr_abs_delta": float(np.max(np.abs(reference_logits - onnx_logits))),
                }
            )
    return results


def export_fpga_lpr(
    weights_dir: Path,
    output_dir: Path,
    *,
    verify_reference: bool = False,
    fixtures_dir: Path = DEFAULT_FIXTURES,
) -> Path:
    """Produce the pinned dual-ONNX manifest only after both graphs validate."""

    cpm, lpr = _load_models(weights_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "MIT-LICENSE.txt").is_file() or not (output_dir / "UPSTREAM.md").is_file():
        raise ValueError("target lacks upstream attribution and license notice")
    with tempfile.TemporaryDirectory(prefix="fpga-lpr-export-", dir=output_dir.parent) as scratch:
        temporary = Path(scratch)
        cpm_file = temporary / "cpm.onnx"
        lpr_file = temporary / "lprnet.onnx"
        torch.onnx.export(
            cpm,
            (torch.zeros(1, 3, 100, 100),),
            str(cpm_file),
            input_names=["input"],
            output_names=["stage", "heatmap"],
            dynamic_axes={"input": {0: "batch"}, "stage": {0: "batch"}, "heatmap": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        torch.onnx.export(
            lpr,
            (torch.zeros(1, 3, 48, 94),),
            str(lpr_file),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        components = {
            "cpm": _specification("cpm.onnx", sha256_file(cpm_file), "cpm"),
            "lprnet": _specification("lprnet.onnx", sha256_file(lpr_file), "lprnet"),
        }
        for name, path in (("cpm", cpm_file), ("lprnet", lpr_file)):
            component = components[name]
            _check_session(path, component["inputs"], component["outputs"])
        parity = _verify_reference(cpm, lpr, cpm_file, lpr_file, fixtures_dir) if verify_reference else None
        manifest = {
            "schema": SCHEMA,
            "model_id": MODEL_ID,
            "source_commit": SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "license_notice": "MIT-LICENSE.txt",
            "charset": CHARS,
            "components": components,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for name in ("cpm.onnx", "lprnet.onnx", "manifest.json"):
            os.replace(temporary / name, output_dir / name)
    load_fpga_manifest(output_dir)
    if parity is not None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / "latest.json"
        report_path.write_text(
            json.dumps(
                {
                    "source_commit": SOURCE_COMMIT,
                    "model_revision": MODEL_REVISION,
                    "provider": "CPUExecutionProvider",
                    "torch_version": torch.__version__,
                    "onnxruntime_version": ort.__version__,
                    "created_at_unix": time.time(),
                    "tolerance": {"rtol": 1e-3, "atol": 1e-4},
                    "fixtures": parity,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    return output_dir / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-reference", action="store_true")
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES)
    args = parser.parse_args()
    manifest_path = export_fpga_lpr(
        args.weights_dir,
        args.output_dir,
        verify_reference=args.verify_reference,
        fixtures_dir=args.fixtures_dir,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
