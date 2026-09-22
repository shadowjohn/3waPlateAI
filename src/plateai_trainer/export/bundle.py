"""Export and parity-check a v1 PyTorch checkpoint before publication."""
from __future__ import annotations
import hashlib, io, json, shutil, uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
import numpy as np
import onnx
import onnxruntime as ort
import torch
from numpy.typing import NDArray
from plateai_shared.bundle import validate_crop_bundle
from plateai_shared.contracts import CharacterSet
from plateai_shared.publication import OutputExistsError, publish_directory_no_replace, remove_owned_staging
from plateai_shared.recognition import CTCCodec, V1_PREPROCESS
from plateai_shared.rules import load_character_set
from plateai_trainer.synthetic.cli import _default_config_path
from plateai_trainer.training.model import PlateCTCNet

class ExportParityError(ValueError): pass
class OrtSession(Protocol):
    def get_inputs(self): ...
    def get_outputs(self): ...
    def run(self, output_names: Sequence[str] | None, input_feed: Mapping[str, NDArray[np.float32]]): ...
@dataclass(frozen=True, slots=True)
class ExportRequest:
    checkpoint: Path
    report: Path
    output: Path
    charset_path: Path | None = None
    rules_path: Path | None = None
def create_cpu_session(path: Path) -> OrtSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _paths():
    return (_default_config_path("configs/charsets/tw_new_style_private_passenger_v1.txt"), _default_config_path("configs/plate_rules/tw_new_style_private_passenger_v1.json"))
def _load_recognizer_checkpoint(snapshot: bytes, charset: CharacterSet, rules_sha256: str) -> PlateCTCNet:
    try:
        checkpoint = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint is not a readable v1 recognizer snapshot") from exc
    class_count = len(charset.symbols) + 1
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("class_count") != class_count
            or checkpoint.get("charset_sha256") != charset.sha256
            or checkpoint.get("rules_sha256") != rules_sha256
            or checkpoint.get("preprocess") != asdict(V1_PREPROCESS)):
        raise ValueError("checkpoint is not compatible with the v1 export contract")
    model = PlateCTCNet(class_count=class_count)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError("checkpoint has incompatible recognizer weights") from exc
    model.eval()
    return model
def _parity(model, session: OrtSession, codec: CTCCodec) -> None:
    for size in (1, 2):
        values = np.linspace(0.0, 1.0, size * 64 * 160, dtype=np.float32).reshape(size, 1, 64, 160)
        with torch.no_grad(): native = model(torch.from_numpy(values)).cpu().numpy()
        exported = session.run(["logits"], {"input": values})[0]
        if native.shape != exported.shape: raise ExportParityError("ONNX logits shape differs")
        for left, right in zip(native.argmax(2), exported.argmax(2), strict=True):
            if codec.decode_greedy(left.tolist()) != codec.decode_greedy(right.tolist()): raise ExportParityError("ONNX greedy decode differs")
        if not np.allclose(native, exported, rtol=1e-4, atol=1e-5): raise ExportParityError("ONNX logits differ")
def export_crop_bundle(request: ExportRequest, session_factory: Callable[[Path], OrtSession] = create_cpu_session) -> Path:
    if request.output.exists(): raise OutputExistsError(f"output already exists: {request.output}")
    default_charset, default_rules = _paths()
    charset_path = request.charset_path or default_charset
    rules_path = request.rules_path or default_rules
    charset = load_character_set(charset_path); rules_hash = _sha(rules_path)
    checkpoint_bytes = request.checkpoint.read_bytes()
    report_bytes = request.report.read_bytes()
    try:
        report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("training report is not valid JSON") from exc
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("checkpoint_sha256") != checkpoint_sha256):
        raise ValueError("training report does not match checkpoint")
    class_count = len(charset.symbols) + 1
    model = _load_recognizer_checkpoint(checkpoint_bytes, charset, rules_hash); torch.set_num_threads(1)
    request.output.parent.mkdir(parents=True, exist_ok=True); staging = request.output.parent / f".{request.output.name}.partial-{uuid.uuid4().hex}"; staging.mkdir()
    try:
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        onnx_path = staging / "recognizer.onnx"
        torch.onnx.export(model, (torch.zeros((2,1,64,160), dtype=torch.float32),), onnx_path, input_names=["input"], output_names=["logits"], opset_version=17, dynamo=True, external_data=False, dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=32)},))
        onnx.checker.check_model(str(onnx_path), full_check=True)
        session = session_factory(onnx_path); inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs)!=1 or len(outputs)!=1 or inputs[0].name!="input" or outputs[0].name!="logits" or inputs[0].type!="tensor(float)" or outputs[0].type!="tensor(float)" or list(inputs[0].shape)[1:]!=[1,64,160] or list(outputs[0].shape)[1:]!=[80,class_count]: raise ExportParityError("ONNX IO contract differs")
        _parity(model, session, CTCCodec.from_charset(charset))
        shutil.copyfile(charset_path, staging / "charset.txt"); shutil.copyfile(rules_path, staging / "plate_rules.json"); (staging / "report.json").write_bytes(report_bytes)
        manifest={"schema_version":1,"contract_version":"1.0","model_id":"twplate-v1-recognizer","version":"1.0.0","created_at":"2026-09-21T00:00:00Z","capabilities":["crop-recognition"],"plate_size":[380,160],"charset":{"file":"charset.txt","sha256":_sha(staging/"charset.txt"),"visible_symbols":len(charset.symbols)},"rules":{"file":"plate_rules.json","sha256":_sha(staging/"plate_rules.json")},"components":{"recognizer":{"file":"recognizer.onnx","format":"onnx","sha256":_sha(onnx_path),"inputs":[{"name":"input","dtype":"float32","shape":["batch",1,64,160]}],"outputs":[{"name":"logits","dtype":"float32","shape":["batch",80,class_count]}],"batch":{"mode":"dynamic","min":1,"opt":8,"max":32}}},"decoder":{"type":"ctc","blank_index":0,"class_count":class_count,"index_mapping":"charset-order-skipping-blank","collapse_repeats":True},"preprocess":{"source_size_wh":[380,160],"color_space":"grayscale","layout":"NCHW","input_size_hw":[64,160],"grayscale":{"reference":"pillow-image-convert-l","library_version":"12.3.0"},"resize":{"mode":"letterbox","reference":"pillow-image-resize","interpolation":"Resampling.BILINEAR","library_version":"12.3.0","resized_size_hw":[64,152],"padding_ltrb":[4,0,4,0],"padding_raw_value":255},"normalization":{"type":"divide","divisor":255.0,"dtype":"float32"}},"provenance":{"training_data":"synthetic","license_reviewed":True,"training_report":{"file":"report.json","sha256":_sha(staging/"report.json")}}}
        (staging/"manifest.json").write_text(__import__("json").dumps(manifest, indent=2)+"\n", encoding="utf-8")
        validate_crop_bundle(staging, Path(__file__).resolve().parents[3] / "schemas/model_manifest.schema.json")
        publish_directory_no_replace(staging, request.output); return request.output
    except BaseException:
        remove_owned_staging(staging, request.output); raise
