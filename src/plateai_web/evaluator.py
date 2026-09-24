"""Fixed-input local OCR benchmarks with explicit timing and model identity."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from plateai_bench import BenchmarkRunner
from plateai_bench.metrics import calculate_polygon_iou, levenshtein_distance
from plateai_reader.fpga_lpr import FpgaLprRecognizer, FpgaLprError
from plateai_reader.fpga_pipeline import FpgaSceneReader, crop_box
from plateai_reader.rectifier import rectify_plate
from plateai_reader.runtime import PlateReader, _default_schema_path
from plateai_shared.bundle import validate_model_bundle
from .paths import workspace_root


ROOT = workspace_root()
MODELS = ("active-v1", "native-preview", "fpga-lpr-mit", "compare")
DATASETS = ("ezcon", "user")


def load_samples(root: Path, dataset: str, limit: int):
    if dataset == "ezcon":
        directory = root / "datasets/restricted/ezcon-taiwan-recognition-test"
        manifest = directory / "reader_v1_eligible_test.jsonl"
    elif dataset == "user":
        directory = root / "datasets/real_benchmarks"
        manifest = directory / "user_cases.jsonl"
    else:
        raise ValueError("未知評測集")
    if not manifest.is_file():
        raise ValueError(f"缺少評測清單：{manifest.relative_to(root)}")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    records.sort(key=lambda r: (r["image"]["path"], r["ground_truth"]["canonical"]))
    samples, identities = [], []
    for record in records[:limit]:
        path = (directory / record["image"]["path"]).resolve()
        path.relative_to(directory.resolve())
        data = path.read_bytes()
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"影像無法讀取：{path.name}")
        gt = record["ground_truth"]
        if "xywhr" in gt:
            x, y, w, h, angle = gt["xywhr"]
            corners = cv2.boxPoints(((x, y), (w, h), math.degrees(angle)))
        else:
            corners = np.asarray(gt.get("corners", gt.get("polygon")), dtype=np.float32)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError(f"無效 GT 四角：{path.name}")
        expected = gt["canonical"].strip().upper()
        if not expected:
            raise ValueError(f"GT 文字為空：{path.name}")
        samples.append((cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), corners, expected))
        identities.append({"path": record["image"]["path"], "sha256": hashlib.sha256(data).hexdigest(),
                           "canonical": expected, "corners": corners.tolist()})
    if not samples:
        raise ValueError("評測集沒有樣本，尚未執行評測")
    fingerprint = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    return samples, fingerprint


class ModelAdapter:
    def __init__(self, root: Path, kind: str):
        self.external = kind == "fpga-lpr-mit"
        bundle = root / "models/bundles" / ("active-v1" if kind == "active-v1" else "candidate-detector-real-v1")
        self.ocr = self.scene = self.native = None
        if self.external:
            self.ocr = FpgaLprRecognizer(root / "third_party/fpga_lpr")
            self.scene = FpgaSceneReader(PlateReader(bundle), self.ocr)
            self.model_id = self.ocr.manifest.model_id
            identity_files = [root / "third_party/fpga_lpr/manifest.json", bundle / "manifest.json"]
        else:
            validate_model_bundle(bundle, _default_schema_path())
            self.native = BenchmarkRunner(bundle)
            self.scene = self.native.reader
            self.model_id = self.native.manifest.get("model_id", kind)
            identity_files = [bundle / "manifest.json"]
        self.identity = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in identity_files}

    def predict(self, image, corners, mode):
        if mode == "crop":
            if self.external:
                box = [*corners.min(axis=0), *corners.max(axis=0)]
                try:
                    return self.ocr.recognize(crop_box(image, box, 0.06)).normalized_text, True
                except FpgaLprError:
                    return "", False
            try:
                crop = rectify_plate(image, corners).image_rgb
            except ValueError:
                return "", False
            return self.native._eval_single_crop(crop)[0], True
        if self.scene is None:
            raise ValueError("此模型缺少 Detector，請選擇 GT Crop 模式")
        result = self.scene.read(image)
        candidates = [(calculate_polygon_iou(p.detection.corners_xy, corners),
                       p.read.normalized_text if self.external else p.decoded.canonical) for p in result.plates]
        best = max(candidates, key=lambda item: item[0]) if candidates else (0, "")
        return (best[1], True) if best[0] >= 0.5 else ("", False)


def measure(adapter, samples, mode, rounds, warmup, progress=None):
    # Warmup is per model/mode, excluded from timing and accuracy.
    for i in range(warmup):
        image, corners, _ = samples[i % len(samples)]
        adapter.predict(image, corners, mode)
    latencies, exact, errors, total_chars, located = [], 0, 0, 0, 0
    total = len(samples) * rounds
    for repeat in range(rounds):
        for index, (image, corners, expected) in enumerate(samples):
            started = time.perf_counter()
            predicted, matched = adapter.predict(image, corners, mode)
            latencies.append((time.perf_counter() - started) * 1000)
            exact += int(matched and predicted == expected)
            located += int(matched)
            errors += levenshtein_distance(predicted, expected)
            total_chars += len(expected)
            if progress:
                progress((repeat * len(samples) + index + 1) / total)
    return {"success": f"{exact}/{total}", "accuracy": f"{100*exact/total:.1f}% (CER {100*errors/total_chars:.1f}%)",
            "sample_count": len(samples), "trials": total, "rounds": rounds, "warmup": warmup,
            "avg_ms": round(float(np.mean(latencies)), 2), "p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2), "max_ms": round(max(latencies), 2),
            "qps": round(1000 / max(float(np.mean(latencies)), 1e-9), 2), "matched_trials": located}


def run_benchmark_task(task_id, tm, target_dataset="ezcon", rounds=1, warmup=2,
                       model_kind="active-v1", sample_limit=20, mode="both"):
    try:
        if model_kind not in MODELS or target_dataset not in DATASETS or mode not in {"both", "crop", "scene"}:
            raise ValueError("無效的模型、資料集或評測模式")
        if not (1 <= rounds <= 20 and 0 <= warmup <= 20 and 1 <= sample_limit <= 100):
            raise ValueError("評測參數超出範圍")
        tm.update_progress(task_id, 1, "核對固定評測集與模型...")
        samples, fingerprint = load_samples(ROOT, target_dataset, sample_limit)
        kinds = ["active-v1", "fpga-lpr-mit"] if model_kind == "compare" else [model_kind]
        modes = ["crop", "scene"] if mode == "both" else [mode]
        tm.append_log(task_id, f"資料集 {target_dataset}：{len(samples)} 個 GT；SHA256 {fingerprint}")
        tm.append_log(task_id, f"每模型／模式 {rounds} 回合；暖機 {warmup} 次。時間僅含伺服器辨識，不含載入、磁碟及 HTTP。")
        tm.append_log(task_id, "GT Crop：原生使用 GT 透視校正；MIT 使用 GT 外接框 + 6% margin，再由 CPM 校正。")
        results = []
        for kind in kinds:
            adapter = ModelAdapter(ROOT, kind)
            for selected_mode in modes:
                slot = len(results)
                tm.append_log(task_id, f"開始 {kind} / {selected_mode}")
                result = measure(adapter, samples, selected_mode, rounds, warmup,
                                 lambda fraction: tm.update_progress(task_id, 5 + int(90*(slot+fraction)/(len(kinds)*len(modes)))))
                result.update(case=f"{kind} / {'GT Crop' if selected_mode == 'crop' else 'End-to-End'}",
                              api="本機推論（非 HTTP）", query=target_dataset, model_id=adapter.model_id,
                              model_manifests=adapter.identity, dataset_sha256=fingerprint)
                results.append(result)
                tm.append_log(task_id, f"{result['case']}：{result['success']}，avg {result['avg_ms']} ms")
        columns = ["case", "query", "success", "accuracy", "avg_ms", "p50_ms", "p95_ms", "max_ms", "qps"]
        markdown = f"資料集 SHA256: {fingerprint}\n樣本: {len(samples)}；rounds={rounds}；warmup={warmup}\n獨立準確率：尚未驗收\n\n"
        markdown += "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n"
        markdown += "\n".join("| " + " | ".join(str(r[c]) for c in columns) + " |" for r in results)
        tm.complete_task(task_id, result={"status": "success", "experimental_diagnostic": True,
                         "m4_5_status": None, "independent_accuracy": "pending", "results": results,
                         "dataset_sha256": fingerprint, "markdown": markdown}, message="所選模型與模式評測完成")
    except Exception as exc:
        tm.fail_task(task_id, f"評測失敗：{exc}")
