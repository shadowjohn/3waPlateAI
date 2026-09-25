"""Local, immutable benchmark thumbnails and paged first-round detail rows."""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from plateai_shared.publication import publish_directory_no_replace


GROUP_RE = r"(?:active-v1|native-preview|fpga-lpr-mit)-(?:crop|scene)"


def _base(root: Path) -> Path:
    base = root / "runs" / "web-benchmarks"
    if base.resolve() != base.absolute():
        raise ValueError("Benchmark 報告目錄不可重新導向")
    return base


def report_directory(root: Path, report_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", report_id):
        raise FileNotFoundError("未知報告")
    path = _base(root) / report_id
    if path.resolve() != path.absolute() or not (path / "summary.json").is_file():
        raise FileNotFoundError("報告不存在或尚未完成")
    return path


def read_rows(root: Path, report_id: str, group: str, offset: int, limit: int, errors_only: bool):
    directory = report_directory(root, report_id)
    if not re.fullmatch(GROUP_RE, group):
        raise FileNotFoundError("未知評測組")
    path = directory / f"{group}.jsonl"
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("此報告沒有該評測組")
    rows, total = [], 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if errors_only and row["correct"]:
                continue
            if offset <= total < offset + limit:
                rows.append(row)
            total += 1
    return {"rows": rows, "total": total, "offset": offset, "limit": limit}


def image_path(root: Path, report_id: str, filename: str) -> Path:
    directory = report_directory(root, report_id)
    if not re.fullmatch(GROUP_RE + r"-\d{5}-(?:scene|crop)\.jpg", filename):
        raise FileNotFoundError("未知縮圖")
    path = directory / filename
    if path.resolve() != path.absolute() or not path.is_file():
        raise FileNotFoundError("縮圖不存在")
    return path


def _box(corners):
    return [*np.min(corners, axis=0).tolist(), *np.max(corners, axis=0).tolist()]


class ReportWriter:
    def __init__(self, root: Path):
        self.report_id = uuid.uuid4().hex
        base = _base(root)
        base.mkdir(parents=True, exist_ok=True)
        self.output = base / self.report_id
        self.staging = base / f".{self.report_id}.partial-{uuid.uuid4().hex}"
        self.staging.mkdir()

    def _save_image(self, filename, rgb, boxes=()):
        height, width = rgb.shape[:2]
        scale = min(1.0, 640 / max(height, width))
        image = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           (max(1, round(width * scale)), max(1, round(height * scale))))
        for box, color, label in boxes:
            if len(box) != 4 or not np.isfinite(box).all():
                continue
            points = np.rint(np.asarray(box) * scale).astype(int)
            x1, y1, x2, y2 = points.tolist()
            x1, x2 = np.clip([x1, x2], 0, image.shape[1] - 1).tolist()
            y1, y2 = np.clip([y1, y2], 0, image.shape[0] - 1).tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("無法產生報告縮圖")
        (self.staging / filename).write_bytes(encoded.tobytes())
        return f"/api/benchmark/reports/{self.report_id}/images/{filename}"

    def add(self, group, mode, index, sample, image, corners, expected, prediction, latency_ms):
        gt_box = _box(corners)
        prefix = f"{group}-{index:05}"
        boxes = [(gt_box, (60, 190, 40), "GT")]
        boxes += [(box, (200, 40, 200), f"P{i + 1}") for i, box in enumerate(prediction.predicted_boxes)]
        scene_url = self._save_image(prefix + "-scene.jpg", image, boxes)
        crop_url = self._save_image(prefix + "-crop.jpg", prediction.crop) if prediction.crop is not None else None
        row = {"index": index, "filename": getattr(sample, "name", f"sample-{index}"),
               "expected": expected, "predicted": prediction.text,
               "text_match": prediction.text == expected,
               "correct": prediction.matched and prediction.text == expected,
               "matched": prediction.matched, "reason": prediction.reason,
               "ocr_score": prediction.score, "score_kind": "uncalibrated" if prediction.score is not None else "not_provided",
               "latency_ms": round(latency_ms, 2), "round": 1,
               "image_url": scene_url, "crop_url": crop_url,
               "gt_box": gt_box, "predicted_boxes": prediction.predicted_boxes,
               "bbox_source": "gt_crop" if mode == "crop" else "detector", "iou": prediction.iou,
               "scene_outputs": prediction.scene_outputs}
        with (self.staging / f"{group}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def finish(self, summary):
        (self.staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        publish_directory_no_replace(self.staging, self.output)
        return self.report_id
