import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from plateai_eval.fpga_protocol import (
    EvaluationInputError,
    evaluate_entries,
    load_audited_manifest,
)
from tools.evaluate_fpga_lpr import tlpd_replay_entries


def _image(tmp_path: Path, name: str) -> tuple[Path, str]:
    image = tmp_path / name
    assert cv2.imwrite(str(image), np.full((20, 40, 3), 255, np.uint8))
    return image, hashlib.sha256(image.read_bytes()).hexdigest()


def _row(image: Path, sha: str, text: str, split: str = "dev", **extra: object) -> dict:
    return {
        "image": str(image), "sha256": sha, "canonical": text,
        "vehicle_class": "v1_passenger", "split": split,
        "crop_xyxy": [0, 0, 40, 20], "source_kind": "audited_unseen", **extra,
    }


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "audit.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("overlap", ["canonical", "sha256"])
def test_cross_split_plate_or_hash_is_rejected(tmp_path: Path, overlap: str) -> None:
    first, first_hash = _image(tmp_path, "first.png")
    second, second_hash = _image(tmp_path, "second.png")
    second_row = _row(second, second_hash, "BBB2222", "holdout")
    second_row[overlap] = _row(first, first_hash, "AAA1111")[overlap]
    with pytest.raises(EvaluationInputError, match="cross_split_overlap"):
        load_audited_manifest(_manifest(tmp_path, [_row(first, first_hash, "AAA1111"), second_row]))


def test_missing_class_and_unreadable_crop_fail_closed(tmp_path: Path) -> None:
    image, sha = _image(tmp_path, "plate.png")
    row = _row(image, sha, "AAA1111")
    del row["vehicle_class"]
    with pytest.raises(EvaluationInputError, match="missing_vehicle_class"):
        load_audited_manifest(_manifest(tmp_path, [row]))
    row = _row(image, sha, "AAA1111", crop_xyxy=[100, 100, 120, 120])
    with pytest.raises(EvaluationInputError, match="invalid_crop"):
        evaluate_entries(_FakeReader(["AAA1111"]), load_audited_manifest(_manifest(tmp_path, [row])), mode="crop")


class _FakeReader:
    providers = ("CPUExecutionProvider",)

    def __init__(self, texts: list[str]) -> None:
        self._texts = iter(texts)

    def recognize(self, image, *, corner_policy="compat"):
        return SimpleNamespace(normalized_text=next(self._texts), timings_ms={"total": 1.25})


def test_exact_empty_and_v1_counts_are_separate(tmp_path: Path) -> None:
    first, sha_first = _image(tmp_path, "one.png")
    second, sha_second = _image(tmp_path, "two.png")
    rows = [
        _row(first, sha_first, "ABC1234"),
        _row(second, sha_second, "MDX9717", vehicle_class="motorcycle"),
    ]
    report = evaluate_entries(
        _FakeReader(["ABC1234", ""]), load_audited_manifest(_manifest(tmp_path, rows)), mode="crop"
    )
    assert (report.overall.total, report.overall.exact, report.overall.empty) == (2, 1, 1)
    assert (report.v1_passenger.total, report.v1_passenger.exact, report.v1_passenger.empty) == (1, 1, 0)
    assert report.independent_accuracy == "pending"  # not an independently frozen holdout


def test_training_source_replay_keeps_its_label(tmp_path: Path) -> None:
    image, sha = _image(tmp_path, "tlpd.png")
    row = _row(image, sha, "ABC1234", source_kind="training_source_replay")
    report = evaluate_entries(_FakeReader(["ABC1234"]), load_audited_manifest(_manifest(tmp_path, [row])), mode="crop")
    assert report.results[0].source_kind == "training_source_replay"
    assert report.independent_accuracy == "pending"


def test_image_hash_is_checked_before_decoding(tmp_path: Path) -> None:
    image, sha = _image(tmp_path, "plate.png")
    entries = load_audited_manifest(_manifest(tmp_path, [_row(image, sha, "ABC1234")]))
    image.write_bytes(b"altered")
    with pytest.raises(EvaluationInputError, match="image_hash_mismatch"):
        evaluate_entries(_FakeReader(["ABC1234"]), entries, mode="crop")


def test_tlpd_filename_is_only_training_source_replay(tmp_path: Path) -> None:
    image = tmp_path / "MDX9717(2)(0)(1).jpg"
    image.write_bytes(b"placeholder")
    rows = tlpd_replay_entries(tmp_path)
    assert len(rows) == 1
    assert rows[0].canonical == "MDX9717"
    assert rows[0].source_kind == "training_source_replay"
    assert rows[0].split == "dev"


def test_scene_exact_requires_spatially_matching_detection(tmp_path: Path) -> None:
    image, sha = _image(tmp_path, "scene.png")
    row = _row(image, sha, "ABC1234", crop_xyxy=[0, 0, 10, 10])
    entries = load_audited_manifest(_manifest(tmp_path, [row]))

    class FakeSceneReader:
        providers = ("CPUExecutionProvider",)

        def read(self, _image):
            return SimpleNamespace(
                plates=(SimpleNamespace(
                    detection=SimpleNamespace(bbox_xyxy=np.array([20, 0, 30, 10], np.float32)),
                    read=SimpleNamespace(normalized_text="ABC1234"),
                ),), timings_ms={"total": 2.0},
            )

    report = evaluate_entries(FakeSceneReader(), entries, mode="scene")
    assert report.overall.exact == 0
    assert report.results[0].predicted == ()
    assert report.results[0].all_scene_predictions == ("ABC1234",)
    assert report.results[0].error == "locator:no_target_match"


def test_scene_exact_uses_matching_box_in_multiplate_image(tmp_path: Path) -> None:
    image, sha = _image(tmp_path, "scene.png")
    entries = load_audited_manifest(_manifest(tmp_path, [
        _row(image, sha, "ABC1234", crop_xyxy=[0, 0, 10, 10]),
    ]))

    class FakeSceneReader:
        def read(self, _image):
            def plate(box, text):
                return SimpleNamespace(
                    detection=SimpleNamespace(bbox_xyxy=np.array(box, np.float32)),
                    read=SimpleNamespace(normalized_text=text),
                )

            return SimpleNamespace(plates=(
                plate([20, 0, 30, 10], "WRONG"),
                plate([0, 0, 10, 10], "ABC1234"),
            ), timings_ms={"total": 2.0})

    report = evaluate_entries(FakeSceneReader(), entries, mode="scene")
    assert report.overall.exact == 1
    assert report.results[0].predicted == ("ABC1234",)
    assert report.results[0].matched_iou == 1.0
