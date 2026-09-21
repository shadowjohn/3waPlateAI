from __future__ import annotations

import json
from dataclasses import replace

import pytest

import plateai_trainer.synthetic.dataset as dataset_module
from plateai_shared.schema_validation import validate_document
from plateai_trainer.synthetic.dataset import (
    InvalidGenerationRequest,
    OutputExistsError,
    generate_dataset,
)

from tests.conftest import ROOT


def test_generate_dataset_writes_images_labels_metadata_and_summary(
    tmp_path, default_request
):
    request = replace(default_request, count=3, output=tmp_path / "dataset")
    summary = generate_dataset(request)
    assert summary.generated == 3
    assert sorted(p.name for p in (request.output / "images").glob("*.png")) == [
        "000000.png",
        "000001.png",
        "000002.png",
    ]
    labels = (request.output / "labels.txt").read_text(encoding="utf-8").splitlines()
    assert len(labels) == 3
    assert all(line.startswith("images/") and "\t" in line for line in labels)
    metadata = (request.output / "metadata.jsonl").read_text(encoding="utf-8")
    assert len(metadata.splitlines()) == 3
    assert (request.output / "generation_config.json").is_file()
    assert (request.output / "summary.json").is_file()


def test_existing_output_is_never_modified(tmp_path, default_request):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(OutputExistsError):
        generate_dataset(replace(default_request, output=output))
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_unicode_output_path_works(tmp_path, default_request):
    output = tmp_path / "台灣車牌資料"
    generate_dataset(replace(default_request, count=1, output=output))
    assert (output / "images/000000.png").is_file()


def test_missing_nested_output_parent_is_created(tmp_path, default_request):
    output = tmp_path / "new" / "nested" / "dataset"
    generate_dataset(replace(default_request, count=1, output=output))
    assert (output / "summary.json").is_file()


def test_encoder_failure_leaves_no_final_or_partial_directory(
    tmp_path, default_request
):
    output = tmp_path / "broken"

    def fail_encoder(_image):
        raise RuntimeError("encode failed")

    with pytest.raises(RuntimeError, match="encode failed"):
        generate_dataset(
            replace(default_request, output=output), encoder=fail_encoder
        )
    assert not output.exists()
    assert list(tmp_path.glob(".broken.partial-*")) == []


def test_publish_race_never_replaces_newly_created_output(
    tmp_path, default_request, monkeypatch
):
    output = tmp_path / "raced"
    real_publish = dataset_module._publish_no_replace

    def create_target_then_publish(staging, target):
        target.mkdir()
        real_publish(staging, target)

    monkeypatch.setattr(
        dataset_module, "_publish_no_replace", create_target_then_publish
    )
    with pytest.raises(OutputExistsError):
        generate_dataset(replace(default_request, count=1, output=output))
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert list(tmp_path.glob(".raced.partial-*")) == []


def test_count_below_one_is_rejected_without_creating_output(default_request):
    with pytest.raises(InvalidGenerationRequest, match="count"):
        generate_dataset(replace(default_request, count=0))
    assert not default_request.output.exists()


def test_output_parent_file_is_not_modified(tmp_path, default_request):
    parent = tmp_path / "not-a-directory"
    parent.write_text("keep", encoding="utf-8")
    with pytest.raises(InvalidGenerationRequest, match="parent"):
        generate_dataset(replace(default_request, output=parent / "dataset"))
    assert parent.read_text(encoding="utf-8") == "keep"


def test_same_request_writes_identical_dataset_files(tmp_path, default_request):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(replace(default_request, output=first))
    generate_dataset(replace(default_request, output=second))
    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


def test_generated_records_and_summary_match_published_schemas(default_request):
    generate_dataset(default_request)
    metadata_schema = ROOT / "schemas/generation_metadata.schema.json"
    summary_schema = ROOT / "schemas/generation_summary.schema.json"
    metadata_lines = (default_request.output / "metadata.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    for line in metadata_lines:
        validate_document(json.loads(line), metadata_schema)
    summary = json.loads(
        (default_request.output / "summary.json").read_text(encoding="utf-8")
    )
    validate_document(summary, summary_schema)
