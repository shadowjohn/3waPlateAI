from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from plateai_trainer.synthetic.dataset import generate_dataset
from plateai_trainer.synthetic.models import GenerationRequest
from plateai_trainer.synthetic.templates import load_template_selector

from tests.conftest import NONE_AUGMENTATION, V1_CHARSET


ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "configs/plate_rules/tw_motorcycle_colours_v1.json"
TEMPLATES = ROOT / "configs/plate_templates/tw_motorcycle_colours_v1.json"

EXPECTED_STYLES = {
    "ordinary-heavy-motorcycle": ((260, 140), (245, 245, 242)),
    "large-heavy-motorcycle-yellow": ((300, 150), (250, 218, 36)),
    "large-heavy-motorcycle-red-over-550cc": ((300, 150), (180, 32, 40)),
    "light-motorcycle-50cc-green": ((260, 140), (22, 122, 72)),
}


def test_motorcycle_colour_templates_are_selected_by_plate_type():
    selector = load_template_selector(TEMPLATES)
    for plate_type, (size, background) in EXPECTED_STYLES.items():
        template = selector.resolve(plate_type)
        assert (template.width, template.height) == size
        assert template.background_rgb == background
        assert template.id == f"tw-motorcycle-colours-v1/{plate_type}"

    with pytest.raises(ValueError, match="has no template"):
        selector.resolve("unconfigured-motorcycle")


def test_generator_renders_each_motorcycle_plate_type_with_its_configured_colour(
    tmp_path,
):
    output = tmp_path / "motorcycle-colours"
    generate_dataset(
        GenerationRequest(
            count=12,
            seed=42,
            output=output,
            charset_path=V1_CHARSET,
            rules_path=RULES,
            template_path=TEMPLATES,
            augmentation_path=NONE_AUGMENTATION,
        )
    )

    records = [
        json.loads(line)
        for line in (output / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {record["plate_type"] for record in records} == set(EXPECTED_STYLES)
    for record in records:
        expected_size, expected_background = EXPECTED_STYLES[record["plate_type"]]
        with Image.open(output / record["image_path"]) as image:
            assert image.size == expected_size
            assert image.convert("RGB").getpixel((6, image.height // 2)) == expected_background
        assert record["renderer"]["template_id"] == (
            f"tw-motorcycle-colours-v1/{record['plate_type']}"
        )
