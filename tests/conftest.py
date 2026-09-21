from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHARSET = ROOT / "configs/charsets/tw_plate_latin_v1.txt"
DEFAULT_RULES = ROOT / "configs/plate_rules/tw_plate_v1.json"
DEFAULT_TEMPLATE = ROOT / "configs/plate_templates/standard_white_v1.json"
DEFAULT_AUGMENTATION = ROOT / "configs/augmentation/standard_v1.json"
NONE_AUGMENTATION = ROOT / "configs/augmentation/none_v1.json"
V1_CHARSET = ROOT / "configs/charsets/tw_new_style_private_passenger_v1.txt"
V1_RULES = ROOT / "configs/plate_rules/tw_new_style_private_passenger_v1.json"
V1_TEMPLATE = ROOT / "configs/plate_templates/new_style_private_passenger_white_v1.json"
V1_NONE_AUGMENTATION = ROOT / "configs/augmentation/none_v1.json"


def write_json(path: Path, value: Any) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def valid_rule_document() -> dict[str, Any]:
    return json.loads(DEFAULT_RULES.read_text(encoding="utf-8"))


@pytest.fixture
def valid_template_document() -> dict[str, Any]:
    return json.loads(DEFAULT_TEMPLATE.read_text(encoding="utf-8"))


@pytest.fixture
def v1_charset():
    from plateai_shared.rules import load_character_set

    return load_character_set(V1_CHARSET)


@pytest.fixture
def default_ruleset():
    from plateai_shared.rules import load_character_set, load_ruleset

    charset = load_character_set(DEFAULT_CHARSET)
    return load_ruleset(DEFAULT_RULES, charset)


@pytest.fixture
def default_template():
    from plateai_trainer.synthetic.templates import load_template

    return load_template(DEFAULT_TEMPLATE)


@pytest.fixture
def none_profile():
    from plateai_trainer.synthetic.augment import load_augment_profile

    return load_augment_profile(NONE_AUGMENTATION)


@pytest.fixture
def standard_profile():
    from plateai_trainer.synthetic.augment import load_augment_profile

    return load_augment_profile(DEFAULT_AUGMENTATION)


@pytest.fixture
def rendered_plate(default_template):
    from plateai_shared.contracts import GeneratedPlate
    from plateai_trainer.synthetic.fonts import resolve_font
    from plateai_trainer.synthetic.renderer import render_plate

    sample = GeneratedPlate("ABC1234", "ABC-1234", "standard-lll-dddd", "standard")
    return render_plate(sample, default_template, resolve_font(None))


@pytest.fixture
def extreme_profile(standard_profile):
    from dataclasses import replace

    return replace(
        standard_profile,
        perspective_probability=1.0,
        max_corner_jitter_ratio=0.2,
        rotation_probability=1.0,
        rotation_degrees=(-15.0, 15.0),
        glare_probability=1.0,
        glare_opacity_range=(0.9, 1.0),
        glare_radius_ratio_range=(0.5, 1.0),
    )


@pytest.fixture
def default_request(tmp_path):
    from plateai_trainer.synthetic.models import GenerationRequest

    return GenerationRequest(
        count=2,
        seed=42,
        output=tmp_path / "dataset",
        charset_path=DEFAULT_CHARSET,
        rules_path=DEFAULT_RULES,
        template_path=DEFAULT_TEMPLATE,
        augmentation_path=DEFAULT_AUGMENTATION,
    )
