from __future__ import annotations

from copy import deepcopy

import pytest

from plateai_shared.schema_validation import DocumentValidationError, validate_document

from tests.conftest import ROOT


BACKGROUND_SCHEMA = ROOT / "schemas/detection_background_manifest.schema.json"
METADATA_SCHEMA = ROOT / "schemas/detection_metadata.schema.json"


def _valid_background_manifest():
    return {
        "schema_version": 1,
        "backgrounds": [
            {"image_path": "backgrounds/toy.png", "sha256": "a" * 64}
        ],
    }


def _valid_metadata():
    return {
        "schema_version": 1,
        "index": 0,
        "image_path": "images/000000.png",
        "image_sha256": "b" * 64,
        "background": {
            "image_path": "backgrounds/toy.png",
            "sha256": "a" * 64,
            "width": 720,
            "height": 480,
        },
        "instances": [
            {
                "bbox_xyxy": [100.0, 120.0, 289.0, 199.0],
                "corners": [
                    [100.0, 120.0],
                    [289.0, 120.0],
                    [289.0, 199.0],
                    [100.0, 199.0],
                ],
                "source_plate": {
                    "canonical": "ABC1235",
                    "display": "ABC-1235",
                    "rule_id": "new-style-private-passenger-lll-dddd",
                    "plate_type": "new-style-private-passenger",
                    "template_id": "new-style-private-passenger-white-v1",
                    "plate_seed": 7,
                    "corners": [
                        [0.0, 0.0],
                        [379.0, 0.0],
                        [379.0, 159.0],
                        [0.0, 159.0],
                    ],
                    "renderer": {
                        "font_kind": "truetype",
                        "font_name": "NotoSansMono[wdth,wght].ttf",
                        "font_variation_axes": [700, 62],
                    },
                },
                "transform": {"homography": [[0.5, 0, 100], [0, 0.5, 120], [0, 0, 1]]},
            }
        ],
    }


def test_background_manifest_requires_sha256():
    document = _valid_background_manifest()
    del document["backgrounds"][0]["sha256"]
    with pytest.raises(DocumentValidationError, match=r"backgrounds\[0\]\.sha256"):
        validate_document(document, BACKGROUND_SCHEMA)


def test_detection_metadata_requires_exactly_four_corners():
    document = _valid_metadata()
    document["instances"][0]["corners"].append([100.0, 120.0])
    with pytest.raises(DocumentValidationError, match="corners"):
        validate_document(document, METADATA_SCHEMA)


def test_detection_metadata_rejects_out_of_frame_corner():
    document = deepcopy(_valid_metadata())
    document["instances"][0]["corners"][0] = [-1.0, 120.0]
    with pytest.raises(DocumentValidationError, match="corners"):
        validate_document(document, METADATA_SCHEMA)


@pytest.mark.parametrize(
    "rooted_path",
    [r"\rooted.png", r"C:drive-relative.png", r"C:\absolute.png", r"\\server\share\bg.png"],
)
def test_background_manifest_rejects_windows_rooted_paths(rooted_path):
    document = _valid_background_manifest()
    document["backgrounds"][0]["image_path"] = rooted_path
    with pytest.raises(DocumentValidationError, match="image_path"):
        validate_document(document, BACKGROUND_SCHEMA)
