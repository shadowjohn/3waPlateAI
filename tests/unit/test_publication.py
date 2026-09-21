from __future__ import annotations

import pytest

from plateai_shared.publication import OutputExistsError, publish_directory_no_replace


def test_publication_refuses_an_existing_target_without_modifying_it(tmp_path):
    output = tmp_path / "bundle"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(OutputExistsError):
        publish_directory_no_replace(tmp_path / "staging", output)

    assert sentinel.read_text(encoding="utf-8") == "keep"
