# Third-party notices

3waPlateAI source is MIT-licensed. That license does not relicense its dependencies or anything users supply, including fonts, datasets, model architectures, trained weights, converters, or runtimes. The dependency versions below are pinned by `requirements/py311.lock.txt`; consult each upstream distribution for its complete license and bundled-component notices.

| Package | Role | Upstream | SPDX license expression |
|---|---|---|---|
| NumPy 2.4.2 | Array representation and deterministic numeric transforms | <https://numpy.org/> | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` |
| opencv-python-headless 5.0.0.93 | Rendering, homography, filtering, and image codecs | <https://github.com/opencv/opencv-python> | `Apache-2.0` |
| Pillow 12.3.0 | TrueType/OpenType rendering and fixture verification | <https://python-pillow.github.io/> | `MIT-CMU` |
| jsonschema 4.26.0 | Draft 2020-12 contract validation | <https://github.com/python-jsonschema/jsonschema> | `MIT` |
| pytest 9.1.1 | Test runner | <https://docs.pytest.org/> | `MIT` |
| Hypothesis 6.168.0 | Property-based tests | <https://hypothesis.works/> | `MPL-2.0` |
| Noto Sans Mono | Default M1 new-style private-passenger glyph rendering | <https://github.com/google/fonts/tree/main/ofl/notosansmono> | `OFL-1.1` |

OpenCV wheels and NumPy distributions can contain separately licensed third-party components. Their installed `LICENSE-3RD-PARTY.txt`, `LICENSES_bundled.txt`, package metadata, and source distribution remain authoritative for the exact wheel in use.

`assets/fonts/NotoSansMono[wdth,wght].ttf` is the unmodified Google Fonts Noto Sans Mono variable font. Its SHA-256 is `2cb2adb378a8f574213e23df697050b83c54c27df465a2015552740b2769a081`; its required copyright and OFL-1.1 text are preserved in `assets/fonts/OFL.txt`. M1 uses the font at weight 700 and width 62. It is an OFL-licensed approximation for the official new-style plate glyphs, not an official Taiwan number-plate font.

Any user-provided font remains local by default. Anyone distributing an additional font must preserve its required copyright and license notice. The same rule applies to datasets and locally trained Model Bundles.
