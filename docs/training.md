# Synthetic dataset and training guide

M1 builds deterministic recognizer-crop datasets on CPU. It does not train or export a model yet; later training code will consume the same labels, charset, rule, and Model Bundle contracts.

## Generate a dataset

From an installed development environment:

```bash
python -m plateai_trainer.synthetic generate \
  --count 10000 \
  --seed 42 \
  --output out/train-seed-42
```

The output path must not exist. Missing parent directories are created, the run is written to a generator-owned sibling staging directory, and the completed directory is published only after every image and manifest succeeds.

## Output contract

```text
out/demo/
├── images/000000.png
├── labels.txt
├── metadata.jsonl
├── generation_config.json
└── summary.json
```

| Path | Purpose |
|---|---|
| `images/*.png` | Default RGB `uint8`, 380×160 new-style private-passenger recognizer crops |
| `labels.txt` | PaddleOCR-compatible `relative/path.png<TAB>CANONICAL_TEXT` records |
| `metadata.jsonl` | One JSON object per image: display text, canonical text, corners, transform parameters, provenance, and PNG hash |
| `generation_config.json` | Seed, count, font identifier, repository-relative input paths, and input hashes |
| `summary.json` | Generated count, rule/type distribution, seed, and configuration hashes |

`CANONICAL_TEXT` contains only recognizer symbols. For example, the renderer may draw `ABC-1234`, while the label is `ABC1234`. The decorative separator is not part of `charset.txt`.

Published schemas live under `schemas/`. `generation_metadata.schema.json` fixes the four-corner order as `left_top`, `right_top`, `right_bottom`, `left_bottom`; the points produced by M1 come directly from its known homography.

## Immutable run inputs

A run consumes four versioned inputs:

- `configs/charsets/tw_new_style_private_passenger_v1.txt` — ordered visible symbols for the default new-style private-passenger profile; never a CTC blank token.
- `configs/plate_rules/tw_new_style_private_passenger_v1.json` — the default 3-4 rule, excluding `I`, `O`, and `4`.
- `configs/plate_templates/new_style_private_passenger_white_v1.json` — 380×160 white background, black glyphs, border, and text box.
- `configs/augmentation/*.json` — bounded geometry and photometric probabilities.

Override them with `--charset`, `--rules`, `--template`, and `--augmentation`. Treat configuration files as immutable inputs to a run. Their SHA-256 values are recorded in the output; edit by creating a new versioned file rather than mutating the history of an established dataset.

The same run seed, sample index, input files, and `requirements/py311.lock.txt` reproduce the same strings, sampled transform parameters, metadata, and encoded images. Reproducibility outside that dependency lock is not promised.

## Fonts

Without `--font`, the renderer uses the bundled, unmodified `NotoSansMono[wdth,wght].ttf` at weight 700 and width 62. The font is covered by SIL OFL-1.1; its SHA-256 and complete license text are retained in `THIRD_PARTY_NOTICES.md` and `assets/fonts/OFL.txt`. It is a legal visual approximation for the default new-style private-passenger profile, not an official Taiwan number-plate font.

Supply a local TrueType or OpenType file when needed:

```bash
python -m plateai_trainer.synthetic generate \
  --count 10000 \
  --seed 42 \
  --font /path/to/font.ttf \
  --output out/train-custom-font
```

The path must name a readable `.ttf` or `.otf` file. Before redistributing a font, verify its license and add its copyright and license text to the distribution. A local font is identified in metadata by filename and SHA-256; its bytes are not copied into the generated dataset.

## From generated data to a local bundle

Users may combine generated data with lawfully obtained real data, implement the later training milestone, and package the result locally according to `schemas/model_manifest.schema.json`. CTC manifests must declare the integer `blank_index`, `class_count = visible symbols + 1`, and `charset-order-skipping-blank`. Recognizers must also declare either dynamic batch limits or a fixed batch with neutral padding and discarded padded outputs.

No generated dataset or resulting model is automatically licensed by this project's MIT license. Dataset collection, privacy, jurisdiction-specific vehicle/plate rules, font rights, compute, evaluation, and model maintenance remain the user's responsibility.
