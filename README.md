# 3waPlateAI

3waPlateAI is a community-first, MIT-licensed toolkit for Taiwan-style license-plate synthesis, training contracts, and high-speed reading infrastructure.

The public repository is deliberately source-only. It contains the code, schemas, configuration, documentation, and four harmless synthetic CI fixtures, but official repositories and releases publish **no checkpoints, trained weights, ONNX models, TensorRT engines, or Model Bundles**. Data rights, legal review, compute, training, tuning, and maintenance of every resulting model remain with the user.

M1 currently delivers a deterministic CPU pipeline for recognizer crops. Training loops, pose detection, the production Reader, and serving layers remain later milestones; see the approved [design specification](docs/superpowers/specs/2026-09-21-3wa-plate-ai-design.md).

## What is here

| Area | Responsibility | M1 status |
|---|---|---|
| `plateai_shared` | Immutable rules and JSON contracts shared across training and inference | Available |
| `plateai_trainer.synthetic` | Rule sampling, clean rendering, deterministic augmentation, and transactional dataset output | Available |
| `plateai_reader` | Detection, geometric normalization, batched recognition, and constrained decoding | Planned |
| Model Bundle | Hash-verified model, charset, rule, tensor, batch, decoder, and rectifier contract | Schema available; no bundle published |

## Quick start

CPython 3.11 on Linux is the supported M1 runtime.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/py311.lock.txt
python -m pip install --no-deps -e .
python -m plateai_trainer.synthetic generate --count 100 --seed 42 --output out/demo
python -m pytest
```

The equivalent installed command is:

```bash
plateai-generate generate --count 100 --seed 42 --output out/demo
```

Generation refuses to overwrite an existing output directory. Publication is transactional: a failed run removes its own partial directory and leaves unrelated files untouched.

## M1 guarantees

- Default RGB `uint8` PNG crops at 380×160 (the new-style private-passenger aspect ratio) with source or transformed four-corner metadata.
- Canonical OCR labels without the decorative hyphen; displayed plate text retains it.
- Per-sample seeds derived from the run seed and index, with no shared random state.
- Repeatable labels, transform parameters, metadata, and images under the locked Python 3.11 dependency set.
- A bundled, unmodified OFL-1.1 Noto Sans Mono font at a recorded width/weight setting. It is a legally redistributable visual approximation, not an official Taiwan number-plate font.
- Default new-style private-passenger rules use white background, black glyphs, the 3-4 layout, and exclude `I`, `O`, and `4`; other plate families remain later versions.
- JSON Schema contracts for generated records, summaries, configuration, and future Model Bundles.

The four PNG files under `tests/fixtures/synthetic/` are algorithmically generated toy inputs used only to prove that code and contracts work. They contain no real vehicle image or identifying plate data and make no accuracy claim.

## Train your own model

Use the public generator, import only lawfully obtained datasets, and train a bundle suitable for your own domain. Real plate photographs, production data, fonts without redistribution permission, and trained artifacts do not belong in this repository. See [the training and dataset guide](docs/training.md) and [the local model policy](models/README.md).

## License

Project source is released under the [MIT License](LICENSE). This does not relicense dependencies, external fonts, datasets, or models; see [third-party notices](THIRD_PARTY_NOTICES.md).
