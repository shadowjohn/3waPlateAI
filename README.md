# 3waPlateAI

3waPlateAI is a community-first, MIT-licensed toolkit for Taiwan-style license-plate synthesis, training contracts, and high-speed reading infrastructure.

The public repository is deliberately source-only. It contains the code, schemas, configuration, documentation, and four harmless synthetic CI fixtures, but official repositories and releases publish **no checkpoints, trained weights, ONNX models, TensorRT engines, or Model Bundles**. Data rights, legal review, compute, training, tuning, and maintenance of every resulting model remain with the user.

M1 delivers deterministic CPU recognizer crops, M2 provides local PyTorch CTC training and ONNX export, M3a provides the deterministic four-corner rectifier, and M3b provides a local native pose-detector workflow. Complete Reader inference and serving remain later milestones; see the approved [design specification](docs/superpowers/specs/2026-09-21-3wa-plate-ai-design.md).

## What is here

| Area | Responsibility | Delivery status |
|---|---|---|
| `plateai_shared` | Immutable rules and JSON contracts shared across training and inference | Available |
| `plateai_trainer.synthetic` | Rule sampling, clean rendering, deterministic augmentation, and transactional dataset output | Available |
| `plateai_reader` | Four-corner geometric normalization and local detector postprocessing; batched recognition and constrained decoding follow later | M3a rectifier and M3b detector handoff available |
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

## M2 local recognition

The optional `training` extra provides a local PyTorch CTC trainer and ONNX exporter for the v1 crop contract. It uses Pillow-golden grayscale preprocessing, 80 CTC timesteps, blank index zero, and validates native-versus-ONNX parity before publishing an ignored local bundle. See `docs/training.md`; datasets, weights, ONNX files, runs, and bundles are not committed.

## M3a deterministic rectification

`plateai_reader.rectifier` accepts a `uint8` RGB source image plus four finite, in-frame corners in any order. It validates the convex hull, rejects degenerate or orientation-ambiguous geometry, uses the long plate edges to determine top/bottom and left/right semantics, then returns a canonical RGB `160×380` crop for M2.

The only warp targets discrete destination pixels `(0,0)`, `(379,0)`, `(379,159)`, and `(0,159)` with OpenCV bilinear interpolation and a white constant border. The identity regression verifies every outer output pixel matches its source coordinate, so this contract does not hide last-row or last-column border mixing. M3a contains no detector data, detector training, detector weight, or detector ONNX export; those are M3b work.

## M3b local pose detection

M3b is a local-only workflow for a user-provided, lawfully usable background manifest. The native detector accepts a 640x640 OpenCV RGB letterbox input and emits pre-NMS candidates shaped `[batch,8400,13]`. After deterministic NMS, each retained four-corner detection is independently passed to M3a's fixed RGB `380x160` crop rectifier, so one invalid pose does not prevent other retained detections from reaching the recognizer boundary.

Run composition, detector training, and full-bundle export from an installed training environment as documented in [the detector workflow](docs/training.md#local-m3b-detector-workflow). Source-tree acceptance deliberately includes no bundled background, detector weight, ONNX artifact, TensorRT benchmark, browser integration, or production recognition metric.

## Train your own model

Use the public generator, import only lawfully obtained datasets, and train a bundle suitable for your own domain. Real plate photographs, production data, fonts without redistribution permission, and trained artifacts do not belong in this repository. See [the training and dataset guide](docs/training.md) and [the local model policy](models/README.md).

## License

Project source is released under the [MIT License](LICENSE). This does not relicense dependencies, external fonts, datasets, or models; see [third-party notices](THIRD_PARTY_NOTICES.md).
