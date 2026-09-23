# Synthetic dataset and training guide

M1 builds deterministic recognizer-crop datasets on CPU. M2 adds optional local PyTorch CTC training and ONNX export.

## Local v1 training and export

Install the isolated training stack, generate train/validation sets with different generator seeds, then use paths that do not already exist:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements\py311.training.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e ".[test,training]"
.\.venv\Scripts\plateai-train --train out\train-seed-42 --validation out\validation-seed-43 --output runs\v1-cpu --epochs 10 --batch-size 32 --seed 42
.\.venv\Scripts\plateai-export --checkpoint runs\v1-cpu\best.pt --report runs\v1-cpu\report.json --output models\bundles\v1-local
```

### GPU acceleration and PyTorch CUDA builds

- **NVIDIA GTX 1080 (Pascal)**: Requires PyTorch built with **`cu118`** (CUDA 11.8). Newer CUDA versions may omit sm_61 architecture support.
- **NVIDIA RTX 5060, RTX 5090 (Blackwell)**: Requires PyTorch built with **`cu128`** (CUDA 12.8+) to target the latest architecture.
- **Automatic device detection**: `plateai-train` defaults to `--device auto`. When CUDA is available, training executes on the detected GPU (e.g. `cuda:0`), displaying the GPU device name and epoch losses. If no GPU is available or CUDA initialization fails, it seamlessly falls back to `cpu`.
- **CPU default**: The locked dependency file (`requirements/py311.training.lock.txt`) installs CPU `torch==2.14.0` for deterministic test verification and CI runs.

### Character sets and logit classes

- **Legacy v1 baseline** (`configs/charsets/tw_new_style_private_passenger_v1.txt`): 33 visible symbols (`0-9` without `4`, `A-Z` without `I` or `O`), `blank_index = 0`, and 34 logits classes (`[batch, 80, 34]`).
- **Unified Taiwan standard profile** (`configs/charsets/tw_standard_v1.txt` + `configs/plate_rules/tw_standard_v1.json`): 34 visible symbols (`0-9` including `4`, `A-Z` without `I` or `O`), `blank_index = 0`, and 35 logits classes (`[batch, 80, 35]`). This profile covers new-style 7-digit, legacy 6-digit (`LL-DDDD` / `DDDD-LL`), motorcycle 6-digit (`LLL-DDD` / `LLD-DDD` / `DLL-DDD`), and legacy 5/4-digit formats.

Both `plateai-train` and `plateai-export` support `--charset <path>` and `--rules <path>` flags to train and package custom or unified models:

```powershell
.\.venv\Scripts\plateai-train --train out\std-train --validation out\std-val --output runs\std-cpu --charset configs\charsets\tw_standard_v1.txt --rules configs\plate_rules\tw_standard_v1.json --epochs 10
.\.venv\Scripts\plateai-export --checkpoint runs\std-cpu\best.pt --report runs\std-cpu\report.json --output models\bundles\std-local --charset configs\charsets\tw_standard_v1.txt --rules configs\plate_rules\tw_standard_v1.json
```

The exact input is `[batch, 1, 64, 160]`: Pillow 12.3.0 `RGB.convert("L")`, bilinear resize to 64x152, raw-white four-pixel side letterbox, then `float32 / 255.0`.

Export runs ONNX checker plus batch-one and batch-two native-versus-ONNX parity (logits and greedy CTC text) before it atomically publishes a hash-verified local bundle. Datasets, checkpoints, ONNX files, runs, and bundles are local, ignored, and not committed. Synthetic metrics are not field accuracy.

## Web Studio background training

Web Studio is intended for occasional local retraining, not a queue or scheduler. Start it with `run_server.bat`; use `run_server.bat --dev-reload` only while developing the Web UI. Closing the Web server does not stop an already accepted training task. Use the training page's stop button and wait for the task to become `cancelled` before assuming that work has stopped.

The task record is an autocommit SQLite database at `runs\.web-training\tasks.sqlite3`; detached worker output is appended to `runs\.web-training\logs\<task-id>.log`. The browser can reconnect after a Web restart by querying that task ID. A computer reboot does not resume training: a task whose worker is conclusively gone is reported as failed on the next Web reconciliation.

The worker reports preparation, training batches, validation batches, epochs, and export. A stop request only sets `cancel_requested`; the worker acknowledges it at a safe boundary and then records `cancelled`. A completed run stays at `runs\<run-name>`, and the resulting bundle is a new `models\bundles\train-<task-id>` directory. Neither path automatically changes `models\bundles\active-v1`; inspect the report and bundle first, then make any model-activation decision manually under your deployment procedure.

## Local M3b detector workflow

Use only a legal, user-provided background manifest whose images you are authorized to process. From an installed training environment, use separate local train and validation manifests/background sets, choose output paths that do not already exist, and run this end-to-end local-only sequence:

```powershell
plateai-compose --background-manifest <local-manifest.json> --count 1000 --seed 42 --output out/detection-train
plateai-detect-train --train out/detection-train --validation out/detection-validation --output runs/detection-v1
plateai-detect-export --recognizer-bundle models/bundles/v1-local --checkpoint runs/detection-v1/best.pt --report runs/detection-v1/report.json --output models/bundles/v1-full-local
```

`plateai-compose` creates one-to-three-plate composite records with source provenance; `plateai-detect-train` trains the native local detector; and `plateai-detect-export` validates and publishes a local full bundle. The detector input is a 640x640 OpenCV RGB letterbox and its pre-NMS output is `[batch,8400,13]`. Reader postprocessing handles each retained four-corner detection independently and sends it to M3a's RGB `380x160` crop rectifier.

Composite instances cycle deterministically through projected shortest-edge sizes `[16,32)`, `[32,64)`, and `[64,infinity)` in source-image pixels, including perspective distortion. Every successfully published run with at least three total instances covers all three strata. A one- or two-instance run uses the first one or two strata and explicitly records `complete: false` plus all three counts in `summary.json`. `generation_config.json` records the `projected-source-edge-round-robin-v1` sampling policy. The default scale range is `[0.1,0.6]`; a custom scale range or canvas that cannot accommodate a scheduled stratum fails without publishing a partial dataset. Seed, configuration and input bytes determine the images and GT; the revised default changes images generated by earlier revisions with the same seed.

Validation retains aggregate metrics and reports `stratum_results.projected_shortest_edge_640px` for `lt16`, `16to31`, `32to63`, and `ge64`. These sizes are measured after the 640-pixel letterbox, so source-size strata may move when resized. Each result includes bbox AP50, semantic corner error, complete-quad precision/recall, M3a acceptance and population counts. NMS and score-ranked one-to-one bbox matching run once per image; matched predictions use their GT's stratum and unmatched false positives use their predicted quad's shortest edge. AP uses scores at least 0.05; corner, complete-quad and rectifier metrics use scores at least 0.25. A complete quad requires every corresponding semantic corner within 8 pixels. Empty AP/ratio populations report zero; no matched corners reports `null`. The report's `metric_config` records matching, boundaries and population rules.

Full-bundle export reads one immutable checkpoint byte snapshot. Both model loading and comparison with the detector report's checkpoint SHA-256 use those same bytes, including when the source path changes during export.

This is source-tree and local-package acceptance only. No background, weight, checkpoint, ONNX artifact, TensorRT benchmark, browser integration, remote CI result, real-image metric, or production recognition metric is bundled or verified by this repository.

## Local M4 Reader

After `plateai-detect-export` has produced a local full bundle, run the Reader
against a local image with persistent ONNX Runtime sessions:

```powershell
.\.venv\Scripts\plateai-read --bundle models\bundles\v1-full-local --image C:\local\frame.png --warmup 5
```

By default the Reader uses the first available provider order TensorRT, CUDA,
then CPU. Repeat `--provider` to explicitly control a provider/fallback order.
It validates every declared bundle file and both ONNX contracts before opening
sessions, then preserves retained NMS order, isolates an invalid four-corner
rectification, chunks valid crops by the manifest's dynamic batch maximum, and
CTC-decodes only strings permitted by the bundled plate rules. The JSON output
records individual detections/rejections and actual detector, rectifier, and
recognizer timings; it is not a benchmark report or accuracy claim.

The `reader` extra contains the ONNX and ONNX Runtime dependencies without
PyTorch: `python -m pip install ".[reader]"`. The existing `training` extra
also includes them for a compose/train/export/Reader workstation. No full
bundle, real frame, performance result, or production service is committed.

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

### Motorcycle colour profiles

The default remains the white-background private-passenger profile required by
the current M2/M4 contract. It does not infer a vehicle class from arbitrary
plate text. To generate a mixed motorcycle visual dataset, use the matching
rule and multi-template files below; the rule's `plate_type` selects the colour
and dimensions before rendering:

```powershell
.\.venv\Scripts\plateai-generate generate `
  --charset configs/charsets/tw_new_style_private_passenger_v1.txt `
  --rules configs/plate_rules/tw_motorcycle_colours_v1.json `
  --template configs/plate_templates/tw_motorcycle_colours_v1.json `
  --output out/motorcycle-colours
```

The bundled profiles represent ordinary heavy motorcycles as white/black
260×140, 250–550cc large heavy motorcycles as yellow/black 300×150, 550cc+
large heavy motorcycles as red/white 300×150, and 50cc light motorcycles as
green/white 260×140. These category colours and nominal sizes follow the
Highway Bureau's coding table; the RGB values and the bundled font remain
visual approximations, not calibrated manufacturing specifications.

The selector is strict: a motorcycle rule whose `plate_type` lacks a template
fails atomically rather than silently falling back to a white plate. A normal,
single-template config still applies that one template to every rule so current
custom workflows remain compatible. These variable-size/type outputs are for
visuals and later multi-profile work; do not feed them to the current M2/M4
private-passenger 380×160 recognizer trainer.

The same run seed, sample index, input files, and `requirements/py311.lock.txt` reproduce the same strings, sampled transform parameters, metadata, and encoded images. Reproducibility outside that dependency lock is not promised.

## Fonts

Without `--font`, the renderer uses the bundled, unmodified `NotoSansMono[wdth,wght].ttf` at weight 700 and width 62. The font is covered by SIL OFL-1.1; its SHA-256 and complete license text are retained in `THIRD_PARTY_NOTICES.md` and `assets/fonts/OFL.txt`. It is a legal visual approximation for the default new-style private-passenger profile, not an official Taiwan number-plate font.

For training data that should use the project-bundled Taiwan plate face, pass
`--font taiwan_plate` (or `--font official`). The run records
`TaiwanPlate-Regular.ttf` and its SHA-256 in `generation_config.json` and
per-image renderer metadata. The face was built from the checked-in Highway
Bureau reference material by `tools/build_plate_font.py`; it is intended for
local synthesis and visual comparison, not a representation that the Highway
Bureau distributes this font as a product or grants general redistribution
rights for derived materials.

```powershell
.\.venv\Scripts\plateai-generate generate `
  --count 10000 `
  --seed 42 `
  --font taiwan_plate `
  --output out/train-taiwan-plate-font
```

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

## Real-scene Detector experiment (local only)

This path trains only the native Detector. It does not generate fonts, train OCR,
change `active-v1`, or confer distribution rights. EZCon's license is unresolved;
use only under the explicit local-experiment acknowledgement described in
`evaluation-sources.md`. Keep the existing recognition-test images out of training.

```powershell
.\.venv\Scripts\python -m pip install -r requirements/py311.data.lock.txt
.\.venv\Scripts\python tools/fetch_ezcon_detection.py `
  --cache datasets/restricted/ezcon-taiwan-detection-raw `
  --output out/ezcon-detector-v1 `
  --holdout-images datasets/restricted/ezcon-taiwan-recognition-test/images `
  --holdout-images datasets/real_benchmarks/images `
  --acknowledge-unreviewed-license
.\.venv\Scripts\plateai-detect-train `
  --train out/ezcon-detector-v1/train `
  --validation out/ezcon-detector-v1/validation `
  --output runs/detector-real-v1 --epochs 30 --batch-size 16 --device cuda
.\.venv\Scripts\python tools/evaluate_detector.py `
  --test out/ezcon-detector-v1/test `
  --baseline runs/detector-run-v1/best.pt `
  --candidate runs/detector-real-v1/best.pt `
  --output out/detector-real-v1-evaluation --device cuda
```

Outputs refuse overwrite. CPU remains supported and tested; CUDA is seeded but
cross-device bitwise determinism is not promised. Real images are decoded once
into an immutable RAM snapshot (~3 GiB for this import), and input hashes are
rechecked before publishing. Full plate polygons become existing multi-instance
targets; geometry/reader contracts are unchanged.

Objectness focal loss is summed and divided by the batch's positive-cell count
(minimum denominator one), not by all 8,400 cells. New heads start with a 1%
objectness prior; existing checkpoint weights override it when loaded. Full-bundle
export combines Detector/Recognizer data provenance and cannot inherit a reviewed
license flag from just the Recognizer.

Select weights on validation, then freeze before scoring test. Compare bbox and
all-four-corner quality separately, and inspect actual top-confidence crops. A
falling loss, fewer candidates, successful ONNX export, or high crop-only OCR
accuracy does not establish correct full-scene localization. Evaluation never
activates a candidate model automatically.

### Detection Debug Mode

For a bbox-only check of one native checkpoint, run:

```powershell
.\.venv\Scripts\python.exe tools\debug_detector.py `
  --image path\to\input.jpg `
  --checkpoint runs\detector-real-v2-geometry\best.pt `
  --output out\detector-debug.jpg `
  --device cuda
```

This runs the existing 640x640 letterbox and detector NMS contracts, maps the
retained bboxes back to the original image, and saves only the boxes and scores.
Its JSON output includes source/resized/padding/model-input sizes plus every
retained bbox's source-space and raw model-input pixel dimensions.
It deliberately does not run corner rectification, OCR, export, activation, or
the frozen test evaluator. The output refuses overwrite. Use
`--score-threshold` and `--nms-iou` only for local diagnostics.
