# 3waPlateAI Design Specification

- Date: 2026-09-21
- Status: Approved for M1 implementation planning; amended after design review
- Repository: shadowjohn/3waPlateAI
- License: MIT for original project code; third-party assets and dependencies retain their own licenses.

## 1. Intent

3waPlateAI is a Taiwan-license-plate-focused engineering project covering the full path from synthetic data generation through training, evaluation, export, high-speed inference, and benchmarking.

The repository contains two product surfaces:

- **Trainer**: generates datasets, trains and evaluates models, and exports versioned Model Bundles.
- **Reader**: loads a Model Bundle and performs detection, perspective correction, recognition, constrained decoding, API serving, and benchmarking.

The boundary between them is the Model Bundle. Trainer internals may change without forcing Reader changes as long as the bundle contract remains compatible.

The public 3wa website is a later showroom and integration consumer. It is not part of this repository's training runtime.

## 2. Success criteria

The project is successful when it can:

1. Generate deterministic synthetic Taiwan plate crops and labels from configurable rules.
2. Fine-tune a compact recognition model with the exact same character set used at inference.
3. Detect a plate and its four ordered corners, then rectify it with a perspective transform.
4. Decode recognition logits under configurable plate grammars instead of validating only after OCR.
5. Export an immutable, checksummed Model Bundle that Reader can validate and load.
6. Report exact-plate accuracy, character accuracy, latency distribution, and failure categories.
7. Run Reader without the full PaddleOCR detection pipeline.
8. Keep real evaluation images and personally identifying plate data out of the public repository.
9. Let a public clone verify the pipeline with committed synthetic fixtures and, after the full Reader exists, a clearly marked toy Model Bundle.

## 3. Non-goals

The first implementation does not:

- use SAM or segmentation in the inference path;
- claim nationwide plate-rule completeness;
- scrape public images or collect identifiable plate data;
- store production recognition events in a database;
- build the 3wa.tw demonstration page;
- package third-party tools under the project's MIT license;
- optimize TensorRT before a correct ONNX baseline exists.
- represent a toy fixture or toy Model Bundle as evidence of commercial recognition accuracy.

## 4. Architectural decisions

### 4.1 One repository, two modules

Trainer and Reader live in one repository because they share plate rules, character ordering, bundle schemas, evaluation formats, and test fixtures. They remain separate Python packages so training dependencies never become Reader runtime dependencies.

### 4.2 Synthetic-first, real-evaluation-required

Synthetic data is the first training source. It provides volume, deterministic labels, and controlled edge cases without collecting personal data.

Synthetic metrics are not accepted as proof of field accuracy. Release-quality models must also be evaluated on a private, legally obtained real-world set split by normal, oblique, night, motion blur, overexposure, low resolution, and partial occlusion conditions.

### 4.3 Pose detector with four corners

The detector returns one bounding box and four keypoints in this fixed order:

1. left_top
2. right_top
3. right_bottom
4. left_bottom

The semantic keypoint labels are an orientation hint, not permission to pass the raw array directly to OpenCV. Rectifier must normalize and validate every quadrilateral before computing a homography:

1. reject non-finite, duplicate, or low-confidence points;
2. compute a four-vertex convex hull and reject self-intersection, concavity, or area below the configured fraction of the detection box;
3. restore a clockwise cyclic order;
4. choose the cyclic rotation/reflection that best preserves the detector's semantic labels while satisfying the geometric checks;
5. return the canonical left_top, right_top, right_bottom, left_bottom order plus a normalization flag and quality score.

Pure geometry cannot reliably distinguish an upright plate from the same quadrilateral rotated 180 degrees. If semantic orientation cannot be recovered confidently, Reader returns an invalid_corners rejection instead of forcing a plausible but potentially upside-down warp.

Only validated canonical corners are passed to the homography. Keypoint visibility and confidence are preserved so low-quality geometry can be rejected or routed to a documented fallback.

### 4.4 Recognition-only OCR runtime

After rectification, Reader invokes only a text-recognition model. It does not run generic text detection or document-orientation stages. The recognition model is trained and exported with the exact character set included in its Model Bundle.

### 4.5 Grammar-constrained decoding

The decoder consumes model logits and a plate grammar. Invalid paths are pruned during decoding rather than corrected only with a regular expression after recognition.

Rules declare:

- output length;
- allowed character class at each position;
- display separator placement;
- optional plate-type association;
- normalization and ambiguity mappings.

The canonical recognition text excludes decorative separators. Formatting is applied after a legal path is selected.

### 4.6 Benchmarking is a release requirement

Every released model records accuracy and latency with enough environment metadata to reproduce the comparison. No release may claim a speedup from a single un-warmed timing sample.

## 5. Runtime data flow

~~~mermaid
flowchart TD
    A["Image or video frame"] --> B["Plate pose detector"]
    B --> C["Four-point perspective warp"]
    C --> D["Plate recognizer"]
    D --> E["Grammar-constrained decoder"]
~~~

Reader returns the normalized result, source geometry, confidence values, inferred plate type, model identity, and per-stage timing.

## 6. Repository layout

~~~text
3waPlateAI/
├── README.md
├── pyproject.toml
├── docs/
│   ├── architecture.md
│   ├── training.md
│   ├── model-contract.md
│   └── superpowers/specs/
├── src/
│   ├── plateai_shared/
│   ├── plateai_trainer/
│   │   ├── synthetic/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── export/
│   └── plateai_reader/
│       ├── detector/
│       ├── rectifier/
│       ├── recognizer/
│       ├── decoder/
│       ├── api/
│       └── benchmark/
├── configs/
│   ├── plate_rules/
│   ├── plate_templates/
│   └── training/
├── schemas/
├── tests/
│   ├── unit/
│   ├── contract/
│   └── integration/
├── samples/
├── scripts/
└── models/
    └── README.md
~~~

Generated datasets, private evaluation images, trained weights, caches, and local benchmark outputs are ignored by Git. Large model artifacts are attached to GitHub Releases or stored by an explicitly configured artifact backend.

## 7. Shared contracts

### 7.1 Character set

The character set is an ordered UTF-8 file with one visible output symbol per line. It never contains the CTC blank token. Its hash is recorded in the bundle manifest. Training, export, and inference must all use that exact file.

For a CTC recognizer, the bundle manifest must declare an integer blank_index, class_count equal to len(charset) + 1, and index_mapping set to charset-order-skipping-blank. This makes index-to-symbol mapping unambiguous whether the training framework places blank first, last, or at another explicit index. Export validation must reject a recognizer output dimension, blank index, or mapping that does not match this contract.

The first milestone supports uppercase Latin letters and decimal digits. Special plate characters such as 軍, 使, 外, 臨, 試, and 電 are added only through an explicit rule and character-set version.

### 7.2 Plate rules

Plate rules are data, not hard-coded branches. Each rule has:

- a stable rule ID;
- a sequence of letter, digit, literal, or explicit-symbol positions;
- a normalized display format;
- an optional plate-type hint;
- generation weight;
- enablement state.

The initial generator includes configurable examples equivalent to LLL-DDDD and DDDD-LL. Before a public model claims broad Taiwan plate support, enabled rules must be checked against an authoritative rule source and represented by real evaluation cases.

### 7.3 Model Bundle

A bundle directory is immutable after release:

~~~text
twplate-v0.1.0/
├── manifest.json
├── charset.txt
├── plate_rules.json
├── detector.onnx
├── recognizer.onnx
├── benchmark.json
└── THIRD_PARTY_LICENSES.md
~~~

A recognizer-only development bundle may omit detector.onnx only when manifest capabilities explicitly declare crop-only input.

The manifest requires:

- schema version, model ID, semantic version, and creation time;
- supported capabilities;
- detector and recognizer filenames, formats, hashes, and tensor contracts;
- recognizer batch mode: either dynamic with min, opt, and max batch sizes, or fixed with batch size and padding policy;
- decoder type, exact CTC blank index, class count, and index-to-character mapping;
- keypoint names and ordering;
- preprocessing and postprocessing parameters;
- normalized plate dimensions;
- character-set and rule-file hashes;
- training-data provenance summary;
- compatible Reader contract version.

Reader validates required files, hashes, schema versions, tensor metadata, decoder semantics, batch contract, and capability compatibility before loading a bundle. A TensorRT engine profile is created ahead of serving and is never rebuilt merely because one frame contains a different number of plates.

### 7.4 Reader result

A successful result contains:

~~~json
{
  "model_id": "twplate-v0.1.0",
  "image_size": [1920, 1080],
  "plates": [
    {
      "bbox": [100, 200, 420, 310],
      "corners": [[110, 205], [412, 201], [418, 306], [104, 309]],
      "raw_text": "ABC1234",
      "plate": "ABC-1234",
      "confidence": 0.982,
      "plate_type": "standard",
      "plate_type_confidence": 0.91,
      "rule_id": "standard-lll-dddd",
      "timings_ms": {
        "detect": 2.8,
        "warp": 0.2,
        "recognize": 4.6,
        "decode": 0.1,
        "total": 7.7
      }
    }
  ]
}
~~~

No-plate results are successful responses with an empty plates array. Invalid bundles, unsupported schemas, corrupt images, and inference failures use stable machine-readable error codes.

## 8. Synthetic data pipeline

The generator is deterministic when given the same configuration, dependency lock, and random seed.

Its stages are:

1. Select a plate rule and generate a legal canonical string.
2. Select a licensed plate template and licensed font.
3. Render plate background, border, glyphs, and optional plate-type styling.
4. Apply photometric effects such as exposure, glare, shadow, noise, JPEG artifacts, dirt, and partial occlusion.
5. Apply geometric effects such as rotation, perspective, crop, motion blur, and resizing.
6. Write the image, recognition label, and JSONL metadata describing every sampled parameter.

Geometry-aware transforms update corner annotations from the same transform matrix used for pixels. Detector training composites are a later milestone; the first milestone generates rectified recognition crops and their labels.

Fonts are loaded from an explicit asset directory. A font is committed only when redistribution is permitted and its license is recorded. The generator fails clearly when no usable font is configured.

## 9. Training and export

Recognition training starts from an appropriate pretrained model only when its head can be fine-tuned for the project's exact character set. Merely swapping an inference dictionary is forbidden.

Trainer owns framework-specific configuration and produces framework-neutral exports. ONNX is the baseline interchange format. Export verification must compare native and ONNX outputs on fixed fixtures within declared numerical tolerances.

Detector training remains behind a detector adapter. A YOLO Pose implementation is acceptable for research, but its code, weights, and packaging are subject to a separate license review before commercial or closed-source redistribution.

## 10. Reader design

Reader uses persistent model sessions and supports provider selection in this order when available:

1. TensorRT execution provider;
2. CUDA execution provider;
3. CPU execution provider.

The baseline first proves correctness with ONNX Runtime. Fixed input sizes, warm-up, FP16, batching, and copy reduction are added only with before-and-after benchmark evidence.

Reader exposes three layers:

- a Python library for direct calls;
- a CLI for local validation and benchmarking;
- an HTTP API in a later milestone for 3wa.tw and other consumers.

The library is the source of truth. CLI and API surfaces call it rather than duplicating pipeline logic.

For a frame with multiple detected plates, Reader preserves detection order, rectifies all accepted crops, and submits recognizer work in batches. A dynamic-batch recognizer chunks crops at the manifest's max batch size. A fixed-batch recognizer chunks at its declared size, pads the last chunk with neutral crops according to the manifest, and discards padded outputs. Batch size one is valid but is treated as an explicit loop contract. Reader must not trigger TensorRT engine rebuilding in the request path. Timing output records real crop count, padded count, chunk count, and recognizer batch sizes.

## 11. Evaluation and performance

Accuracy metrics include:

- exact normalized plate accuracy;
- character accuracy;
- per-rule accuracy;
- per-condition accuracy;
- detection recall and precision;
- normalized corner error;
- plate-type accuracy;
- reject rate and false-accept rate.

Latency reports include warm-up count, batch size, image size, plate count, provider, precision, device, software versions, p50, p95, p99, and throughput.

The engineering target is an end-to-end path below 10–15 ms per plate on a suitable NVIDIA GPU after warm-up. This is a target, not a published claim, until measured on named hardware and fixtures.

## 12. Testing strategy

- Unit tests cover grammar generation, normalization, constrained decoding, transforms, and bundle validation.
- Property tests generate many legal plates and prove they round-trip through normalization and formatting.
- Contract tests reject missing, modified, or incompatible bundle files, ambiguous CTC blank mappings, and invalid dynamic/fixed batch declarations.
- Rectifier tests cover all point permutations, crossed semantic labels, degenerate hulls, low area, low confidence, and unresolvable orientation.
- Reader batching tests cover zero, one, exact-batch, partial-batch, and over-max plate counts without engine rebuilding.
- Integration tests run a small deterministic synthetic fixture set through the available pipeline.
- Export tests compare native and ONNX model outputs.
- Benchmark tests are separate from correctness tests and never use fragile wall-clock thresholds in normal CI.

Image tests assert semantic properties and transformed coordinates. They do not depend solely on platform-sensitive whole-image hashes.

## 13. Privacy, security, and repository hygiene

- Real vehicle images are private by default and excluded from Git.
- Public samples use synthetic or explicitly consented data.
- Dataset metadata records provenance and allowed use.
- Uploaded images in a future demo have size, type, decoding, and retention limits.
- Model bundles are checksum-validated before loading.
- Untrusted bundle paths cannot escape their bundle directory.
- Generated files, environments, caches, weights, and credentials are ignored.

## 14. Dependency and license policy

Original 3waPlateAI code is MIT. That does not relicense external fonts, datasets, model weights, training frameworks, or generated artifacts.

Each releasable bundle includes third-party notices and provenance. CI will eventually run dependency and artifact license checks. Any component with incompatible, unclear, or commercial-use-restricted terms remains optional and is not included in a public release.

### 14.1 Public distribution profile

The public repository distributes the complete original MIT source for Trainer, Reader, the synthetic engine, constrained decoder, contracts, and benchmarks.

It also commits a small deterministic fixture set under `tests/fixtures/synthetic/`. Every fixture is generated entirely by this project, contains no observed vehicle identifier, and has a manifest recording generator version, seed, canonical label, and SHA-256. These fixtures exist only to keep unit and integration tests runnable from a clean clone.

After detector, recognizer, and Reader integration exist, GitHub Releases may include a lightweight `toy-sample-bundle`. It must execute the real inference contract, but it is trained only enough to verify installation and pipeline wiring. It is not a production accuracy artifact.

Every toy bundle manifest must declare:

- `distribution.tier: "toy"`;
- `distribution.intended_use` containing `pipeline-validation` and `ci`;
- `distribution.production_ready: false`;
- `distribution.accuracy_claimed: false`;
- synthetic-only training provenance and complete third-party notices.

Commercial-grade datasets, real plate imagery, and production-trained weights are not distributed by default. README must state that the project provides the synthesis, training, and high-speed inference engine, while production accuracy requires users to generate suitable synthetic data or import a lawfully obtained dataset and train their own bundle.

## 15. Milestones

### M1 — Runnable synthetic recognizer dataset

Deliver:

- Python project and package skeleton;
- schema-driven plate rules and templates;
- deterministic generator CLI;
- licensed-font discovery and clear missing-font errors;
- clean plus configurable photometric and geometric augmentation;
- PaddleOCR-style recognition labels and JSONL metadata;
- character-set, rule, and Model Bundle schemas;
- a small manifest-backed synthetic toy fixture set for CI;
- unit and property tests;
- concise setup and generation documentation.

Acceptance:

~~~text
python -m plateai_trainer.synthetic generate \
  --count 100 \
  --seed 42 \
  --output out/demo

pytest
~~~

The command produces 100 valid images, a labels file, metadata JSONL, and a generation summary. Repeating it with the same locked environment and seed produces the same plate strings and transform parameters.

### M2 — Recognition training and ONNX export

Fine-tune a compact recognizer using the shared visible character set, declare and verify its CTC blank index and class mapping, evaluate it, export ONNX, verify parity, and create a crop-only development bundle with an explicit batch contract.

### M3 — Four-corner plate detector

Generate or label detector data, train the pose detector, evaluate boxes and corners, and add validated corner normalization plus perspective rectification.

### M4 — High-speed Reader

Load full bundles, run the end-to-end pipeline, add constrained decoding, implement manifest-driven multi-plate batching, benchmark providers and precision modes, expose CLI/library interfaces, and publish a synthetic-only `toy-sample-bundle` for pipeline validation.

### M5 — API and 3wa showroom

Expose a hardened HTTP API and build the 3wa.tw visualization showing bounding boxes, corners, rectified crops, text, confidence, plate type, and timings.

## 16. Initial implementation constraints

- Target Python is CPython 3.11 on Linux for training and inference development.
- CPU execution remains supported for correctness and CI.
- GPU acceleration is optional in M1.
- The first milestone implements the smallest useful vertical slice and avoids empty placeholder subsystems.
- Public accuracy claims wait for a versioned real-world evaluation set and reproducible benchmark report.
