# Project History

This file is an append-only record of important implementation decisions, local verification evidence, and known validation boundaries.

## 2026-09-21 - Project baseline and M1 recorded

### Repository state

- Branch: `codex/initial-design`, tracking `origin/codex/initial-design`.
- Baseline commit: `eab53b2` (`feat: implement M1 synthetic dataset pipeline`).
- Working tree was clean before this history file was added.
- The public repository is source-only under MIT. It must not publish real plate imagery, trained weights, ONNX files, TensorRT engines, or Model Bundles. The committed PNG fixtures are four harmless, synthetic contract fixtures.

### Architecture and contracts

- `plateai_shared` is the stable boundary used by Trainer and the planned Reader. It loads the ordered charset and data-driven plate rules, produces legal canonical/display text, and validates JSON Schema contracts (including CTC, batch, manifest, and finite-number cross-field rules).
- The Trainer and Reader communicate through a versioned, hash-verified Model Bundle contract. The Reader package, detector, rectifier, recognizer, decoder, API, and benchmark layers are designed but not implemented in M1.
- M1 supports uppercase Latin letters and decimal digits using the configurable examples equivalent to `LLL-DDDD` and `DDDD-LL`. Decorative hyphens are display-only; canonical recognition labels exclude them.
- Model manifests already define future CTC blank/index mapping, class-count, tensor, batch, capability, filename, hash, and four-corner requirements so M2-M4 can preserve a stable public contract.

### M1 implementation delivered

- `plateai_trainer.synthetic` produces deterministic CPU recognition crops. Every sample derives an independent seed from the run seed and index; no shared random state is used.
- Rendering uses an OpenCV Hershey fallback, avoiding an unlicensed redistributable font. It emits RGB `uint8` 320x96 PNGs and four ordered source/transformed corners.
- Configurable photometric and geometric augmentation records its sampled parameters in metadata. Geometry is retried only while finite, clockwise, in-frame, and sufficiently non-degenerate; otherwise it records a safe fallback.
- Dataset generation writes into an owned sibling staging directory, validates records and summary data, then publishes once without replacing an existing or raced output target. Failures remove only the owned partial directory.
- The CLI is `python -m plateai_trainer.synthetic generate` (or installed `plateai-generate`). It produces `images/`, `labels.txt`, `metadata.jsonl`, `generation_config.json`, and `summary.json`; existing output returns exit code 3 and is never overwritten.
- Default configuration assets and JSON schemas cover charset, plate rules, template, augmentation profile, generated metadata/summary, and future model manifests.

### Test and build evidence

- The checked-in test suite covers rule loading/generation, rendering, deterministic augmentation, schema and Model Bundle contract validation, safe synthetic fixtures, transactional dataset generation, and CLI exit behavior.
- GitHub Actions is configured for Ubuntu CPython 3.11: locked dependency install, editable install, `pytest -q`, package build/install, and a three-image smoke generation. Its remote run status was not inspected here.
- Local test execution is **not yet verified**. The default `python` is CPython 3.12.0, while the project supports CPython 3.11 only. A direct local CPython 3.11.15 runtime exists under the user-managed `uv` location, but it has no `pytest` installed. No dependencies, virtual environments, generated datasets, or source files were changed during this baseline check.

### Next approved roadmap boundary

- M2 is recognition training and ONNX export: use the shared charset, explicitly verify CTC blank/class mapping, evaluate the recognizer, check native-versus-ONNX parity, and make only a local crop-development bundle. No trained artifact may be committed or attached to a release.
- M3 adds a four-corner pose detector and validated perspective normalization. M4 loads full local bundles for high-speed constrained recognition. M5 adds the hardened API and 3wa showroom.
- Browser, real-data, field-accuracy, production deployment, and remote CI acceptance remain unverified. Synthetic output is not evidence of real-world plate recognition accuracy.

## 2026-09-21 - Local M1 environment established and verified

- Created the ignored `.venv` with the local CPython 3.11.15 runtime managed by `uv`; the Windows `py -3.11` launcher alias is unavailable, so use `.venv\Scripts\python.exe` from this checkout.
- Installed the exact `requirements/py311.lock.txt` dependency set and the project as an editable package. `pip check` reported no broken requirements.
- Fresh local test evidence: `.venv\Scripts\python.exe -m pytest -q` completed with `47 passed in 8.26s`.
- Fresh CLI smoke evidence: generated three synthetic PNG crops with `--count 3 --seed 42`; the JSON summary reported `generated: 3`, `standard-lll-dddd: 3`, and `standard: 3`.
- This verifies the local Windows M1 synthetic pipeline and its committed contracts. It does not verify GitHub Actions, Linux-only support, real-world accuracy, model training, or any production deployment.

## 2026-09-21 - Official new-style plate font comparison

- The M1 default is OpenCV `FONT_HERSHEY_SIMPLEX` (`opencv-hershey-simplex`), not a Taiwan number-plate font. The local training guide already classifies it as safe for development and CI only, not representative of every production plate font.
- The Highway Bureau publishes an explicit two-page reference, `Font of English letter and number used in new type number plate.pdf`, under its number-plate materials. The current Hershey glyphs have not been traced from, or matched against, that official reference and must not be described as identical.
- The current `standard-white-v1` canvas is 320x96 (3.33:1), whereas the official new-style self-use passenger-car plate is 380x160 mm (2.375:1). It is a recognizer-crop template, not a physical plate-spec template.
- The official current coding table excludes `I`, `O`, and `4` from new-style allocation. The M1 generic charset and rules still include all A-Z and 0-9, so they are deliberately broader than the formal allocation set.
- Before claiming visual fidelity, add a separately versioned official-reference-derived glyph/profile only after confirming reproduction permission or creating independently licensed vector glyphs; retain Hershey as the public CI fallback.

## 2026-09-21 - New-style private-passenger visual profile v1

- Per project decision, the default M1 CLI profile is now restricted to new-style private passenger cars: white background, black glyphs, `LLL-DDDD` display, 380x160 output, and no `I`, `O`, or `4` in its versioned charset/rule files. Motorcycle, commercial, and other plate families remain out of scope.
- Default glyphs use the bundled, unmodified Google Fonts Noto Sans Mono variable font (`OFL-1.1`), SHA-256 `2cb2adb378a8f574213e23df697050b83c54c27df465a2015552740b2769a081`, at weight 700 and width 62. The complete license text is in `assets/fonts/OFL.txt`; notices and package data rules preserve it in source and wheel distributions.
- This is a lawful visual approximation, not an official Taiwan plate font or traced derivative. A user can still pass `--font` with a local `.ttf` or `.otf`; run configuration records the selected font name, SHA-256, and variation axes.
- The prior generic charset, rules, template, and 320x96 toy fixtures remain as compatibility/test material, but the CLI defaults now select the v1 private-passenger profile.
- This decision supersedes the earlier provisional suggestion to retain Hershey as the default public CI fallback; the bundled OFL font is now the default while Hershey remains only a renderer compatibility branch.
- Verification evidence: the new default-font and default-CLI tests first failed against the former Hershey/320x96 behavior, then passed after implementation; full suite passed with `49 passed in 6.61s`. A fresh sdist/wheel build included the new configs, TTF, and OFL text. After force-installing that wheel, the CLI loaded from `.venv\Lib\site-packages`, generated a 380x160 crop, and recorded the expected font provenance.
- Not verified: exact official glyph equivalence, official-font reproduction rights, real-world recognition accuracy, other plate categories, or production deployment.

## 2026-09-21 - M2 PyTorch CTC design approved

- M2 is limited to the v1 new-style private-passenger `LLL-DDDD` crop profile. It will use native PyTorch CTC with the 33 visible symbols in `tw_new_style_private_passenger_v1.txt`, CTC `blank_index: 0`, and `class_count: 34`; `4`, `I`, and `O` are not legal recognition outputs.
- The approved preprocessing contract is RGB 380x160 source crop -> grayscale -> aspect-preserving 64x152 resize -> 64x160 letterbox with raw white 255 padding of four pixels on each horizontal side -> `float32 / 255.0` NCHW `[batch, 1, 64, 160]`. The source ratio is 2.375:1, not 3:1.
- The Model Bundle's existing `manifest.json` will carry the preprocessing, CTC, dynamic batch, tensor, checksum, and provenance contracts; no duplicate `meta.json` is planned. ONNX uses a dynamic batch only and must pass native-versus-ONNX tensor tolerance and decoded-text parity before atomic local bundle publication.
- Formal design record: `docs/superpowers/specs/2026-09-21-m2-pytorch-ctc-design.md`. This decision is approved for design documentation, but implementation planning and code changes await written-spec review.

## 2026-09-21 - M2 implementation plan prepared

- The approved design has been decomposed into `docs/superpowers/plans/2026-09-21-m2-pytorch-ctc.md`: isolated training dependencies, shared CTC/preprocessing and manifest contracts, validated M1 v1 data loading, native CRNN training, ONNX export/parity, atomic bundle publication, and final local verification.
- The plan includes a red-green test step for every task and keeps M1's generic fixtures and lightweight generator install intact. It is planning evidence only; no M2 dependency, training code, checkpoint, ONNX file, bundle, or model output has been created.

## 2026-09-21 - M2 plan review amendments

- Preprocessing now names Pillow 12.3.0 as the golden reference: `Image.fromarray(..., mode="RGB").convert("L")` followed by `.resize((152, 64), resample=Image.Resampling.BILINEAR)`. The handwritten luminance/rounding formula was removed. A synthetic input/output conformance vector will prevent a future Reader or OpenCV implementation from silently drifting by one 8-bit value.
- The recognizer no longer compresses width to 40 CTC steps or uses a BiLSTM. It retains 80 horizontal steps by using horizontal stride 2, then uses residual depthwise-separable 1D convolutional temporal blocks (dilations 1, 2, 4). The legal repeat-heavy `AAA8888` target needs 12 CTC positions and is now an explicit decoder, dataset, training, and export smoke boundary.
- These plan corrections were subsequently implemented as M2.

## 2026-09-21 - M2 local PyTorch CTC and ONNX implementation

- Implemented the 33-symbol, blank-zero CTC contract; Pillow 12.3.0 golden preprocessing; 80-step depthwise temporal CNN; validated v1 source reader; CPU trainer; and hash-verified atomic ONNX bundle exporter.
- `AAA8888` is covered as a 12-step CTC target through codec, data loading, training, and ONNX parity tests. Export checks batch sizes one and two, ONNX checker, logits tolerance, and greedy decoded-text parity before publication.
- Local verification passed: `pytest -q` reported `75 passed in 25.94s`; sdist/wheel build succeeded; the wheel was force-installed with `pip check` reporting no broken requirements; `plateai-train --help` and `plateai-export --help` returned successfully.
- The local Windows PyTorch 2.14.0+cpu convolution path emitted access-violation traces at its default six threads under pytest. Training/export now pin Torch CPU work to one thread; the same model test completed cleanly with one thread. This favors reproducibility over throughput pending independent runtime investigation.
- Not verified: remote CI, GPU behavior, real photographs, official-font equivalence, field accuracy, Reader integration, browser/API/IIS/DB, or production deployment. All datasets, checkpoints, ONNX files, runs, and bundles remain local ignored artifacts.

## 2026-09-21 - M3a deterministic four-corner rectifier

- Added `plateai_reader.rectifier` as the deterministic boundary between a future pose detector and M2. `normalize_corners` accepts four finite, in-frame points in arbitrary order, constructs a convex hull first, and returns exact clockwise `left_top`, `right_top`, `right_bottom`, `left_bottom` semantics or an `InvalidCornersError` reason code.
- The normalizer rejects non-finite, duplicate, out-of-frame, non-convex, low-area, and geometry-ambiguous candidates before homography creation. A raw bow-tie ordering of an otherwise convex point set is deliberately normalized rather than rejected: original ordering is unavailable from an unordered set, and accepting arbitrary order is the public contract.
- Top/bottom selection uses opposite long-edge midpoint projections along the image-down normal; left/right selection uses projections along the long plate axis. It does not rely on `min(x+y)` or `min(x)`, avoiding the diamond/high-tilt corner swap hazard.
- `rectify_plate` accepts only RGB `uint8` input and maps normalized source corners to the fixed discrete destination `[[0,0],[379,0],[379,159],[0,159]]` using OpenCV `INTER_LINEAR` and white `BORDER_CONSTANT`. Its output is exactly RGB `uint8 [160,380,3]`, directly consumable by M2's existing Pillow-golden preprocessing.
- Local synthetic evidence includes every permutation of a valid trapezoid, malformed geometry rejections, a M1-rendered plate passed through a known perspective fixture and M2 preprocessing, and an identity warp whose all four outer rows/columns exactly equal the corresponding source pixels. The latter confirms no observed one-pixel outer-edge mixing for the chosen discrete destination convention.
- Final review added severe-tilt permutation coverage, `reordered` metadata coverage, and stable `invalid_image_size` handling for a non-sequence image-size argument.
- Full local verification: `.venv\Scripts\python.exe -m pytest -q` completed with `89 passed in 25.90s`; source distribution and wheel builds succeeded and included `plateai_reader`; `.venv\Scripts\python.exe -m pip check` reported no broken requirements.
- Not verified: M3b detector training/data/keypoint output, detector ONNX compatibility, GPU behavior, real images, field accuracy, full Reader decoding, browser/API/IIS/DB, remote CI, or production deployment. No detector artifact, model weight, or real plate image was added.

## 2026-09-21 - M3b native multi-plate pose-detector design approved

- M3b is approved as a native-PyTorch, single-class, anchor-free YOLO-style pose detector with a lightweight CSP-Darknet-tiny-style P3/P4/P5 path. It uses no Ultralytics code, package, model, or export pipeline; repository-owned detector code retains the project source-only boundary while normal third-party runtime notices still apply.
- The detector accepts RGB source images through one OpenCV-golden 640x640 letterbox contract and supports multiple plates per image. Its fixed-spatial ONNX output is dynamic-batch `[batch,8400,13]`: bbox, confidence, and four semantic corners for every candidate; NMS remains a deterministic pure-NumPy Reader operation.
- M3b composite data will place one to three M1-rendered v1 plates per user-authorized local background, preserving exact per-instance semantic corners and bbox GT. Background identity/hashes and disjoint train/validation background sets are required; no background, real plate image, dataset, checkpoint, or ONNX file is committed.
- Reader inverse letterboxing must transform bbox and all eight corner values, then clip emitted points to discrete `[0,width-1]` and `[0,height-1]` bounds before M3a. A rejected M3a instance cannot prevent other NMS survivors from rectifying.
- Formal spec: `docs/superpowers/specs/2026-09-21-m3b-native-pose-detector-design.md`. Implementation, model training, ONNX export, TensorRT benchmark, real-image accuracy, and deployment remain unstarted and unverified.

## 2026-09-21 - M3b implementation plan prepared

- The approved M3b design is decomposed in `docs/superpowers/plans/2026-09-21-m3b-native-pose-detector.md` into seven atomic, test-first commits: shared OpenCV letterbox contract; 1-3-instance transactional composites; native pose network/loss; deterministic local training; NumPy Reader NMS and isolated M3a handoff; strict ONNX/full-bundle export; then local workflow documentation and package regression.
- The plan explicitly covers odd-padding inverse mapping for all corners, semantic-corner/non-overlap composition, deterministic NMS tie handling, train/validation background SHA-256 separation, native-versus-ONNX NMS parity, and full-bundle rejection cases. It preserves M2 crop-only validation and M3a's 380x160 rectifier boundary.
- This is planning evidence only. No detector source implementation, dependency change, composite dataset, user background, checkpoint, ONNX model, bundle, TensorRT benchmark, real-image result, browser/API integration, remote CI result, or deployment validation has been added.

## 2026-09-21 - M3b local detector workflow and package regression

- M3b implementation is recorded by commits `3cfe883`, `9d68742`, `d4a53c6`, `34fdd87`, `1c55356`, `457612a`, `a9b637f`, `440a336`, `be9cd91`, `af537c5`, and `207f620`. This final documentation/regression task adds no runtime implementation, background, dataset, checkpoint, ONNX file, or bundle.
- The installed-command regression parametrizes `plateai-compose.exe`, `plateai-detect-train.exe`, and `plateai-detect-export.exe`; each must return `--help` usage through the active virtual environment's `Scripts` directory. Its first execution in the inherited Task 6 environment was already green (`9 passed in 13.02s`) because that task had force-installed a wheel. After temporarily uninstalling only the local `3wa-plate-ai` distribution, the same command failed at all three new cases with `FileNotFoundError` for the absent scripts (the five existing subprocess CLI tests also failed because the package was absent). Restoring the editable distribution made the regression green (`9 passed in 10.51s`).
- Fresh local Windows evidence: `.venv\Scripts\python.exe -m pytest -q` completed with `243 passed in 93.71s`; `.venv\Scripts\python.exe -m build` built `3wa_plate_ai-0.1.0.dev0.tar.gz` and `3wa_plate_ai-0.1.0.dev0-py3-none-any.whl`; force-installing that resolved wheel succeeded; `.venv\Scripts\python.exe -m pip check` reported `No broken requirements found`; and the installed CLI regression completed with `9 passed in 13.29s`. `git -c core.whitespace=cr-at-eol diff --check` exited successfully with no output.
- The package distribution is named `3wa-plate-ai`, so its wheel is `dist/3wa_plate_ai-*.whl`; the task brief's `dist/plateai_trainer-*.whl` does not exist, and PowerShell passes an unexpanded wildcard to `pip`. The local package gate therefore resolved the actual single wheel before calling `pip install --force-reinstall`.
- Local-only documentation now records the user-authorized background-manifest compose/train/export sequence, the 640x640 OpenCV RGB letterbox input, pre-NMS `[batch,8400,13]` candidates, and independent M3a RGB `380x160` rectification per retained detection.
- Not verified: user-provided or real backgrounds/images, GPU behavior or performance, a trained detector/checkpoint/ONNX/full bundle, TensorRT, real-world or production recognition metrics, browser/API/IIS/DB integration, remote CI, or production deployment. The source tree remains artifact-free.

## 2026-09-21 - M3b consolidated final-review fixes

- Fixed installed detector-command tests to resolve Windows `.exe` and Unix entry points while retaining real Windows `--help` execution. A platform regression reproduced three Linux-name failures and three passing Windows cases before the fix.
- Composite generation now cycles the actual perspective-projected source shortest edge through `[16,32)`, `[32,64)`, and `[64,infinity)`; any successfully published run with at least three instances covers all three. One/two-instance runs explicitly record partial coverage, and an incompatible custom scale range or canvas fails atomically. The default scale range is now `[0.1,0.6]`, so output pixels for a prior seed change with this revision. New coverage provenance is checked against metadata; old v1 provenance remains loadable.
- Validation now reports AP50, semantic corner error, complete-quad precision/recall, rectifier acceptance and denominators for every projected-edge stratum after the 640-pixel letterbox. Matching runs globally once; matched predictions inherit the GT stratum, unmatched false positives use the predicted quad edge. Empty ratios/AP report zero and empty corner populations report null. Source-pixel generation and resized validation strata are explicitly distinguished in the report configuration and guide.
- Detector export loads and hashes the same immutable checkpoint bytes. The ABA regression actually swaps checkpoint A for valid B during `torch.load`, restores A, and proves the exported model and report still correspond to A; real ONNX parity, bundle validation and staging cleanup remain covered.
- RED evidence: 15 composite cases failed on missing size coverage/provenance/refusal, seven metric cases failed on missing per-stratum results, and the exporter ABA case caught B being loaded while the report hashed A. Four additional compatibility/provenance tests exposed the old dataset field whitelist before it was extended.
- Fresh local verification: affected tests passed `114 passed in 97.59s`; the single full-suite run passed `276 passed in 103.93s`; editable installation with `--no-deps` succeeded; `pip check` reported no broken requirements; `git -c core.whitespace=cr-at-eol diff --check` passed. Existing file encodings/newline styles were retained. No dependencies, model/data/ONNX artifacts or Ultralytics code were added.
- Boundary: Windows source/editable-install and procedural synthetic tests only. Linux path selection was simulated, not executed on Ubuntu; a fresh distributable build, remote CI, real backgrounds/images, GPU/TensorRT, real-world accuracy, browser/API/IIS/DB and production deployment were not validated by this fix wave.

## 2026-09-21 - Windows one-command build gate

- Added `build.ps1` as the Windows build entry point and `build.bat` as its cmd/double-click wrapper. The default gate establishes an ignored CPython 3.11 `.venv`, installs `requirements/py311.training.lock.txt` plus the editable test/training extras, runs the full suite, builds `dist/`, force-installs the resulting wheel, generates three synthetic smoke images in an owned temporary directory, and runs `pip check`.
- `build.ps1` prefers `uv venv --seed --python 3.11`, so a host without a registered Python 3.11 can bootstrap the required interpreter and pip. It falls back to `py -3.11`, rejects an incompatible existing virtual environment with an explicit `-RecreateVenv` recovery path, and limits that option's removal scope to the ignored `.venv` directory.
- Fresh clean-environment Windows evidence: the first run downloaded CPython 3.11.15 through `uv`, installed the exact lock, completed `276 passed in 203.07s`, built `3wa_plate_ai-0.1.0.dev0.tar.gz` and `3wa_plate_ai-0.1.0.dev0-py3-none-any.whl`, passed wheel-installed three-image smoke generation, and reported `No broken requirements found`. `build.bat -BootstrapOnly` also completed successfully.
- The gate does not train, download, publish, or package models; `dist/` contains source-package artifacts only. It remains a local Windows acceptance path, not evidence of remote CI, GPU/TensorRT, real-image accuracy, browser/API/IIS/DB, or production deployment.

## 2026-09-21 - M4 local full-bundle Reader core

- Added `plateai_reader.runtime.PlateReader` and `plateai-read`. Construction validates a supplied local full bundle before session creation, then holds detector and recognizer ONNX Runtime sessions for repeated reads. Default provider selection is TensorRT, CUDA, then CPU when available; callers may explicitly select providers.
- The Reader applies the M3b 640x640 letterbox, deterministic NumPy NMS, inverse mapping, and independent M3a rectification. Valid RGB 380x160 crops are preprocessed with the pinned M2 contract and chunked by the manifest's dynamic recognizer maximum. An invalid quad becomes an isolated result rejection and does not discard later detections.
- Added rule-constrained CTC Viterbi decoding. Its state tracks the enabled manifest rule position and previous raw CTC index, so it supports blank-separated repeats such as `AAA8888` without enumerating every legal plate string. Results expose canonical/display strings, rule/type, detection geometry/confidence, CTC log probability, provider list, batch sizes, and observed stage timings.
- The separate `reader` extra contains only `onnx==1.23.0` and `onnxruntime==1.30.0`, avoiding a PyTorch dependency for a local Reader-only installation. `build.ps1` now validates `plateai-read.exe --help` after reinstalling the built wheel, in addition to the existing installed-package synthetic smoke.
- Fresh local Windows evidence: focused Reader/CLI/package regression passed `43 passed in 15.07s`; the full build gate passed `285 passed in 210.68s`, produced the sdist/wheel, confirmed the wheel-installed Reader command, completed synthetic smoke generation, and reported `No broken requirements found`. A no-test package rerun after the added wheel-level Reader command also completed successfully.
- Boundary: Reader pipeline behavior is validated with deterministic fake ONNX session outputs because the repository correctly contains no full bundle. Real trained detector/recognizer behavior, actual provider availability/performance, TensorRT/CUDA execution, real images/accuracy, benchmark reports, API/IIS/DB, remote CI, and production deployment remain unverified.

## 2026-09-22 - Taiwan plate font, coloured motorcycle profiles, and repository consolidation

- Commit `23cd32c` adds the project-bundled `TaiwanPlate-Regular.ttf`, Highway Bureau reference PDFs/images, reproducible `tools/build_plate_font.py`, and direct renderer coverage. The generated font is a project asset built from the checked-in reference material for local synthesis and visual comparison; it is not a claim that the Highway Bureau distributes it as a product or grants general redistribution rights for derived materials.
- `plateai-generate generate --font taiwan_plate` and `--font official` now select that face reliably. The CLI parses `--font` as a `Path`, so the resolver normalizes aliases before comparison; it also locates packaged fonts from either a source checkout or an installed wheel data directory. Metadata records `TaiwanPlate-Regular.ttf` and SHA-256 `4ac89c39eb57045d5466e711ece3c901b13ca6b7c338c7c55efef9079f745075`. The default remains the OFL-licensed Noto Sans Mono profile, preserving M1/M2/M4 compatibility.
- A strict `PlateTemplateSelector` adds a versioned Taiwan motorcycle visual profile. `plate_type` chooses ordinary-heavy white/black 260×140, 250–550cc yellow/black 300×150, 550cc-plus red/white 300×150, or 50cc green/white 260×140. A missing selector entry fails atomically; existing one-template configuration files continue to work unchanged. These variable-size visual outputs are explicitly outside the current fixed 380×160 M2/M4 recognizer-training contract.
- `docs/evaluation-sources.md` and `tools/fetch_ezcon_taiwan_eval.py` document a pinned, explicit-acknowledgement local-only EZCon OCR evaluation fetcher and the separate TLPD detection/rectification boundary. Real images, labels, datasets, metrics, checkpoints, and model bundles remain ignored and are not release content. `tools/generate_plate_visual_gallery.py` provides a transactional, clean visual-inspection gallery for supplied plate text without claiming the text is allocated or visually official.
- `README.md` is now maintained in Traditional Chinese and links the source-only boundary, Windows build gate, Taiwan-font generation command, motorcycle profile constraints, local training, Reader, and evaluation-source guidance.
- Fresh local Windows verification: focused font/CLI/motorcycle tests passed `29 passed in 14.45s`; the complete `build.ps1` gate passed `292 passed in 198.75s`, built `3wa_plate_ai-0.1.0.dev0.tar.gz` and `3wa_plate_ai-0.1.0.dev0-py3-none-any.whl`, force-installed the wheel, completed installed Reader and three-image generator smoke checks, and reported `No broken requirements found`. From outside the source checkout, the wheel-installed `plateai-generate` successfully generated a one-image run with `--font taiwan_plate` and recorded the expected font identity/hash.
- Not verified: exact official glyph equivalence, permission for broad redistribution of reference-derived material, real-photo recognition accuracy, motorcycle allocation/prefix correctness beyond the configured visual categories, remote CI, GPU/TensorRT, web/API/IIS/DB, or production deployment.

## 2026-09-22 - Direction 1 & 3: Unified standard profile, full model training, export parity, and PyTorch GPU guidance

- **Direction 3 (Unified standard profile & strict contracts)**:
  - Extended character set (`configs/charsets/tw_standard_v1.txt`) to 34 visible symbols (including `4`), yielding 35 CTC classes (`[batch, 80, 35]`).
  - Added unified plate rules (`configs/plate_rules/tw_standard_v1.json`) covering 11 formats (new-style 7-digit, legacy 6-digit `LL-DDDD` / `DDDD-LL`, motorcycle `LLL-DDD`, `DDD-LLL`, `LLD-DDD`, `DLL-DDD`, and legacy 5/4-digit).
  - Enforced strict contract validation in `M1CropDataset`: validates that `rule_id` exists in the enabled ruleset, matches `plate_type`, validates canonical character tokens and length, and checks display separator placement.
- **Direction 1 (End-to-end model training and ONNX export)**:
  - Synthesized 8,000 training crops and 1,000 validation crops balanced across all 11 standard rules using `TaiwanPlate-Regular.ttf`.
  - Trained `PlateCTCNet` with 35 classes, achieving 100.0% exact plate accuracy on validation (1,000 / 1,000, loss: 0.00095).
  - Exported recognizer crop bundle `models/bundles/tw-std-v1-recognizer` (600 KB) with verified PyTorch / ONNX Runtime numerical parity.
  - Generated deterministic disjoint background sets via `tools/generate_backgrounds.py`, synthesized 1,000 training and 200 validation composite multi-plate scenes.
  - Trained `PlatePoseNet` multi-plate detector (best epoch 5, validation `bbox_ap50 = 0.438`), exported full model bundle `models/bundles/tw-std-v1-full` (detector: 2.17 MB, recognizer: 600 KB).
  - Fixed `PlateReader` crop batching (`np.stack` instead of `np.concatenate`) to preserve rank-4 `[batch, 1, 64, 160]`, and handled Windows console UTF-8 reconfigure.
  - Validated inference on test plates (`LAB-6531`, `LAG-5618`, `XHU-013`, `HJ9-037`, `AQ-560` inverted) with 100% exact match under constrained CTC.
- **PyTorch GPU hardware guidance**:
  - NVIDIA GTX 1080 (Pascal architecture, Compute Capability 6.1) requires PyTorch built with **`cu118`** (CUDA 11.8).
  - NVIDIA RTX 5060, RTX 5090 (Blackwell architecture) requires PyTorch built with **`cu128`** (CUDA 12.8+).
  - The clean source checkout and deterministic testing/build gate lock remain pinned to CPU `torch==2.14.0`.
- Fresh local verification: all 302 unit, integration, and contract tests pass in 216s.

## 2026-09-22 - Occasional background training specification

- Added `docs/superpowers/specs/2026-09-22-background-training-design.md` for review. The user scoped training as an occasional, single-machine operation: normally reuse the stable model and retrain manually when recognition coverage needs improvement.
- The proposed design uses one independent worker per training run, SQLite autocommit (`isolation_level=None`) without application-managed transactions, an OS file lock for single-run exclusion, and persistent logs/status so Web restarts can reconnect. SQLite's internal per-statement transactions remain enabled; no queue, scheduler, service, automatic retry, or reboot resume is proposed.
- The spec defines real batch/epoch progress, acknowledged cancellation, retained failure evidence, isolated model outputs, and preservation of the currently used model. It also records that the trainer publishes its final run directory only after success, so absence of that directory alone does not prove inactivity.
- Evidence and boundary: reviewed the current launcher, task manager, trainer, engine, generation/export interfaces, and training-page behavior, and checked Python 3.11/SQLite transaction semantics. This is documentation only: no production code, database, runtime, model, or running service was changed; all implementation/browser/GPU acceptance checks in the spec remain pending.

## 2026-09-22 - Background training implementation plan

- The user approved the background-training spec and requested the plan. Added `docs/superpowers/plans/2026-09-22-background-training.md` with six ordered tasks: autocommit task storage, training callbacks/cancellation, direct train/export pipeline, detached worker/process lifecycle, Web/API/launcher integration, and end-to-end verification/documentation.
- The plan keeps the agreed single-machine, occasional-training scope and no explicit transaction policy. It specifies worker-owned OS locking and process identity, pending/cancellation races, honest progress, preserved published artifacts, and no automatic activation of a new model.
- Planning checks used the current API/tests and training/export interfaces plus official Windows process/locking references. The existing Web tests must isolate build/release targets before broad regression runs. Windows console closure and actual browser reconnection are explicit acceptance items, separate from CPU artifacts and helper-process tests.
- This turn changed documentation only. The implementation, automated tests, database creation, live service restarts, browser checks, and GPU training have not been executed by this plan-writing task.

## 2026-09-22 - Native background Web training implementation

- Replaced Web Studio's process-local training path with a single-machine detached worker backed by `runs\.web-training\tasks.sqlite3` in SQLite autocommit mode. Task rows, cancellation requests, heartbeat, bounded logs, worker PID/creation-time identity, and task-local output locations survive a Web restart. The implementation has no application-managed `BEGIN`/`COMMIT`/`ROLLBACK`, no queue, scheduler, automatic retry, reboot resume, or automatic activation of a model.
- Training now reports real generation, batch, validation, epoch, and export events. A completed task retains a new `runs\<run-name>` and `models\bundles\train-<task-id>` bundle, including checkpoint/report paths; a late cancellation preserves any already-published result instead of replacing `active-v1`.
- Web launchers now default to `reload=False`; `--dev-reload` is explicit. The training page has an independent one-request-at-a-time poller with 1/2/5/10-second recovery, persistent-task reconnect, acknowledged cancellation, batch display before the first epoch, and text-safe status/log/path rendering.
- Fresh local Windows evidence: store/worker/API/launcher/UI-focused checks passed, then `.venv\Scripts\python.exe -m pytest -q` passed **352 tests in 126.82s** (25 warnings: existing Starlette/httpx deprecations, Torch deterministic `warn_only` CUDA warnings, ONNX exporter FutureWarnings, and `multipart` pending deprecation). The isolated 4+4 fixture CPU worker performed real one-epoch train, ONNX export, parity/bundle validation, and preserved an `active-v1` sentinel.
- Isolated browser/server evidence: the page showed `batch 100/2000` and `尚未驗證` before an epoch, reconnected after a page reload at `batch 1600/2000`, kept polling after a stop request until `cancelled`, displayed the capped reconnect status during intentional server downtime, and after server restart showed the completed bundle path with `已匯出，尚未啟用`. Console errors during that interval were only intentional `ERR_CONNECTION_REFUSED` fetches; no JavaScript exception was observed.
- Still unverified: user-operated `run_server.bat` console close/reload on the normal project root, GPU throughput/accuracy, power-loss/reboot behavior, IIS, production data/model activation, real vehicle imagery, and remote CI. The browser test used an isolated temporary root and CPU-only helper server; it did not change user datasets or model bundles.

## 2026-09-22 - GPU utilization monitor

- Replaced the training chart's primary signal from aggregate VRAM occupancy to GPU 0 compute utilization. The existing `/api/system/gpu_memory` response remains compatible and now also reports `gpu_utilization_percent`, `memory_utilization_percent`, and `metrics_source` from local `nvidia-smi`; unavailable metrics remain explicit `null` values rather than being fabricated.
- The chart now puts `GPU 運算 (%)` on the primary 0-100% axis and retains VRAM GB as a secondary contextual series. This makes short CUDA workloads visible even when their small VRAM allocation is visually flat on a 16 GB scale.
- RED/GREEN evidence: a Web API contract first failed because the utilization fields were absent, then passed with an `nvidia-smi` fixture. `node --check web/js/app.js` passed; a live local call reported RTX 5060 Ti with `metrics_source: nvidia-smi`; the full suite passed **352 tests in 129.63s** (the same 25 existing dependency/Torch warnings).
- Still unverified: the currently user-operated server must be restarted (default launcher has reload disabled) and the revised chart has not yet been observed during a fresh GPU training run. No model, dataset, active bundle, IIS, or production service was changed.

## 2026-09-22 - Real-photo Web recognition diagnosis

- The user requested a recommendation between immediate image-processing changes and detector training after two motorcycle-photo failures. Inspection and local diagnostic inference only were performed; existing uncommitted Web changes and local images were preserved.
- The current `active-v1` manifest declares only `crop-recognition`. All three locally present bundle manifests are recognizer-only. The Web predictor calls nonexistent `PlateReader.load` and `read_image` methods and refers to `detection.box`/`polygon` rather than the actual `PlateReader(bundle_dir).read`, `bbox_xyxy`, and `corners_xy` contract. Loader exceptions are swallowed. Fresh local import confirmed `reader is None` and an active recognizer session, leaving contour-based candidate extraction in use.
- The active report's validation accuracy is 100% over 500 generated crops in `out/val-default`, using the same configured synthetic template/font family; this is not measured real-photo or full-pipeline accuracy. Historical 99.8% epoch values likewise do not establish real-image performance.
- On the first supplied screenshot's 480x480 photo region, the current locator emitted one unrelated box and missed the visible `MDX-9717` plate. Manually specifying its visible four corners and applying the existing rectifier produced `MDX-771`; a diagnostic grayscale CLAHE variant (`clipLimit=2`, 8x8 tiles) produced `MDX-971`. Neither recovered the full plate. These results indicate recognizer coverage also needs investigation after localization is corrected.
- Single local diagnostic timings were 5.3 ms for candidate extraction, 1.88-4.42 ms for recognizer preprocessing/inference, and 836-874 ms for constrained decoding per crop. These are screenshot-region measurements, not the original phone-photo request or a repeatable latency benchmark; resizing alone cannot be assumed to deliver a 0.2-0.3 second end-to-end response.
- Recommended next sequence, not implemented by this diagnosis: repair and expose the Web/Reader/bundle integration; establish separate real-image localization, manually corrected-crop OCR, and end-to-end baselines; improve detector and recognizer coverage using properly separated real data; evaluate downsampled localization with original-resolution crops, decoding optimization, and optional CLAHE against that baseline. No model was retrained/activated and no live service was restarted.

## 2026-09-22 - Step 1: Web and PlateReader interface contract repair & diagnostic transparency

- Repaired Web predictor (`src/plateai_web/predictor.py`) alignment with `PlateReader(bundle_dir).read(image_rgb)`:
  - Corrected manifest check to require both `crop-recognition` and `plate-detection` alongside the `detector` component before attempting full-pipeline reader execution.
  - Aligned coordinate extraction to `detection.bbox_xyxy` and `detection.corners_xy`.
  - When the bundle is recognizer-only, honestly reports `pipeline_mode: "hybrid_heuristic"`, `locator_type: "opencv_contour_v1"`, and `detector_available: false`.
- Exposed granular latency breakdown (`locator_ms`, `rectifier_ms`, `onnx_inference_ms`, `ctc_decoding_ms`, and `total_ms`) and sent JPEG data URLs of exact rectified crops (`crop_base64`) along with `raw_greedy_text` in API responses.
- Implemented `POST /api/model/activate` in `src/plateai_web/app.py` allowing one-click promotion of trained bundles to `active-v1` with in-memory predictor hot-reloading.
- Enhanced Web Studio UI (`web/index.html`, `web/js/app.js`, `web/css/app.css`) with `#infer-diagnostic-bar`, crop preview thumbnail buttons opening `#modal-crop-inspect`, greedy text display, timing tags, and mascot one-click activation button.
- Verification: Web API tests expanded in `tests/test_web_api.py` (9 passed); full regression test suite passed (353 passed in 175.47s); live server restarted with `--dev-reload` on port 1688 and verified via API calls.

## 2026-09-22 - Step 2: Road real-photo benchmark suite & empirical baseline establishment

- Created `plateai_bench` package (`src/plateai_bench/`):
  - `metrics.py`: implements Levenshtein edit distance, `compute_cer`, analytical convex polygon IoU via `cv2.convexHull` and `cv2.intersectConvexConvex`, and objective failure attribution (`SUCCESS`, `LOCATOR_MISSED`, `LOCATOR_BAD_CROP`, `RECOGNIZER_MISREAD`, `RULE_FILTERED`).
  - `runner.py`: implements `BenchmarkRunner` supporting dual-mode evaluation: Mode A (Oracle Crop OCR from GT corners) and Mode B (End-to-End detection and recognition with IoU matching).
- Built CLI tool `tools/evaluate_real_benchmark.py` and prepared `datasets/real_benchmarks/user_cases.jsonl` (with `213-NSK` and `MDX-9717` ground-truth corners), plus EZCon car (`reader_v1_eligible_test.jsonl`) and motorcycle (`motorcycle_test.jsonl`) splits.
- Replaced mock simulation sleep and hardcoded accuracy in `src/plateai_web/evaluator.py` with real `BenchmarkRunner` calls.
- Baseline findings on active recognizer (`active-v1`) across 62 real road samples written to `out/benchmarks/baseline_report.json`:
  - User motorcycle cases: Oracle Acc 0.0% (CER 61.5%), E2E Acc 0.0% (100% `RECOGNIZER_MISREAD`).
  - EZCon motorcycles (30): Oracle Acc 0.0% (CER 77.8%), E2E Recall 46.7%, E2E Acc 0.0% (47% `RECOGNIZER_MISREAD`, 43% `LOCATOR_BAD_CROP`, 10% `LOCATOR_MISSED`).
  - EZCon passenger cars (30): Oracle Acc 3.3% (CER 50.5%), E2E Recall 53.3%, E2E Acc 3.3% (50% `RECOGNIZER_MISREAD`, 27% `LOCATOR_BAD_CROP`, 20% `LOCATOR_MISSED`).
- Verification: unit tests added in `tests/test_real_benchmark.py` (4 passed); full test suite passed (**357 passed in 140.57s**).


