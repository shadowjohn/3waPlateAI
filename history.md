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
