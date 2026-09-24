# Maintained MIT CPM + LPRNet Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the attributed, MIT-labeled author CPM + LPRNet a hash-validated, maintained ONNX candidate that users can compare on real photos without replacing `active-v1`, with a crop-only distributable service and honest evaluation boundaries.

**Architecture:** Keep the fixed native M2/M3 bundle contract untouched. Track a separate two-model external manifest and ONNX artifacts, implement a plate-ROI recognizer, compose it with the existing native detector for local full-scene comparison, and expose an allowlisted Web choice. The release service uses only the MIT OCR pair on a supplied plate crop; it does not redistribute the native detector trained on license-unreviewed data.

**Tech Stack:** CPython `>=3.11,<3.12`, PyTorch `2.14.0` for developer-only conversion, ONNX Runtime `1.30.0` for inference, NumPy `2.4.2`, OpenCV `5.0.0.93`, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-maintained-mit-lpr-adapter-design.md`

## Global Constraints

- Pin author source commit `574667ca7f5730d17b4b6fcda3ec568521bcbcd8` and model revision `51b9606b174eedbc091aec844c288a72aa9cd25b`. Verify original CPM SHA-256 `70450f3e24570bf714e53d0c169ac52948e448c3945f6748d19909d525420fa4` and LPRNet SHA-256 `21cda2d7f095d958eb725ac2dc05e87cbd342aed5fae7c0cf304477fe5b5a42b` before loading with `weights_only=True`.
- The upstream notebook uses OpenCV **BGR** input: BGR ROI → 100×100 CPM → 94×48 BGR warp → LPRNet `[B,37,18]`; CPM yields two `[B,4,50,50]` tensors. Our public `roi_rgb` contract converts RGB to BGR once before following that path and converts the returned display crop back to RGB. The final `-` class (index 36) is **blank**, not a displayed hyphen. Native v1 remains grayscale `[B,1,64,160]`, `[B,80,35]`, blank `0`. Keep the 37-character tuple as validated runtime data in the manifest/adapter, not imported from the PyTorch architecture module. Never pass external logits to `decode_constrained_ctc_v1` or copy the external manifest to `active-v1`.
- Include minimal attributed architecture source and converted ONNX CPM/LPRNet runtime files in Git and any OCR release. Do not include duplicate `.pth` files, TLPD images/labels, AGPL external detector code/weights, or unreviewed EZCon data/derived weights. Preserve the available original copyright notice and full MIT permission notice; do not fabricate a missing upstream holder.
- Keep `candidate-detector-real-v1` as the existing default Web preview. `candidate-v2-retest` is not currently a Web choice. The new `fpga-lpr-mit` choice is allowlisted and preview-only; no automatic activation, default change, push, or deployment.
- Keep unrelated dirty README/Web/release/resource-center changes. Before touching an overlapping file, inspect its live diff; if it cannot be preserved, stop and request direction. All local test results must distinguish TLPD training-source replay, manually audited unseen crop OCR, and full-scene end-to-end accuracy.

## File Map

| Unit | Files and ownership |
| --- | --- |
| Assets and provenance | `third_party/fpga_lpr/{model_arch.py,UPSTREAM.md,MIT-LICENSE.txt,manifest.json,cpm.onnx,lprnet.onnx}`, `.gitignore`, `THIRD_PARTY_NOTICES.md`; pinned architecture, hashes, notices, ONNX artifacts. |
| Build and validation | `tools/build_fpga_lpr_onnx.py`, `src/plateai_reader/fpga_assets.py`; developer conversion, manifest validation, no runtime PyTorch. |
| ROI recognizer | `src/plateai_reader/fpga_lpr.py`; original-path normalization, CPM, warp, LPRNet and decode, plus named safe-corners mode. |
| Scene composition | `src/plateai_reader/fpga_pipeline.py`; native `detect()` → bounded ROI → independent external OCR per plate. |
| Local Web | `src/plateai_web/{external_predictor.py,app.py}`, `web/{index.html,js/app.js}`; fixed candidate selection and diagnostics, no active overwrite. |
| Evaluation | `src/plateai_eval/fpga_protocol.py`, `tools/evaluate_fpga_lpr.py`; audited split contract, per-image scores and latency. |
| Crop-only service | `src/plateai_web/mit_lpr_crop_api.py`, `tools/build_mit_lpr_crop_release.py`; separate allowlisted source+ONNX output under `out/mit-lpr-crop-release`, no hidden full-scene claim. |

## Review Focus

1. A one-byte-modified ONNX or wrong tensor metadata must prevent model loading, not fall back silently — `tests/contract/test_fpga_lpr_assets.py::test_modified_onnx_fails_closed` (Task 1) and `tests/integration/test_fpga_lpr_export.py::test_exported_onnx_signatures` (Task 2).
2. A repeated symbol such as `AA` and the final `-` blank class must decode without duplicate-collapse or separator errors — `tests/unit/test_fpga_lpr_adapter.py::test_repeat_requires_blank_and_dash_is_blank` (Task 3).
3. An invalid CPM quadrilateral for one of two plates must reject only that plate — `tests/integration/test_fpga_lpr_pipeline.py::test_bad_first_plate_keeps_second` (Task 4).
4. A crafted form value resembling a path or bundle name must not select arbitrary artifacts or activate the external manifest — `tests/test_web_api.py::test_external_preview_is_allowlisted_and_does_not_activate` (Task 5).
5. Duplicate photos or the same plate string in development and holdout metadata must fail evaluation before scoring — `tests/unit/test_fpga_lpr_protocol.py::test_cross_split_plate_or_hash_is_rejected` (Task 6).

---

### Task 1: Asset manifest, notices, and fail-closed loader

**Files:** Create `src/plateai_reader/fpga_assets.py`, `tests/contract/test_fpga_lpr_assets.py`, `third_party/fpga_lpr/UPSTREAM.md`, `third_party/fpga_lpr/MIT-LICENSE.txt`; modify `.gitignore`, `THIRD_PARTY_NOTICES.md`.

**Interfaces:** Consumes a directory containing `manifest.json`, `cpm.onnx`, `lprnet.onnx`. Produces `FpgaLprManifest` and `load_fpga_manifest(assets_dir: Path) -> FpgaLprManifest`; raises `FpgaAssetError` for missing files, hash mismatch, wrong version, missing license/provenance, invalid 37-class charset, or unsupported tensor declarations. Task 2 creates the real artifact directory; this task's tests use tiny synthetic ONNX-file bytes to exercise the contract offline.

- [ ] **Step 1: Write the failing contract tests.** Use `tmp_path` to write `b"cpm"` and `b"lpr"`, hashes, and a JSON manifest with `schema="fpga-lpr-onnx-v1"`, `inputs` and `outputs` for both models. Assert `load_fpga_manifest(tmp_path).model_id == "fpga-lpr-mit-v1"`; mutate one byte and assert `FpgaAssetError("cpm sha256 mismatch")`. Add the same check for absent notice and unknown schema.

```python
with pytest.raises(FpgaAssetError, match="cpm sha256 mismatch"):
    load_fpga_manifest(tmp_path)
```

- [ ] **Step 2: Verify red.** Run `.venv\Scripts\python.exe -m pytest -q tests/contract/test_fpga_lpr_assets.py`; expect import failure because `plateai_reader.fpga_assets` does not exist.
- [ ] **Step 3: Implement the minimal manifest loader.** Read the manifest as UTF-8 JSON, compare an explicit schema and fixed relative filenames, reject `..`/absolute paths, hash each file with streaming SHA-256, and return a frozen dataclass. `FpgaLprManifest` carries `schema`, `model_id`, `source_commit`, `model_revision`, `components` (filename, SHA-256, input/output tensor names and dimensions), `charset` (the pinned exact 37-entry tuple ending in `-`), and `license_notice`; `from_document` validates these fields and `load_fpga_manifest` resolves only the two fixed model filenames. Add a `third_party/fpga_lpr/UPSTREAM.md` with the pinned URLs, revisions, original hashes, model-card MIT claim and a local-change ledger; copy the MIT text and preserve all upstream copyright notices actually present. Record the exception to the prior no-weight rule in `THIRD_PARTY_NOTICES.md`. Add an explicit `.gitignore` exception for `third_party/fpga_lpr/*.onnx`, keeping global `*.onnx` ignore intact elsewhere.

```python
def load_fpga_manifest(assets_dir: Path) -> FpgaLprManifest:
    data = json.loads((assets_dir / "manifest.json").read_text(encoding="utf-8"))
    if data.get("schema") != "fpga-lpr-onnx-v1":
        raise FpgaAssetError("unsupported external model schema")
    for name in ("cpm", "lprnet"):
        filename = {"cpm": "cpm.onnx", "lprnet": "lprnet.onnx"}[name]
        if sha256_file(assets_dir / filename) != data["components"][name]["sha256"]:
            raise FpgaAssetError(f"{name} sha256 mismatch")
    return FpgaLprManifest.from_document(data)
```

- [ ] **Step 4: Verify green.** Run the contract tests and `git diff --check`. Inspect the notice text against the pinned upstream README/model metadata; if a copied architecture dependency's license remains unclear, do not proceed to artifact check-in.
- [ ] **Step 5: Commit this independent contract.** Stage only the files in this task and commit `feat(lpr): add attributed external asset contract`.

### Task 2: Reproducible PyTorch-to-ONNX conversion and parity

**Files:** Create `third_party/fpga_lpr/model_arch.py`, `tools/build_fpga_lpr_onnx.py`, `tests/integration/test_fpga_lpr_export.py`, and generated `third_party/fpga_lpr/{manifest.json,cpm.onnx,lprnet.onnx}`; update `third_party/fpga_lpr/UPSTREAM.md` with converted hashes and exact local modifications.

**Interfaces:** Consumes the two pinned original `.pth` files via explicit `--weights-dir`; calls Task 1's manifest contract. Produces `verify_original_weights(weights_dir: Path) -> None` (checks both fixed SHA-256 digests before either load) and `export_fpga_lpr(weights_dir: Path, output_dir: Path) -> Path` returning the manifest path. No network or arbitrary notebook execution. Output names: CPM `stage`, `heatmap`; LPRNet `logits`. Batch axis is dynamic; CPM tensors `[B,4,50,50]`, LPRNet `[B,37,18]`.

- [ ] **Step 1: Write failing tests for source checks and output signatures.** `verify_original_weights(tmp_path)` must reject `best_val_loss.pth` with the wrong digest *before* `torch.load`. A checked-in artifact test opens the two ONNX sessions and asserts their names/shapes, including CPM's two outputs and the 37-class LPRNet output.

```python
with pytest.raises(ValueError, match="CPM source sha256 mismatch"):
    verify_original_weights(tmp_path)
assert [o.name for o in cpm_session.get_outputs()] == ["stage", "heatmap"]
assert lpr_session.get_outputs()[0].shape == ["batch", 37, 18]
```

- [ ] **Step 2: Verify red.** Run `.venv\Scripts\python.exe -m pytest -q tests/integration/test_fpga_lpr_export.py`; expect missing converter or missing checked-in artifacts.
- [ ] **Step 3: Vendor the six required model classes and `CHARS` from the pinned `model_utils.py` into the attributed `model_arch.py`, dropping unused notebook imports only.** Instantiate `CPMLicensePlateNet(num_stages=6)` and `LPRNet(lpr_max_len=7, phase=False, class_num=37, dropout_rate=0.5)`. Verify original hashes, load `weights_only=True`, export with opset 17 and dynamic batch to temporary files, validate ONNX signatures, then atomically place the ONNX files and a manifest containing their new hashes and the exact upstream `CHARS` tuple. The model manifest must identify original and converted revisions separately. The converter may import `model_arch.py`; the shipped adapter/release must not import it or require PyTorch.

```python
torch.onnx.export(cpm.eval(), (torch.zeros(1, 3, 100, 100),), cpm_path,
                  input_names=["input"], output_names=["stage", "heatmap"],
                  dynamic_axes={"input": {0: "batch"}, "stage": {0: "batch"}, "heatmap": {0: "batch"}},
                  opset_version=17, dynamo=False)
```

- [ ] **Step 4: Verify green and parity.** Run the integration tests, then run `.venv\Scripts\python.exe tools/build_fpga_lpr_onnx.py --weights-dir runs/external-lpr-bakeoff/upstream --output-dir third_party/fpga_lpr --verify-reference`. The script must compare PyTorch and ORT CPM heatmaps and LPR logits on fixed plate crops with `rtol=1e-3, atol=1e-4`, assert identical greedy strings, and emit a local report under ignored `runs/fpga-lpr-parity/`. Check model load and batch 1/2; do not describe a skipped reference check as passing.
- [ ] **Step 5: Commit the exact vetted artifacts.** Verify each file is under GitHub's normal file-size limit, stage only `third_party/fpga_lpr/` and the converter/test files, and inspect `git diff --cached --name-only` to exclude `.pth`, datasets, and unrelated binaries. Commit `feat(lpr): vendor pinned MIT ONNX pair with parity evidence`.

### Task 3: Plate-ROI recognizer with explicit author and safe modes

**Files:** Create `src/plateai_reader/fpga_lpr.py`, `tests/unit/test_fpga_lpr_adapter.py`; update `src/plateai_reader/__init__.py` only for intended public exports.

**Interfaces:** Consumes `FpgaLprManifest` and two ONNX sessions from Tasks 1–2. Produces `FpgaLprRecognizer(assets_dir: Path, providers: Sequence[str] | None = None, session_factory: SessionFactory | None = None)` with `recognize(roi_rgb: NDArray[np.uint8], *, corner_policy: Literal["compat", "safe"] = "compat") -> FpgaLprRead`. `FpgaLprRead` contains `raw_text`, `normalized_text`, `aligned_rgb`, `roi_corners_xy`, `timings_ms`, and `score_kind="uncalibrated"`; reject invalid geometry with `FpgaLprError(reason)`. Export a runtime `FPGA_CHARS` tuple matching the validated manifest, without importing `third_party/fpga_lpr/model_arch.py` or `torch`.

- [ ] **Step 1: Write failing tests for decode, image contract, and geometry.** A class sequence `[10,36,10,36]` decodes to two copies of `FPGA_CHARS[10]`, `[10,10,36]` to one; final class 36 never prints `-`. The local test helper `logits_for(indices)` fills a `(37,18)` float tensor with a low baseline and sets the listed class to the maximum at each timestep (remaining positions blank). Invalid uint8/RGB shape raises `FpgaLprError("invalid_roi")`. Use fake ONNX sessions returning known `[1,4,50,50]` heatmaps and `[1,37,18]` logits; duplicate/collinear corner positions in safe mode raise `FpgaLprError("invalid_corners")`, while a shuffled valid quadrilateral is reordered and accepted. A deliberately asymmetric red/blue ROI must show that the CPM input receives OpenCV BGR channel order.

```python
assert decode_fpga_logits(logits_for([10, 36, 10, 36]), FPGA_CHARS) == FPGA_CHARS[10] * 2
assert decode_fpga_logits(logits_for([10, 10, 36]), FPGA_CHARS) == FPGA_CHARS[10]
```

- [ ] **Step 2: Verify red.** Run `.venv\Scripts\python.exe -m pytest -q tests/unit/test_fpga_lpr_adapter.py`; expect missing adapter functions/classes.
- [ ] **Step 3: Implement the author-compatible inference path.** Convert the public RGB ROI to OpenCV BGR first, then resize to 100×100, divide by 255, transpose to NCHW; run CPM, bilinearly upsample its final heatmap to 100×100 with the same coordinate rule, derive four points in upstream sum/difference order, warp **BGR** to 94×48, normalize `(pixel - 127.5) * 0.0078125`, run LPRNet and collapse repeat/blank with final index 36. Convert the display crop back to RGB. The named `safe` policy applies `normalize_corners(..., (100,100))` before warp and rejects unsafe quadrilaterals. Check tensor names, dtypes, finite values and shapes before indexing; do not use the native CTC decoder or claim calibrated confidence. `collapse_ctc` starts with a sentinel previous value, emits only nonblank classes differing from the previous timestep, and updates previous on **every** timestep (including blank).

```python
def decode_fpga_logits(logits: np.ndarray, chars: Sequence[str]) -> str:
    if logits.shape != (37, 18) or len(chars) != 37:
        raise FpgaLprError("invalid_lpr_logits")
    indices = logits.argmax(axis=0)
    return "".join(chars[i] for i in collapse_ctc(indices, blank_index=36))
```

- [ ] **Step 4: Verify green.** Run adapter tests, then compare `compat` decoded strings on selected known TLPD plate crops with the ignored prior PyTorch replay. Record mode differences separately; do not demand `safe` equal `compat` where topology defense intentionally changes corners.
- [ ] **Step 5: Commit only adapter and tests.** `git add src/plateai_reader/fpga_lpr.py src/plateai_reader/__init__.py tests/unit/test_fpga_lpr_adapter.py` and commit `feat(lpr): recognize plate ROIs with attributed ONNX pair`.

### Task 4: Compose native detection with per-plate external recognition

**Files:** Create `src/plateai_reader/fpga_pipeline.py`, `tests/integration/test_fpga_lpr_pipeline.py`.

**Interfaces:** Consumes an object implementing `detect(image_rgb) -> DetectionResult` and Task 3's recognizer. Produces `FpgaSceneReader(detector, recognizer, roi_margin: float = 0.06)` and `read(image_rgb) -> FpgaSceneResult` with ordered `plates`, `rejections`, detector/recognizer IDs, and per-stage timings. `FpgaScenePlate` has `detection: PlateDetection`, `read: FpgaLprRead`, and a `text` property forwarding `read.normalized_text`; `FpgaSceneRejection` has `detection`, `stage`, and `reason`. `crop_box(image_rgb, bbox_xyxy, margin)` returns a nonempty RGB view or raises `FpgaLprError("invalid_bbox")`. Uses original detected bbox/polygon for overlays; CPM corners are stored separately as ROI-local evidence.

- [ ] **Step 1: Write failing multi-plate and edge tests.** A fake detector returns two `PlateDetection` objects; a fake recognizer rejects the first ROI and returns `ABC1234` on the second. Assert one result and one rejection in input order. Also assert a bbox touching x=0 or y=0 clips safely, a zero-width box is rejected, and empty detector output does not invoke OCR or a contour fallback.

```python
result = FpgaSceneReader(detector, recognizer).read(rgb)
assert [p.text for p in result.plates] == ["ABC1234"]
assert [r.reason for r in result.rejections] == ["invalid_corners"]
```

- [ ] **Step 2: Verify red.** Run `.venv\Scripts\python.exe -m pytest -q tests/integration/test_fpga_lpr_pipeline.py`; expect missing `FpgaSceneReader`.
- [ ] **Step 3: Implement bounded ROI extraction and independent error handling.** Expand bbox by 6% per axis, clip to `[0,width]×[0,height]`, require a nonempty slice, pass one RGB ROI to the recognizer, and retain all detector confidence/corner data. Catch only expected `FpgaLprError` per plate; a detector-wide failure returns an explicit pipeline error, not a fabricated recognition or heuristic retry.

```python
for detection in detected.detections:
    try:
        roi = crop_box(image_rgb, detection.bbox_xyxy, margin=self.roi_margin)
        plates.append(FpgaScenePlate(detection, self.recognizer.recognize(roi)))
    except FpgaLprError as error:
        rejections.append(FpgaSceneRejection(detection, error.reason))
```

- [ ] **Step 4: Verify green.** Run the integration tests and the native `tests/integration/test_reader_detection_rectification.py` regression. A bad plate must not erase another plate; changing the margin must never index outside the source image.
- [ ] **Step 5: Commit the composition.** Stage the two files and commit `feat(lpr): compose native detections with external OCR`.

### Task 5: Preserve Web defaults while adding an allowlisted A/B choice

**Files:** Create `src/plateai_web/external_predictor.py`; modify `src/plateai_web/app.py`, `web/index.html`, `web/js/app.js`, `tests/test_web_api.py`, `tests/unit/test_web_predictor.py`. Inspect all four existing dirty files before editing; preserve unrelated user changes.

**Interfaces:** `ExternalPredictorEngine(root: Path)` owns cached validated external sessions and a native `PlateReader.detect` source. `_fpga_lpr_predictor() -> ExternalPredictorEngine` caches a fixed instance for `ROOT`; it accepts no user path. `predict_image(image_bytes: bytes) -> dict` uses the same top-level response keys as native Web prediction, with `diagnostics.recognizer_type="fpga-lpr-mit"`, `locator_type="plate_pose_net"`, `score_kind="uncalibrated"`, and no false confidence percentage. `/api/predict/compare` accepts `candidate_kind: Literal["native-preview", "fpga-lpr-mit"] = Form("native-preview")`; the old request/response defaults remain unchanged.

- [ ] **Step 1: Write failing API/predictor tests.** Submit multipart `candidate_kind=fpga-lpr-mit` and assert active and external both receive identical bytes, `preview.activation_changed is False`, and active files remain unchanged. Submit `candidate_kind=../../models/bundles/active-v1` and expect HTTP 422 without invoking any candidate. With a missing/corrupt external asset, assert active result remains available and candidate returns `model_error` diagnostics rather than HTTP 500. Assert `/api/model/activate` cannot activate `fpga-lpr-mit` assets. Verify the existing no-kind test still returns `candidate-detector-real-v1`.

```python
response = client.post("/api/predict/compare", data={"candidate_kind": "fpga-lpr-mit"}, files=files)
assert response.json()["preview"]["activation_changed"] is False
assert client.post("/api/predict/compare", data={"candidate_kind": "../active-v1"}, files=files).status_code == 422
```

- [ ] **Step 2: Verify red.** Run `.venv\Scripts\python.exe -m pytest -q tests/test_web_api.py tests/unit/test_web_predictor.py`; expect the new selection/diagnostic assertions to fail while old tests remain green.
- [ ] **Step 3: Add a fixed engine and UI choice.** Cache one `ExternalPredictorEngine` for the pinned `third_party/fpga_lpr` assets, using the existing native detector only locally. Route the explicit enum to that engine; preserve `_candidate_predictor()` for the existing preview. Add a candidate select beside the current switch and append its literal value to `FormData` only in compare mode. Show a `not independently validated` warning, source links, recognizer ID, crop text, and rejected plate reasons. Do not change `predictor = PredictorEngine()` or copy any external files to `active-v1`.

```python
candidate = _candidate_predictor() if candidate_kind == "native-preview" else _fpga_lpr_predictor()
candidate_result = candidate.predict_image(image_bytes)
return {"active": active_result, "candidate": candidate_result,
        "preview": {"active_bundle": "active-v1", "candidate_bundle": CANDIDATE_BUNDLE if candidate_kind == "native-preview" else "fpga-lpr-mit",
                    "activation_changed": False}}
```

- [ ] **Step 4: Verify green.** Run the named tests, `node --check web/js/app.js`, and the full `tests/test_web_api.py` suite. Manually upload a plate crop and a full-scene photo in the local browser; note that full-scene localization quality remains a separate measured stage. Do not claim browser behavior from API tests alone.
- [ ] **Step 5: Commit only owned hunks.** Review `git diff -- web/index.html web/js/app.js src/plateai_web/app.py` for user changes before staging. Stage the intended paths/hunks without `git add -A`, then commit `feat(web): compare maintained MIT OCR without activation`.

### Task 6: Audited evaluation and one-factor improvement loop

**Files:** Create `src/plateai_eval/fpga_protocol.py`, `tools/evaluate_fpga_lpr.py`, `tests/unit/test_fpga_lpr_protocol.py`; update `docs/evaluation-sources.md`, `history.md` only with observed evidence. Store image manifests, per-image output, and any tuned thresholds under ignored `runs/fpga-lpr-eval/`.

**Interfaces:** `AuditedPlate` holds `image: Path`, `sha256`, `canonical`, `vehicle_class`, `split` (`dev` or `holdout`), `crop_xyxy`, and `source_kind` (`audited_unseen` or `training_source_replay`). `load_audited_manifest(path: Path) -> tuple[AuditedPlate, ...]` requires those JSONL fields except that `source_kind` defaults to `audited_unseen` only for manually supplied rows. `EvalReport` holds overall and v1-passenger exact/empty/total counts, per-image errors, per-stage timings, hardware/provider and source-kind provenance. `evaluate_entries(reader, entries, mode) -> EvalReport` never upgrades filename-derived TLPD strings to audited labels.

- [ ] **Step 1: Write failing protocol tests.** A manifest with the same `canonical` or identical image SHA-256 across dev/holdout raises `EvaluationInputError("cross_split_overlap")`; missing `vehicle_class` or unreadable crop raises a named input error. A fake reader's two rows must yield exactly one exact match and one empty result, with separate v1-passenger counts. A TLPD replay row marked `source_kind="training_source_replay"` must print that label, not `held_out`.

```python
with pytest.raises(EvaluationInputError, match="cross_split_overlap"):
    load_audited_manifest(manifest_path)
assert report.v1_passenger.exact == 1
```

- [ ] **Step 2: Verify red.** Run `.venv\Scripts\python.exe -m pytest -q tests/unit/test_fpga_lpr_protocol.py`; expect the new module contract to be absent.
- [ ] **Step 3: Implement the offline evaluator.** Verify each image SHA-256 before reading pixels, normalize expected text without confusing vehicle class with a 3-4 regex, count exact strings and empty outputs, write JSONL plus summary atomically under an explicit output directory, and record hardware/provider/timing scope. CLI flags choose `corner_policy=compat|safe` and `roi_margin=0.06|0.10`; run one factor at a time against the **dev** split. Read holdout once only after selecting settings, never use holdout to choose a setting. If no audited unseen set exists, print `independent_accuracy=pending` and stop short of promotion.

```python
if {r.canonical for r in dev} & {r.canonical for r in holdout}:
    raise EvaluationInputError("cross_split_overlap")
report = evaluate_entries(reader, entries, mode="crop")
```

- [ ] **Step 4: Verify green and collect only defensible evidence.** Run protocol tests, the 3,032-image TLPD compatibility replay (label it training-source), and the available manually checked independent photos. Keep `MDX-9717` in motorcycle diagnostics, not the private-passenger v1 score. Compare `compat` versus `safe`, then ROI margin, with per-image changes; do not add CLAHE or fine-tune weights unless dev-set evidence and rights support that next experiment.
- [ ] **Step 5: Commit evaluator, not private photos/results.** Stage only evaluator code, tests and evidence-qualified docs/history; commit `test(lpr): add leakage-safe real-photo scorecards`.

### Task 7: Separate MIT crop-OCR release from the source-only full-scene scaffold

**Files:** Create `src/plateai_web/mit_lpr_crop_api.py`, `tools/build_mit_lpr_crop_release.py`, `tests/unit/test_mit_lpr_crop_release.py`; update `THIRD_PARTY_NOTICES.md` and the new release README template. Do **not** overwrite the already-dirty `release/3wa_plate_api` files or change `src/plateai_web/release_packager.py` in this task.

**Interfaces:** `create_crop_api(recognizer: FpgaLprRecognizer | None) -> FastAPI` exposes `GET /api/health` with `inference_ready` and `POST /api/recognize/crop` accepting a supplied plate-centric image; `None` represents a failed asset load and must yield a 503, not a silent fallback. `decode_limited_rgb(data: bytes) -> NDArray[np.uint8]` rejects excessive bytes/pixels before RGB conversion, returning a 400 on malformed images. `build_mit_lpr_crop_release(root: Path, destination: Path) -> Path` creates `out/mit-lpr-crop-release` with an allowlisted copy of `fpga_assets.py`, `fpga_lpr.py`, `rectifier.py`, `mit_lpr_crop_api.py`, minimal empty package initializers, `app.py` launcher, pinned runtime requirements, the two validated ONNX files, manifest, repository `LICENSE`, upstream MIT notice and attribution. The test helper `packaged_hash(name)` hashes `destination / "third_party" / "fpga_lpr" / name`. It never includes other project modules, native detector artifacts, `.pth`, datasets or a fabricated full-scene `/api/predict` result.

- [ ] **Step 1: Write failing service and package tests.** With a fake recognizer, a valid crop returns text plus uncalibrated score-kind; a malformed image returns 400; missing/corrupt ONNX makes health `inference_ready=false` and inference return 503. A built package contains exactly the two ONNX files matching manifest hashes and both notices. `POST /api/predict` must not claim full-scene inference: return 503 with an explicit crop-only explanation.

```python
assert client.post("/api/recognize/crop", files=files).json()["raw_text"] == "ABC1234"
assert client.post("/api/predict", files=files).status_code == 503
assert packaged_hash("cpm.onnx") == manifest.components["cpm"].sha256
```

- [ ] **Step 2: Verify red.** Run `.venv\Scripts\python.exe -m pytest -q tests/unit/test_mit_lpr_crop_release.py`; expect missing app/builder contracts.
- [ ] **Step 3: Implement the bounded crop API and package assembly.** Decode uploads with OpenCV, enforce image size/byte limits, call Task 3's recognizer, and return only JSON-safe scalar fields (`raw_text`, `normalized_text`, `model_id`, `score_kind`, `timings_ms`); do not serialize NumPy crop/corner arrays. Assemble a minimal source package from the exact module allowlist in the interface; use empty `plateai_reader/__init__.py` and `plateai_web/__init__.py` inside the package so the existing package-level imports cannot pull in the native detector or trainer. Include runtime-only pinned requirements and an `app.py` launcher that imports `create_crop_api`, validates adjacent assets, and never executes a notebook or loads `.pth`. Refuse nonempty/symlink targets rather than deleting user data. Generate a README stating that the service recognizes already-cropped plates and cannot locate a plate in a whole vehicle image. The existing 1788 scaffold and its source-only meaning remain unchanged.

```python
@app.post("/api/recognize/crop")
async def recognize_crop(file: UploadFile = File(...)):
    rgb = decode_limited_rgb(await file.read())
    read = recognizer.recognize(rgb)
    return {"raw_text": read.raw_text, "normalized_text": read.normalized_text,
            "model_id": "fpga-lpr-mit-v1", "score_kind": read.score_kind,
            "timings_ms": read.timings_ms}
```

- [ ] **Step 4: Verify green and regression.** Run the new tests, existing `tests/unit/test_release_packager.py`, `tests/test_web_api.py`, full `.venv\Scripts\python.exe -m pytest -q`, `node --check web/js/app.js`, and `git diff --check`. Build the crop package locally, install/run it in an isolated environment, call health and one valid/invalid crop, then inspect the package file inventory and source/weight licenses. Treat Web browser, IIS, and production as separate unverified boundaries.
- [ ] **Step 5: Commit without publishing.** Stage only the crop-service code/tests/notices and evidence-qualified `history.md`; commit `feat(release): package attributed MIT crop OCR service`. Push or publish only on explicit user request. A distributable whole-scene service remains a separate later task requiring a detector with reviewed redistribution rights.

## Final Review and Handoff

- Re-read the spec requirement by requirement: two model hashes/parity (Tasks 1–2), unlike-v1 adapter (Task 3), per-plate full-scene composition (Task 4), Web A/B without activation (Task 5), independent real-photo evaluation/tuning (Task 6), and attributed crop-only release (Task 7). Do not promote `active-v1` unless new independent evidence warrants a separately authorized change.
- Inspect `git diff --name-status`, `git ls-files third_party/fpga_lpr`, and release inventory for accidental TLPD images, `.pth`, native/AGPL detector weights, or unrelated dirty files. Report the exact committed ONNX hashes and source revisions.
- Report local unit/integration/full-suite results separately from real-photo OCR and full-scene scores. If audited unseen photos are insufficient, deliver the working candidate but leave its real-world accuracy and default-promotion gate pending; do not substitute the TLPD 98.09% replay.
