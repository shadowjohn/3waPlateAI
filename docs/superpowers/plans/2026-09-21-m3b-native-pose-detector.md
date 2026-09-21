# M3b Native Multi-Plate Pose Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native-PyTorch, multi-plate detector whose ONNX candidates are deterministically postprocessed into independently rectified M3a crops.

**Architecture:** A shared OpenCV-golden letterbox contract feeds both the native `PlatePoseNet` trainer and ONNX Reader. The Trainer creates transactional 1-3-instance synthetic composite datasets and a small anchor-free P3/P4/P5 pose model; the Reader runs deterministic NumPy NMS, inverse-maps every bbox and corner, then isolates each M3a rectification failure. A detector ONNX is merged atomically with an existing validated M2 crop bundle into a full local bundle.

**Tech Stack:** CPython 3.11, PyTorch 2.14.0, NumPy 2.4.2, OpenCV 5.0.0.93, ONNX 1.23.0, ONNX Runtime 1.30.0, pytest, Hypothesis.

**Spec:** `docs/superpowers/specs/2026-09-21-m3b-native-pose-detector-design.md`

## File structure

| Path | Responsibility |
|---|---|
| `src/plateai_shared/detection.py` | Exact 640 letterbox affine, inverse/clip helpers, immutable detector value objects. |
| `src/plateai_trainer/detection/composite.py` | Transactional multi-instance synthetic compositor and background-manifest validation. |
| `src/plateai_trainer/detection/model.py` | Native CSP-tiny/FPN pose network, candidate decode, targets, and loss. |
| `src/plateai_trainer/detection/dataset.py` | Validated composite-dataset loading and batch collation. |
| `src/plateai_trainer/detection/engine.py` | Deterministic CPU-first detector training and checkpoint/report publication. |
| `src/plateai_trainer/detection/export.py` | Detector ONNX parity checks and full-bundle staging/merge. |
| `src/plateai_reader/detector.py` | NumPy NMS, inverse letterbox, and isolated M3a handoff. |
| `schemas/*.schema.json` and `src/plateai_shared/bundle.py` | Composite schemas plus full-bundle contract/hash validation. |

## Global Constraints

- Use only repository-owned native PyTorch/NumPy detector code; do not add `ultralytics`, its models, its checkpoints, or copied code.
- The detector input golden reference is OpenCV 5.0.0.93 RGB `INTER_LINEAR` letterbox to 640x640 with `floor(value + 0.5)` resize rounding, `(114,114,114)` padding, NCHW `float32 / 255.0`, and dynamic batch only.
- The sole ONNX candidate output is `float32 [batch,8400,13]`, in the exact `cx,cy,w,h,confidence,LT,RT,RB,LB` order defined by the M3b spec. Export ONNX opset 17 with no NMS node.
- Every Reader inverse transform applies to the bbox and all eight corner values, then clips public source values to `[0,width-1] x [0,height-1]` before M3a.
- Synthetic composites contain one through three non-overlapping, fully in-frame v1 plates. Real backgrounds stay user-supplied outside Git; tests use procedural toy backgrounds only.
- Generated datasets, checkpoints, ONNX, reports, full bundles, and TensorRT artifacts remain ignored. Publication never replaces an existing destination.
- Preserve M1/M2 behavior, the 380x160 M2 crop contract, M3a `convex-hull-semantic-v1`, and the existing 33-symbol CTC manifest contract.

## Review Focus

1. An odd-padding, non-square source must invert all four corners exactly enough to remain in-frame; Task 1 and Task 5.
2. A 1/2/3-instance composite must preserve each semantic GT corner and cannot silently overlap instances; Task 2.
3. Equal-confidence candidates must have stable candidate-index tie breaking, while an invalid candidate must not hide an independent valid plate; Task 5.
4. Train and validation datasets reusing even one background SHA-256 must be rejected before training; Task 4.
5. A full bundle with wrong detector IO, missing postprocess data, or an undeclared detector file must be rejected; Task 6.

---

### Task 1: Shared detector letterbox and source-coordinate contract

**Files:**
- Create: `src/plateai_shared/detection.py`
- Modify: `src/plateai_shared/__init__.py`
- Create: `tests/unit/test_detection_contract.py`

**Interfaces:** Produces `LetterboxTransform`, `PlateDetection`, `letterbox_rgb_v1(image_rgb) -> tuple[NDArray[np.float32], LetterboxTransform]`, `map_points_to_letterbox(points_xy, transform)`, and `inverse_and_clip_points(points_xy, transform)`. Tasks 2, 4, and 5 consume these exact names.

- [ ] **Step 1: Write the failing contract tests**

```python
def test_letterbox_v1_has_golden_odd_padding_and_round_trip_corners():
    image = np.zeros((333, 1000, 3), dtype=np.uint8)
    tensor, transform = letterbox_rgb_v1(image)

    assert tensor.shape == (3, 640, 640)
    assert transform.resized_size_wh == (640, 213)
    assert transform.padding_ltrb == (0, 213, 0, 214)
    points = np.float32([[0, 0], [999, 0], [999, 332], [0, 332]])
    np.testing.assert_allclose(
        inverse_and_clip_points(map_points_to_letterbox(points, transform), transform),
        points,
        atol=1e-5,
    )

def test_letterbox_rejects_non_uint8_or_non_rgb_image():
    with pytest.raises(ValueError, match="uint8 RGB"):
        letterbox_rgb_v1(np.zeros((640, 640), dtype=np.uint8))
```

- [ ] **Step 2: Verify RED**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_contract.py -q`

Expected: FAIL because `plateai_shared.detection` does not exist.

- [ ] **Step 3: Implement the minimal shared contract**

```python
@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    source_size_wh: tuple[int, int]
    resized_size_wh: tuple[int, int]
    padding_ltrb: tuple[int, int, int, int]
    scale: float

def inverse_and_clip_points(points_xy, transform) -> NDArray[np.float32]:
    source_w, source_h = transform.source_size_wh
    points = (np.asarray(points_xy, dtype=np.float32) - np.float32([transform.pad_left, transform.pad_top])) / transform.scale
    points[:, 0] = np.clip(points[:, 0], 0.0, float(source_w - 1))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(source_h - 1))
    return points
```

Use `math.floor(value + 0.5)`, `cv2.copyMakeBorder`, raw pad `(114,114,114)`, and a contiguous CHW float32 output. Reject empty dimensions and non-finite input coordinates with stable `ValueError` messages.

- [ ] **Step 4: Verify GREEN**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_contract.py -q`

Expected: PASS with golden padding, round trip, clipping, and invalid-image cases green.

- [ ] **Step 5: Commit**

```bash
git add src/plateai_shared/detection.py src/plateai_shared/__init__.py tests/unit/test_detection_contract.py
git commit -m "feat: add detector letterbox contract"
```

### Task 2: Transactional 1-3-instance composite data

**Files:**
- Create: `configs/detection/composite_v1.json`
- Create: `schemas/detection_background_manifest.schema.json`
- Create: `schemas/detection_metadata.schema.json`
- Create: `src/plateai_trainer/detection/__init__.py`
- Create: `src/plateai_trainer/detection/contracts.py`
- Create: `src/plateai_trainer/detection/composite.py`
- Create: `src/plateai_trainer/detection/cli.py`
- Create: `tests/unit/test_detection_composite.py`
- Create: `tests/contract/test_detection_schemas.py`
- Modify: `pyproject.toml`

**Interfaces:** Produces `CompositeGenerationRequest`, `CompositeInstance`, `CompositeGenerationSummary`, and `generate_composite_dataset(request)`. Tasks 3 and 4 consume the published `images/`, `metadata.jsonl`, `generation_config.json`, and `summary.json` layout. The installed command is `plateai-compose`.

- [ ] **Step 1: Write the failing composite and schema tests**

```python
@pytest.mark.parametrize("instances", [1, 2, 3])
def test_composite_records_exact_semantic_instances(tmp_path, background_manifest, instances):
    output = tmp_path / f"composite-{instances}"
    summary = generate_composite_dataset(
        CompositeGenerationRequest(
            output=output, count=1, seed=7, background_manifest=background_manifest,
            instances_per_image=(instances, instances),
        )
    )
    record = json.loads((output / "metadata.jsonl").read_text(encoding="utf-8"))

    assert summary.generated == 1
    assert len(record["instances"]) == instances
    assert max_pairwise_bbox_iou(record["instances"]) == 0.0
    assert all(is_semantic_quad(item["corners"]) for item in record["instances"])

def test_composite_rejects_existing_output_without_deleting_it(tmp_path, background_manifest):
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(OutputExistsError):
        generate_composite_dataset(
            CompositeGenerationRequest(output=output, count=1, seed=1, background_manifest=background_manifest)
        )
```

Add schema tests for a missing background SHA-256, five corners, and an out-of-frame corner.

- [ ] **Step 2: Verify RED**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_composite.py tests/contract/test_detection_schemas.py -q`

Expected: FAIL because the detection package and schemas do not exist.

- [ ] **Step 3: Implement deterministic composition and publication**

```python
@dataclass(frozen=True, slots=True)
class CompositeInstance:
    bbox_xyxy: tuple[float, float, float, float]
    corners_xy: NDArray[np.float32]  # [LT, RT, RB, LB]
    source_plate: Mapping[str, JsonValue]

def generate_composite_dataset(request: CompositeGenerationRequest) -> CompositeGenerationSummary:
    backgrounds = _load_and_hash_background_manifest(request.background_manifest)
    rng = np.random.default_rng(request.seed)
    staging = _create_owned_staging(request.output)
    try:
        records = [
            _compose_one_image(backgrounds, rng, request, staging / "images", index)
            for index in range(request.count)
        ]
        _write_jsonl(staging / "metadata.jsonl", records)
        _write_generation_provenance(staging, request, backgrounds)
        publish_directory_no_replace(staging, request.output)
    except BaseException:
        remove_owned_staging(staging, request.output)
        raise
    return CompositeGenerationSummary(generated=request.count, output=request.output)
```

The background manifest contains manifest-relative image paths and SHA-256 values; resolve and hash-check every background before use. Render with the existing M1 v1 template/font, sample one through three instances, compose with known homographies, derive boxes from transformed corners, and record source-M1 identity plus transforms. Retry a placement only when all four corners are in frame and bbox IoU with every accepted instance is `<= 0.0`. Use `publish_directory_no_replace` and `remove_owned_staging`; add `plateai-compose = "plateai_trainer.detection.cli:main"` to `[project.scripts]`.

- [ ] **Step 4: Verify GREEN**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_composite.py tests/contract/test_detection_schemas.py -q`

Expected: PASS for exact 1/2/3 instance counts, deterministic metadata/hash output, semantic corners, non-overlap, transactional refusal, and malformed schema rejection.

- [ ] **Step 5: Commit**

```bash
git add configs/detection/composite_v1.json schemas/detection_background_manifest.schema.json schemas/detection_metadata.schema.json src/plateai_trainer/detection/contracts.py src/plateai_trainer/detection/composite.py src/plateai_trainer/detection/cli.py src/plateai_trainer/detection/__init__.py pyproject.toml tests/unit/test_detection_composite.py tests/contract/test_detection_schemas.py
git commit -m "feat: generate multi-plate synthetic composites"
```

### Task 3: Native anchor-free pose network, targets, and loss

**Files:**
- Create: `src/plateai_trainer/detection/model.py`
- Create: `src/plateai_trainer/detection/targets.py`
- Create: `src/plateai_trainer/detection/loss.py`
- Create: `tests/unit/test_detection_model.py`

**Interfaces:** Produces `PlatePoseNet`, `DetectionTargets`, `assign_detection_targets(instances)`, and `detection_loss(predictions, targets) -> DetectorLoss`. Tasks 4 and 6 consume `PlatePoseNet` and checkpoint metadata.

- [ ] **Step 1: Write failing model, assignment, and gradient tests**

```python
def test_plate_pose_net_emits_fixed_candidate_layout():
    candidates = PlatePoseNet()(torch.zeros((2, 3, 640, 640), dtype=torch.float32))
    assert tuple(candidates.shape) == (2, 8400, 13)
    assert torch.all((0.0 <= candidates[..., 4]) & (candidates[..., 4] <= 1.0))

def test_assigner_uses_expected_pyramid_level_and_semantic_corner_order():
    instance = instance_with_short_side(80.0)
    targets = assign_detection_targets([instance])
    assert targets.positive_level_indices.tolist() == [1]  # P4
    np.testing.assert_array_equal(targets.corners_xy[0], instance.corners_xy)

@pytest.mark.parametrize("short_side, expected_level", [(63.0, 0), (64.0, 1), (127.0, 1), (128.0, 2)])
def test_assigner_uses_inclusive_scale_boundaries(short_side, expected_level):
    targets = assign_detection_targets([instance_with_short_side(short_side)])
    assert targets.positive_level_indices.tolist() == [expected_level]

def test_assigner_breaks_overlapping_positive_cell_ties_by_smaller_area_then_instance_index():
    targets = assign_detection_targets(overlapping_instances_with_equal_area())
    assert targets.matched_instance_indices[target_cell_index()] == 0

def test_detector_loss_is_finite_and_updates_a_parameter():
    model = PlatePoseNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = next(model.parameters()).detach().clone()
    result = detection_loss(model(torch.rand((1, 3, 640, 640))), synthetic_targets())
    optimizer.zero_grad(); result.total.backward(); optimizer.step()
    assert math.isfinite(float(result.total))
    assert not torch.equal(before, next(model.parameters()).detach())
```

- [ ] **Step 2: Verify RED**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_model.py -q`

Expected: FAIL because the model, target assigner, and loss are absent.

- [ ] **Step 3: Implement the model and exact training math**

```python
class PlatePoseNet(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _validate_detector_images(images)
        p3, p4, p5 = self.neck(self.backbone(images))
        return torch.cat(
            (self._decode(p3, stride=8), self._decode(p4, stride=16), self._decode(p5, stride=32)),
            dim=1,
        )

def detection_loss(predictions: torch.Tensor, targets: DetectionTargets) -> DetectorLoss:
    return DetectorLoss(
        total=objectness + 5.0 * box_ciou + 2.0 * corner_smooth_l1,
        objectness=objectness,
        box_ciou=box_ciou,
        corner_smooth_l1=corner_smooth_l1,
    )
```

Use a repository-owned CSP-tiny backbone and FPN/PAN with P3 80x80, P4 40x40, and P5 20x20 grids. Decode sole-class confidence through sigmoid. Assign from letterbox-pixel bbox short side: P3 `<64`, P4 `64-127`, P5 `>=128`; positives lie inside the bbox and at Chebyshev grid distance at most one from its centre cell. When multiple GTs nominate a cell, choose the smaller bbox area and then the lower metadata instance index. Normalize each corner residual by matched bbox width/height and calculate corner loss only for positives.

- [ ] **Step 4: Verify GREEN**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_model.py -q`

Expected: PASS with 8,400 candidates, exact P3/P4/P5 boundary behavior, bounded confidence, finite loss/gradient, and a changed parameter.

- [ ] **Step 5: Commit**

```bash
git add src/plateai_trainer/detection/model.py src/plateai_trainer/detection/targets.py src/plateai_trainer/detection/loss.py tests/unit/test_detection_model.py
git commit -m "feat: add native plate pose model and loss"
```

### Task 4: Validated detector dataset, CPU training, and local CLI

**Files:**
- Create: `src/plateai_trainer/detection/dataset.py`
- Create: `src/plateai_trainer/detection/engine.py`
- Create: `src/plateai_trainer/detection/train_cli.py`
- Create: `tests/unit/test_detection_dataset.py`
- Create: `tests/integration/test_detection_training_cli.py`
- Modify: `pyproject.toml`

**Interfaces:** Consumes Task 1 affine helpers, Task 2 composite layout, and Task 3 model/loss. Produces `DetectionDataset`, `DetectorTrainingConfig`, `train_detector(config) -> DetectorTrainingRun`, `validate_train_validation_pair(train, validation)`, and installed `plateai-detect-train`. Task 6 consumes its checkpoint and report.

- [ ] **Step 1: Write failing data-isolation, training, and CLI tests**

```python
def test_detection_dataset_rejects_shared_train_validation_background_hash(train_dir, validation_dir):
    copy_background_hash(train_dir / "metadata.jsonl", validation_dir / "metadata.jsonl")
    with pytest.raises(DetectionDataError, match="background SHA-256"):
        validate_train_validation_pair(DetectionDataset(train_dir), DetectionDataset(validation_dir))

def test_detector_training_smoke_writes_checkpoint_and_finite_report(train_dir, validation_dir, tmp_path):
    run = train_detector(
        DetectorTrainingConfig(train_dir, validation_dir, tmp_path / "run", epochs=2, batch_size=1, seed=11)
    )
    assert run.best_checkpoint.is_file()
    assert math.isfinite(run.report["train"]["loss"])
    assert run.report["overfit"]["last_loss"] < run.report["overfit"]["first_loss"]

def test_installed_plateai_detect_train_shows_help():
    result = subprocess.run([detect_train_exe(), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
```

- [ ] **Step 2: Verify RED**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_dataset.py tests/integration/test_detection_training_cli.py -q`

Expected: FAIL because the dataset, trainer, and CLI do not exist.

- [ ] **Step 3: Implement deterministic dataset loading and training publication**

```python
@dataclass(frozen=True, slots=True)
class DetectorTrainingConfig:
    train_directory: Path
    validation_directory: Path
    output_directory: Path
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "cpu"

def train_detector(config: DetectorTrainingConfig) -> DetectorTrainingRun:
    train_dataset = DetectionDataset(config.train_directory)
    validation_dataset = DetectionDataset(config.validation_directory)
    validate_train_validation_pair(train_dataset, validation_dataset)
    _seed_detector_training(config.seed)
    return _train_and_publish_detector_run(config, train_dataset, validation_dataset)
```

Validate image hash, metadata schema, 1-3 instance limit, in-frame corners, semantic order, and exact bbox enclosure before tensor conversion. Use Task 1 letterboxing for pixels and every target coordinate. Pin Torch CPU execution to one thread as M2 does; use a seeded `DataLoader` with `num_workers=0`; atomically publish `best.pt` and `report.json`. The report records train loss, validation bbox AP50, average 640-pixel corner error, NMS complete-quad precision/recall, rectifier acceptance, per-stratum counts, all input/config hashes, and fixed overfit-smoke losses. Add `plateai-detect-train = "plateai_trainer.detection.train_cli:main"`.

- [ ] **Step 4: Verify GREEN**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_dataset.py tests/integration/test_detection_training_cli.py tests/unit/test_detection_model.py -q`

Expected: PASS for tampered data rejection, disjoint background enforcement, finite deterministic training output, decreasing overfit smoke loss, checkpoint/report publication, and CLI help.

- [ ] **Step 5: Commit**

```bash
git add src/plateai_trainer/detection/dataset.py src/plateai_trainer/detection/engine.py src/plateai_trainer/detection/train_cli.py pyproject.toml tests/unit/test_detection_dataset.py tests/integration/test_detection_training_cli.py
git commit -m "feat: train native plate pose detector"
```

### Task 5: NumPy NMS, source-coordinate recovery, and isolated rectification

**Files:**
- Create: `src/plateai_reader/detector.py`
- Modify: `src/plateai_reader/__init__.py`
- Create: `tests/unit/test_reader_detector.py`
- Create: `tests/integration/test_reader_detection_rectification.py`

**Interfaces:** Consumes `LetterboxTransform`, `PlateDetection`, M3a `rectify_plate`, and Task 3/6 candidate arrays. Produces `numpy_nms_v1(candidates, score_threshold, iou_threshold, max_detections)`, `postprocess_candidates(candidates, transform, postprocess) -> list[PlateDetection]`, and `rectify_detections(image_rgb, detections) -> DetectionRectificationResult`.

- [ ] **Step 1: Write failing Reader tests**

```python
def test_postprocess_inverse_maps_and_clips_bbox_and_every_corner():
    detections = postprocess_candidates(two_candidates(), odd_padding_transform(), default_postprocess())
    assert len(detections) == 2
    assert detections[0].bbox_xyxy.tolist() == [0.0, 0.0, 999.0, 332.0]
    assert np.all((detections[0].corners_xy[:, 0] >= 0) & (detections[0].corners_xy[:, 0] <= 999))
    assert np.all((detections[0].corners_xy[:, 1] >= 0) & (detections[0].corners_xy[:, 1] <= 332))

def test_numpy_nms_breaks_equal_scores_by_candidate_index():
    assert numpy_nms_v1(overlapping_equal_score_candidates(), 0.25, 0.50, 100) == [0]

def test_one_invalid_rectification_does_not_discard_another_valid_detection():
    result = rectify_detections(rgb_image(), [invalid_detection(), valid_detection()])
    assert len(result.rectified) == 1
    assert result.rejections[0].reason == "duplicate"
```

- [ ] **Step 2: Verify RED**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_reader_detector.py tests/integration/test_reader_detection_rectification.py -q`

Expected: FAIL because the Reader detector module is absent.

- [ ] **Step 3: Implement deterministic postprocessing and failure isolation**

```python
def numpy_nms_v1(candidates: NDArray[np.float32], score_threshold: float,
                 iou_threshold: float, max_detections: int) -> list[int]:
    eligible = _eligible_candidate_indices(candidates, score_threshold)
    ordered = sorted(eligible, key=lambda index: (-float(candidates[index, 4]), index))
    retained: list[int] = []
    for index in ordered:
        if all(_xyxy_iou(_candidate_xyxy(candidates[index]), _candidate_xyxy(candidates[kept])) <= iou_threshold
               for kept in retained):
            retained.append(index)
        if len(retained) == max_detections:
            break
    return retained

def rectify_detections(image_rgb, detections) -> DetectionRectificationResult:
    accepted, rejections = [], []
    for detection in detections:
        try:
            accepted.append(rectify_plate(image_rgb, detection.corners_xy))
        except InvalidCornersError as error:
            rejections.append(DetectionRejection(detection=detection, reason=error.reason))
    return DetectionRectificationResult(tuple(accepted), tuple(rejections))
```

Convert candidate `cxcywh` to `xyxy` before NMS. Apply Task 1 inverse/clip helpers separately to bbox corners and the four semantic points; never use an x-only transform or inclusive width/height clip. Validate postprocess thresholds before evaluating candidates: score in `[0,1]`, IoU in `(0,1]`, and positive maximum.

- [ ] **Step 4: Verify GREEN**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_reader_detector.py tests/integration/test_reader_detection_rectification.py tests/unit/test_rectifier.py -q`

Expected: PASS for odd-padding inverse mapping, eight-corner clipping, stable NMS ties, invalid numeric candidates, and per-instance M3a failure isolation.

- [ ] **Step 5: Commit**

```bash
git add src/plateai_reader/detector.py src/plateai_reader/__init__.py tests/unit/test_reader_detector.py tests/integration/test_reader_detection_rectification.py
git commit -m "feat: postprocess multi-plate detector candidates"
```

### Task 6: Full-bundle manifest, detector ONNX export, and parity

**Files:**
- Modify: `schemas/model_manifest.schema.json`
- Modify: `src/plateai_shared/schema_validation.py`
- Modify: `src/plateai_shared/bundle.py`
- Modify: `src/plateai_trainer/export/__init__.py`
- Create: `src/plateai_trainer/detection/export.py`
- Create: `src/plateai_trainer/detection/export_cli.py`
- Modify: `pyproject.toml`
- Modify: `tests/contract/test_model_bundle.py`
- Modify: `tests/contract/test_schemas.py`
- Create: `tests/integration/test_detection_export.py`

**Interfaces:** Consumes `PlatePoseNet`, its checkpoint/report, and an existing validated M2 crop bundle. Produces `validate_model_bundle(bundle_dir, schema_path)`, backward-compatible `validate_crop_bundle(bundle_dir, schema_path)`, `DetectionExportRequest`, `export_full_bundle(request)`, and installed `plateai-detect-export`.

- [ ] **Step 1: Write failing manifest, validator, and ONNX parity tests**

```python
def test_full_manifest_requires_exact_detector_io_and_numpy_postprocess():
    manifest = valid_full_manifest()
    validate_model_manifest(manifest, SCHEMA, visible_charset_symbol_count=33)
    manifest["components"]["detector"]["outputs"][0]["shape"] = ["batch", 8401, 13]
    with pytest.raises(DocumentValidationError, match="detector.outputs"):
        validate_model_manifest(manifest, SCHEMA, visible_charset_symbol_count=33)

def test_full_manifest_requires_dynamic_detector_batch_and_detector_report():
    manifest = valid_full_manifest()
    manifest["components"]["detector"]["batch"] = {"mode": "fixed", "size": 1}
    with pytest.raises(DocumentValidationError, match="components.detector.batch"):
        validate_model_manifest(manifest, SCHEMA, visible_charset_symbol_count=33)
    manifest = valid_full_manifest()
    del manifest["provenance"]["detector_training_report"]
    with pytest.raises(DocumentValidationError, match="provenance.detector_training_report"):
        validate_model_manifest(manifest, SCHEMA, visible_charset_symbol_count=33)

def test_detector_export_merges_valid_crop_bundle_and_matches_onnx(tmp_path, exported_crop_bundle, detector_run):
    bundle = export_full_bundle(
        DetectionExportRequest(exported_crop_bundle, detector_run.best_checkpoint,
                               detector_run.report_path, tmp_path / "full")
    )
    manifest = validate_model_bundle(bundle, SCHEMA)
    assert manifest["components"]["detector"]["outputs"][0]["shape"] == ["batch", 8400, 13]

def test_full_bundle_rejects_undeclared_detector_file(full_bundle):
    (full_bundle / "extra.onnx").write_bytes(b"unexpected")
    with pytest.raises(DocumentValidationError, match="undeclared"):
        validate_model_bundle(full_bundle, SCHEMA)

def test_native_and_onnx_candidates_keep_the_same_nms_survivors(detector_run, exported_full_bundle):
    native = detector_run.model(letterbox_test_batch())
    onnx = run_detector_onnx(exported_full_bundle / "detector.onnx", letterbox_test_batch())
    assert numpy_nms_v1(native[0].detach().numpy(), 0.25, 0.50, 100) == numpy_nms_v1(onnx[0], 0.25, 0.50, 100)
```

Add an injected ORT session that perturbs one candidate tensor and asserts parity failure does not publish output.

- [ ] **Step 2: Verify RED**

Run: `./.venv/Scripts/python.exe -m pytest tests/contract/test_model_bundle.py tests/contract/test_schemas.py tests/integration/test_detection_export.py -q`

Expected: FAIL because detector postprocess schema fields, full-bundle validation, and detector exporter are absent.

- [ ] **Step 3: Implement strict schema extension and atomic exporter**

```python
@dataclass(frozen=True, slots=True)
class DetectionExportRequest:
    recognizer_bundle: Path
    detector_checkpoint: Path
    detector_report: Path
    output: Path

def export_full_bundle(request: DetectionExportRequest) -> Path:
    crop_manifest = validate_crop_bundle(request.recognizer_bundle, _model_manifest_schema_path())
    detector_model = _load_detector_checkpoint(request.detector_checkpoint)
    staging = _create_owned_staging(request.output)
    try:
        _copy_declared_crop_bundle(request.recognizer_bundle, crop_manifest, staging)
        detector_path = _export_detector_onnx(detector_model, staging / "detector.onnx")
        _assert_detector_onnx_parity(detector_model, detector_path, batches=(1, 2))
        _write_full_manifest_and_report(staging, crop_manifest, request.detector_report)
        validate_model_bundle(staging, _model_manifest_schema_path())
        publish_directory_no_replace(staging, request.output)
    except BaseException:
        remove_owned_staging(staging, request.output)
        raise
    return request.output
```

Extend `components.detector` to require exactly one `images [batch,3,640,640]` input, one `candidates [batch,8400,13]` output, dynamic batch `{mode: dynamic, min: 1, opt: 8, max: 32}`, and the exact four keypoint names. When `plate-detection` is present, require exactly the two capabilities `crop-recognition` and `plate-detection`, `rectifier.normalization_strategy = convex-hull-semantic-v1`, and `provenance.detector_training_report` with declared file and SHA-256. Require a closed detector postprocess object with candidate format, preprocessor, `numpy-nms-v1`, score `0.25`, IoU `0.50`, and maximum `100`. Keep crop-only manifests valid and make `validate_crop_bundle` reject detector-bearing bundles. Full validation must hash-check recognizer, recognizer report, detector, and detector report, reject every undeclared regular file, reject ONNX external-data references, and reject a detector model larger than 8 MiB.

- [ ] **Step 4: Verify GREEN**

Run: `./.venv/Scripts/python.exe -m pytest tests/contract/test_model_bundle.py tests/contract/test_schemas.py tests/integration/test_export_bundle.py tests/integration/test_detection_export.py -q`

Expected: PASS for crop compatibility, full-bundle hashes/files, exact detector contract, batch-one/two native-versus-ONNX parity, and refusal to publish perturbed-parity output.

- [ ] **Step 5: Commit**

```bash
git add schemas/model_manifest.schema.json src/plateai_shared/schema_validation.py src/plateai_shared/bundle.py src/plateai_trainer/export/__init__.py src/plateai_trainer/detection/export.py src/plateai_trainer/detection/export_cli.py pyproject.toml tests/contract/test_model_bundle.py tests/contract/test_schemas.py tests/integration/test_detection_export.py
git commit -m "feat: export verified full plate detector bundles"
```

### Task 7: Local workflow documentation, history, and regression gate

**Files:**
- Modify: `README.md`
- Modify: `docs/training.md`
- Modify: `models/README.md`
- Modify: `history.md`
- Modify: `tests/integration/test_cli.py`

**Interfaces:** Documents the installed `plateai-compose`, `plateai-detect-train`, and `plateai-detect-export` commands from Tasks 2, 4, and 6. It does not add a production-serving API, a detector model weight, a user background, or TensorRT support.

- [ ] **Step 1: Write the failing installed-command regression test**

```python
@pytest.mark.parametrize(
    "executable",
    ("plateai-compose.exe", "plateai-detect-train.exe", "plateai-detect-export.exe"),
)
def test_detector_cli_help_is_installed(executable):
    result = subprocess.run([venv_scripts() / executable, "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
```

- [ ] **Step 2: Verify RED**

Run: `./.venv/Scripts/python.exe -m pytest tests/integration/test_cli.py -q`

Expected: FAIL because the three detector command entry points are not installed.

- [ ] **Step 3: Document the local, reproducible workflow**

Add a single end-to-end local-only sequence using a legal, user-provided background manifest:

```powershell
plateai-compose --background-manifest <local-manifest.json> --count 1000 --seed 42 --output out/detection-train
plateai-detect-train --train out/detection-train --validation out/detection-validation --output runs/detection-v1
plateai-detect-export --recognizer-bundle models/bundles/v1-local --checkpoint runs/detection-v1/best.pt --report runs/detection-v1/report.json --output models/bundles/v1-full-local
```

Explain that the detector accepts 640x640 OpenCV RGB letterbox input, yields pre-NMS `[batch,8400,13]` candidates, and sends each retained four-corner detection independently into M3a's 380x160 crop rectifier. State that no bundled background, weight, ONNX artifact, TensorRT benchmark, browser integration, or production recognition metric is part of source-tree acceptance. Append to `history.md` only after the implementation checks below have passed: commit IDs, exact local commands/results, and these remaining boundaries.

- [ ] **Step 4: Verify GREEN and run the complete local regression gate**

Run:

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m build
./.venv/Scripts/python.exe -m pip install --force-reinstall dist/plateai_trainer-*.whl
./.venv/Scripts/python.exe -m pytest tests/integration/test_cli.py -q
git -c core.whitespace=cr-at-eol diff --check
```

Expected: all repository tests pass; the distributable wheel installs; all three detector commands answer `--help`; and the staged implementation has no whitespace errors. Report M3b as locally validated only after these commands succeed.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/training.md models/README.md history.md tests/integration/test_cli.py
git commit -m "docs: record M3b local detector workflow"
```

## Plan Self-Review

- Task 1 owns the OpenCV-golden affine and the odd-padding inverse contract consumed everywhere else.
- Task 2 owns legal-input validation, transactional one-to-three plate composition, and auditable ground-truth metadata.
- Tasks 3 and 4 own the native model, exact assignment/loss math, and deterministic local training evidence.
- Task 5 owns source-coordinate recovery, stable NumPy NMS, and per-instance M3a failure isolation.
- Task 6 owns the strict ONNX and full-bundle boundary while preserving the existing crop-only validator.
- Task 7 owns installed-command regression coverage, user-operable documentation, history evidence, package build, and the local-only acceptance boundary.
