# Detector v2 Geometry Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Stage 1 whole-plate four-corner localization on local real-scene data without changing the public detector, Reader, Rectifier, or active-bundle contracts.

**Architecture:** Keep the native three-scale `PlatePoseNet` and its decoded `[batch, 8400, 13]` output. Training will make geometry supervision acceptance-aligned: large plates receive an absolute-pixel corner penalty as well as the existing relative penalty, and large/middle instances receive positives on the next finer FPN level. Checkpoint selection will use validation complete-quad recall before bbox AP50. The existing frozen test split remains evaluation-only.

**Tech Stack:** Python 3.11, PyTorch CUDA, NumPy, OpenCV, existing local-only EZCon real-scene snapshot.

**Spec:** `docs/plans/2026-09-22-detector-real-scene-repair.md` and `docs/superpowers/specs/2026-09-21-m3b-native-pose-detector-design.md`

## Global Constraints

- Preserve RGB 640×640 letterbox preprocessing, P3/P4/P5 decoded 8,400×13 tensor order, NumPy NMS, semantic `LT, RT, RB, LB` ordering, and M3a's 380×160 canonical output.
- Use only `out/ezcon-detector-v1/train` and `validation` for development; `out/ezcon-detector-v1/test` is frozen evaluation-only and must be rejected by `train_detector`.
- EZCon data, checkpoints, comparison outputs, and exported candidate bundles remain local-only and ignored; do not distribute or activate them.
- Do not alter `models/bundles/active-v1`, score/NMS thresholds, Recognizer weights, or OCR acceptance policy.
- Promotion requires frozen-test bbox operating precision and recall each ≥0.90 plus complete-quad recall ≥0.80 at all four semantic corners within 8 px in the 640-pixel letterbox.

## Review Focus

- A large (P5) plate must receive next-finer P4 positives without losing its canonical P5 positives; test this in Task 2.
- A small P3 plate must not acquire an invalid nonexistent finer level; test this in Task 2.
- Identical relative corner residuals must cost more when their absolute pixel error exceeds the 8 px acceptance tolerance; test this in Task 1.
- Validation checkpoint selection must prefer a geometrically better epoch over a slightly higher bbox AP50 epoch; test this in Task 3.
- Test-split paths and active-bundle paths must remain impossible inputs to the training command; retain existing dataset/config validation in Task 4.

---

### Task 1: Add acceptance-aligned absolute corner loss

**Files:**
- Modify: `src/plateai_trainer/detection/loss.py`
- Modify: `tests/unit/test_detection_model.py`
- Modify: `docs/superpowers/specs/2026-09-21-m3b-native-pose-detector-design.md`

**Interfaces:**
- `DetectorLoss` gains `corner_absolute_smooth_l1: torch.Tensor` while retaining `corner_smooth_l1` for the existing relative loss.
- `detection_loss(predictions, targets)` returns `total = objectness + 5 * box_ciou + 2 * relative_corner + 2 * absolute_corner`.
- Absolute residuals are divided by `8.0`, the existing complete-quad tolerance, before Smooth L1 so the term is dimensionless and explicitly tied to the gate.

- [ ] **Step 1: Write the failing loss test.**

```python
def test_detection_loss_penalizes_equal_relative_error_more_for_large_plates():
    small_target = assign_detection_targets([instance((200., 200., 232., 232.))])
    large_target = assign_detection_targets([instance((200., 200., 360., 360.))])
    small_predictions = perfect_predictions(small_target)
    large_predictions = perfect_predictions(large_target)
    small_predictions[0, small_target.positive_indices, 5:] += 4.0
    large_predictions[0, large_target.positive_indices, 5:] += 20.0
    small = detection_loss(small_predictions, small_target)
    large = detection_loss(large_predictions, large_target)
    assert small.corner_smooth_l1 == pytest.approx(large.corner_smooth_l1)
    assert large.corner_absolute_smooth_l1 > small.corner_absolute_smooth_l1
```

- [ ] **Step 2: Run RED.**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_model.py -q`

Expected: FAIL because `DetectorLoss` has no absolute-corner component.

- [ ] **Step 3: Implement only the hybrid corner term.**

```python
absolute_residual = (positive_predictions[:, 5:].reshape(-1, 4, 2) - matched_corners) / 8.0
corner_absolute_smooth_l1 = F.smooth_l1_loss(absolute_residual, torch.zeros_like(absolute_residual))
total = objectness + 5.0 * box_ciou + 2.0 * corner_smooth_l1 + 2.0 * corner_absolute_smooth_l1
```

For an empty-positive batch, make the new term a differentiable zero derived from the corner prediction tensor. Update `test_corner_loss_normalizes_xy_by_bbox_dimensions_and_ignores_negatives` to include the absolute component in its total-loss assertion. Update the M3b training formula in the spec exactly.

- [ ] **Step 4: Run GREEN.**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_model.py -q`

Expected: PASS with finite loss and gradients retained by existing tests.

- [ ] **Step 5: Leave the verified source change uncommitted.**

Commit or push only after the user separately approves the measured v2 outcome.

### Task 2: Add next-finer positives for geometry supervision

**Files:**
- Modify: `src/plateai_trainer/detection/targets.py`
- Modify: `tests/unit/test_detection_model.py`
- Modify: `docs/superpowers/specs/2026-09-21-m3b-native-pose-detector-design.md`

**Interfaces:**
- `assign_detection_targets(instances)` continues returning `DetectionTargets` with the same fields and row numbering.
- A P3 object remains P3-only; a P4 object gets P3 and P4 positives; a P5 object gets P4 and P5 positives.
- Existing area/index conflict ordering applies independently per level/cell.

- [ ] **Step 1: Write failing level-assignment tests.**

```python
def test_large_instance_uses_p4_and_p5_geometry_positives():
    targets = assign_detection_targets([instance_with_short_side(160)])
    assert set(targets.positive_level_indices) == {1, 2}


def test_small_instance_remains_p3_only():
    targets = assign_detection_targets([instance_with_short_side(32)])
    assert set(targets.positive_level_indices) == {0}
```

- [ ] **Step 2: Run RED.**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_model.py -q`

Expected: FAIL because each instance is currently assigned only to one scale.

- [ ] **Step 3: Implement the explicit level list.**

```python
canonical_level = 0 if short_side < 64 else 1 if short_side < 128 else 2
levels = (canonical_level,) if canonical_level == 0 else (canonical_level - 1, canonical_level)
for level in levels:
    stride, offset = ((8, 0), (16, 6400), (32, 8000))[level]
    # retain the existing centre-neighbour cell assignment and conflict rule
```

Keep `matched`, `levels`, positives, boxes, and semantic corners aligned after sorting by row index. Update the spec's one-level assignment statement to describe this two-level geometry supervision.

- [ ] **Step 4: Run GREEN.**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_model.py -q`

Expected: PASS; no output tensor or Reader contract changes.

- [ ] **Step 5: Leave the verified source change uncommitted.**

Commit or push only after the user separately approves the measured v2 outcome.

### Task 3: Select validation checkpoints by full-quad quality

**Files:**
- Modify: `src/plateai_trainer/detection/engine.py`
- Modify: `tests/integration/test_detection_training_cli.py`
- Modify: `docs/plans/2026-09-22-detector-real-scene-repair.md`

**Interfaces:**
- Add `_checkpoint_selection_key(metrics: dict[str, float]) -> tuple[float, float, float, float]`.
- The returned key is `(complete_quad_recall, complete_quad_precision, bbox_ap50, -loss)`.
- The run report records the exact geometry-first selection sentence.

- [ ] **Step 1: Write the failing deterministic selection test.**

```python
def test_checkpoint_selection_prefers_complete_quad_recall_over_bbox_ap50():
    geometry_better = metrics(complete_quad_recall=.60, complete_quad_precision=.70, bbox_ap50=.94, loss=.8)
    bbox_better = metrics(complete_quad_recall=.40, complete_quad_precision=.80, bbox_ap50=.97, loss=.7)
    assert _checkpoint_selection_key(geometry_better) > _checkpoint_selection_key(bbox_better)
```

- [ ] **Step 2: Run RED.**

Run: `./.venv/Scripts/python.exe -m pytest tests/integration/test_detection_training_cli.py -q`

Expected: FAIL because no geometry-first helper exists and the engine chooses `(bbox_ap50, -loss)`.

- [ ] **Step 3: Implement the helper and report text.**

```python
def _checkpoint_selection_key(metrics):
    return (metrics['complete_quad_recall'], metrics['complete_quad_precision'], metrics['bbox_ap50'], -metrics['loss'])
```

Use it in `train_detector`; do not inspect the frozen test split to select a checkpoint.

- [ ] **Step 4: Run GREEN.**

Run: `./.venv/Scripts/python.exe -m pytest tests/integration/test_detection_training_cli.py tests/unit/test_detector_evaluation.py -q`

Expected: PASS and report provenance remains complete.

- [ ] **Step 5: Leave the verified source change uncommitted.**

Commit or push only after the user separately approves the measured v2 outcome.

### Task 4: Run local v2 training and frozen evaluation

**Files:**
- Create locally ignored: `runs/detector-real-v2-geometry/`
- Create locally ignored: `out/detector-real-v2-evaluation/`
- Modify: `history.md`

**Interfaces:**
- Train input: `out/ezcon-detector-v1/train` and `out/ezcon-detector-v1/validation`.
- Train output: a new, no-replace `runs/detector-real-v2-geometry` directory.
- Evaluation compares `runs/detector-real-v1/best.pt` with the new v2 checkpoint on `out/ezcon-detector-v1/test` exactly once after v2 validation selection.

- [ ] **Step 1: Run the changed unit and integration tests before CUDA.**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_detection_model.py tests/integration/test_detection_training_cli.py tests/unit/test_real_detection_dataset.py -q`

Expected: PASS with no test-split path accepted by training.

- [ ] **Step 2: Train a new local-only candidate.**

```powershell
.\.venv\Scripts\python.exe -m plateai_trainer.detection.train_cli `
  --train out\ezcon-detector-v1\train `
  --validation out\ezcon-detector-v1\validation `
  --output runs\detector-real-v2-geometry `
  --epochs 60 --batch-size 16 --learning-rate 0.001 --seed 42 --device cuda
```

Record only validation history while training. Stop on a non-finite loss/gradient or data hash mismatch; do not reuse a partial output directory.

- [ ] **Step 3: Freeze the selected v2 checkpoint and run the test comparison once.**

```powershell
.\.venv\Scripts\python.exe tools\evaluate_detector.py `
  --test out\ezcon-detector-v1\test `
  --baseline runs\detector-real-v1\best.pt `
  --candidate runs\detector-real-v2-geometry\best.pt `
  --output out\detector-real-v2-evaluation `
  --device cuda
```

Do not change loss weights, epochs, confidence thresholds, or the checkpoint after opening this test report.

- [ ] **Step 4: Record outcome without automatic promotion.**

Append test metrics, size strata, data hashes, and whether all three gates passed to `history.md`. If any gate fails, retain the candidate only as a local diagnostic; do not export/activate it.

### Task 5: Package only a qualified local candidate

**Files:**
- Create locally ignored only when all gates pass: `models/bundles/candidate-detector-real-v2/`
- Test: existing `tests/integration/test_detection_export.py`

**Interfaces:**
- Input is a v2 checkpoint that passed Task 4's frozen-test gates.
- Output is a local bundle whose manifest says `training_data: mixed`, `license_reviewed: false`, and contains an export report; `active-v1` remains unchanged.

- [ ] **Step 1: Gate export from the recorded frozen test result.**

If precision, recall, or complete-quad recall is below its threshold, stop this task and record no export. This is an intentional fail-closed result, not a training error.

- [ ] **Step 2: When eligible, run the existing export parity tests.**

Run: `./.venv/Scripts/python.exe -m pytest tests/integration/test_detection_export.py -q`

Expected: PASS before creating the new local candidate bundle.

- [ ] **Step 3: Export and validate without activation.**

Use the existing `plateai-detect-export` workflow with a new `candidate-detector-real-v2` path, then run the existing manifest/schema/hash and ONNX parity checks. Record any all-candidate numeric-parity exception separately from retained-detection parity.

- [ ] **Step 4: Run final regression and review.**

Run: `./.venv/Scripts/python.exe -m pytest -q`

Expected: PASS. Then inspect `git status --short` and `git -c core.whitespace=cr-at-eol diff --check`; stage no local datasets, runs, bundles, test photos, or output panels.
