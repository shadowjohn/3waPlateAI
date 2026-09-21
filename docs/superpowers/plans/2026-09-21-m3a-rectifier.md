# M3a Rectifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic four-corner perspective rectifier whose output is the exact M2 v1 RGB `380 x 160` source crop.

**Architecture:** `plateai_reader.rectifier` separates hull-first corner normalization from the only OpenCV warp. It produces clockwise `left_top`, `right_top`, `right_bottom`, `left_bottom` points or a stable rejection; M2 receives the crop unchanged.

**Tech Stack:** CPython 3.11, NumPy, OpenCV, Pillow, pytest, Hypothesis.

**Spec:** `docs/superpowers/specs/2026-09-21-m3a-rectifier-design.md`

## Global Constraints

- Only RGB `uint8 [height, width, 3]` plus exactly four finite in-frame points are valid input.
- Convex-hull validation precedes semantic assignment and homography creation.
- Select top/bottom from opposite long-edge midpoints and long-axis projections; never use `min(x+y)` or `min(x)`.
- Reject ambiguity, low area, duplicate, non-convex, and homography-invalid input with `InvalidCornersError` reason codes. A raw self-intersecting ordering of a convex point set is normalized, because the public API accepts arbitrary corner order.
- Output is always RGB `uint8 [160, 380, 3]`, `INTER_LINEAR`, white constant border, and no extra scaling.
- Models, real images, generated datasets, and benchmarks remain ignored and uncommitted.

## Review Focus

1. Every permutation of a valid trapezoid normalizes identically; Task 1.
2. A raw bow-tie ordering normalizes safely; duplicate, non-finite, out-of-frame, non-convex, and low-area candidates fail before OpenCV; Task 1.
3. Near-square/diamond geometry is rejected as ambiguous rather than arbitrarily flipped; Task 1.
4. Valid warps are exactly `380x160` and feed M2 preprocessing directly; Task 2.
5. M3a does not introduce detector data, weights, training, or ONNX; Task 3.

---

### Task 1: Hull-first corner normalization

**Files:**
- Create: `src/plateai_reader/__init__.py`
- Create: `src/plateai_reader/rectifier.py`
- Create: `tests/unit/test_rectifier.py`

**Interfaces:** `normalize_corners(points_xy, image_size_wh) -> NormalizedCorners`, `InvalidCornersError(reason)`, and `NormalizedCorners(points_xy, area, reordered, long_axis_xy)`.

- [ ] **Step 1: Write failing tests**

```python
def test_every_trapezoid_permutation_has_one_canonical_order():
    expected = np.array([[30, 20], [350, 45], [330, 130], [50, 110]], np.float32)
    for item in itertools.permutations(expected):
        np.testing.assert_allclose(normalize_corners(np.asarray(item), (380, 160)).points_xy, expected)

@pytest.mark.parametrize("points, reason", [(nonfinite, "non_finite"), (duplicates, "duplicate"), (diamond, "ambiguous")])
def test_invalid_corners_reject(points, reason):
    with pytest.raises(InvalidCornersError, match=reason):
        normalize_corners(points, (380, 160))
```

- [ ] **Step 2: Verify RED**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_rectifier.py -q`

Expected: FAIL because the Rectifier module does not exist.

- [ ] **Step 3: Implement normalization**

```python
# validate finite/unique/in-frame points; cv2.convexHull must return four vertices
# compare opposite edge-pair means and reject insufficient long/short separation
# select top/bottom long edges by midpoint projection and left/right by long-axis projection
# return clockwise [left_top, right_top, right_bottom, left_bottom]
```

Require a source-area-relative minimum hull area and stable reason tokens. No raw detector order can bypass geometric validation.

- [ ] **Step 4: Verify and commit**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_rectifier.py -q`

```bash
git add src/plateai_reader/__init__.py src/plateai_reader/rectifier.py tests/unit/test_rectifier.py
git commit -m "feat: normalize defensive plate corners"
```

### Task 2: Exact M2 crop rectification

**Files:**
- Modify: `src/plateai_reader/rectifier.py`
- Modify: `tests/unit/test_rectifier.py`
- Create: `tests/integration/test_rectifier_m2_contract.py`

**Interfaces:** `rectify_plate(image_rgb, points_xy) -> RectifiedPlate`, where `RectifiedPlate` exposes `image_rgb`, normalized corners, and transform.

- [ ] **Step 1: Write failing warp tests**

```python
def test_rectifier_warps_m1_perspective_fixture_to_m2_source_contract():
    result = rectify_plate(warped_rgb, shuffled_ground_truth_corners)
    assert result.image_rgb.shape == (160, 380, 3)
    assert result.image_rgb.dtype == np.uint8
    assert preprocess_v1_rgb(result.image_rgb).shape == (1, 64, 160)

def test_invalid_corners_never_call_opencv(monkeypatch):
    monkeypatch.setattr(cv2, "getPerspectiveTransform", pytest.fail)
    with pytest.raises(InvalidCornersError):
        rectify_plate(rgb, invalid_points)
```

- [ ] **Step 2: Verify RED**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_rectifier.py tests/integration/test_rectifier_m2_contract.py -q`

Expected: FAIL because no crop API exists.

- [ ] **Step 3: Implement the guarded warp**

```python
destination = np.float32([[0, 0], [379, 0], [379, 159], [0, 159]])
transform = cv2.getPerspectiveTransform(normalized.points_xy.astype(np.float32), destination)
crop = cv2.warpPerspective(image_rgb, transform, (380, 160), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
```

Build the fixture from an M1-rendered synthetic plate with known perspective corners. Validate image source type/shape and finite transform coefficients.

- [ ] **Step 4: Verify and commit**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_rectifier.py tests/integration/test_rectifier_m2_contract.py tests/unit/test_recognition.py -q`

```bash
git add src/plateai_reader/rectifier.py tests/unit/test_rectifier.py tests/integration/test_rectifier_m2_contract.py
git commit -m "feat: rectify plate crops for M2 recognition"
```

### Task 3: M3a documentation and branch verification

**Files:**
- Modify: `README.md`
- Modify: `models/README.md`
- Modify: `history.md`
- Modify: `tests/contract/test_schemas.py`

**Interfaces:** Documents M3a/M3b boundary and pins the existing manifest strategy `convex-hull-semantic-v1`.

- [ ] **Step 1: Write a detector-manifest regression test**

```python
def test_detection_manifest_keeps_explicit_rectifier_strategy():
    manifest = valid_v1_manifest()
    manifest["capabilities"] = ["crop-recognition", "plate-detection"]
    manifest["rectifier"] = {"normalization_strategy": "convex-hull-semantic-v1"}
    assert validate_document(manifest, SCHEMAS / "model_manifest.schema.json") is None
```

- [ ] **Step 2: Verify, document, and commit**

Run: `./.venv/Scripts/python.exe -m pytest -q; ./.venv/Scripts/python.exe -m build; ./.venv/Scripts/python.exe -m pip check; git -c core.whitespace=cr-at-eol diff --check`

Record only local geometry/build evidence, explicit `380x160` M2 coupling, and remaining detector/GPU/real-image/production boundaries.

```bash
git add README.md models/README.md history.md tests/contract/test_schemas.py
git commit -m "docs: record M3a rectifier boundary"
```

## Spec coverage self-review

Task 1 owns topology and orientation defense; Task 2 owns the sole warp and M2 coupling; Task 3 owns the M3a/M3b boundary. No detector data/model work is present.
