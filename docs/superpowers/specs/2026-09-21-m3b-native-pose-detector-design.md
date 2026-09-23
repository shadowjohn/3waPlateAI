# M3b — Native multi-plate pose-detector design

## Goal and scope

M3b adds the learned half of v1 detection: a native-PyTorch, single-class,
anchor-free, YOLO-style pose detector that locates one or more new-style
private-passenger plates in an RGB image and predicts each axis-aligned box plus
four semantic corners. It produces candidates for M3a; M3a remains the only
component that validates corner geometry and creates M2's canonical crop.

V1 supports multiple instances. Its synthetic compositor emits one through
three plates per image. M3b does not collect or commit real images, publicly
release checkpoints, models, TensorRT engines, or bundles, claim field accuracy,
or add serving. Such artifacts remain local, ignored, and the user's
responsibility.

## License and architecture

The detector, loss, target assigner, compositor, exporter, and NumPy Reader
postprocessor are repository-owned native PyTorch/NumPy code. M3b must not
import, invoke, copy from, or export a model/checkpoint associated with
Ultralytics. Normal third-party notices still cover PyTorch, NumPy, OpenCV,
ONNX, and ONNX Runtime; this design does not grant rights for user data or
trained artifacts.

`PlatePoseNet` uses a small CSP-Darknet-tiny-style backbone, FPN/PAN fusion,
and P3/P4/P5 heads at strides 8, 16, and 32. Its one-class head emits
objectness, an axis-aligned box, and four fixed corner pairs. The target is an
uncompressed FP32 ONNX file no larger than 8 MiB; it is not a device-latency
claim.

## Preprocessing and detector tensor contract

The detector golden reference is OpenCV `opencv-python-headless` 5.0.0.93.
For RGB `uint8 [height, width, 3]` input:

1. `scale = min(640 / width, 640 / height)`; each resized dimension is
   `floor(original_dimension * scale + 0.5)`.
2. Use `cv2.resize(..., interpolation=cv2.INTER_LINEAR)`.
3. Divide each remaining padding total using `floor(total / 2)` on left/top
   and `ceil(total / 2)` on right/bottom; raw RGB padding is `(114,114,114)`.
4. Convert RGB HWC to NCHW and divide `float32` values by `255.0`.

This preprocessor exposes exact `scale`, `pad_left`, and `pad_top`. Training
annotations use the same affine mapping. The fixed-spatial, dynamic-batch IO is:

```text
images:     float32 [batch, 3, 640, 640]
candidates: float32 [batch, 8400, 13]
```

`8400 = 80*80 + 40*40 + 20*20` from P3/P4/P5. Every decoded candidate uses
640-pixel letterbox coordinates in this exact order:

```text
[center_x, center_y, width, height, confidence,
 left_top_x, left_top_y, right_top_x, right_top_y,
 right_bottom_x, right_bottom_y, left_bottom_x, left_bottom_y]
```

The model output includes decoded coordinates and sigmoid confidence in `[0,1]`
but excludes score thresholding, NMS, inverse letterboxing, and M3a. Export uses
ONNX opset 17, external data disabled, one input named `images`, one output
named `candidates`, and dynamic batch only.

## Reader contract and M3a handoff

The Reader owns deterministic, Torch-free `numpy-nms-v1`. It rejects
non-finite candidates and boxes with non-positive width/height, filters by the
manifest score threshold, then sorts descending confidence and ascending
candidate index before class-agnostic bbox NMS. This tie breaking is contractual.

For each survivor, the Reader applies `(coordinate - pad) / scale` to its bbox
and **all eight corner values**. It clips public source coordinates to
`[0, source_width - 1]` and `[0, source_height - 1]`, not inclusive `width` or
`height`: M3a requires discrete in-frame coordinates. It returns:

```python
@dataclass(frozen=True)
class PlateDetection:
    bbox_xyxy: np.ndarray      # float32 [4], source x1, y1, x2, y2
    confidence: float
    corners_xy: np.ndarray     # float32 [4, 2], semantic source coordinates
```

The Reader loops over `List[PlateDetection]` and calls M3a per instance. One
`InvalidCornersError` produces a rejection record only for that detection; all
other survivors continue to rectification and later recognition.

## Synthetic composite data

M3b composes M1-rendered v1 `380x160` RGB plates on a user-supplied local
background canvas. Each image has one, two, or three instances. A sampled
perspective homography, rotation, and scale transforms each M1 source corner
into exact semantic GT; `bbox_xyxy` is its enclosing axis-aligned box.

V1 requires four in-frame GT corners and bbox IoU `<= 0.0` between instances.
Occlusion, truncation, and visibility-labelled corners are later profiles, not
silent v1 behavior. Every run samples projected shortest-edge strata of 16–31
pixels (small), 32–63 pixels (medium), and at least 64 pixels (large), and
validation reports results for every stratum.

Backgrounds must be legally supplied outside Git. Runs record a background
manifest identity and SHA-256, seed, and config hashes; train and validation
background SHA-256 sets must be disjoint. The repository holds only procedural
toy backgrounds for tests. Dataset publication follows M1's transactional,
new-directory-only policy and records image hash, every instance's GT and
transform, source-M1 provenance, and background identity.

## Training contract

The native dataset applies the exact 640 affine to image, bbox, and all four
corners. Using the letterbox-pixel shortest bbox side, the scale-aware
anchor-free assigner selects canonical P3 for `<64`, P4 for `64–127`, and P5
for `>=128`; P4/P5 instances additionally supervise the immediately finer
level (P3/P4 respectively) for corner geometry. Positive cells are inside the
GT bbox and at Chebyshev grid distance at most one from its centre cell. On
overlap, the smaller-area GT wins, then lower instance index breaks ties.
Positive loss is:

```text
L = 1.0 * focal_bce(objectness) + 5.0 * ciou(bbox)
    + 2.0 * smooth_l1(normalized_corners)
    + 2.0 * smooth_l1((predicted_corners - target_corners) / 8 px)
```

Corner residuals are normalized independently by matched bbox width and height;
the additional absolute term is normalized by the 8 px complete-quad acceptance
tolerance; both corner terms are positive-only. GT order is always left-top, right-top,
right-bottom, left-bottom. Run metadata captures seed, dependency lock,
model/data/config hashes, metrics, and checkpoint compatibility. A deterministic
small-data smoke proves finite forward/loss/gradients, checkpoint read/write,
and short overfit behavior; it is not a real-image accuracy claim.

## Full local bundle and ONNX validation

A detector checkpoint alone is not a Model Bundle. Full-bundle export starts
from a validated M2 crop-recognition bundle, copies its declared recognizer,
charset, rules, and report into owned staging, adds `detector.onnx` and detector
report, checks all hashes, and atomically publishes only to a new local path.

The manifest retains `crop-recognition`, adds `plate-detection`, retains
`convex-hull-semantic-v1`, and requires detector postprocessing:

```json
{
  "candidate_format": "cxcywh-confidence-corners-letterbox-px-v1",
  "preprocess": "opencv-rgb-letterbox-640-v1",
  "nms": "numpy-nms-v1",
  "score_threshold": 0.25,
  "iou_threshold": 0.50,
  "max_detections": 100
}
```

These are versioned defaults read from the bundle, never substituted silently.
M3b extends the crop-only validator into a full-bundle validator that rejects
undeclared detector files, incompatible recognizer input, hash mismatch, and a
missing postprocess contract.

M3b acceptance requires all of the following:

1. Deterministic, transactional 1/2/3-instance composites with exact GT,
   semantic ordering, non-overlap, and train/validation background separation.
2. Model tests for `[batch,3,640,640] -> [batch,8400,13]`, finite loss and
   gradients, plus checkpoint-compatible overfit smoke.
3. `onnx.checker`, named IO/dtype/shape checks for batches one and two, then
   native/ONNX candidate parity at `rtol=1e-4, atol=1e-5`.
4. Native and ONNX candidates yielding identical stable NMS survivors, bbox,
   and four corners; known affine fixtures verify inverse letterbox and clipping
   for bbox plus all eight corner values.
5. A multi-instance integration where M3a rejects one invalid candidate while
   another detection remains available for M2 preprocessing.
6. Full-project pytest, sdist/wheel build, and `pip check` success.

TensorRT benchmarks, GPU latency guarantees, real-image evaluation, tracking,
full OCR orchestration, browser/API serving, remote CI, and production
deployment are explicitly out of scope.
