# M3a — Deterministic four-corner Rectifier design

## Goal and boundary

M3a establishes the deterministic geometric boundary between a future plate-pose
detector and M2's crop recognizer. It accepts a source RGB image and four
possibly unordered plate-corner candidates, either returns one canonical RGB
`380 x 160` crop or returns a stable rejection. It does not create detector
training data, train a detector, load a detector, add model weights, or claim
recognition accuracy. Those belong to M3b.

## Input and output contract

`normalize_corners(points_xy, image_size_wh) -> NormalizedCorners` accepts a
finite numeric `[4, 2]` point array and source width/height. It returns points
in this exact order: `left_top`, `right_top`, `right_bottom`, `left_bottom`.
`rectify_plate(image_rgb, points_xy) -> RectifiedPlate` accepts only RGB
`uint8 [height, width, 3]` input and returns an RGB `uint8 [160, 380, 3]`
canonical crop. This is exactly M2's source-image contract; it is passed to
`preprocess_v1_rgb` without resize, padding, or color conversion in M3a.

The destination correspondence is fixed to `(0,0)`, `(379,0)`, `(379,159)`,
and `(0,159)`. The warp uses `cv2.getPerspectiveTransform` and
`cv2.warpPerspective` with `INTER_LINEAR` and a white constant border. M3a
does not silently substitute any other output ratio or resolution.

## Geometry defense

1. Reject non-finite points, duplicate points, points outside the source image,
   and hulls that do not contain exactly four vertices.
2. Build the convex hull before assigning semantic positions. Reject a hull
   below an explicit source-image-relative area threshold; a concave,
   self-intersecting, or bow-tie candidate therefore cannot reach OpenCV.
3. Put the hull in cyclic order. Compare the two pairs of opposite edges and
   require an unambiguous longer pair. The longer pair is the projected plate
   horizontal axis; the other pair is the vertical axis.
4. Classify the two long edges as top and bottom by their midpoint along the
   image-down normal. Classify each endpoint left/right along the long-axis
   vector. This uses the plate's elongated geometry rather than `min(x+y)` or
   `min(x)`, which can swap `left_top` and `left_bottom` around a diamond-like
   or severely tilted boundary.
5. Reject an ambiguous long/short-edge ratio, coincident top/bottom projections,
   non-clockwise canonical result, or a homography that OpenCV cannot create.
   Geometry alone cannot resolve a 180-degree orientation when its top/bottom
   evidence is ambiguous; rejection is safer than an upside-down crop.

The normalized result records whether input ordering changed, the convex area,
and the long-axis orientation. Error messages use `InvalidCornersError` with a
stable reason code suitable for a future Reader response.

## M1/M2 integration and acceptance

M1's perspective-augmented synthetic output and its known corners are the M3a
ground-truth test source. Tests cover every permutation of a valid quadrilateral,
extreme trapezoids, near-diamond ambiguity, bow ties, concavity, duplicates,
non-finite points, out-of-frame points, and low area. A successful known-corner
warp must retain the source plate's correspondence and produce exactly
`380 x 160` RGB. The output must pass M2's `preprocess_v1_rgb` shape/margin
contract.

An optional local integration test may feed a user-supplied M2 bundle through
the rectified output. Its prediction is useful smoke evidence only: no model,
dataset, ONNX file, or claimed real-world accuracy is committed by M3a.

## Verification boundary

M3a is locally accepted when pure-geometry, synthetic-M1, M2-preprocessing,
and full-project tests pass. Detector keypoint quality, detector ONNX formats,
GPU behavior, real images, real vehicle identity data, field recognition
accuracy, Reader/API behavior, and production deployment remain M3b or later.
