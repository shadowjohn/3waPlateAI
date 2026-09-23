# Detector whole-plate repair (local experiment)

## Approved boundary

Repair fragmented whole-scene localization in native PyTorch PlatePoseNet, without
changing the 640x640 RGB letterbox, 8,400x13 candidate contract, NumPy NMS or M3a
380x160 crop. No Ultralytics, new fonts, automatic active-bundle replacement or
push. Recognition accuracy is a separate gate. User explicitly approved EZCon
data for local experiments only; neither data nor derived weights may be shared.

## Evidence and minimum changes

- Existing active detector: 8,115/8,400 positive raw rows on F6S-992; maximum raw
  bbox IoU .258. Native/ONNX match; inverse letterbox is identity on this image.
- Objectness focal mean over 8,400 cells is diluted relative to positive-only
  geometry. Four-image overfit experiment: positive normalization raises bbox
  AP50 from .0018 to 1.0. This is an optimization diagnostic, not real accuracy.
- Normalize objectness by positive count, initialize objectness probability .01.
- Add a strict real-scene dataset adapter, CUDA support and explicit provenance.
- Fetch only immutable EZCon parquet revision
  `ab64ba1e86615c8371e1b5617792a130d45028e8`; verify byte sizes and LFS SHA256.
- Preserve official splits, exclude existing held-out images first, then prevent
  duplicate overlap with priority test > validation > train. Match decoded RGB
  hashes and pHash Hamming distance <=4; report all exclusions. This heuristic
  does not prove independence of every scene/vehicle.
- Retain every valid plate instance in each image. Reject an entire image if any
  annotation is invalid/ambiguous rather than teach an unlabeled plate as background.

## Verification and acceptance

1. Regression tests: objectness normalization/gradient, initial prior, multi-plate
   coordinates, decoded duplicate exclusion, CPU/CUDA training and report provenance.
2. Train from scratch, select checkpoint on validation only, then freeze selection
   before independent test evaluation. Do not tune on the user's evaluation photos.
3. Select checkpoints by validation complete-quad recall, then complete-quad
   precision, bbox AP50, and minimum validation loss. Report bbox AP50, operating precision/recall, semantic corner errors, complete
   quad precision/recall (all four <=8px at 640), NMS count and rectifier acceptance.
   Compare old and new weights on identical retained test data.
4. Whole-plate crop evidence for user photos, independently of OCR. Fewer false
   positives alone is not acceptance. Initial local promotion targets: bbox
   precision/recall >=.90 and complete-quad recall >=.80; these are development
   gates, not production guarantees. Failed gates remain explicit.
5. ONNX shape/numeric/NMS parity and existing regression suite before handoff.
   Do not automatically replace active-v1 even if local gates pass.

## Implementation sequence

Loss/prior regression -> pinned import and leak exclusions -> CPU/CUDA adapter ->
real-scene training -> frozen test comparison -> ONNX/provenance -> history.

## Known limits

EZCon license is undeclared, not reviewed. Its vehicle/plate-type mix is not a
change to the original passenger-car v1 product scope. Existing recognizer font
provenance and real-photo OCR quality remain unresolved. Highly ambiguous quads
are excluded by the existing rectifier semantics and counted in the import report.
