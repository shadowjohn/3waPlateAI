# External evaluation sources

Real plate images and their labels stay in ignored local folders. They are not
fixtures, source artifacts, or release content. Keep the source's license,
privacy, and redistribution conditions with every local copy.

## Ready locally

### EZCon Taiwan recognition test split — OCR evaluation candidate, rights pending

- Source: <https://huggingface.co/datasets/EZCon/taiwan-license-plate-recognition>
- Pinned revision: `15fd0d583d88c4a87a837cc2ef31b15c6a1b3719`.
- Upstream schema: 640×640 image, `xywhr`, `license_number`, and
  `is_electric_car`; its published test split contains 259 rows.
- The Dataset Card/API did **not** declare a license when this source was
  reviewed. It may be kept locally for provisional engineering evaluation only;
  do not redistribute it, train a commercial model from it, publish metrics that
  expose examples, or claim a right to use it until EZCon confirms its terms.
- Fetch it reproducibly, into the already ignored `datasets/` tree, with:

  ```powershell
  .\.venv\Scripts\python.exe .\tools\fetch_ezcon_taiwan_eval.py `
    --output .\datasets\restricted\ezcon-taiwan-recognition-test `
    --acknowledge-unreviewed-license
  ```

  The output has `all_test.jsonl`, image SHA-256 values, ground-truth text and
  rotated boxes. `reader_v1_eligible_test.jsonl` is the exact subset matching
  the current `LLL-DDDD`, no-`I`/`O`/`4` Reader v1 contract. The script refuses
  to silently follow an upstream revision change or overwrite an existing run.

## Useful, but not OCR-text ground truth

### EZCon Taiwan detection — explicitly approved local-only experiment

- Source: <https://huggingface.co/datasets/EZCon/taiwan-license-plate-detection>
- Pinned revision: `ab64ba1e86615c8371e1b5617792a130d45028e8`.
- Official splits: 2,346 train / 671 validation / 336 test; embedded images,
  class-zero labels and normalized four-point polygons, including multiple plates.
- No license is declared. On 2026-09-22 the user explicitly approved local
  experimental training only, with no redistribution of data or resulting weights.
  This is not a license review or permission for commercial deployment.
- `tools/fetch_ezcon_detection.py` retrieves fixed-revision parquet, verifies LFS
  SHA-256 and byte size, and records unresolved rights in every imported split.
  It does not execute upstream dataset scripts. PyArrow is optional import tooling.
- Existing recognition-test/user photos take precedence over all imported splits;
  test precedes validation, which precedes train. Decoded-RGB hashes and pHash
  Hamming distance <=4 exclude exact/recompressed/near-duplicate photos, without
  moving examples across official splits. This does not establish independence of
  every vehicle, recording session or differently cropped scene.
- The first local import retained 2,000 / 591 / 295 images, excluding 341 duplicate
  or near-duplicate images and 126 invalid/ambiguous annotations. In particular,
  the F6S-992 evaluation photo also appeared as upstream training row 1257 and
  was excluded. Full reasons and held-out hashes are in local `import_report.json`.

### TLPD — MIT Taiwan detector and rectifier evaluation

- Source: <https://huggingface.co/datasets/evan6007/TLPD>
- Pinned revision: `00f9ae2fa3d186bcf74e7f6ef4f48280181d6e85`.
- The Dataset Card declares MIT; it contains 3,032 Taiwan vehicle images with
  paired LabelMe plate polygons.
- The annotations label the polygon as `carplate`; they do not contain the
  displayed plate string. Use it to score M3b detection and M3a four-corner
  rectification, not recognizer exact-match accuracy.
- A spaced sample of the pinned images contains tight plate crops, not just
  full-vehicle scenes. Do not treat the dataset name as proof that it supplies
  full-scene Detector training coverage; inspect image framing before using it.
- Local integrity check at the pinned revision found 3,032 image files and
  3,032 JSON files, but 1,463 JSON `imagePath` values point at a different
  filename. Do not run its supplied `dataset.py` as-is for a metric: pair
  `labels/<stem>.json` with `images/<stem>.jpg`, then independently check the
  embedded dimensions and polygon bounds before scoring.

## Sources deliberately not imported

| Source | Why it is not a Reader v1 OCR benchmark now |
| --- | --- |
| [AOLP](https://github.com/AvLab-CV/AOLP) | 2,049 Taiwan images and recognition ground truth are valuable, but download needs a university-email request, is research-only/non-commercial, and forbids redistribution. Obtain permission before importing. |
| [CCPD](https://github.com/detectRecog/CCPD) | MIT and filename-embedded text/geometry, but it is mainland-Chinese format (including province characters) and its full archive is about 13 GB. It is a later cross-domain detection stress set, not a Taiwan Reader v1 OCR score. |
| [UFPR-ALPR](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/) | 4,500 fully annotated Brazilian images, but acquisition is academic/non-commercial and request-controlled; its format does not match the Taiwan profile. |

The public MIT [TLPD Dataset Card](https://huggingface.co/datasets/evan6007/TLPD)
documents polygon annotations rather than text labels. The CCPD publisher
documents filename-embedded ground truth and its MIT grant. The UFPR publisher
documents its request-controlled non-commercial terms.
