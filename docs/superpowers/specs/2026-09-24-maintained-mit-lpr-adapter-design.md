# Maintained MIT CPM + LPRNet Adapter — Design

Date: 2026-09-24

## Decision and outcome

Adopt the public `evan6007/FPGA-LPR` CPM + LPRNet as a **maintained, attributed** recognition option for 3waPlateAI. The project retains ownership of its detector, adapter, validation, and release decisions; the upstream authors retain credit for their original model/code. A user can compare this option with `active-v1` on the same image without changing the active model. Promotion to a default is a separate decision based on previously unseen real photos, not the 3,032-image TLPD replay.

The user chose the author's public model as the improvement target and approved bringing MIT-labeled model artifacts into the project. The implementation assumption for review is: track the minimal architecture source and **converted ONNX CPM/LPRNet weights** in Git and include those same runtime artifacts in an actual inference release. Do not duplicate the original `.pth` weights or include TLPD images/JSON in Git or release. If the user wants a different artifact boundary, revise this spec before planning.

## Source, license, and provenance

- Source code: <https://github.com/evan6007/FPGA-LPR>, pinned commit `574667ca7f5730d17b4b6fcda3ec568521bcbcd8`. Its README declares MIT.
- Weight source: <https://huggingface.co/evan6007/FPGA-LPR>, pinned revision `51b9606b174eedbc091aec844c288a72aa9cd25b`; its model metadata declares `license:mit` and it is not gated. Original SHA-256 values: CPM `70450f3e24570bf714e53d0c169ac52948e448c3945f6748d19909d525420fa4`, LPRNet `21cda2d7f095d958eb725ac2dc05e87cbd342aed5fae7c0cf304477fe5b5a42b`.
- Keep the source's available copyright/license statements, the full MIT permission notice, author/repository links, pinned revisions, original and converted file hashes, and a description of local changes in third-party notices and a machine-readable model manifest. Do not invent an upstream copyright holder where none is stated. MIT permits reuse and modification but requires preservation of the copyright and permission notice; a name-only credit is insufficient.
- Vendored minimal model architecture and project-authored adapter stay separate. Do not execute the upstream notebook, its data loader, or `torch.load` on untrusted files in the shipped runtime. Conversion is a pinned developer operation; the runtime loads ONNX only. Check transitive copied code/dependency notices before shipping.
- This is a deliberate exception to the prior source-only/no-weight packaging policy **for this reviewed MIT option only**. It does not change the status of AGPL-labeled external detectors or license-unreviewed EZCon data/derived weights.

## Architecture and contract

The existing v1 recognizer contract is fixed at grayscale `[B,1,64,160]`, 80 CTC steps and blank index 0. The author's notebook reads images through OpenCV in **BGR** order and uses a plate-centric color ROI, CPM at 100×100, a 94×48 perspective crop, and a 37-entry class list including `I` and `O`; its final `-` entry is used as the blank class, not as a printed separator. The project adapter's public input is RGB, so it must convert RGB to BGR once before following the author path and convert the aligned display crop back to RGB. Do **not** mislabel these ONNX models as a v1 `recognizer.onnx` or feed their logits to `decode_constrained_ctc_v1`.

Add a versioned external-recognizer manifest and a narrowly scoped adapter. It validates two ONNX hashes, input/output metadata, charset, preprocessing version, source attribution, and provider availability before use. The adapter accepts one RGB plate-centric ROI and returns raw text, normalized text, confidence/evidence marked as uncalibrated, the aligned crop, timings, and a rejection reason. It applies the pinned author resize/normalization/greedy path for parity; a separately identified safe-corners mode uses the existing convex-hull/topology validation before warping, with per-plate fail-closed rejection rather than a frame-wide exception. Neither mode silently changes the author's decoding rules.

For a full-scene photo, keep detector and OCR responsibilities separate: use the existing native detector's boxes/corners, extract a bounded ROI with configurable margin, and run the external adapter on each ROI. One bad ROI does not discard other detections. This model is **not** a replacement whole-image locator; no AGPL external detector is imported by this feature. The native M3a/CTC path remains unchanged.

The Web comparison path adds a fixed allowlisted option for this adapter alongside `active-v1` and the existing fixed `candidate-detector-real-v1` preview; `candidate-v2-retest` remains a local bundle, not an existing Web choice. Request data cannot name an arbitrary filesystem path. The comparison reports which detector and recognizer produced every result and keeps rejected candidates visible in diagnostics. The existing activation route must not copy this unlike-schema bundle onto `active-v1`; activation/default changes require their own explicit gate after evidence is reviewed. The source-only 1788 API scaffold is not silently upgraded to a working service by this change; a later release integration must include the inference path, notices, and model files together. Until a distributable full-scene detector is available, such a release may offer plate-crop recognition only and must state that boundary explicitly.

## Improvement loop and evaluation

1. Reproduce the pinned PyTorch implementation and convert CPM and LPRNet to ONNX. Compare per-stage tensors and decoded strings on fixed plate-centric fixtures, including repeated characters, skew, blur, and low light. Record exact environment, numerical tolerance, and mismatches.
2. Preserve the 3,032-image TLPD replay only as a compatibility regression: local prior result was `2,974/3,032 = 98.09%` with filename-derived, incompletely audited labels. The author's weights were trained on this source, so this is **not** a held-out accuracy estimate. Do not tune on this set and then call it a blind test.
3. Build a separate, consented/appropriately licensed real-photo development set and an untouched held-out set. Audit plate text, vehicle class, near duplicates, and ROI readability; report v1 new-style private-passenger plates separately from motorcycles and other formats. A 3-4 string alone does not establish v1 vehicle class.
4. Only after the original-path baseline is fixed, try one factor at a time: ROI margin, CPM versus validated M3a corners, contrast/white balance, or decoder calibration. Keep an ablation only if the development set improves without degrading held-out results. Weight fine-tuning is optional and requires additional training data with clear rights and a new untouched holdout; do not claim a TLPD split is blind for the author's already-trained weights.
5. Report two scorecards: given a verified plate crop, exact-string OCR/empty-output rate; and full scene, localization plus end-to-end exact-string results. Compare latency on the same provider/hardware and scope. No promised 98% real-world result or production readiness.

## Acceptance and safety gates

- The checked-in ONNX files load by hash and match the pinned PyTorch reference on fixtures; corrupt, missing, or wrong-contract artifacts fail closed with actionable diagnostics.
- Crop-level and multi-plate full-scene tests cover valid detections, invalid CPM corners, bounds, repeated characters, non-v1 text, and partial failures. Native `active-v1` behavior is unchanged when the option is not selected.
- A fixed set of user-supplied or independently sourced real photos is manually labeled and scored; the earlier `MDX-9717` motorcycle example is diagnostic, not evidence for private-passenger v1. TLPD replay and real-photo results are always labeled separately.
- Unit/integration tests, full local suite, ONNX parity, Web comparison, release packaging, and included notice/hash checks pass before promotion. Browser/IIS/production claims require their own evidence; no automatic push or deployment is implied.

## Out of scope

No new whole-scene detector, AGPL integration, TLPD image distribution, unlicensed EZCon model distribution, automatic `active-v1` replacement, retraining guarantee, or 98% production claim. Keep unrelated dirty Web/release/resource-center work untouched until its own review.
