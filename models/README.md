# Local model workspace

This directory is documentation only. No checkpoint, trained weight, ONNX file, TensorRT engine, dataset, run, or Model Bundle is committed here or attached to an official 3waPlateAI release. Local v1 bundles are published only after native-versus-ONNX parity and hash verification.

Train and tune locally, then create an immutable directory outside Git such as:

```text
twplate-v0.1.0/
├── manifest.json
├── charset.txt
├── plate_rules.json
├── detector.onnx
├── recognizer.onnx
├── benchmark.json
└── THIRD_PARTY_LICENSES.md
```

A crop-recognition development bundle may omit the detector only when its manifest declares `crop-recognition`. A full detection bundle must declare the four semantic keypoints in this order:

1. `left_top`
2. `right_top`
3. `right_bottom`
4. `left_bottom`

It must also declare `rectifier.normalization_strategy` as `convex-hull-semantic-v1`, so raw pose output is never treated as already normalized. Recognizers declare dynamic batch limits or fixed-batch padding behavior, and CTC decoders declare an explicit integer blank index.

M3a implements that strategy as a local deterministic component: it converts an arbitrary ordering of a convex four-corner set into `left_top`, `right_top`, `right_bottom`, `left_bottom`, then warps exactly to the M2 RGB `380×160` source contract. M3b's native detector consumes a 640x640 OpenCV RGB letterbox and exposes pre-NMS `[batch,8400,13]` candidates. Deterministic Reader postprocessing sends every retained four-corner detection independently to M3a, so one rejected pose does not discard the other retained detections.

Build a full local bundle only with the [M3b detector workflow](../docs/training.md#local-m3b-detector-workflow), a user-authorized background manifest, and an existing local recognizer bundle. This repository still publishes no background, detector weight, checkpoint, ONNX artifact, TensorRT benchmark, browser integration, or production recognition metric; source-tree acceptance is local only.

Validate local manifests with `schemas/model_manifest.schema.json` plus `plateai_shared.schema_validation.validate_model_manifest`. Users who distribute a bundle must include a bundle-specific `THIRD_PARTY_LICENSES.md` covering the trained artifact, training data, fonts, model architecture/code, converters, runtimes, and other incorporated material.

The repository's `.gitignore` blocks common model extensions and `models/bundles/`. Keep private tuning results, production bundles, and benchmark output in access-controlled storage appropriate to your own project.
