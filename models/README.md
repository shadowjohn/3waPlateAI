# Local model workspace

This directory is documentation only. No checkpoint, trained weight, ONNX file, TensorRT engine, or Model Bundle is committed here or attached to an official 3waPlateAI release.

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

Validate local manifests with `schemas/model_manifest.schema.json` plus `plateai_shared.schema_validation.validate_model_manifest`. Users who distribute a bundle must include a bundle-specific `THIRD_PARTY_LICENSES.md` covering the trained artifact, training data, fonts, model architecture/code, converters, runtimes, and other incorporated material.

The repository's `.gitignore` blocks common model extensions and `models/bundles/`. Keep private tuning results, production bundles, and benchmark output in access-controlled storage appropriate to your own project.
