# M2 PyTorch CTC Recognition and ONNX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a local-only v1 Taiwan private-passenger crop recognizer with native PyTorch CTC, export a hash-verified ONNX bundle, and prove native/ONNX parity.

**Architecture:** Keep framework-neutral CTC, preprocessing, manifest, bundle-hash, and atomic-publication contracts in `plateai_shared`. Add a PyTorch-only Trainer package for validated M1 dataset loading, convolutional temporal training, and evaluation, plus a separate exporter package that creates an ONNX Runtime-verified local bundle. The recognizer sees only the approved 380x160 v1 crop profile; no detector, rectifier, Reader, or real-world dataset enters this milestone.

**Tech Stack:** CPython 3.11; NumPy 2.4.2; Pillow 12.3.0; PyTorch 2.14.0; ONNX 1.23.0; ONNX Runtime 1.30.0; ONNX Script 0.7.2; pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-09-21-m2-pytorch-ctc-design.md`

## Global Constraints

- Support only `tw-new-style-private-passenger-v1`: white background, black glyphs, `LLL-DDDD` display, and seven-symbol canonical labels without the hyphen.
- The only visible symbols are `012356789ABCDEFGHJKLMNPQRSTUVWXYZ`, in the exact order stored by `configs/charsets/tw_new_style_private_passenger_v1.txt`.
- CTC has `blank_index = 0`, `class_count = 34`, and `charset-order-skipping-blank`; visible `3` has class index 4 and visible `4` has no class.
- M2 accepts RGB width-by-height `380x160` input and creates only `float32` NCHW `[batch, 1, 64, 160]` tensors through Pillow 12.3.0 `Image.convert("L")`, Pillow 12.3.0 `Image.resize(..., Resampling.BILINEAR)`, and white letterboxing.
- ONNX opset is 17. Its only dynamic axis is batch, constrained in the manifest to min 1, opt 8, max 32; logits are `[batch, 80, 34]`.
- Training code is an optional Python 3.11 dependency set. The existing synthetic generator remains usable without PyTorch, ONNX, or ONNX Runtime installed.
- Models, checkpoints, ONNX files, Model Bundles, generated datasets, caches, and real images remain ignored local artifacts. Never add them to Git or a release.
- Preserve existing generic M1 fixtures and their 320x96 compatibility purpose. M2 defaults and M2 tests use the v1 380x160 profile.
- Each code task starts with a targeted failing test, ends with its targeted passing test and the full relevant test subset, and stages only the files named in that task's commit command.

## Review Focus

1. Adjacent duplicate CTC symbols such as `AAA8888` need 12 CTC positions and must retain one symbol from each run after blank removal; Tasks 2, 4, and 5 pin a forced-logit decoder vector, a rendered repeat-heavy synthetic record, and an 80-timestep training batch.
2. A data directory with a valid-looking label but a changed PNG hash, 320x96 image, or non-v1 plate type must be rejected before an epoch starts; Task 4 mutates each case independently.
3. The letterbox must contain four raw-value-255 pixels on each side before division, making the final margin exactly `1.0`; Task 2 asserts every resulting margin pixel and an interior grayscale value.
4. A manifest whose input says 3 channels, output says 33 or 35 classes, or preprocess dimensions disagree with tensors must fail cross-field validation; Task 3 covers all three variants.
5. An ONNX graph that is valid but emits numerically divergent logits or a different greedy decode must not publish a bundle; Task 6 injects each parity failure and checks that no output directory exists.

---

## File structure

```text
requirements/
  py311.training.direct.txt             # Direct M2 packages, exact versions
  py311.training.lock.txt               # Full CPython 3.11 M1+M2 dependency lock
src/
  plateai_shared/
    recognition.py                      # Charset-indexed CTC codec and image preprocessing
    publication.py                      # No-clobber owned-directory publication shared by M1 and M2
    bundle.py                           # Hash-verified, schema-validated crop-bundle loading
    schema_validation.py                # M2 tensor and preprocessing cross-field checks
  plateai_trainer/
    training/
      dataset.py                        # Verified M1 crop-dataset reader and PyTorch collation
      model.py                          # Compact fixed-shape CRNN
      engine.py                         # Deterministic epoch, checkpoint, and evaluation orchestration
      cli.py                            # `plateai-train` / module CLI
      __main__.py
    export/
      bundle.py                         # ONNX export, graph checks, parity, atomic bundle assembly
      cli.py                            # `plateai-export` / module CLI
      __main__.py
tests/
  contract/test_training_packaging.py
  unit/test_recognition.py
  unit/test_publication.py
  contract/test_model_bundle.py
  unit/test_training_dataset.py
  unit/test_training_model.py
  integration/test_training_cli.py
  integration/test_export_bundle.py
```

The package entry points `plateai-train` and `plateai-export` call their own
CLI modules. Neither is imported by `plateai-generate`; importing the base M1
package therefore does not import PyTorch.

### Task 1: Pin the optional CPU training toolchain and CI path

**Files:**
- Create: `requirements/py311.training.direct.txt`
- Create: `requirements/py311.training.lock.txt`
- Create: `tests/contract/test_training_packaging.py`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/test.yml`

**Interfaces:**
- Consumes: existing CPython 3.11 bounds, `requirements/py311.lock.txt`, and the M1 `test` extra.
- Produces: a `training` extra with exact direct packages and a full lock that CI installs before M2 tests. Task 5 adds `plateai-train`; Task 6 adds `plateai-export` with their real CLI modules.

- [ ] **Step 1: Write the failing packaging-contract test**

```python
def test_installed_distribution_exposes_the_isolated_training_extra():
    requirements = importlib.metadata.distribution("3wa-plate-ai").requires or []
    training = {
        requirement.name: str(requirement.specifier)
        for raw in requirements
        if "extra == 'training'" in raw
        for requirement in [packaging.requirements.Requirement(raw)]
    }
    assert training == {
        "torch": "==2.14.0",
        "onnx": "==1.23.0",
        "onnxruntime": "==1.30.0",
        "onnxscript": "==0.7.2",
    }
```

- [ ] **Step 2: Run the contract test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/contract/test_training_packaging.py -q`

Expected: FAIL because the `training` extra and direct lock do not yet exist.

- [ ] **Step 3: Add the isolated dependency contract and reproducible lock**

Add this exact project extra without console entry points; the entry points are
added alongside their executable implementations in Tasks 5 and 6:

```toml
[project.optional-dependencies]
training = [
  "torch==2.14.0",
  "onnx==1.23.0",
  "onnxruntime==1.30.0",
  "onnxscript==0.7.2",
]
```

Write `py311.training.direct.txt` with those four exact lines. In a fresh
CPython 3.11 virtual environment, install `py311.lock.txt` plus the direct
file, run `python -m pip freeze --exclude-editable`, and save the complete
sorted result as `py311.training.lock.txt`. Do not include the editable
project itself in the lock. Update CI to install that training lock before
`pip install --no-deps -e ".[test,training]"`, then retain the existing build
and generator smoke stages. Reinstall the editable package after editing
`pyproject.toml` so the test reads the new distribution metadata.

- [ ] **Step 4: Run the targeted and base tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/contract/test_training_packaging.py tests/unit/test_rules.py -q`

Expected: PASS, proving the exact optional dependency declaration does not
change the M1 rule loader.

- [ ] **Step 5: Commit the dependency boundary**

```bash
git add pyproject.toml requirements/py311.training.direct.txt requirements/py311.training.lock.txt .github/workflows/test.yml tests/contract/test_training_packaging.py
git commit -m "build: add isolated PyTorch training dependencies"
```

### Task 2: Implement the v1 CTC codec and Pillow-golden image preprocessing

**Files:**
- Create: `src/plateai_shared/recognition.py`
- Create: `tests/unit/test_recognition.py`
- Modify: `src/plateai_shared/__init__.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: `CharacterSet` from `plateai_shared.contracts`, NumPy RGB `uint8` arrays, and the v1 charset file.
- Produces: `CTCCodec.from_charset(charset) -> CTCCodec`, `CTCCodec.encode(text: str) -> tuple[int, ...]`, `CTCCodec.decode_greedy(indices: Sequence[int]) -> str`, `V1_PREPROCESS`, and `preprocess_v1_rgb(image_rgb: NDArray[np.uint8]) -> NDArray[np.float32]`.

- [ ] **Step 1: Write failing codec and preprocessing tests**

```python
def test_v1_ctc_indices_are_blank_zero_and_charset_order_skips_blank(v1_charset):
    codec = CTCCodec.from_charset(v1_charset)
    assert codec.blank_index == 0
    assert codec.class_count == 34
    assert codec.encode("035AZ") == (1, 4, 5, 10, 33)
    with pytest.raises(ValueError, match="not in charset"):
        codec.encode("A4I")


def test_greedy_ctc_decoder_preserves_repeat_heavy_plate_with_blank_runs(v1_charset):
    codec = CTCCodec.from_charset(v1_charset)
    indices = [10, 0, 10, 0, 10, 8, 0, 8, 0, 8, 0, 8]
    assert codec.required_timesteps(codec.encode("AAA8888")) == 12
    assert codec.decode_greedy(indices) == "AAA8888"


def test_preprocess_v1_matches_committed_pillow_12_3_golden_vector():
    image = np.asarray(Image.open(PREPROCESS_FIXTURE).convert("RGB"))
    tensor = preprocess_v1_rgb(image)
    expected = np.load(PREPROCESS_EXPECTED)
    assert tensor.shape == (1, 64, 160)
    assert tensor.dtype == np.float32
    np.testing.assert_array_equal(tensor, expected)
    np.testing.assert_array_equal(tensor[:, :, :4], np.ones((1, 64, 4)))
    np.testing.assert_array_equal(tensor[:, :, -4:], np.ones((1, 64, 4)))
```

- [ ] **Step 2: Run the unit test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_recognition.py -q`

Expected: FAIL because `plateai_shared.recognition` does not exist.

- [ ] **Step 3: Add the pure shared contract implementation**

Implement a frozen `PreprocessSpec` with these exact values:

```python
V1_PREPROCESS = PreprocessSpec(
    source_size_wh=(380, 160),
    input_size_hw=(64, 160),
    resized_size_hw=(64, 152),
    padding_ltrb=(4, 0, 4, 0),
    padding_raw_value=255,
    grayscale_reference="pillow-image-convert-l",
    resize_reference="pillow-image-resize-bilinear",
    pillow_version="12.3.0",
)
```

`CTCCodec.from_charset` creates a one-based symbol mapping and rejects a
charset containing a duplicate or a blank. `decode_greedy` first collapses
only adjacent identical indices, then removes index zero, then rejects every
remaining index outside `0..33`. Add `required_timesteps(target)` as
`len(target) + count(adjacent equal pairs)`, so `AAA8888` returns 12.
`preprocess_v1_rgb` rejects arrays other than `uint8` RGB `[160, 380, 3]`,
calls exactly `Image.fromarray(image_rgb, mode="RGB").convert("L")`, calls
exactly `.resize((152, 64), resample=Image.Resampling.BILINEAR)`, places it in
a raw `uint8` white `(64, 160)` canvas at x=4, divides by 255 after padding,
and returns a C-contiguous `[1, 64, 160]` `float32` array. The conformance
test constructs a deterministic synthetic RGB vector in source and checks the
SHA-256 of its C-contiguous preprocessed `float32` bytes, generated with Pillow
12.3.0 during this task. Later implementations may replace Pillow only after
producing that identical digest.

Add `V1_CHARSET`, `V1_RULES`, `V1_TEMPLATE`, and `V1_NONE_AUGMENTATION`
constants plus a `v1_charset` fixture to `tests.conftest.py`; point them at
the versioned v1 configuration files rather than changing the generic M1
fixtures.

- [ ] **Step 4: Run the targeted tests and M1 renderer test**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_recognition.py tests/unit/test_renderer.py -q`

Expected: PASS, including the exact index-4, repeated-symbol, source-shape,
and white-margin assertions.

- [ ] **Step 5: Commit the framework-neutral recognition contract**

```bash
git add src/plateai_shared/__init__.py src/plateai_shared/recognition.py tests/conftest.py tests/unit/test_recognition.py
git commit -m "feat: add v1 CTC and preprocessing contracts"
```

### Task 3: Extend the Model Bundle schema and cross-field validator for M2

**Files:**
- Modify: `schemas/model_manifest.schema.json`
- Modify: `src/plateai_shared/schema_validation.py`
- Modify: `tests/contract/test_schemas.py`
- Create: `tests/contract/test_model_bundle.py`

**Interfaces:**
- Consumes: `V1_PREPROCESS`, `CTCCodec.class_count`, the existing Draft 2020-12 validator, and a JSON manifest dictionary.
- Produces: `validate_model_manifest(document, schema_path, visible_charset_symbol_count=33) -> None` that accepts only the M2 crop-recognition tensor, batch, decoder, and preprocessing contracts.

- [ ] **Step 1: Write failing M2 manifest tests**

```python
def test_v1_crop_manifest_accepts_exact_preprocess_and_tensor_contract():
    validate_model_manifest(valid_v1_manifest(), SCHEMA, visible_charset_symbol_count=33)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("components", "recognizer", "inputs", 0, "shape"), ["batch", 3, 64, 160], "inputs"),
        (("components", "recognizer", "outputs", 0, "shape"), ["batch", 80, 33], "outputs"),
        (("preprocess", "resize", "padding_ltrb"), [3, 0, 5, 0], "padding_ltrb"),
    ],
)
def test_v1_manifest_rejects_incompatible_cross_fields(path, value, message):
    manifest = valid_v1_manifest()
    set_path(manifest, path, value)
    with pytest.raises(DocumentValidationError, match=message):
        validate_model_manifest(manifest, SCHEMA, visible_charset_symbol_count=33)
```

- [ ] **Step 2: Run the manifest tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/contract/test_schemas.py tests/contract/test_model_bundle.py -q`

Expected: FAIL because `preprocess` still accepts the former RGB scale-only
shape and the validator does not inspect recognizer tensors.

- [ ] **Step 3: Replace the ambiguous preprocessing schema and enforce invariants**

Make `preprocess` require the exact object in the approved specification:
`source_size_wh`, `color_space`, `layout`, `input_size_hw`, `grayscale`,
`resize`, and `normalization`. Constrain `color_space` to `grayscale`,
layout to `NCHW`, source `[380, 160]`, input `[64, 160]`, grayscale reference
`pillow-image-convert-l` at library version `12.3.0`, resize reference
`pillow-image-resize` with `Resampling.BILINEAR` at library version `12.3.0`,
resize mode `letterbox`, resized `[64, 152]`, padding `[4, 0, 4, 0]`, raw
padding 255, float32 output, and divisor 255.0.
Extend `provenance` with a required `training_report` object containing a
schema-approved filename and SHA-256; the bundle must checksum its report just
as it does the charset, rules, and recognizer.

In `validate_model_manifest`, after schema validation, require all of these
cross-field facts:

```python
assert document["plate_size"] == [380, 160]
assert recognizer["inputs"] == [{
    "name": "input", "dtype": "float32", "shape": ["batch", 1, 64, 160]
}]
assert recognizer["outputs"] == [{
    "name": "logits", "dtype": "float32", "shape": ["batch", 80, 34]
}]
assert decoder["blank_index"] == 0
assert decoder["class_count"] == visible_charset_symbol_count + 1 == 34
assert batch == {"mode": "dynamic", "min": 1, "opt": 8, "max": 32}
```

Use explicit `DocumentValidationError` messages instead of Python `assert` so
the CLI and tests receive deterministic path-addressed errors. Update
`valid_crop_only_manifest` to the v1 33-symbol fixture while retaining the
existing detector-capability tests with a separately valid recognizer. Rename
the test helper to `valid_v1_manifest()` and define its `set_path` helper in
the same test module.

- [ ] **Step 4: Run all schema and shared-contract tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/contract/test_schemas.py tests/contract/test_model_bundle.py tests/unit/test_recognition.py -q`

Expected: PASS; malformed preprocess metadata, nonzero blank, wrong class
count, and non-dynamic batch contracts all report a stable failure.

- [ ] **Step 5: Commit the M2 manifest contract**

```bash
git add schemas/model_manifest.schema.json src/plateai_shared/schema_validation.py tests/contract/test_schemas.py tests/contract/test_model_bundle.py
git commit -m "feat: define M2 recognizer bundle contract"
```

### Task 4: Reuse no-clobber publication and load only validated M1 v1 datasets

**Files:**
- Create: `src/plateai_shared/publication.py`
- Modify: `src/plateai_trainer/synthetic/dataset.py`
- Create: `src/plateai_trainer/training/__init__.py`
- Create: `src/plateai_trainer/training/dataset.py`
- Create: `tests/unit/test_publication.py`
- Create: `tests/unit/test_training_dataset.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: an M1 directory containing `generation_config.json`, `summary.json`, `labels.txt`, `metadata.jsonl`, and image files; the v1 charset/rules SHA-256 values; `preprocess_v1_rgb` and `CTCCodec`.
- Produces: `publish_directory_no_replace(staging: Path, output: Path) -> None`, `remove_owned_staging(staging: Path, output: Path) -> None`, `M1CropDataset(root, charset, rules_path)`, and `collate_crop_samples(samples) -> tuple[Tensor, Tensor, Tensor]`.

- [ ] **Step 1: Write failing publication and dataset-contract tests**

```python
def test_publication_refuses_an_existing_target_without_modifying_it(tmp_path):
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "sentinel.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(OutputExistsError):
        publish_directory_no_replace(tmp_path / "staging", output)
    assert (output / "sentinel.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("mutation", ["wrong_image_size", "changed_png", "wrong_plate_type"])
def test_m1_v1_dataset_rejects_tampered_or_incompatible_inputs(v1_dataset, mutation):
    mutate_dataset(v1_dataset, mutation)
    with pytest.raises(TrainingDataError):
        M1CropDataset(v1_dataset, V1_CHARSET, V1_RULES)


def test_repeat_heavy_synthetic_record_is_a_valid_12_step_ctc_target(v1_repeat_dataset):
    dataset = M1CropDataset(v1_repeat_dataset, V1_CHARSET, V1_RULES)
    _, target = dataset[0]
    assert CTCCodec.from_charset(load_character_set(V1_CHARSET)).required_timesteps(target) == 12
```

- [ ] **Step 2: Run the test files to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_publication.py tests/unit/test_training_dataset.py -q`

Expected: FAIL because shared publication helpers and the M1 dataset reader do
not exist.

- [ ] **Step 3: Extract publication safely and implement eager input validation**

Move the existing platform-specific no-clobber rename logic and owned-sibling
cleanup guard from `synthetic.dataset` into `plateai_shared.publication`; keep
M1 exception behavior and adapt the M1 tests if imports move. Add v1 constants
to `tests.conftest.py` for the charset, rule, template, and no-augmentation
profile so test datasets are generated by the real M1 pipeline.

Add a `v1_repeat_dataset` fixture that renders `GeneratedPlate("AAA8888",
"AAA-8888", "new-style-private-passenger-lll-dddd",
"new-style-private-passenger")` with the real v1 template and bundled OFL
font, then writes the five M1 files with matching image/configuration hashes.
This is a synthetic test-only record, not a production rule or committed model
artifact.

`M1CropDataset.__init__` must read and validate every required document before
creating a PyTorch dataset object: one tab per `labels.txt` line; no duplicate
relative image paths; labels that encode with the 33-symbol codec; one metadata
record per label with equal canonical text; type
`new-style-private-passenger`; actual PNG SHA-256 equal to metadata; RGB
mode and `(380, 160)` Pillow size; `generation_config.config_sha256.charset`
and `.rules` equal to the requested files; and summary hashes equal the same
values. Expose the generated seed as a read-only dataset property. Reject
absolute, parent-escaping, missing, or extra label image paths.

`__getitem__` reads only a previously verified path, returns the shared
preprocessed NumPy image and integer targets, and `collate_crop_samples`
creates `torch.float32 [batch, 1, 64, 160]`, concatenated `torch.int64`
targets, and `torch.int64` target lengths. Import PyTorch inside collation so
base M1 module imports remain free of training dependencies.

- [ ] **Step 4: Run M1 transactional and training dataset tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_dataset.py tests/unit/test_publication.py tests/unit/test_training_dataset.py -q`

Expected: PASS; M1 preserves its transactional semantics and the training
reader rejects all three review-focus mutations before collation.

- [ ] **Step 5: Commit validated-data and shared-publication work**

```bash
git add src/plateai_shared/publication.py src/plateai_trainer/synthetic/dataset.py src/plateai_trainer/training/__init__.py src/plateai_trainer/training/dataset.py tests/conftest.py tests/unit/test_dataset.py tests/unit/test_publication.py tests/unit/test_training_dataset.py
git commit -m "feat: validate v1 crops before PyTorch training"
```

### Task 5: Build the compact CRNN, deterministic trainer, metrics, and train CLI

**Files:**
- Create: `src/plateai_trainer/training/model.py`
- Create: `src/plateai_trainer/training/engine.py`
- Create: `src/plateai_trainer/training/cli.py`
- Create: `src/plateai_trainer/training/__main__.py`
- Create: `tests/unit/test_training_model.py`
- Create: `tests/integration/test_training_cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `M1CropDataset`, `collate_crop_samples`, `CTCCodec`, and PyTorch 2.14.0.
- Produces: `ConvBlock`, `TemporalDepthwiseBlock`, `PlateCTCNet(class_count: int = 34)`, `train_one_batch(model, batch, optimizer) -> BatchResult`, `train_recognizer(config: TrainingConfig) -> TrainingRun`, `evaluate_recognizer(model, loader, codec, device) -> EvaluationReport`, and `plateai-train`.

- [ ] **Step 1: Write failing model, metric, and CLI smoke tests**

```python
def test_plate_ctc_net_produces_80_timestep_34_class_logits():
    logits = PlateCTCNet()(torch.zeros((2, 1, 64, 160), dtype=torch.float32))
    assert tuple(logits.shape) == (2, 80, 34)


def test_ctc_training_step_is_finite_and_updates_a_parameter(v1_batch):
    torch.manual_seed(7)
    model = PlateCTCNet()
    before = next(model.parameters()).detach().clone()
    result = train_one_batch(model, v1_batch, torch.optim.AdamW(model.parameters(), lr=1e-3))
    assert math.isfinite(result.loss)
    assert not torch.equal(before, next(model.parameters()).detach())


def test_repeat_heavy_ctc_batch_has_80_steps_for_a_12_step_alignment(v1_repeat_batch):
    model = PlateCTCNet()
    logits = model(v1_repeat_batch.images)
    assert logits.shape[1] == 80
    assert v1_repeat_batch.required_timesteps == [12]


def test_train_cli_writes_checkpoint_and_synthetic_metrics(v1_train_dir, v1_validation_dir, tmp_path):
    result = main(["--train", str(v1_train_dir), "--validation", str(v1_validation_dir), "--output", str(tmp_path / "run"), "--epochs", "1", "--batch-size", "2", "--seed", "7"])
    assert result == 0
    assert (tmp_path / "run" / "best.pt").is_file()
    assert json.loads((tmp_path / "run" / "report.json").read_text(encoding="utf-8"))["validation"]["samples"] == 4
```

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_training_model.py tests/integration/test_training_cli.py -q`

Expected: FAIL because the model, training engine, and CLI do not exist.

- [ ] **Step 3: Implement a fixed-shape CRNN and deterministic training loop**

Implement `PlateCTCNet` as this fixed topology:

```python
self.encoder = nn.Sequential(
    ConvBlock(1, 32, pool=(2, 2)),   # [B, 32, 32, 80]
    ConvBlock(32, 64, pool=(2, 1)),  # [B, 64, 16, 80]
    ConvBlock(64, 128, pool=(2, 1)), # [B, 128, 8, 80]
    nn.AdaptiveAvgPool2d((1, 80)),
)
self.temporal = nn.Sequential(
    TemporalDepthwiseBlock(128, dilation=1),
    TemporalDepthwiseBlock(128, dilation=2),
    TemporalDepthwiseBlock(128, dilation=4),
)
self.classifier = nn.Conv1d(128, 34, kernel_size=1)
```

Define `ConvBlock(in_channels, out_channels, pool)` as a `Conv2d` with
kernel 3 and padding 1, followed by `BatchNorm2d`, `ReLU(inplace=True)`, and
`MaxPool2d(pool)` only when `pool` is not `None`. Define
`TemporalDepthwiseBlock(channels, dilation)` as a residual `Conv1d` sequence:
depthwise `Conv1d(channels, channels, kernel_size=3, padding=dilation,
dilation=dilation, groups=channels, bias=False)`, `BatchNorm1d`, ReLU,
pointwise `Conv1d(channels, channels, kernel_size=1, bias=False)`,
`BatchNorm1d`, then `ReLU(residual + branch)`. `train_one_batch` owns the
single `zero_grad(set_to_none=True) -> loss.backward() -> optimizer.step()`
sequence and returns `BatchResult(loss: float)`.

`forward` rejects any shape other than `[batch, 1, 64, 160]`, runs the
encoder, squeezes the one-pixel height to `[batch, 128, 80]`, applies the
three temporal blocks and pointwise classifier, then transposes to
`[batch, 80, 34]`. `TrainingConfig` has exactly these CLI-backed
fields: train directory, validation directory, output directory, epochs,
batch size, learning rate, seed, and device. Default to CPU, epochs 10,
batch size 32, learning rate `1e-3`, seed 42.

Set Python, NumPy, and Torch seeds; use a seeded training DataLoader
generator and `num_workers=0`. Apply `F.log_softmax(logits, dim=2)`, transpose
to time-major, and call `nn.CTCLoss(blank=0, zero_infinity=False)`. Before the
loss, reject a target whose required CTC path length exceeds 80. Run the
repeat-heavy `AAA8888` fixture through a finite CTC-loss smoke path. Save only the
best validation exact-plate checkpoint atomically in the run directory with
the model state dict, class count, charset SHA-256, rules SHA-256, and input
preprocessing object. `report.json` records train loss plus validation sample
count, exact canonical-plate accuracy, character accuracy, rejected count,
seed, input hashes, PyTorch version, and resolved config.

The CLI refuses an existing output directory, a train/validation seed match,
and nonpositive epochs, batch size, or learning rate. It catches only expected
data/configuration errors, prints the stable message to stderr, and returns
exit code 2; unexpected exceptions remain visible to CI.

Add `plateai-train = "plateai_trainer.training.cli:main"` to
`[project.scripts]` only after the module exists, reinstall the editable
package, and assert the installed command's `--help` exits zero in the CLI
integration test.

- [ ] **Step 4: Run model, CLI, and M1 integration tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_training_model.py tests/integration/test_training_cli.py tests/integration/test_cli.py -q`

Expected: PASS; the short CPU run emits a local checkpoint/report, while the
existing generator CLI retains its overwrite refusal behavior.

- [ ] **Step 5: Commit the native PyTorch trainer**

```bash
git add pyproject.toml src/plateai_trainer/training/model.py src/plateai_trainer/training/engine.py src/plateai_trainer/training/cli.py src/plateai_trainer/training/__main__.py tests/unit/test_training_model.py tests/integration/test_training_cli.py
git commit -m "feat: train v1 recognizer with PyTorch CTC"
```

### Task 6: Export, verify, and atomically publish a local ONNX crop bundle

**Files:**
- Create: `src/plateai_shared/bundle.py`
- Create: `src/plateai_trainer/export/__init__.py`
- Create: `src/plateai_trainer/export/bundle.py`
- Create: `src/plateai_trainer/export/cli.py`
- Create: `src/plateai_trainer/export/__main__.py`
- Create: `tests/integration/test_export_bundle.py`
- Modify: `tests/contract/test_model_bundle.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: a M2 `best.pt` checkpoint, `report.json`, the exact v1 charset and rules files, `PlateCTCNet`, `V1_PREPROCESS`, `validate_model_manifest`, ONNX 1.23.0, and ONNX Runtime CPU.
- Produces: `ExportRequest`, `export_crop_bundle(request: ExportRequest, session_factory: Callable[[Path], OrtSession] = create_cpu_session) -> Path`, and `validate_crop_bundle(bundle_dir: Path, schema_path: Path) -> Mapping[str, JsonValue]`.

`OrtSession` is a local `typing.Protocol` in `plateai_trainer.export.bundle`
with `get_inputs()`, `get_outputs()`, and
`run(output_names: Sequence[str] | None, input_feed: Mapping[str, NDArray[np.float32]])`.
`create_cpu_session` returns an `onnxruntime.InferenceSession` configured with
`providers=["CPUExecutionProvider"]`.

- [ ] **Step 1: Write failing export and corruption tests**

```python
def test_exported_bundle_has_v1_manifest_and_native_onnx_parity(trained_run, tmp_path):
    bundle = export_crop_bundle(export_request(trained_run, tmp_path / "bundle"))
    manifest = validate_crop_bundle(bundle, SCHEMA)
    assert manifest["decoder"]["blank_index"] == 0
    assert manifest["components"]["recognizer"]["outputs"][0]["shape"] == ["batch", 80, 34]


@pytest.mark.parametrize("failure", ["logit_divergence", "decode_divergence"])
def test_export_refuses_parity_failure_without_publishing_output(trained_run, tmp_path, failure):
    with pytest.raises(ExportParityError):
        export_crop_bundle(export_request(trained_run, tmp_path / "bad"), session_factory=failing_session(failure))
    assert not (tmp_path / "bad").exists()
```

Define `failing_session(failure)` in this test module as a wrapper around the
real CPU session: for `logit_divergence`, add `0.01` to one logit; for
`decode_divergence`, replace every timestep argmax with class 10. Both wrappers
retain the real output tensor shape so the exporter exercises parity rather
than a shape-error branch.

- [ ] **Step 2: Run export tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/integration/test_export_bundle.py tests/contract/test_model_bundle.py -q`

Expected: FAIL because no exporter, bundle reader, or ONNX parity path exists.

- [ ] **Step 3: Implement graph checks, parity, bundle hashing, and safe publication**

Load checkpoint weights with `torch.load(..., map_location="cpu", weights_only=True)`
and reject mismatched class count, charset hash, rules hash, or preprocessing
object before export. Put the model in `eval()` mode and export with:

```python
torch.onnx.export(
    model,
    (torch.zeros((1, 1, 64, 160), dtype=torch.float32),),
    onnx_path,
    input_names=["input"],
    output_names=["logits"],
    opset_version=17,
    dynamo=True,
    dynamic_shapes={"input": {0: torch.export.Dim("batch", min=1, max=32)}},
)
```

Run `onnx.checker.check_model(onnx_path, full_check=True)`, create an
ONNX Runtime CPU session, and inspect its input/output names, dtype, ranks,
and fixed dimensions. For deterministic preprocessed batches of sizes one and
two, run native and ONNX logits; require equal shapes,
`numpy.allclose(rtol=1e-4, atol=1e-5)`, and identical `CTCCodec` greedy text.

Write a sibling staging directory containing exactly `manifest.json`,
`charset.txt`, `plate_rules.json`, `recognizer.onnx`, and `report.json`.
Hash each file and store its name/hash in `charset`, `rules`, `components`,
and `provenance.training_report`. Validate the staged directory with
`validate_crop_bundle`, then publish through `publish_directory_no_replace`.
`validate_crop_bundle` loads only schema-approved filenames, confirms every
required file is a regular file inside the bundle, hashes each declared file,
loads the exact 33-symbol charset, and calls `validate_model_manifest` with
its actual symbol count.

Add `plateai-export = "plateai_trainer.export.cli:main"` to
`[project.scripts]` only after the module exists, reinstall the editable
package, and assert the installed command's `--help` exits zero in the export
integration test.

- [ ] **Step 4: Run export, schema, and package tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/integration/test_export_bundle.py tests/contract/test_model_bundle.py tests/contract/test_schemas.py -q`

Expected: PASS; batch-one and batch-two parity pass, corrupt hashes fail, and
both injected parity mismatches leave no final bundle directory.

- [ ] **Step 5: Commit the ONNX exporter and bundle verifier**

```bash
git add pyproject.toml src/plateai_shared/bundle.py src/plateai_trainer/export/__init__.py src/plateai_trainer/export/bundle.py src/plateai_trainer/export/cli.py src/plateai_trainer/export/__main__.py tests/integration/test_export_bundle.py tests/contract/test_model_bundle.py
git commit -m "feat: export verified local ONNX recognizer bundles"
```

### Task 7: Document the executable local workflow and complete M2 verification

**Files:**
- Modify: `README.md`
- Modify: `docs/training.md`
- Modify: `models/README.md`
- Modify: `history.md`

**Interfaces:**
- Consumes: implemented `plateai-train` and `plateai-export` commands, an M1-generated v1 train/validation pair, and their generated local output paths.
- Produces: user-facing instructions that make the optional training environment, CPU run, export/parity acceptance, artifact policy, and unverified boundaries explicit.

- [ ] **Step 1: Write failing documentation-contract assertions**

```python
def test_training_docs_publish_the_v1_input_and_onnx_parity_contract():
    text = (ROOT / "docs/training.md").read_text(encoding="utf-8")
    assert "[batch, 1, 64, 160]" in text
    assert "blank_index = 0" in text
    assert "native-versus-ONNX parity" in text
    assert "not committed" in text
```

- [ ] **Step 2: Run the documentation test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/contract/test_training_packaging.py::test_training_docs_publish_the_v1_input_and_onnx_parity_contract -q`

Expected: FAIL because the current guide describes M1 only.

- [ ] **Step 3: Write complete local commands and operational boundaries**

Document this CPU-only sequence, using distinct synthetic seeds and nonexisting
output paths:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements\py311.training.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e ".[test,training]"
.\.venv\Scripts\plateai-train --train out\train-seed-42 --validation out\validation-seed-43 --output runs\v1-cpu --epochs 10 --batch-size 32 --seed 42
.\.venv\Scripts\plateai-export --checkpoint runs\v1-cpu\best.pt --report runs\v1-cpu\report.json --output models\bundles\v1-local --model-id tw-private-v1 --version 0.1.0
```

Explain the 33 visible symbols, blank index zero, 34 output classes, `380x160`
source, the exact grayscale/letterbox/normalization procedure, CPU parity
requirements, and the fact that synthetic metrics are not field accuracy.
State in all three documents that weights, ONNX files, datasets, runs, and
bundles remain local and ignored. Append M2 implementation evidence, commands,
and unverified production/GPU/real-data boundaries to `history.md` only after
the checks in Step 4 have succeeded.

- [ ] **Step 4: Run the complete local verification matrix**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps .\dist\3wa_plate_ai-0.1.0.dev0-py3-none-any.whl
.\.venv\Scripts\python.exe -m pip check
git -c core.whitespace=cr-at-eol diff --check
```

Expected: all tests pass with the training stack installed; wheel build and
force-installed CLI smoke pass; `pip check` reports no broken requirements;
and Git reports no whitespace errors. Record only these local facts, not
remote CI, browser, GPU, production, or real-world accuracy claims.

- [ ] **Step 5: Commit documentation and verified M2 history**

```bash
git add README.md docs/training.md models/README.md history.md tests/contract/test_training_packaging.py
git commit -m "docs: document local v1 recognizer training workflow"
```

## Spec coverage self-review

| Specification requirement | Implementing task |
|---|---|
| Exact 33-symbol ordering, blank zero, 34 classes, and CTC repeat behavior | Task 2 |
| Fixed 380x160 RGB to grayscale 64x160 NCHW preprocessing | Tasks 2 and 3 |
| Compact 80-step convolutional CTC recognizer and CPU deterministic PyTorch training | Task 5 |
| M1 source hashes, dimensions, labels, metadata, plate type, and split seed validation | Task 4 |
| ONNX opset 17, dynamic batch only, tensor contract, checker, and ORT parity | Task 6 |
| Atomic local bundle, filename/hash verification, and no-clobber failure | Tasks 4 and 6 |
| Optional dependency boundary, reproducible lock, CI, commands, and public artifact policy | Tasks 1 and 7 |
| Local-only acceptance and explicit unverified boundaries | Task 7 |

Plan self-review completed: every specification requirement maps to a task;
all defined function names and tensor shapes are consistent across tasks; the
five review-focus conditions have a named owner and test; and the plan contains
no unresolved placeholders.
