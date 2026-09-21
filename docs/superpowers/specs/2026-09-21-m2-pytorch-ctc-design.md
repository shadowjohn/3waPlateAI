# M2 PyTorch CTC Recognition and ONNX Design

- Date: 2026-09-21
- Status: Design approved; amended after implementation-plan review
- Scope: M2 local crop recognizer only

## Intent and boundaries

M2 trains a compact, native PyTorch recognizer from the versioned M1 synthetic
crop dataset, evaluates it, exports a compatible ONNX recognizer, and assembles
a local-only crop-recognition Model Bundle. It does not implement a detector,
perspective rectifier, Reader runtime, grammar-constrained decoder, API, or
real-world accuracy claim. Checkpoints, ONNX files, generated datasets, and
Model Bundles remain ignored local artifacts and are never committed.

The only supported plate family is `tw-new-style-private-passenger-v1`: white
background, black glyphs, `LLL-DDDD` display, with canonical labels omitting the
decorative hyphen. Old plates, motorcycles, commercial plates, and other rule
sets are deliberately outside M2.

## Character and CTC contract

The bundle contains an unmodified copy of
`tw_new_style_private_passenger_v1.txt` and records its SHA-256. It is the
single source of truth for training, native decoding, export verification, and
future Reader decoding.

The ordered visible symbols are:

```text
0 1 2 3 5 6 7 8 9 A B C D E F G H J K L M N P Q R S T U V W X Y Z
```

This is 9 digits plus 24 letters, for 33 visible symbols. It excludes `4`,
`I`, and `O`, consistent with the approved v1 allocation profile. The file
contains no CTC blank symbol.

PyTorch CTC uses `blank_index = 0`; each line in the charset maps to its
one-based line number. The model therefore has 34 output classes:

```text
0: <blank>
1..9:  0 1 2 3 5 6 7 8 9
10..33: A B C D E F G H J K L M N P Q R S T U V W X Y Z
```

`class_count` is always 34. In particular, class index 4 means visible symbol
`3`; it does not make `4` legal. The manifest declares:

```json
{
  "type": "ctc",
  "blank_index": 0,
  "class_count": 34,
  "index_mapping": "charset-order-skipping-blank",
  "collapse_repeats": true
}
```

Encoding rejects characters outside the charset. Decoding takes the argmax
class at every timestep, collapses adjacent duplicate classes, removes index
0, and returns canonical text without a hyphen. The repeat-heavy legal label
`AAA8888` needs 12 CTC positions (seven targets plus five blank-separated
adjacent repeats). The recognizer produces 80 timesteps, leaving more than a
sixfold alignment budget rather than relying on the former 40-step minimum.

## Preprocessing and tensor contract

M1 produces RGB `380 x 160` crops in width-by-height order. M2 accepts only
that source size. A future rectifier must first produce this canonical crop;
it must not silently substitute a different plate ratio.

Each crop is transformed identically for training, native inference, and ONNX
inference:

1. Use Pillow 12.3.0 as the golden reference: construct
   `Image.fromarray(image_rgb, mode="RGB")` and call `.convert("L")`. M2
   does not reimplement the grayscale coefficients, rounding, or fixed-point
   behavior with OpenCV or NumPy.
2. On that `L` image, call
   `.resize((152, 64), resample=Image.Resampling.BILINEAR)` from Pillow 12.3.0
   to preserve aspect ratio from
   `160 x 380` height-by-width to `64 x 152`.
3. Letterbox it on a `64 x 160` height-by-width canvas with four pixels of raw
   white (`255`) on both left and right sides. No crop, vertical padding, or
   aspect-ratio stretch is permitted.
4. Convert to `float32` and divide every value, including padding, by `255.0`.
   No mean subtraction or standard-deviation normalization is applied in M2.
5. Arrange the tensor as `NCHW`, `[batch, 1, 64, 160]`.

The source plate aspect ratio is exactly `380 / 160 = 2.375`; it is not 3:1.
The fixed model canvas ratio is 2.5:1, with the narrow difference represented
only by the two white letterbox margins.

The Model Bundle uses the existing `manifest.json`, not a second `meta.json`,
and represents this procedure in its `preprocess` member. M2 extends the
schema so that the following exact structure is required:

```json
{
  "source_size_wh": [380, 160],
  "color_space": "grayscale",
  "layout": "NCHW",
  "input_size_hw": [64, 160],
  "grayscale": {
    "reference": "pillow-image-convert-l",
    "library_version": "12.3.0"
  },
  "resize": {
    "mode": "letterbox",
    "reference": "pillow-image-resize",
    "interpolation": "Resampling.BILINEAR",
    "library_version": "12.3.0",
    "resized_size_hw": [64, 152],
    "padding_ltrb": [4, 0, 4, 0],
    "padding_raw_value": 255
  },
  "normalization": {
    "type": "divide",
    "divisor": 255.0,
    "dtype": "float32"
  }
}
```

The image is converted to `float32` only after resize and letterboxing;
`padding_raw_value` is therefore unambiguously pre-normalization. A committed,
synthetic RGB preprocessing vector and the SHA-256 of its C-contiguous
preprocessed `float32` bytes are the conformance test for any future non-Pillow
Reader implementation. This avoids a separate metadata file drifting from the
exported graph.

## Native model and training

`plateai_trainer.training` provides a compact convolutional CTC recognizer
implemented only with PyTorch modules:

- a grayscale convolutional encoder with horizontal stride 2, which produces
  80 horizontal timesteps from the 160-pixel model canvas;
- three residual depthwise-separable one-dimensional temporal blocks with
  dilations 1, 2, and 4, which provide local sequence context without an RNN
  operator; and
- a pointwise 34-class classifier at every timestep.

The model is trained from scratch in M2. It imports no pretrained weights and
does not make an accuracy or license claim about an external recognition
model. It emits logits in `[batch, 80, 34]`; training applies
`log_softmax(class_dimension)` and PyTorch `CTCLoss(blank=0)` after converting
to `[80, batch, 34]`. Invalid targets or target lengths are rejected before
loss evaluation rather than being masked as successful samples.

The trainer accepts separate train and validation M1 output directories. It
checks each directory's labels, image dimensions, charset hash, rules hash,
and supported plate type before use. It records the resolved input hashes,
random seed, PyTorch version, command configuration, epoch metrics, and best
validation checkpoint in an ignored run directory. The validation dataset must
be generated separately from training with a different seed.

M2 evaluation reports exact canonical-plate accuracy, character accuracy,
sample count, invalid/rejected sample count, and CTC greedy-decoder output.
It does not claim production accuracy; synthetic metrics only measure the
local synthetic validation set.

## ONNX export and local bundle

The exporter loads a specified native checkpoint and the exact training
configuration, exports opset 17 ONNX with only the batch dimension dynamic,
and names tensors `input` and `logits`:

```text
input:  float32 [batch, 1, 64, 160]
logits: float32 [batch, 80, 34]
batch:  dynamic { min: 1, opt: 8, max: 32 }
```

Before publication, the exporter runs CPU PyTorch and ONNX Runtime on a
deterministic fixture batch of one and two canonical crops. It requires equal
tensor shapes, `numpy.allclose` with `rtol=1e-4` and `atol=1e-5`, and identical
greedy-decoded canonical strings. Any mismatch fails export and leaves no
published bundle directory.

The atomic local bundle contains `manifest.json`, `charset.txt`,
`plate_rules.json`, `recognizer.onnx`, and a JSON training/evaluation report.
The manifest records each file SHA-256, crop-recognition capability, dynamic
batch contract, the 34-class CTC contract, and the full preprocessing contract.
The shared validator rejects a missing file, mismatched hash, 33-or-35-class
recognizer, nonzero CTC blank, conflicting charset count, or incompatible
tensor/preprocessing declaration.

## Dependencies, tests, and acceptance

Training-only dependencies are declared as an explicit Python 3.11 extra:
PyTorch, ONNX, and ONNX Runtime. The base M1 generator installation remains
free of those heavyweight runtime dependencies. CPU tests use the same lock
profile as the local environment; GPU use is optional and never required for
correctness.

Tests are written before implementation and cover:

- the complete visible-symbol/index mapping, invalid-character rejection, CTC
  repeat collapsing, and blank removal;
- exact grayscale, resize, padding, normalization, NCHW shape, and
  reproducibility of preprocessing against the Pillow 12.3.0 vector;
- a repeat-heavy `AAA8888` CTC smoke path, including its required 12-timestep
  alignment and greedy decoding with blank-separated adjacent symbols;
- training dataset contract rejection for incompatible M1 metadata;
- native CRNN output shape and a small deterministic overfit smoke case;
- manifest schema and cross-field rejection for every M2-specific contract;
- ONNX tensor and decoded-output parity for batch sizes one and two; and
- atomic bundle publication and hash verification.

M2 is accepted locally when the complete test suite, a small CPU training run,
ONNX parity validation, and `pip check` pass. Browser, detector/rectifier,
GPU performance, real-world plate accuracy, remote CI, and production
deployment remain explicitly unverified.
