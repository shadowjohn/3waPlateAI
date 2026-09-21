# 3waPlateAI M1 Synthetic Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic command-line generator that emits synthetic Taiwan-style plate crops, PaddleOCR recognition labels, transform metadata, validated shared contracts, and a tiny harmless fixture set for public CI without requiring a GPU.

**Architecture:** `plateai_shared` owns immutable rules and cross-module contracts. `plateai_trainer.synthetic` composes rule generation, clean rendering, deterministic augmentation, and transactional dataset writing. Configuration stays in versioned JSON files; M1 produces recognizer crops only and leaves model training, pose detection, Reader runtime, and the website demo to separate plans.

**Tech Stack:** CPython 3.11, stdlib `argparse`/`dataclasses`/`pathlib`, NumPy 2.4.x, OpenCV Headless 5.0.x, Pillow 12.x, JSON Schema Draft 2020-12, pytest 9.x, Hypothesis 6.x.

**Spec:** `docs/superpowers/specs/2026-09-21-3wa-plate-ai-design.md`

## Global Constraints

- Target CPython is exactly 3.11 on Linux for M1 development and CI.
- CPU execution must support every M1 feature; GPU acceleration is optional and unused by this plan.
- Internal image arrays are `numpy.ndarray` values with shape `(height, width, 3)`, dtype `uint8`, and RGB channel order.
- Recognition labels contain canonical text without decorative separators; rendered images contain display text with separators.
- `charset.txt` contains visible output symbols only. A CTC blank is never written into that file; every recognizer manifest declares its exact integer `blank_index`, `class_count`, and `charset-order-skipping-blank` mapping.
- Corner order is always `left_top`, `right_top`, `right_bottom`, `left_bottom`, stored as four `[x, y]` float pairs.
- M1 corners come from known synthetic transforms. Detector-output corner normalization belongs to M3/M4; the M1 manifest schema must still reserve the full-bundle rectifier contract so raw pose output can never be treated as normalized implicitly.
- The same run seed, sample index, configuration files, and dependency lock must reproduce the same plate strings and transform parameters.
- Deterministic output files must not contain wall-clock timestamps, UUIDs, absolute repository paths, or staging-directory names.
- Real vehicle images and personally identifying plate data must never enter the repository or M1 fixtures.
- Generated datasets remain ignored except for the intentionally committed, manifest-backed files under `tests/fixtures/synthetic/`.
- Do not commit a TTF/OTF file unless redistribution permission and its license notice are committed in the same change.
- M1 may emit PaddleOCR-style labels but must not depend on PaddleOCR, Ultralytics, CUDA, ONNX Runtime, or TensorRT.
- Generated datasets, local environments, caches, weights, and local benchmark outputs stay ignored by Git.
- No command overwrites an existing output directory. The user must choose a new path or remove the old path explicitly.

## Review Focus

1. **Malformed configuration or bundle contract:** unknown/empty character class, duplicate charset symbols, illegal separator position, disabled-only rules, missing CTC blank semantics, or invalid recognizer batch declarations must fail before use. Tasks 1 and 6 pin these cases.
2. **Unsafe output target:** count zero, a pre-existing output directory, or an output parent that exists as a file must return a stable error without modifying existing content; missing parent directories are created for the requested output. Tasks 4 and 5 pin these cases.
3. **Unicode filesystem paths:** a Traditional-Chinese output path must generate readable PNG, label, metadata, and summary files. Task 4 pins this case.
4. **Extreme transform sampling:** maximum configured perspective jitter and glare must still produce finite, clockwise, in-frame corners and a valid `uint8` image. Task 3 pins this case.
5. **Mid-run failure:** an encoder or write failure must leave no finalized dataset and remove only the generator-owned partial directory. Task 4 pins this case.

---

## File Map

### Project and packaging

- `pyproject.toml` — package metadata, Python/dependency bounds, and test configuration.
- `requirements/py311.lock.txt` — exact M1 dependency versions used for reproducibility.
- `.gitignore` — environments, caches, generated datasets, weights, and temporary output.
- `README.md` — starts as a package-installable project stub in Task 1 and becomes the complete M1 quick start in Task 7.

### Shared contracts

- `src/plateai_shared/__init__.py` — public shared exports.
- `src/plateai_shared/contracts.py` — immutable `CharacterSet`, `RuleToken`, `PlateRule`, `PlateRuleset`, and `GeneratedPlate` dataclasses.
- `src/plateai_shared/rules.py` — JSON loading, validation, weighted rule selection, canonical generation, and display formatting.
- `src/plateai_shared/schema_validation.py` — Draft 2020-12 schema loading and validation helpers.

### Synthetic trainer

- `src/plateai_trainer/__init__.py` — trainer package marker.
- `src/plateai_trainer/synthetic/__init__.py` — synthetic module exports.
- `src/plateai_trainer/synthetic/__main__.py` — `python -m plateai_trainer.synthetic` entry point.
- `src/plateai_trainer/synthetic/models.py` — template, font, augmentation, request, record, and summary dataclasses.
- `src/plateai_trainer/synthetic/templates.py` — plate-template loading and validation.
- `src/plateai_trainer/synthetic/fonts.py` — built-in Hershey fallback and optional TTF/OTF resolution.
- `src/plateai_trainer/synthetic/renderer.py` — clean RGB plate rendering and source corners.
- `src/plateai_trainer/synthetic/augment.py` — deterministic geometric and photometric transforms.
- `src/plateai_trainer/synthetic/encoder.py` — RGB-to-PNG bytes conversion.
- `src/plateai_trainer/synthetic/dataset.py` — orchestration and transactional dataset writer.
- `src/plateai_trainer/synthetic/cli.py` — argument parsing, stable exit codes, and concise JSON result output.

### Configuration and schemas

- `configs/charsets/tw_plate_latin_v1.txt` — ordered digits and uppercase Latin symbols, one per line.
- `configs/plate_rules/tw_plate_v1.json` — M1 `LLL-DDDD` and `DDDD-LL` rules.
- `configs/plate_templates/standard_white_v1.json` — 320×96 white plate template.
- `configs/augmentation/none_v1.json` — identity profile.
- `configs/augmentation/standard_v1.json` — bounded perspective, exposure, noise, blur, glare, and JPEG profile.
- `schemas/plate_rules.schema.json` — plate-rule document contract.
- `schemas/plate_template.schema.json` — renderer-template contract.
- `schemas/augmentation_profile.schema.json` — augmentation profile contract.
- `schemas/generation_metadata.schema.json` — one JSONL record contract.
- `schemas/generation_summary.schema.json` — completed-run summary contract.
- `schemas/model_manifest.schema.json` — version-one Model Bundle manifest contract.

### Tests and automation

- `tests/unit/test_rules.py` — rule loading, generation, validation, and property tests.
- `tests/conftest.py` — repository paths, JSON helpers, and reusable valid configuration fixtures.
- `tests/unit/test_renderer.py` — image shape, text, corners, and font resolution.
- `tests/unit/test_augment.py` — determinism, geometry, dtype, and extreme transforms.
- `tests/unit/test_dataset.py` — layout, labels, metadata, Unicode paths, and cleanup.
- `tests/contract/test_schemas.py` — valid and invalid JSON Schema examples.
- `tests/integration/test_cli.py` — module CLI success and stable error behavior.
- `tests/contract/test_toy_fixtures.py` — committed synthetic fixture manifest, hashes, labels, and image decoding.
- `tests/fixtures/synthetic/manifest.json` — provenance and SHA-256 for the public toy fixtures.
- `tests/fixtures/synthetic/*.png` — four deterministic, synthetic-only 320×96 RGB fixtures.
- `.github/workflows/test.yml` — Python 3.11 install, tests, package build, and three-image smoke generation.
- `docs/training.md` — dataset layout, reproducibility, font policy, and configuration reference.
- `models/README.md` — why weights stay out of Git and what future bundles contain.
- `THIRD_PARTY_NOTICES.md` — runtime/test dependency license inventory and font rule.

---

### Task 1: Establish packaging and the deterministic plate-rule engine

**Files:**
- Create: `pyproject.toml`
- Create: `requirements/py311.lock.txt`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/plateai_shared/__init__.py`
- Create: `src/plateai_shared/contracts.py`
- Create: `src/plateai_shared/rules.py`
- Create: `src/plateai_trainer/__init__.py`
- Create: `src/plateai_trainer/synthetic/__init__.py`
- Create: `configs/charsets/tw_plate_latin_v1.txt`
- Create: `configs/plate_rules/tw_plate_v1.json`
- Create: `tests/conftest.py`
- Test: `tests/unit/test_rules.py`

**Interfaces:**
- Consumes: UTF-8 character-set and plate-rule files.
- Produces: `load_character_set(path: Path) -> CharacterSet`, `load_ruleset(path: Path, charset: CharacterSet) -> PlateRuleset`, and `generate_plate(ruleset: PlateRuleset, rng: random.Random) -> GeneratedPlate`.

- [ ] **Step 1: Add the Python 3.11 package and dependency contract**

Create `pyproject.toml` with these bounds:

```toml
[build-system]
requires = ["setuptools>=80,<81"]
build-backend = "setuptools.build_meta"

[project]
name = "3wa-plate-ai"
version = "0.1.0.dev0"
description = "Taiwan license plate synthetic data, training, and high-speed reading tools"
readme = "README.md"
requires-python = ">=3.11,<3.12"
license = { text = "MIT" }
dependencies = [
  "numpy>=2.4.2,<2.5",
  "opencv-python-headless>=5.0.0.93,<5.1",
  "Pillow>=12.3.0,<13",
  "jsonschema>=4.26.0,<5",
]

[project.optional-dependencies]
test = [
  "pytest>=9.1.1,<10",
  "hypothesis>=6.168.0,<7",
  "build>=1.2,<2",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-ra"
```

Create an initial `README.md` containing the project title, the one-sentence M1 goal, and a link to the approved design spec so editable installation never references a missing readme.

After the first successful environment installation, generate `requirements/py311.lock.txt` with `python -m pip freeze --exclude-editable`. The lock must contain every resolved transitive dependency and include exact entries for `numpy==2.4.2`, `opencv-python-headless==5.0.0.93`, `Pillow==12.3.0`, `jsonschema==4.26.0`, `pytest==9.1.1`, and `hypothesis==6.168.0`. Keep resolution changes visible by regenerating and reviewing this file deliberately rather than installing latest packages silently.

- [ ] **Step 2: Write failing rule tests**

Create `tests/conftest.py` with `ROOT = Path(__file__).resolve().parents[1]`, constants for the default charset/rules/template/augmentation paths, a `write_json(path, value)` helper that writes UTF-8 JSON with `ensure_ascii=False`, and fixtures that load a fresh mutable JSON document for each test. The `default_ruleset` fixture calls the public loaders rather than constructing internal dataclasses.

Create tests for successful generation and every Review Focus rule failure:

```python
def test_default_rules_generate_canonical_and_display_text(default_ruleset):
    result = generate_plate(default_ruleset, random.Random(7))
    assert result.rule_id in {"standard-lll-dddd", "legacy-dddd-ll"}
    assert "-" not in result.canonical
    assert result.display.count("-") == 1
    assert result.display.replace("-", "") == result.canonical


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["rules"][0].update(tokens=[{"class": "UNKNOWN"}]), "unknown character class"),
        (lambda d: d["character_classes"].update(L=""), "must not be empty"),
        (lambda d: d["rules"][0].update(separator_after=[99]), "separator position"),
        (lambda d: [r.update(enabled=False) for r in d["rules"]], "no enabled rules"),
    ],
)
def test_invalid_rules_fail_before_generation(tmp_path, valid_rule_document, mutate, message):
    mutate(valid_rule_document)
    path = write_json(tmp_path / "rules.json", valid_rule_document)
    with pytest.raises(ValueError, match=message):
        load_ruleset(path, load_character_set(DEFAULT_CHARSET))


def test_charset_rejects_blank_or_duplicate_symbols(tmp_path):
    path = tmp_path / "charset.txt"
    path.write_text("A\nB\nA\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate|blank"):
        load_character_set(path)


@given(st.integers(min_value=-(2**31), max_value=2**31 - 1))
def test_generated_plate_round_trips_through_display(seed):
    charset = load_character_set(DEFAULT_CHARSET)
    ruleset = load_ruleset(DEFAULT_RULES, charset)
    result = generate_plate(ruleset, random.Random(seed))
    assert result.display.replace("-", "") == result.canonical
    assert all(symbol in charset.symbols for symbol in result.canonical)
```

Run: `python -m pytest tests/unit/test_rules.py -v`

Expected: FAIL because `plateai_shared.rules` does not exist.

- [ ] **Step 3: Implement immutable contracts and strict loaders**

Implement these exact dataclasses in `contracts.py`:

```python
@dataclass(frozen=True, slots=True)
class CharacterSet:
    id: str
    symbols: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class RuleToken:
    kind: Literal["class", "literal"]
    value: str


@dataclass(frozen=True, slots=True)
class PlateRule:
    id: str
    tokens: tuple[RuleToken, ...]
    separator: str
    separator_after: tuple[int, ...]
    plate_type: str
    weight: float
    enabled: bool


@dataclass(frozen=True, slots=True)
class PlateRuleset:
    schema_version: int
    id: str
    character_classes: Mapping[str, str]
    rules: tuple[PlateRule, ...]


@dataclass(frozen=True, slots=True)
class GeneratedPlate:
    canonical: str
    display: str
    rule_id: str
    plate_type: str


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
```

Implement weighted selection and display formatting in `rules.py`:

```python
def generate_plate(ruleset: PlateRuleset, rng: random.Random) -> GeneratedPlate:
    enabled = tuple(rule for rule in ruleset.rules if rule.enabled)
    rule = rng.choices(enabled, weights=[rule.weight for rule in enabled], k=1)[0]
    chars: list[str] = []
    for token in rule.tokens:
        if token.kind == "literal":
            chars.append(token.value)
        else:
            chars.append(rng.choice(ruleset.character_classes[token.value]))
    canonical = "".join(chars)
    pieces: list[str] = []
    for index, char in enumerate(canonical, start=1):
        pieces.append(char)
        if index in rule.separator_after:
            pieces.append(rule.separator)
    return GeneratedPlate(
        canonical=canonical,
        display="".join(pieces),
        rule_id=rule.id,
        plate_type=rule.plate_type,
    )
```

Validate schema version `1`, file readability, one Unicode code point per charset line, no blanks or duplicates, character classes as non-empty subsets of the charset, token references, separator positions from `1` through `len(tokens)-1`, unique rule IDs, finite positive enabled weights, and at least one enabled rule.

Wrap loaded character-class mappings with `types.MappingProxyType` before storing them in the frozen dataclass so callers cannot mutate an allegedly immutable ruleset.

- [ ] **Step 4: Add the M1 character set and two data-driven rules**

Write `configs/charsets/tw_plate_latin_v1.txt` in this stable order, one symbol per line:

```text
0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
```

The actual file must contain 36 visible symbols on 36 lines, not the space-separated presentation above. It must not contain a CTC blank label. Define `L` as `ABCDEFGHIJKLMNOPQRSTUVWXYZ`, `D` as `0123456789`, and rules with token sequences `LLLDDDD` plus separator after position 3, and `DDDDLL` plus separator after position 4. Name them `standard-lll-dddd` and `legacy-dddd-ll`; assign both `plate_type: "standard"` and positive weights.

- [ ] **Step 5: Run the rule test suite and commit**

Run:

```bash
python -m pip install -e ".[test]"
python -m pip freeze --exclude-editable > requirements/py311.lock.txt
python -m pytest tests/unit/test_rules.py -v
```

Expected: PASS with no warnings from project code.

Commit:

```bash
git add pyproject.toml requirements/py311.lock.txt .gitignore README.md src/plateai_shared src/plateai_trainer configs tests/conftest.py tests/unit/test_rules.py
git commit -m "feat: add deterministic plate rule engine"
```

---

### Task 2: Render clean plate crops with a safe font fallback

**Files:**
- Create: `src/plateai_trainer/synthetic/models.py`
- Create: `src/plateai_trainer/synthetic/templates.py`
- Create: `src/plateai_trainer/synthetic/fonts.py`
- Create: `src/plateai_trainer/synthetic/renderer.py`
- Create: `configs/plate_templates/standard_white_v1.json`
- Modify: `tests/conftest.py`
- Test: `tests/unit/test_renderer.py`

**Interfaces:**
- Consumes: `GeneratedPlate` from Task 1 and a template JSON document.
- Produces: `load_template(path: Path) -> PlateTemplate`, `resolve_font(value: str | Path | None) -> FontSpec`, and `render_plate(sample: GeneratedPlate, template: PlateTemplate, font: FontSpec) -> RenderedPlate`.

- [ ] **Step 1: Write failing renderer tests**

Extend `tests/conftest.py` with `DEFAULT_TEMPLATE`, `valid_template_document`, and `default_template`; the last fixture must call `load_template(DEFAULT_TEMPLATE)`.

```python
def test_hershey_renderer_returns_rgb_plate_and_ordered_corners(default_template):
    sample = GeneratedPlate("ABC1234", "ABC-1234", "standard-lll-dddd", "standard")
    rendered = render_plate(sample, default_template, resolve_font(None))
    assert rendered.image_rgb.shape == (96, 320, 3)
    assert rendered.image_rgb.dtype == np.uint8
    assert rendered.corners.tolist() == [
        [0.0, 0.0], [319.0, 0.0], [319.0, 95.0], [0.0, 95.0]
    ]
    assert np.count_nonzero(rendered.image_rgb < 80) > 500
    assert rendered.metadata["rendered_text"] == "ABC-1234"


def test_missing_explicit_font_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="font file does not exist"):
        resolve_font(tmp_path / "missing.ttf")


def test_template_rejects_out_of_bounds_text_box(tmp_path, valid_template_document):
    valid_template_document["text_box"] = [0, 0, 999, 96]
    path = write_json(tmp_path / "template.json", valid_template_document)
    with pytest.raises(ValueError, match="text_box"):
        load_template(path)
```

Run: `python -m pytest tests/unit/test_renderer.py -v`

Expected: FAIL because renderer modules do not exist.

- [ ] **Step 2: Implement template and render result models**

```python
@dataclass(frozen=True, slots=True)
class PlateTemplate:
    id: str
    width: int
    height: int
    background_rgb: tuple[int, int, int]
    foreground_rgb: tuple[int, int, int]
    border_rgb: tuple[int, int, int]
    border_width: int
    corner_radius: int
    text_box: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class FontSpec:
    kind: Literal["hershey", "truetype"]
    name: str
    path: Path | None


@dataclass(frozen=True, slots=True)
class RenderedPlate:
    image_rgb: NDArray[np.uint8]
    corners: NDArray[np.float32]
    metadata: Mapping[str, JsonValue]
```

The default `resolve_font(None)` returns a built-in OpenCV `FONT_HERSHEY_SIMPLEX` specification. A supplied path must exist, be a regular file, and end in `.ttf` or `.otf`; the renderer loads it with Pillow and reports `cannot load font` on `OSError`.

- [ ] **Step 3: Implement fit-to-box text rendering**

Create a white 320×96 RGB canvas from the template, draw the rounded border, and fit `sample.display` inside `text_box`. For Hershey text, find a scale from `cv2.getTextSize`; for TrueType, binary-search point sizes `8..template.height * 2` with `ImageDraw.textbbox`. Center the chosen glyph box horizontally and vertically.

Use this source-corner array exactly:

```python
corners = np.asarray(
    [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
    dtype=np.float32,
)
```

If the text cannot fit at the minimum supported size, raise `ValueError("display text does not fit template text_box")`; never silently crop characters.

- [ ] **Step 4: Add and validate the standard white template**

Use a 320×96 canvas, near-white background `[245, 245, 242]`, near-black foreground `[18, 18, 18]`, two-pixel border, eight-pixel corner radius, and text box `[18, 12, 302, 84]`. Keep visual dimensions in JSON so later templates can add green, yellow, or red plate variants without changing renderer code.

- [ ] **Step 5: Run renderer tests and commit**

Run: `python -m pytest tests/unit/test_renderer.py -v`

Expected: PASS.

Commit:

```bash
git add src/plateai_trainer/synthetic configs/plate_templates tests/unit/test_renderer.py
git commit -m "feat: render configurable synthetic plate crops"
```

---

### Task 3: Add deterministic geometry and photometric augmentation

**Files:**
- Modify: `src/plateai_trainer/synthetic/models.py`
- Create: `src/plateai_trainer/synthetic/augment.py`
- Create: `configs/augmentation/none_v1.json`
- Create: `configs/augmentation/standard_v1.json`
- Modify: `tests/conftest.py`
- Test: `tests/unit/test_augment.py`

**Interfaces:**
- Consumes: `RenderedPlate` from Task 2, `AugmentProfile`, and an integer sample seed.
- Produces: `load_augment_profile(path: Path) -> AugmentProfile`, `derive_sample_seed(run_seed: int, index: int) -> int`, and `apply_augmentations(rendered: RenderedPlate, profile: AugmentProfile, sample_seed: int) -> AugmentedPlate`.

- [ ] **Step 1: Write failing determinism and geometry tests**

Extend `tests/conftest.py` with `none_profile`, `standard_profile`, and `rendered_plate`. Define `extreme_profile` by replacing the standard profile's geometry/glare fields with their maximum still-valid values rather than loading an untracked configuration file.

```python
def test_same_seed_reproduces_pixels_corners_and_metadata(rendered_plate, standard_profile):
    first = apply_augmentations(rendered_plate, standard_profile, sample_seed=99)
    second = apply_augmentations(rendered_plate, standard_profile, sample_seed=99)
    assert np.array_equal(first.image_rgb, second.image_rgb)
    assert np.array_equal(first.corners, second.corners)
    assert first.metadata == second.metadata


def test_none_profile_is_identity(rendered_plate, none_profile):
    result = apply_augmentations(rendered_plate, none_profile, sample_seed=123)
    assert np.array_equal(result.image_rgb, rendered_plate.image_rgb)
    assert np.array_equal(result.corners, rendered_plate.corners)


def test_extreme_valid_profile_keeps_finite_clockwise_in_frame_corners(rendered_plate, extreme_profile):
    result = apply_augmentations(rendered_plate, extreme_profile, sample_seed=4)
    height, width = result.image_rgb.shape[:2]
    assert result.image_rgb.dtype == np.uint8
    assert np.isfinite(result.corners).all()
    assert ((0 <= result.corners[:, 0]) & (result.corners[:, 0] < width)).all()
    assert ((0 <= result.corners[:, 1]) & (result.corners[:, 1] < height)).all()
    assert cv2.contourArea(result.corners.astype(np.float32), oriented=True) > 0
```

Run: `python -m pytest tests/unit/test_augment.py -v`

Expected: FAIL because `augment.py` does not exist.

- [ ] **Step 2: Define bounded augmentation contracts**

Add these exact dataclasses:

```python
@dataclass(frozen=True, slots=True)
class AugmentProfile:
    id: str
    perspective_probability: float
    max_corner_jitter_ratio: float
    rotation_probability: float
    rotation_degrees: tuple[float, float]
    brightness_probability: float
    brightness_range: tuple[float, float]
    gamma_probability: float
    gamma_range: tuple[float, float]
    noise_probability: float
    noise_std_range: tuple[float, float]
    motion_blur_probability: float
    motion_blur_kernels: tuple[int, ...]
    glare_probability: float
    glare_opacity_range: tuple[float, float]
    glare_radius_ratio_range: tuple[float, float]
    jpeg_probability: float
    jpeg_quality_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class AugmentedPlate:
    image_rgb: NDArray[np.uint8]
    corners: NDArray[np.float32]
    metadata: Mapping[str, JsonValue]
```

Validate probabilities in `[0, 1]`, ordered numeric ranges, odd positive blur kernels, JPEG quality in `[1, 100]`, and `max_corner_jitter_ratio <= 0.20`.

Derive sample seeds without shared RNG state:

```python
def derive_sample_seed(run_seed: int, index: int) -> int:
    payload = f"3waPlateAI:{run_seed}:{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
```

- [ ] **Step 3: Implement geometry with one homography**

Sample destination corners inside the image bounds, apply the requested rotation around image center, and compute one 3×3 homography. Warp RGB pixels with `cv2.warpPerspective` and transform the original four corners with `cv2.perspectiveTransform`. Reject a sampled quadrilateral if its area is below 40% of the original or its point order changes; retry at most eight times, then use identity geometry and record `geometry_fallback: true`.

- [ ] **Step 4: Implement ordered photometric transforms**

Apply transforms in this stable order: brightness, gamma, Gaussian noise, motion blur, glare, JPEG round-trip. Each stage records whether it ran and its sampled parameters. Perform arithmetic in `float32`, clip once per stage to `[0, 255]`, and return RGB `uint8`. JPEG encoding/decoding must explicitly convert RGB↔BGR at the OpenCV boundary.

- [ ] **Step 5: Add identity and standard profiles, run tests, and commit**

The identity profile sets every probability to zero and neutral numeric ranges. The standard profile uses conservative M1 bounds: perspective jitter ≤8%, rotation ≤4°, brightness `0.75..1.25`, gamma `0.80..1.20`, noise sigma `0..8`, blur kernels `1/3/5`, glare opacity ≤0.35, and JPEG quality `60..95`.

Run: `python -m pytest tests/unit/test_augment.py -v`

Expected: PASS.

Commit:

```bash
git add src/plateai_trainer/synthetic/models.py src/plateai_trainer/synthetic/augment.py configs/augmentation tests/unit/test_augment.py
git commit -m "feat: add deterministic plate augmentation"
```

---

### Task 4: Generate a complete dataset transactionally

**Files:**
- Modify: `src/plateai_trainer/synthetic/models.py`
- Create: `src/plateai_trainer/synthetic/encoder.py`
- Create: `src/plateai_trainer/synthetic/dataset.py`
- Modify: `tests/conftest.py`
- Test: `tests/unit/test_dataset.py`

**Interfaces:**
- Consumes: rule, template, font, augmentation, count, seed, and output configuration.
- Produces: `encode_png(image_rgb: NDArray[np.uint8]) -> bytes` and `generate_dataset(request: GenerationRequest, *, encoder: ImageEncoder = encode_png) -> GenerationSummary`.

- [ ] **Step 1: Write failing output-layout and safety tests**

Extend `tests/conftest.py` with `default_request`, pointing at all committed default configuration files and a per-test temporary output path.

```python
def test_generate_dataset_writes_images_labels_metadata_and_summary(tmp_path, default_request):
    request = replace(default_request, count=3, output=tmp_path / "dataset")
    summary = generate_dataset(request)
    assert summary.generated == 3
    assert sorted(p.name for p in (request.output / "images").glob("*.png")) == [
        "000000.png", "000001.png", "000002.png"
    ]
    labels = (request.output / "labels.txt").read_text(encoding="utf-8").splitlines()
    assert len(labels) == 3
    assert all(line.startswith("images/") and "\t" in line for line in labels)
    assert len((request.output / "metadata.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_existing_output_is_never_modified(tmp_path, default_request):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(OutputExistsError):
        generate_dataset(replace(default_request, output=output))
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_unicode_output_path_works(tmp_path, default_request):
    output = tmp_path / "台灣車牌資料"
    generate_dataset(replace(default_request, count=1, output=output))
    assert (output / "images/000000.png").is_file()


def test_missing_nested_output_parent_is_created(tmp_path, default_request):
    output = tmp_path / "new" / "nested" / "dataset"
    generate_dataset(replace(default_request, count=1, output=output))
    assert (output / "summary.json").is_file()


def test_encoder_failure_leaves_no_final_or_partial_directory(tmp_path, default_request):
    output = tmp_path / "broken"
    def fail_encoder(_image):
        raise RuntimeError("encode failed")
    with pytest.raises(RuntimeError, match="encode failed"):
        generate_dataset(replace(default_request, output=output), encoder=fail_encoder)
    assert not output.exists()
    assert list(tmp_path.glob(".broken.partial-*")) == []
```

Run: `python -m pytest tests/unit/test_dataset.py -v`

Expected: FAIL because dataset generation does not exist.

- [ ] **Step 2: Define request, record, summary, and stable errors**

```python
@dataclass(frozen=True, slots=True)
class GenerationRequest:
    count: int
    seed: int
    output: Path
    charset_path: Path
    rules_path: Path
    template_path: Path
    augmentation_path: Path
    font: str | Path | None = None


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    schema_version: int
    generated: int
    seed: int
    rule_counts: Mapping[str, int]
    plate_type_counts: Mapping[str, int]
    charset_sha256: str
    rules_sha256: str
    template_sha256: str
    augmentation_sha256: str


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    schema_version: int
    index: int
    sample_seed: int
    image_path: str
    canonical: str
    display: str
    rule_id: str
    plate_type: str
    corners: tuple[tuple[float, float], ...]
    renderer: Mapping[str, JsonValue]
    augmentation: Mapping[str, JsonValue]
    image_sha256: str


ImageEncoder: TypeAlias = Callable[[NDArray[np.uint8]], bytes]
```

Convert mutable renderer, augmentation, and count mappings to `MappingProxyType` at model boundaries. Serialize records explicitly rather than relying on `dataclasses.asdict` to understand proxies.

Define `GenerationError`, `InvalidGenerationRequest`, and `OutputExistsError`. Reject count values below one, a parent that exists but is not a directory, and any existing target before creating a staging directory. Create missing parent directories with `mkdir(parents=True, exist_ok=True)` for the explicitly requested output path.

- [ ] **Step 3: Implement PNG encoding and per-sample records**

`encode_png` verifies shape, dtype, and RGB order, converts RGB to BGR, calls `cv2.imencode(".png", ...)`, checks its success flag, and returns bytes.

Each metadata JSONL record must include `schema_version`, `index`, `sample_seed`, `image_path`, `canonical`, `display`, `rule_id`, `plate_type`, four corners, renderer metadata, augmentation metadata, and SHA-256 of the encoded PNG.

`generation_config.json` stores run seed, count, font identifier, repository-relative configuration paths, and the four configuration SHA-256 values. It must not store absolute paths or a generation timestamp.

- [ ] **Step 4: Implement transactional directory publication**

Create a sibling staging path named `.<output-name>.partial-<uuid>`. Write `images/*.png`, `labels.txt`, `metadata.jsonl`, `generation_config.json`, and `summary.json` there. Flush and close every file, then publish with `os.replace(staging, output)` only after all samples and the summary succeed.

On failure, remove only a staging path that satisfies both conditions:

```python
staging.parent.resolve() == output.parent.resolve()
staging.name.startswith(f".{output.name}.partial-")
```

Never delete, merge into, or overwrite `output`.

- [ ] **Step 5: Run dataset tests and commit**

Run: `python -m pytest tests/unit/test_dataset.py -v`

Expected: PASS.

Commit:

```bash
git add src/plateai_trainer/synthetic/models.py src/plateai_trainer/synthetic/encoder.py src/plateai_trainer/synthetic/dataset.py tests/unit/test_dataset.py
git commit -m "feat: write synthetic datasets transactionally"
```

---

### Task 5: Expose the M1 command-line interface

**Files:**
- Create: `src/plateai_trainer/synthetic/cli.py`
- Create: `src/plateai_trainer/synthetic/__main__.py`
- Modify: `src/plateai_trainer/synthetic/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/integration/test_cli.py`

**Interfaces:**
- Consumes: command-line arguments and Task 4's `generate_dataset`.
- Produces: `build_parser() -> argparse.ArgumentParser`, `main(argv: Sequence[str] | None = None) -> int`, module command `python -m plateai_trainer.synthetic generate`, and console command `plateai-generate generate`.

- [ ] **Step 1: Write failing CLI integration tests**

Define `run_cli(*args: str)` in the test module as a `subprocess.run` wrapper around `[sys.executable, "-m", "plateai_trainer.synthetic", *args]` with `text=True`, `capture_output=True`, and `check=False`.

```python
def test_generate_command_creates_requested_count(tmp_path):
    output = tmp_path / "demo"
    result = subprocess.run(
        [sys.executable, "-m", "plateai_trainer.synthetic", "generate",
         "--count", "3", "--seed", "42", "--output", str(output)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["generated"] == 3
    assert payload["output"] == str(output)


def test_zero_count_returns_usage_error_without_output(tmp_path):
    output = tmp_path / "zero"
    result = run_cli("generate", "--count", "0", "--output", str(output))
    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert not output.exists()


def test_existing_output_returns_stable_exit_code(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    result = run_cli("generate", "--count", "1", "--output", str(output))
    assert result.returncode == 3
    assert "output already exists" in result.stderr
```

Run: `python -m pytest tests/integration/test_cli.py -v`

Expected: FAIL because there is no module entry point.

- [ ] **Step 2: Implement command parsing and repository defaults**

The `generate` subcommand accepts `--count`, `--seed`, `--output`, `--charset`, `--rules`, `--template`, `--augmentation`, and `--font`. Resolve default configuration paths from the repository root detected as `Path(__file__).resolve().parents[3]`; explicit paths always win.

Use a custom positive-integer parser for count. Keep these exit codes stable:

- `0`: success;
- `2`: argparse or configuration/request validation error;
- `3`: output already exists;
- `4`: image generation or filesystem failure.

On success, print exactly one JSON object to stdout containing `output`, `generated`, `seed`, `rule_counts`, and `plate_type_counts`. Send concise human-readable errors to stderr without a traceback unless `--debug` is supplied.

- [ ] **Step 3: Wire both module and console entry points**

`__main__.py` must contain:

```python
from .cli import main

raise SystemExit(main())
```

Add this entry point to `pyproject.toml`:

```toml
[project.scripts]
plateai-generate = "plateai_trainer.synthetic.cli:main"
```

- [ ] **Step 4: Run integration and full tests, then commit**

Run:

```bash
python -m pytest tests/integration/test_cli.py -v
python -m pytest -q
```

Expected: all tests PASS.

Commit:

```bash
git add pyproject.toml src/plateai_trainer/synthetic tests/integration/test_cli.py
git commit -m "feat: add synthetic dataset CLI"
```

---

### Task 6: Publish and validate JSON contracts

**Files:**
- Create: `src/plateai_shared/schema_validation.py`
- Create: `schemas/plate_rules.schema.json`
- Create: `schemas/plate_template.schema.json`
- Create: `schemas/augmentation_profile.schema.json`
- Create: `schemas/generation_metadata.schema.json`
- Create: `schemas/generation_summary.schema.json`
- Create: `schemas/model_manifest.schema.json`
- Test: `tests/contract/test_schemas.py`

**Interfaces:**
- Consumes: JSON-compatible mappings, Draft 2020-12 schema paths, and the visible charset size for model manifests.
- Produces: `load_schema(path: Path) -> Mapping[str, JsonValue]`, `validate_document(document: Mapping[str, JsonValue], schema_path: Path) -> None`, and `validate_model_manifest(document: Mapping[str, JsonValue], schema_path: Path, *, visible_charset_symbol_count: int) -> None`.

- [ ] **Step 1: Write failing schema contract tests**

In the contract test module, define `valid_generation_record()` from an actual one-image identity-profile run so property names cannot drift from Task 4. Use this complete crop-only manifest factory:

```python
def valid_crop_only_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract_version": "1.0",
        "model_id": "twplate-dev-rec-v001",
        "version": "0.1.0",
        "created_at": "2026-09-21T00:00:00Z",
        "capabilities": ["crop-recognition"],
        "plate_size": [320, 96],
        "charset": {"file": "charset.txt", "sha256": "0" * 64, "visible_symbols": 36},
        "rules": {"file": "plate_rules.json", "sha256": "1" * 64},
        "components": {
            "recognizer": {
                "file": "recognizer.onnx",
                "format": "onnx",
                "sha256": "2" * 64,
                "inputs": [{"name": "image", "dtype": "float32", "shape": ["batch", 3, 96, 320]}],
                "outputs": [{"name": "logits", "dtype": "float32", "shape": ["batch", "time", 37]}],
                "batch": {"mode": "dynamic", "min": 1, "opt": 8, "max": 32},
            }
        },
        "decoder": {
            "type": "ctc",
            "blank_index": 0,
            "class_count": 37,
            "index_mapping": "charset-order-skipping-blank",
            "collapse_repeats": True,
        },
        "preprocess": {
            "color_order": "RGB",
            "dtype": "float32",
            "scale": 0.00392156862745098,
        },
        "provenance": {"training_data": "synthetic", "license_reviewed": True},
    }
```

```python
@pytest.mark.parametrize(
    ("document_path", "schema_path"),
    [
        (DEFAULT_RULES, SCHEMAS / "plate_rules.schema.json"),
        (DEFAULT_TEMPLATE, SCHEMAS / "plate_template.schema.json"),
        (DEFAULT_STANDARD_AUGMENT, SCHEMAS / "augmentation_profile.schema.json"),
    ],
)
def test_default_documents_match_published_schemas(document_path, schema_path):
    validate_document(read_json(document_path), schema_path)


def test_model_manifest_rejects_path_traversal():
    manifest = valid_crop_only_manifest()
    manifest["components"]["recognizer"]["file"] = "../recognizer.onnx"
    with pytest.raises(DocumentValidationError, match="components.recognizer.file"):
        validate_document(manifest, SCHEMAS / "model_manifest.schema.json")


def test_ctc_manifest_requires_explicit_blank_mapping():
    manifest = valid_crop_only_manifest()
    del manifest["decoder"]["blank_index"]
    with pytest.raises(DocumentValidationError, match="decoder.blank_index"):
        validate_model_manifest(
            manifest,
            SCHEMAS / "model_manifest.schema.json",
            visible_charset_symbol_count=36,
        )


@pytest.mark.parametrize(
    "batch",
    [
        {"mode": "dynamic", "min": 8, "opt": 4, "max": 2},
        {"mode": "fixed", "size": 8},
    ],
)
def test_manifest_rejects_invalid_batch_contract(batch):
    manifest = valid_crop_only_manifest()
    manifest["components"]["recognizer"]["batch"] = batch
    with pytest.raises(DocumentValidationError, match="batch"):
        validate_model_manifest(
            manifest,
            SCHEMAS / "model_manifest.schema.json",
            visible_charset_symbol_count=36,
        )


def test_metadata_requires_exactly_four_ordered_corners():
    metadata = valid_generation_record()
    metadata["corners"] = [[0, 0], [1, 0], [1, 1]]
    with pytest.raises(DocumentValidationError, match="corners"):
        validate_document(metadata, SCHEMAS / "generation_metadata.schema.json")
```

Run: `python -m pytest tests/contract/test_schemas.py -v`

Expected: FAIL because schemas and validation helpers do not exist.

- [ ] **Step 2: Implement deterministic schema error reporting**

Use `Draft202012Validator.check_schema` when loading a schema. Reject non-finite floats recursively before JSON Schema evaluation. Sort validation failures by their absolute JSON path and raise one `DocumentValidationError` whose message begins with the dotted path followed by the validator message. Do not expose a raw stack trace for user configuration mistakes.

`validate_model_manifest` first calls `validate_document`, then checks cross-field rules JSON Schema cannot express: dynamic `min <= opt <= max`, `blank_index` within class range, and `class_count == visible_charset_symbol_count + 1`.

- [ ] **Step 3: Define configuration and generated-record schemas**

Every schema sets `"$schema": "https://json-schema.org/draft/2020-12/schema"`, requires `schema_version: 1`, rejects unknown top-level fields with `additionalProperties: false`, and mirrors the exact dataclass property names from Tasks 1–4. `corners` requires exactly four pairs of finite numbers. SHA-256 values use `^[0-9a-f]{64}$`.

- [ ] **Step 4: Define the crop-only-capable Model Bundle manifest schema**

Require `schema_version`, `contract_version`, `model_id`, semantic `version`, UTC `created_at`, `capabilities`, `plate_size`, `charset`, `rules`, `components`, `decoder`, `preprocess`, and `provenance`. A crop-only M1 development manifest may declare only a recognizer component; a full bundle that declares `plate-detection` must also provide detector metadata, the fixed four keypoint names, and `rectifier.normalization_strategy: "convex-hull-semantic-v1"`.

Component filenames must match `^[A-Za-z0-9][A-Za-z0-9._-]*$`, preventing separators and `..`. Component hashes use lowercase SHA-256. Tensor entries explicitly name input/output tensors, dtypes, and shapes.

The recognizer component requires exactly one of these batch contracts:

```json
{"mode": "dynamic", "min": 1, "opt": 8, "max": 32}
```

```json
{"mode": "fixed", "size": 8, "padding": "neutral-image", "discard_padded_outputs": true}
```

The schema enforces positive sizes and the implementation-level validator enforces `min <= opt <= max`. A request-time TensorRT rebuild is not a valid batch policy.

For CTC recognition, require this decoder shape, using an integer rather than the word `last`:

```json
{
  "type": "ctc",
  "blank_index": 0,
  "class_count": 37,
  "index_mapping": "charset-order-skipping-blank",
  "collapse_repeats": true
}
```

The implementation-level validator checks `0 <= blank_index < class_count` and `class_count == visible_charset_symbol_count + 1`. A future non-CTC decoder must use a different decoder schema branch rather than overloading these fields.

- [ ] **Step 5: Validate generated output against schemas and commit**

Extend Task 4's tests to validate every metadata record and `summary.json` with the published schemas.

Run:

```bash
python -m pytest tests/contract/test_schemas.py tests/unit/test_dataset.py -v
```

Expected: PASS.

Commit:

```bash
git add src/plateai_shared/schema_validation.py schemas tests/contract/test_schemas.py tests/unit/test_dataset.py
git commit -m "feat: publish M1 data and model contracts"
```

---

### Task 7: Document, automate, and verify the M1 vertical slice

**Files:**
- Modify: `README.md`
- Create: `docs/training.md`
- Create: `models/README.md`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `.github/workflows/test.yml`
- Create: `tests/contract/test_toy_fixtures.py`
- Create: `tests/fixtures/synthetic/manifest.json`
- Create: `tests/fixtures/synthetic/000000.png`
- Create: `tests/fixtures/synthetic/000001.png`
- Create: `tests/fixtures/synthetic/000002.png`
- Create: `tests/fixtures/synthetic/000003.png`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: the completed CLI and tests.
- Produces: reproducible setup/generation instructions, a manifest-backed four-image synthetic fixture set, and a Linux Python 3.11 CI gate.

- [ ] **Step 1: Write the failing committed-fixture contract test**

Create `tests/contract/test_toy_fixtures.py` so it loads `tests/fixtures/synthetic/manifest.json`, requires exactly four unique entries, verifies every relative path stays inside the fixture directory, checks every file SHA-256, decodes each image with Pillow as RGB 320×96, and verifies canonical/display labels contain only the configured visible charset plus the decorative hyphen.

Run: `python -m pytest tests/contract/test_toy_fixtures.py -v`

Expected: FAIL because the fixture manifest and images do not exist.

- [ ] **Step 2: Generate and commit the harmless synthetic fixtures**

Generate four identity-profile images with seeds `42000`, `42001`, `42002`, and `42003` into a temporary directory. Copy only their PNG bytes into `tests/fixtures/synthetic/`, then write a stable `manifest.json` containing `schema_version`, `generator_version`, `synthetic_only: true`, and one entry per image with relative path, seed, canonical, display, rule ID, and lowercase SHA-256. Do not include timestamps or absolute paths.

Run: `python -m pytest tests/contract/test_toy_fixtures.py -v`

Expected: PASS for four fixtures.

- [ ] **Step 3: Write the README quick start and scope boundary**

Document these commands verbatim:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/py311.lock.txt
python -m pip install --no-deps -e .
python -m plateai_trainer.synthetic generate --count 100 --seed 42 --output out/demo
python -m pytest
```

Explain Trainer versus Reader, the Model Bundle boundary, M1's recognizer-crop scope, the built-in Hershey development fallback, and why real plate images and production model weights are excluded from Git.

Include this policy in equivalent clear wording: the repository provides the complete MIT plate synthesis, training, and high-speed inference engine; committed fixtures verify the code and contracts, but the project publishes no checkpoints, trained weights, ONNX models, TensorRT engines, or Model Bundles. Every user generates suitable synthetic data or imports a lawfully obtained dataset and trains their own bundle.

- [ ] **Step 4: Document output and configuration contracts**

In `docs/training.md`, show the exact output tree and field purpose:

```text
out/demo/
├── images/000000.png
├── labels.txt
├── metadata.jsonl
├── generation_config.json
└── summary.json
```

State that labels use `relative/path.png<TAB>CANONICAL_TEXT`, configurations are immutable inputs to a run, and reproducibility covers strings and sampled transform parameters under the locked dependency set. Document how `--font /path/to/font.ttf` works and that redistributable font files require their own notice.

- [ ] **Step 5: Add third-party and model-storage notices**

List NumPy, OpenCV/opencv-python-headless, Pillow, jsonschema, pytest, and Hypothesis with package name, role, upstream URL, and SPDX license expression. State that the repository's MIT license does not relicense dependencies, external fonts, datasets, or model weights. `models/README.md` states that no binary model artifact is committed or attached to official releases; it documents local bundle layout and requires `THIRD_PARTY_LICENSES.md` for any bundle users create themselves.

- [ ] **Step 6: Add Python 3.11 CI**

Create one Ubuntu workflow triggered by pushes and pull requests. It must:

```yaml
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
  with:
    python-version: "3.11"
- run: python -m pip install --upgrade pip
- run: python -m pip install -r requirements/py311.lock.txt
- run: python -m pip install --no-deps -e .
- run: python -m pytest -q
- run: python -m build
- run: python -m plateai_trainer.synthetic generate --count 3 --seed 42 --output "$RUNNER_TEMP/plateai-smoke"
```

Do not upload generated plates as public CI artifacts in M1.

- [ ] **Step 7: Run fresh-environment verification**

From the repository root:

```bash
python3.11 -m venv .venv-verify
. .venv-verify/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/py311.lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
python -m build
python -m plateai_trainer.synthetic generate --count 100 --seed 42 --output out/verify-seed-42
python -m plateai_trainer.synthetic generate --count 100 --seed 42 --output out/verify-seed-42-repeat
diff out/verify-seed-42/labels.txt out/verify-seed-42-repeat/labels.txt
diff out/verify-seed-42/metadata.jsonl out/verify-seed-42-repeat/metadata.jsonl
python - <<'PY'
import json
from pathlib import Path
from PIL import Image

root = Path("out/verify-seed-42")
records = [json.loads(line) for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(records) == 100
assert len(list((root / "images").glob("*.png"))) == 100
for image_path in (root / "images").glob("*.png"):
    with Image.open(image_path) as image:
        assert image.mode == "RGB"
        assert image.size == (320, 96)
print("M1 verification OK")
PY
```

Expected: tests and build PASS, labels match, both runs contain 100 valid RGB PNG files, and the script prints `M1 verification OK`.

- [ ] **Step 8: Commit the M1 documentation, fixtures, and CI gate**

```bash
git add README.md docs/training.md models/README.md THIRD_PARTY_NOTICES.md .github/workflows/test.yml .gitignore tests/contract/test_toy_fixtures.py tests/fixtures/synthetic
git commit -m "docs: add M1 usage and verification workflow"
```

---

## Final branch verification

- [ ] Run `python -m pytest -q` and record the exact pass count in the pull request.
- [ ] Run `python -m build` and confirm both sdist and wheel are created.
- [ ] Run the 100-image acceptance command from the design spec.
- [ ] Confirm `labels.txt` and sampled transform metadata repeat for seed 42.
- [ ] Confirm no generated datasets, fonts without licenses, real plates, model weights, credentials, or virtual environments are tracked by `git status --short`.
- [ ] Review the branch diff against `docs/superpowers/specs/2026-09-21-3wa-plate-ai-design.md` and keep M2–M5 implementation out of this branch.
- [ ] Update the pull request description with setup, test, build, and smoke-generation evidence.
