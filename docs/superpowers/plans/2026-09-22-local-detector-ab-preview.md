# Local Detector A/B Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the local Live Inference page visibly compare `active-v1` against the unqualified `candidate-detector-real-v1` on the same uploaded image, without changing active model files.

**Architecture:** A second, bundle-specific cached `PredictorEngine` is created only for the fixed candidate bundle. `POST /api/predict/compare` runs both engines against exactly the same decoded upload bytes and returns named results. The existing `/api/predict` behavior stays unchanged. The frontend opt-in renders active output normally and candidate pose output as a visually distinct overlay and summary.

**Tech Stack:** FastAPI, existing `PredictorEngine` / `PlateReader`, jQuery/Bootstrap/canvas, pytest, local Playwright fallback (Browser plugin unavailable).

**Spec:** User-approved local A/B preview described in the 2026-09-22 conversation; [Detector real-scene repair plan](../../plans/2026-09-22-detector-real-scene-repair.md).

## Global Constraints

- Candidate bundle is exactly `models/bundles/candidate-detector-real-v1`; caller input never selects a filesystem path.
- Preview is local-only and must state its unreviewed data/license and failed promotion gate.
- Do not call `/api/model/activate`, copy a model file, mutate `active-v1`, alter OCR rules, or silently fall back to active output.
- Existing `/api/predict` response contract remains unchanged.
- Preserve Windows CRLF/BOM format in pre-existing files; do not stage or commit unrelated work.

## Review Focus

- Candidate bundle missing/corrupt: active inference still responds and the candidate panel shows the actual load error.
- Candidate detector finds a plate but OCR rejects it: A/B overlay must still draw that candidate as diagnostic-only, never report it as an accepted plate.
- Regular upload after A/B upload: candidate panel and dashed overlays must clear rather than show stale results.
- Candidate name/path injection: endpoint exposes only the repository-owned fixed candidate name.
- Candidate warning: unreviewed/local-only and non-promoted status must remain visible even when it returns an OCR result.

---

### Task 1: Fixed-bundle Predictor and comparison API

**Files:**
- Modify: `src/plateai_web/predictor.py`
- Modify: `src/plateai_web/app.py`
- Test: `tests/test_web_api.py`
- Test: `tests/unit/test_web_predictor.py`

**Interfaces:**
- Consumes: `(image_bytes: bytes)` from existing multipart/base64 parsing.
- Produces: `POST /api/predict/compare -> {"active": PredictResult, "candidate": PredictResult, "preview": {...}}`.
- Preserves: `POST /api/predict -> PredictResult`.

- [x] **Step 1: Write the failing tests**

```python
def test_compare_api_returns_named_active_and_candidate_without_activation(web_client, monkeypatch):
    monkeypatch.setattr(app_module, 'predictor', StubPredictor('active-v1'))
    monkeypatch.setattr(app_module, '_candidate_predictor', lambda: StubPredictor('candidate-detector-real-v1'))
    response = web_client.post('/api/predict/compare', files={'file': ('plate.png', b'png', 'image/png')})
    assert response.status_code == 200
    assert response.json()['active']['diagnostics']['bundle_name'] == 'active-v1'
    assert response.json()['candidate']['diagnostics']['bundle_name'] == 'candidate-detector-real-v1'
```

```python
def test_candidate_predictor_marks_unreviewed_bundle_as_local_preview(monkeypatch, tmp_path):
    value = PredictorEngine(root=tmp_path, bundle_name='candidate-detector-real-v1', preview_only=True)
    assert '本機 A/B 候選' in value.model_warnings[0]
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv\\Scripts\\python -m pytest tests/test_web_api.py::test_compare_api_returns_named_active_and_candidate_without_activation tests/unit/test_web_predictor.py::test_candidate_predictor_marks_unreviewed_bundle_as_local_preview -q`

Expected: FAIL because the route and bundle-specific constructor do not exist.

- [x] **Step 3: Implement the smallest fixed-bundle comparison path**

```python
class PredictorEngine:
    def __init__(self, root=None, *, bundle_name='active-v1', preview_only=False, ...):
        self.bundle_name = bundle_name
        self.preview_only = preview_only

    def _bundle_path(self) -> Path:
        return self.root / 'models' / 'bundles' / self.bundle_name
```

```python
@app.post('/api/predict/compare')
async def compare_image_upload(...):
    image_bytes = await _uploaded_image_bytes(file, image_base64)
    return {'active': predictor.predict_image(image_bytes),
            'candidate': _candidate_predictor().predict_image(image_bytes),
            'preview': {'active_bundle': 'active-v1', 'candidate_bundle': CANDIDATE_BUNDLE}}
```

- [x] **Step 4: Run focused tests to verify they pass**

Run: `.venv\\Scripts\\python -m pytest tests/test_web_api.py tests/unit/test_web_predictor.py -q`

Expected: PASS; no activation endpoint call and existing predictor behavior remains covered.

### Task 2: Opt-in A/B visualization and browser proof

**Files:**
- Modify: `web/index.html`
- Modify: `web/js/app.js`
- Test: local Playwright temporary script outside the repository

**Interfaces:**
- Consumes: the Task 1 named `active`/`candidate` response.
- Produces: opt-in candidate preview control, named summary, purple candidate polygons and diagnostic dashed candidate rejections.

- [x] **Step 1: Add the opt-in control and empty A/B summary host**

```html
<input id="toggle-candidate-preview" type="checkbox">
<label for="toggle-candidate-preview">本機 A/B 候選定位預覽（不啟用）</label>
<div id="infer-ab-summary" hidden></div>
```

- [x] **Step 2: Route checked uploads and render both outputs**

```javascript
const url = $('#toggle-candidate-preview').prop('checked')
  ? '/api/predict/compare' : '/api/predict';
// Main table/canvas uses response.active. Candidate detections are purple.
// Candidate rejections with polygon/box are purple dashed diagnostic-only.
```

- [x] **Step 3: Verify rendered behavior with Playwright**

Run a temporary script against `http://127.0.0.1:1688`:

```javascript
await page.setInputFiles('#file-input', fixturePath);
await page.check('#toggle-candidate-preview');
await page.setInputFiles('#file-input', fixturePath);
await expect(page.locator('#infer-ab-summary')).toContainText('候選');
```

Expected: active and candidate labels are visible, candidate is marked local-only/non-promoted, no console error is introduced, and regular mode clears the A/B summary.

### Task 3: Regression and evidence

**Files:**
- Modify: `history.md`

- [x] **Step 1: Run full verification**

Run: `.venv\\Scripts\\python -m pytest -q`, `.venv\\Scripts\\python -m pip check`, `node --check web/js/app.js`, and `git -c core.whitespace=cr-at-eol diff --check`.

- [x] **Step 2: Append observed local evidence and boundaries**

Record only verified browser/API evidence, exact candidate/active separation, no activation, and remaining candidate-gate limitations in `history.md`.

## Self-Review

- Spec coverage: fixed local candidate, explicit warnings, same-byte comparison, active-contract preservation, visible pose evidence, candidate rejection visibility, and no activation are all assigned above.
- Placeholder scan: no TBD/TODO or undefined interface names remain.
- Type consistency: API uses `active`, `candidate`, and `preview` throughout; existing results retain their current schema.
- Review focus: missing bundle, rejection-only candidate, mode reset, path choice, and warning visibility have route/UI checks.

## Execution Handoff

The user explicitly approved the proposed A/B preview. Implement natively in this session with the TDD sequence above; do not use a worktree because the user needs the existing local 1688 project context.
