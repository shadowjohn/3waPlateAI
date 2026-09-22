"""Regressions for false plate acceptance and hidden Web inference fallbacks."""
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from plateai_reader.runtime import DecodedPlate, PlateRead, ReaderResult, ReaderTiming
from plateai_shared.detection import PlateDetection
from plateai_web.predictor import PredictorEngine


def image_bytes():
    return cv2.imencode('.png', np.full((100, 200, 3), 255, np.uint8))[1].tobytes()


def result(score=-0.05):
    detection = PlateDetection(
        bbox_xyxy=np.array([10, 20, 190, 80], np.float32), confidence=0.9,
        corners_xy=np.array([[10, 20], [190, 20], [190, 80], [10, 80]], np.float32),
    )
    plate = PlateRead(detection, DecodedPlate('ABC5678', 'ABC-5678', 'v1', 'standard', score))
    timing = ReaderTiming(12, 1, 300, 1, 1, (1,))
    return ReaderResult((plate,), (), ('CPUExecutionProvider',), timing)


def engine(monkeypatch, tmp_path, read):
    # Only substitute model loading/inference, retaining real Web decisions.
    monkeypatch.setattr(PredictorEngine, '_load_active_model', lambda self: None)
    value = PredictorEngine()
    value.root = tmp_path
    value.reader = SimpleNamespace(read=read)
    value.recognizer_session = object()
    value.codec = object()
    return value


def test_low_score_is_rejected_instead_of_clamped_to_five_percent(monkeypatch, tmp_path):
    value = engine(monkeypatch, tmp_path, lambda image: result(-100))
    response = value.predict_image(image_bytes())
    assert response['detections'] == []
    assert response['rejections'][0]['reason'] == 'low_recognition_score'
    assert response['rejections'][0]['recognition_score'] < 0.001
    assert response['diagnostics']['pipeline_mode'] == 'neural_full_pipeline'


def test_empty_neural_result_does_not_invoke_contour_fallback(monkeypatch, tmp_path):
    value = engine(monkeypatch, tmp_path, lambda image: replace(result(), plates=()))
    def forbidden(*args):
        pytest.fail('neural empty result must remain empty')
    monkeypatch.setattr(value, '_detect_and_recognize_with_timings', forbidden)
    response = value.predict_image(image_bytes())
    assert response['count'] == 0
    assert response['diagnostics']['pipeline_mode'] == 'neural_full_pipeline'


def test_neural_failure_is_reported_without_contour_fallback(monkeypatch, tmp_path):
    def fail(image):
        raise RuntimeError('invalid output shape')
    value = engine(monkeypatch, tmp_path, fail)
    monkeypatch.setattr(value, '_detect_and_recognize_with_timings', lambda *a: pytest.fail('hidden fallback'))
    response = value.predict_image(image_bytes())
    assert response['status'] == 'inference_error'
    assert 'invalid output shape' in response['diagnostics']['error']
    assert response['count'] == 0


def test_unchanged_full_bundle_reuses_sessions(monkeypatch, tmp_path):
    import json
    import plateai_web.predictor as module
    bundle = tmp_path / 'models/bundles/active-v1'
    bundle.mkdir(parents=True)
    manifest = {'capabilities': ['crop-recognition', 'plate-detection'], 'components': {'detector': {}}}
    (bundle / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    created = []
    def factory(path):
        created.append(path)
        return SimpleNamespace(manifest=manifest, codec=object(), ruleset=object(),
                               _recognizer_session=object(), providers=('CPUExecutionProvider',))
    monkeypatch.setattr(module, 'PlateReader', factory)
    value = PredictorEngine.__new__(PredictorEngine)
    value.__init__(root=tmp_path)
    first = value.reader
    value._load_active_model()
    value._load_active_model()
    assert value.reader is first
    assert len(created) == 1
    # A changed invalid manifest must invalidate the cached sessions, not keep
    # serving the old model while advertising the new active bundle.
    (bundle / 'manifest.json').write_text('{broken', encoding='utf-8')
    response = value.predict_image(image_bytes())
    assert response['status'] == 'model_error'
    assert value.reader is value.recognizer_session is None


def test_greedy_is_not_replaced_by_constrained_text(monkeypatch, tmp_path):
    r = result()
    plate = replace(r.plates[0], raw_greedy_text='ABC567', crop_rgb=np.zeros((160, 380, 3), np.uint8))
    value = engine(monkeypatch, tmp_path, lambda image: replace(r, plates=(plate,)))
    response = value.predict_image(image_bytes())
    assert response['detections'] == []
    rejected = response['rejections'][0]
    assert rejected['raw_greedy_text'] == 'ABC567'
    assert rejected['canonical'] == 'ABC5678'
    assert rejected['reason'] == 'decoder_disagreement'


def test_valid_high_score_plate_survives_with_separate_timings(monkeypatch, tmp_path):
    r = result()
    plate = replace(r.plates[0], raw_greedy_text='ABC5678', crop_rgb=np.zeros((160, 380, 3), np.uint8))
    timing = replace(r.timing, onnx_inference_ms=4.0, ctc_decoding_ms=295.0, preprocess_ms=1.0)
    value = engine(monkeypatch, tmp_path, lambda image: replace(r, plates=(plate,), timing=timing))
    response = value.predict_image(image_bytes())
    assert response['count'] == 1
    assert response['detections'][0]['raw_greedy_text'] == 'ABC5678'
    assert response['diagnostics']['timing_breakdown']['onnx_inference_ms'] == 4
    assert response['diagnostics']['timing_breakdown']['ctc_decoding_ms'] == 295
    assert response['detections'][0]['timings']['onnx_ms'] is None  # shared batch, not per-plate total


def test_corrupt_bundle_cannot_load_an_unvalidated_recognizer(tmp_path):
    bundle = tmp_path / 'models/bundles/active-v1'
    bundle.mkdir(parents=True)
    (bundle / 'manifest.json').write_text('{broken', encoding='utf-8')
    value = PredictorEngine(root=tmp_path)
    response = value.predict_image(image_bytes())
    assert response['status'] == 'model_error'
    assert response['model_status'] == 'not_loaded'
    assert response['diagnostics']['error']
