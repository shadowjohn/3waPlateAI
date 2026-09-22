"""Detector-only frozen holdout comparison; no OCR and no model activation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from plateai_reader.detector import postprocess_candidates
from plateai_reader.rectifier import InvalidCornersError, rectify_plate
from plateai_shared.detection import letterbox_rgb_v1
from plateai_shared.publication import publish_directory_no_replace, remove_owned_staging
from plateai_trainer.detection.engine import _collate, _evaluate, _METRIC_CONFIG, _iou
from plateai_trainer.detection.export import _load_detector_checkpoint, _POSTPROCESS
from plateai_trainer.detection.real_dataset import RealDetectionDataset


def operating_summary(metrics):
    matched = metrics['corner_matched_instances']
    precision = matched / metrics['nms_predictions'] if metrics['nms_predictions'] else 0.
    recall = matched / metrics['instances'] if metrics['instances'] else 0.
    return {'bbox_precision': precision, 'bbox_recall': recall,
            'local_promotion_gate': precision >= .90 and recall >= .90 and metrics['complete_quad_recall'] >= .80}


def _write_json(path, document):
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def _probe_cases(manifests, labels):
    for manifest in manifests:
        for row in (json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()):
            truth = row['ground_truth']
            if labels and truth['display'] not in labels:
                continue
            image_path = (manifest.parent / row['image']['path']).resolve()
            image_path.relative_to(manifest.parent.resolve())
            rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), 1), cv2.COLOR_BGR2RGB)
            if 'corners' in truth:
                corners = np.asarray(truth['corners'], dtype=np.float32)
            else:
                cx, cy, width, height, radians = truth['xywhr']
                corners = cv2.boxPoints(((cx, cy), (width, height), float(np.degrees(radians))))
            yield truth['display'], rgb, corners, image_path


def _probes(models, manifests, labels, output, device):
    results = []
    for i, (label, rgb, gt_corners, path) in enumerate(_probe_cases(manifests, labels)):
        image, transform = letterbox_rgb_v1(rgb)
        panels = []
        record = {'label': label, 'image': str(path), 'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'models': {}}
        gt_box = np.r_[gt_corners.min(axis=0), gt_corners.max(axis=0)]
        for name, model in models.items():
            with torch.no_grad():
                rows = model(torch.from_numpy(image[None]).to(device))[0].cpu().numpy()
            detections = postprocess_candidates(rows, transform, _POSTPROCESS)
            boxes = np.asarray([d.bbox_xyxy for d in detections], dtype=np.float32).reshape(-1, 4)
            values = _iou(gt_box, boxes)
            result = {'raw_above_025': int((rows[:, 4] >= .25).sum()), 'nms_count': len(detections),
                      'best_retained_bbox_iou': float(values.max()) if len(values) else 0.,
                      'top_score_bbox_iou': float(values[0]) if len(values) else 0.,
                      'top_score_crop': None}
            panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.polylines(panel, [np.round(gt_corners).astype(np.int32)], True, (255, 100, 0), 2)
            for detection in detections:
                cv2.polylines(panel, [np.round(detection.corners_xy).astype(np.int32)], True, (0, 200, 0), 1)
            if detections:
                try:
                    crop = rectify_plate(rgb, detections[0].corners_xy).image_rgb
                    filename = f'probe-{i:02d}-{name}-top-crop.png'
                    cv2.imwrite(str(output / filename), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    result['top_score_crop'] = filename
                except InvalidCornersError as error:
                    result['top_score_crop_rejection'] = error.reason
            cv2.putText(panel, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
            panels.append(panel)
            record['models'][name] = result
        cv2.imwrite(str(output / f'probe-{i:02d}-comparison.png'), np.hstack(panels))
        results.append(record)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--probe-manifest', type=Path, action='append', default=[])
    parser.add_argument('--probe-label', action='append', default=[])
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output exists')
    torch.set_num_threads(1)
    dataset = RealDetectionDataset(args.test)
    if dataset.provenance['split'] != 'test':
        parser.error('independent test split required; select checkpoint using validation first')
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=_collate)
    models, reports = {}, {}
    for name, path in (('baseline', args.baseline), ('candidate', args.candidate)):
        payload = path.read_bytes()
        model = _load_detector_checkpoint(payload).to(args.device).eval()
        models[name] = model
        metrics = _evaluate(model, loader)
        reports[name] = {'checkpoint': str(path), 'sha256': hashlib.sha256(payload).hexdigest(),
                         'metrics': metrics, **operating_summary(metrics)}
        print(json.dumps({name: {key: value for key, value in reports[name].items() if key != 'metrics'}, 'metrics': metrics}), flush=True)
    dataset.verify_unchanged()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.parent / f'.{args.output.name}.partial-{uuid.uuid4().hex}'
    staging.mkdir()
    try:
        probes = _probes(models, args.probe_manifest, args.probe_label, staging, args.device)
        _write_json(staging / 'comparison.json', {'models': reports, 'probes': probes,
            'test_input_hashes': dataset.input_hashes, 'metric_config': _METRIC_CONFIG,
            'probe_postprocess': _POSTPROCESS, 'local_only': True, 'license_reviewed': False,
            'boundary': 'Frozen detector-only local holdout; OCR/production not evaluated. Test NMS .45, public Reader probe NMS .50. No model activation.'})
        publish_directory_no_replace(staging, args.output)
    except BaseException:
        remove_owned_staging(staging, args.output)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
