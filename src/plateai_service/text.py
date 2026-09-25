"""Conservative OCR row selection, not a vehicle registration/rule database."""
from __future__ import annotations

import re

import numpy as np

# Text shapes help detect competing OCR rows; they do not establish legal allocation.
SHAPES = [re.compile(pattern) for pattern in (
    r'([A-Z]{2,3})([0-9]{3,4})', r'([0-9]{3,4})([A-Z]{2,3})',
    r'([A-Z]{2})([0-9]{2})', r'([0-9]{2})([A-Z]{2})',
)]


def normalize_text(raw: str) -> str:
    ascii_width = ''.join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in raw)
    return ''.join(c for c in ascii_width.upper() if c.isalnum())


def shape_parts(text: str):
    return next((match.groups() for pattern in SHAPES if (match := pattern.fullmatch(text))), None)


def suggest_b8_by_shape(text: str) -> str:
    if shape_parts(text):
        return text
    candidates = {
        text[:index] + ('8' if char == 'B' else 'B') + text[index + 1:]
        for index, char in enumerate(text) if char in 'B8'
        if shape_parts(text[:index] + ('8' if char == 'B' else 'B') + text[index + 1:])
    }
    return candidates.pop() if len(candidates) == 1 else text


def warning(code: str) -> dict:
    messages = {
        'format_unknown': '未載入適用的完整法規編碼表，請複核。',
        'format_suggested_b8': '依字母／數字位置建議 B／8 校正，請複核原始 OCR。',
        'special_plate_unverified': '特殊字頭未驗證，請保留原文複核。',
        'unexpected_character': '包含一般英數以外字元，未自動刪除或猜改。',
        'ocr_charset_limited': '本版英文 OCR 未驗證中文或特殊車牌能力。',
    }
    return {'code': code, 'message': messages[code]}


def read_plate_line(texts, polygons) -> dict:
    result = {
        'text': None, 'raw_text': None, 'display_text': None,
        'status': 'unreadable', 'reason': 'ocr_empty',
        'format_check': {'status': 'unknown', 'rule_id': None,
                         'ruleset_version': 'unverified-20260925',
                         'allocation_status': 'unknown', 'registration_status': 'not_checked'},
        'warnings': [warning('ocr_charset_limited')],
    }
    if not texts:
        return result
    if len(texts) != len(polygons):
        raise ValueError('invalid_ocr_geometry')
    boxes = [np.asarray(p, dtype=float) for p in polygons]
    if any(p.shape != (4, 2) or not np.isfinite(p).all() for p in boxes):
        raise ValueError('invalid_ocr_geometry')
    heights = [(np.linalg.norm(p[3]-p[0]) + np.linalg.norm(p[2]-p[1])) / 2 for p in boxes]
    remaining = {i for i, text in enumerate(texts) if str(text).strip() and heights[i] > 0}
    rows = []
    while remaining:
        anchor = max(remaining, key=lambda i: heights[i])
        q = boxes[anchor]
        axis = q[1]-q[0]
        axis /= max(float(np.linalg.norm(axis)), 1e-6)
        if axis[0] < 0:
            axis = -axis
        normal = np.array([-axis[1], axis[0]])
        center, height = q.mean(axis=0), heights[anchor]
        indices = [i for i in remaining if heights[i] >= .5*height and
                   abs(float(np.dot(boxes[i].mean(axis=0)-center, normal))) <= .5*height]
        indices.sort(key=lambda i: float(np.dot(boxes[i].mean(axis=0), axis)))
        raw = ' '.join(str(texts[i]) for i in indices)
        rows.append((height, raw, normalize_text(raw)))
        remaining.difference_update(indices)
    if not rows:
        return result
    plausible = [row for row in rows if shape_parts(row[2])]
    if len(plausible) > 1:
        result.update(status='ambiguous', reason='multiple_text_candidates')
        return result
    # Preserve the probe's tallest-row policy; no GT or score voting.
    _, raw, text = max(rows, key=lambda row: row[0])
    if not text:
        return result
    suggested = suggest_b8_by_shape(text)
    was_suggested = suggested != text
    text = suggested
    parts = shape_parts(text)
    result.update(text=text, raw_text=raw, display_text='-'.join(parts) if parts else None,
                  status='recognized' if parts else 'unverified_format',
                  reason=None if parts else 'format_unknown')
    result['warnings'].append(warning('format_unknown'))
    if was_suggested:
        result['warnings'].append(warning('format_suggested_b8'))
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        result['warnings'].append(warning('special_plate_unverified'))
    if not re.fullmatch('[A-Z0-9]+', text):
        result['warnings'].append(warning('unexpected_character'))
    return result
