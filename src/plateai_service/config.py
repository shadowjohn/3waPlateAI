"""Trusted deployment manifest; request payloads never supply asset paths."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class ServiceConfig:
    model_id: str
    detector_path: Path
    detector_sha256: str
    ocr_detection_dir: Path
    ocr_recognition_dir: Path
    ocr_files: dict[str, dict[str, str]]
    device: str = 'auto'
    threads: int = 4

    @classmethod
    def from_file(cls, path: Path) -> 'ServiceConfig':
        path = Path(path).resolve()
        obj = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(obj, dict) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', str(obj.get('model_id', ''))):
            raise ValueError('invalid_model_manifest')
        def asset_path(entry):
            value = entry['path']
            if not isinstance(value, str) or not value:
                raise ValueError('invalid_model_manifest')
            return (path.parent / value).resolve()
        def digest(value):
            if not isinstance(value, str) or not re.fullmatch('[a-fA-F0-9]{64}', value):
                raise ValueError('invalid_model_manifest')
            return value.lower()
        try:
            files = {}
            for key in ('ocr_detection', 'ocr_recognition'):
                mapping = obj[key]['files']
                required = {'inference.json', 'inference.pdiparams', 'inference.yml'}
                if not isinstance(mapping, dict) or not required.issubset(mapping):
                    raise ValueError('invalid_model_manifest')
                # Only explicit files inside the model directory; no traversal/absolute names.
                if any(Path(name).name != name or ':' in name or name in ('.', '..') for name in mapping):
                    raise ValueError('invalid_model_manifest')
                files[key] = {name: digest(value) for name, value in mapping.items()}
            device, threads = obj.get('device', 'auto'), obj.get('threads', 4)
            if device not in ('auto', 'cpu') or type(threads) is not int or not 1 <= threads <= 32:
                raise ValueError('invalid_model_manifest')
            return cls(obj['model_id'], asset_path(obj['detector']), digest(obj['detector']['sha256']),
                       asset_path(obj['ocr_detection']), asset_path(obj['ocr_recognition']), files, device, threads)
        except (KeyError, TypeError) as exc:
            raise ValueError('invalid_model_manifest') from exc


def file_sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_assets(config: ServiceConfig) -> None:
    assets = [(config.detector_path, config.detector_sha256)]
    for key, folder in (('ocr_detection', config.ocr_detection_dir), ('ocr_recognition', config.ocr_recognition_dir)):
        assets.extend((folder/name, expected) for name, expected in config.ocr_files[key].items())
    for path, expected in assets:
        if not path.is_file():
            raise ValueError('asset_missing')
        if file_sha256(path) != expected:
            raise ValueError('asset_hash_mismatch')


def validate_binding(host: str, token: str | None) -> None:
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == 'localhost'
    if not loopback and (not token or len(token) < 24):
        raise ValueError('non_loopback_requires_token_at_least_24_characters')
