"""Dataset download routines for EZCon and TLPD."""
from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from .paths import workspace_root
from .tasks import TaskManager, task_manager


def download_ezcon_task(task_id: str, tm: TaskManager, *, acknowledged: bool = False):
    if not acknowledged:
        tm.fail_task(task_id, "請先確認 EZCon 授權尚未審核，僅供本機實驗")
        return
    root = workspace_root()
    target_dir = root / "datasets" / "restricted" / "ezcon-taiwan-recognition-test"
    
    tm.update_progress(task_id, 10, "正在檢查 EZCon 目錄與權限...")
    tm.append_log(task_id, "=== 開始下載 EZCon 台灣真實車牌測試集 ===")
    tm.append_log(task_id, f"目標目錄: {target_dir}")
    
    if target_dir.exists():
        tm.append_log(task_id, "目標目錄已存在，準備進行完整性或更新檢驗...")
        summary_file = target_dir / "summary.json"
        if summary_file.exists():
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            tm.append_log(task_id, f"EZCon 已就緒！總樣本數: {data.get('total_samples')}, 符合 v1 規則樣本數: {data.get('reader_v1_eligible_samples')}")
            tm.complete_task(task_id, result={"status": "ready", "path": str(target_dir), "samples": data.get("total_samples")}, message="EZCon 資料集已就緒")
            return
            
    # Run fetch_ezcon_taiwan_eval.py
    fetch_script = root / "tools" / "fetch_ezcon_taiwan_eval.py"
    py_exec = sys.executable
    cmd = [
        py_exec,
        str(fetch_script),
        "--output", str(target_dir),
        "--acknowledge-unreviewed-license",
    ]
    
    tm.update_progress(task_id, 25, "正在連線 Hugging Face 下載測試資料...")
    rc = tm.run_subprocess_command(task_id, cmd, cwd=str(root))
    if rc == 0:
        tm.update_progress(task_id, 100, "EZCon 下載完成！")
        tm.complete_task(task_id, result={"status": "downloaded", "path": str(target_dir)}, message="EZCon 資料集下載成功")
    else:
        tm.fail_task(task_id, f"EZCon 下載腳本退出，Exit Code: {rc}")


TLPD_REVISION = '00f9ae2fa3d186bcf74e7f6ef4f48280181d6e85'
TLPD_EXPECTED_COUNT = 3032


def tlpd_is_ready(target: Path) -> bool:
    """A verified download marker plus complete unchanged file sizes is required.

    The downloader rechecks byte hashes on reuse. This inexpensive dashboard
    check establishes download completeness, not annotation/training quality.
    """
    try:
        marker = json.loads((target / 'download-complete.json').read_text(encoding='utf-8'))
        if marker['revision'] != TLPD_REVISION or marker['count'] != TLPD_EXPECTED_COUNT:
            return False
        sizes = marker['file_sizes']
        images = {p.stem for p in (target / 'images').glob('*.jpg')}
        labels = {p.stem for p in (target / 'labels').glob('*.json')}
        if images != labels or len(images) != TLPD_EXPECTED_COUNT or len(sizes) != 2 * TLPD_EXPECTED_COUNT:
            return False
        for name, size in sizes.items():
            if not _tlpd_safe_path(name):
                return False
            path = target / name
            if path.is_symlink() or path.resolve().parent != (target / name.split('/')[0]).resolve() or path.stat().st_size != size:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _tlpd_safe_path(name: str) -> bool:
    parts = name.split('/')
    return (len(parts) == 2 and parts[0] in ('images', 'labels')
            and bool(re.fullmatch(r'[A-Za-z0-9_.() -]+', parts[1]))
            and parts[1] not in ('.', '..')
            and parts[1].endswith('.jpg' if parts[0] == 'images' else '.json'))


def _tlpd_entries(directory: str) -> list[dict]:
    url = f'https://huggingface.co/api/datasets/evan6007/TLPD/tree/{TLPD_REVISION}/{directory}'
    entries, visited = [], set()
    while url:
        if url in visited or len(visited) >= 100:
            raise ValueError('invalid TLPD listing pagination')
        visited.add(url)
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.netloc != 'huggingface.co' or not parsed.path.startswith(f'/api/datasets/evan6007/TLPD/tree/{TLPD_REVISION}/'):
            raise ValueError('unexpected TLPD listing URL')
        with urlopen(Request(url, headers={'User-Agent': '3waPlateAI'}), timeout=30) as response:
            entries.extend(item for item in json.load(response) if item.get('type') == 'file')
            link = re.search(r'<([^>]+)>;\s*rel="?next"?', response.headers.get('Link', ''))
            url = link.group(1) if link else None
    return entries


def _tlpd_bytes(name: str) -> bytes:
    url = f'https://huggingface.co/datasets/evan6007/TLPD/resolve/{TLPD_REVISION}/{quote(name, safe="/")}'
    with urlopen(Request(url, headers={'User-Agent': '3waPlateAI'}), timeout=60) as response:
        data = response.read(32 * 1024 * 1024 + 1)
    if len(data) > 32 * 1024 * 1024:
        raise ValueError('TLPD file exceeds download limit')
    return data


def _tlpd_matches(data: bytes, entry: dict) -> bool:
    if len(data) != entry['size']:
        return False
    if entry.get('lfs'):
        return hashlib.sha256(data).hexdigest() == entry['lfs']['oid']
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == entry['oid']


def download_tlpd_task(task_id: str, tm: TaskManager, *, root: Path | None = None):
    root = Path(root) if root is not None else workspace_root()
    target = root / 'datasets/tlpd-taiwan-detector'
    try:
        tm.update_progress(task_id, 1, '正在取得固定版本的 TLPD 圖片與標註清單...')
        entries = _tlpd_entries('images') + _tlpd_entries('labels')
        paths = [entry['path'] for entry in entries]
        if len(set(paths)) != len(paths) or not all(_tlpd_safe_path(p) for p in paths):
            raise ValueError('invalid or duplicate TLPD paths')
        images = {Path(p).stem for p in paths if p.startswith('images/')}
        labels = {Path(p).stem for p in paths if p.startswith('labels/')}
        if images != labels or len(images) != TLPD_EXPECTED_COUNT:
            raise ValueError('TLPD image/label pairs are incomplete')
        tm.append_log(task_id, f'TLPD revision {TLPD_REVISION}; {len(images)} image/label pairs')
        for directory in ('images', 'labels'):
            destination = target / directory
            if destination.is_symlink():
                raise ValueError('TLPD destination must not be a symlink')
            destination.mkdir(parents=True, exist_ok=True)

        def transfer(entry):
            path = target / entry['path']
            if path.is_symlink():
                raise ValueError('TLPD files must not be symlinks')
            if path.is_file() and _tlpd_matches(path.read_bytes(), entry):
                return
            data = _tlpd_bytes(entry['path'])
            if not _tlpd_matches(data, entry):
                raise ValueError(f'TLPD checksum mismatch: {entry["path"]}')
            temporary = path.with_suffix(path.suffix + '.part')
            if temporary.is_symlink():
                raise ValueError('TLPD temporary file must not be a symlink')
            temporary.write_bytes(data)
            temporary.replace(path)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(transfer, entry) for entry in entries]
            try:
                for completed, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    if completed % 50 == 0 or completed == len(entries):
                        tm.update_progress(task_id, int(99 * completed / len(entries)), f'已驗證 {completed}/{len(entries)} 個檔案')
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        marker = target / 'download-complete.json'
        if marker.is_symlink():
            raise ValueError('TLPD marker must not be a symlink')
        marker.write_text(json.dumps({'revision': TLPD_REVISION, 'count': len(images),
            'file_sizes': {entry['path']: entry['size'] for entry in entries}}, indent=2), encoding='utf-8')
        if not tlpd_is_ready(target):
            raise ValueError('TLPD download completeness check failed')
        tm.complete_task(task_id, result={'status': 'ready', 'count': len(images), 'path': str(target)},
                         message=f'TLPD {len(images)} 組圖片與標註已下載並通過雜湊驗證')
    except Exception as error:
        tm.fail_task(task_id, f'TLPD 下載未完成：{error}；可重試以續傳已驗證檔案。')
