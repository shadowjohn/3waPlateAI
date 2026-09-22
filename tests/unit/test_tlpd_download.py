import hashlib

import pytest

from plateai_web import downloader
from plateai_web.tasks import TaskManager, TaskStatus


@pytest.mark.parametrize('name,want', [('images/0000C6(2)(0)(1).jpg', True),
    ('labels/0000C6(2)(0)(1).json', True), ('images/../../x.jpg', False),
    ('images/a\\x.jpg', False), ('C:/x.jpg', False)])
def test_pinned_dataset_names_are_safe_and_supported(name, want):
    assert downloader._tlpd_safe_path(name) is want


def test_empty_tlpd_directories_are_not_ready(tmp_path):
    root = tmp_path / 'datasets/tlpd-taiwan-detector'
    (root / 'images').mkdir(parents=True)
    (root / 'labels').mkdir()
    assert not downloader.tlpd_is_ready(root)


def test_download_transfers_and_verifies_paired_files_before_success(tmp_path, monkeypatch):
    image = b'image-download-fixture'
    label = b'{"shapes": []}'
    def entry(path, content):
        return {'type': 'file', 'path': path, 'size': len(content),
                'oid': hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()}
    monkeypatch.setattr(downloader, 'TLPD_EXPECTED_COUNT', 1)
    monkeypatch.setattr(downloader, '_tlpd_entries', lambda directory: [
        entry('images/a.jpg', image) if directory == 'images' else entry('labels/a.json', label)])
    monkeypatch.setattr(downloader, '_tlpd_bytes', lambda path: image if path.endswith('.jpg') else label)
    manager = TaskManager()
    task = manager.create_task('download')
    downloader.download_tlpd_task(task, manager, root=tmp_path)
    target = tmp_path / 'datasets/tlpd-taiwan-detector'
    assert (target / 'images/a.jpg').read_bytes() == image
    assert (target / 'labels/a.json').read_bytes() == label
    assert manager.get_task(task).status == TaskStatus.COMPLETED
    assert downloader.tlpd_is_ready(target)
    (target / 'labels/a.json').unlink()
    assert not downloader.tlpd_is_ready(target)


def test_download_failure_never_publishes_ready_marker(tmp_path, monkeypatch):
    def fail(directory):
        raise OSError('network unavailable')
    monkeypatch.setattr(downloader, '_tlpd_entries', fail)
    manager = TaskManager()
    task = manager.create_task('download')
    downloader.download_tlpd_task(task, manager, root=tmp_path)
    assert manager.get_task(task).status == TaskStatus.FAILED
    assert not downloader.tlpd_is_ready(tmp_path / 'datasets/tlpd-taiwan-detector')


def test_download_checksum_mismatch_never_reports_success(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, 'TLPD_EXPECTED_COUNT', 1)
    monkeypatch.setattr(downloader, '_tlpd_entries', lambda directory: [{
        'path': f'{directory}/a.' + ('jpg' if directory == 'images' else 'json'),
        'size': 4, 'lfs': {'oid': hashlib.sha256(b'good').hexdigest()},
    }])
    monkeypatch.setattr(downloader, '_tlpd_bytes', lambda path: b'evil')
    manager = TaskManager()
    task = manager.create_task('download')
    downloader.download_tlpd_task(task, manager, root=tmp_path)
    assert manager.get_task(task).status == TaskStatus.FAILED
    assert not (tmp_path / 'datasets/tlpd-taiwan-detector/download-complete.json').exists()
