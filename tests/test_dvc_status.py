import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.training.dvc_status import (
    compute_file_md5,
    get_dvc_status,
    read_dvc_sidecar,
    sync_checkpoint_to_dvc,
    write_dvc_sidecar,
)


def test_read_and_write_dvc_sidecar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights-v1")

    md5 = compute_file_md5(checkpoint)
    write_dvc_sidecar(checkpoint, md5=md5, size=checkpoint.stat().st_size)
    sidecar = read_dvc_sidecar(checkpoint)

    assert sidecar is not None
    assert sidecar["md5"] == md5
    assert sidecar["size"] == len(b"weights-v1")


def test_get_dvc_status_out_of_sync(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"old")
    md5 = compute_file_md5(checkpoint)
    write_dvc_sidecar(checkpoint, md5=md5, size=len(b"old"))

    checkpoint.write_bytes(b"new-version")
    status = get_dvc_status(checkpoint, storage=None)

    assert status["status"] == "out_of_sync"
    assert status["disk_md5"] != status["tracked_md5"]


def test_get_dvc_status_quick_skips_md5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights-v1")
    md5 = compute_file_md5(checkpoint)
    write_dvc_sidecar(checkpoint, md5=md5, size=checkpoint.stat().st_size)

    def fail_md5(*_args, **_kwargs):
        raise AssertionError("compute_file_md5 should not run in quick mode")

    monkeypatch.setattr(
        "app.training.dvc_status.compute_file_md5",
        fail_md5,
    )

    status = get_dvc_status(checkpoint, storage=None, quick=True)

    assert status["status"] == "in_sync"
    assert status["disk_md5"] == md5


def test_get_dvc_status_quick_detects_size_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"old")
    md5 = compute_file_md5(checkpoint)
    write_dvc_sidecar(checkpoint, md5=md5, size=len(b"old"))
    checkpoint.write_bytes(b"new-version")

    status = get_dvc_status(checkpoint, storage=None, quick=True)

    assert status["status"] == "out_of_sync"
    assert status["disk_md5"] is None


def test_sync_checkpoint_to_dvc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    payload = b"sync-me" * 1000
    checkpoint.write_bytes(payload)
    md5 = hashlib.md5(payload).hexdigest()

    storage = MagicMock()
    storage.client.stat_object.side_effect = Exception("not found")
    storage.upload_file.return_value = (
        f"s3://mlops-models/dvc/files/md5/{md5[:2]}/{md5[2:]}"
    )

    monkeypatch.setenv("DVC_BUCKET", "mlops-models")
    monkeypatch.setenv("DVC_PREFIX", "dvc")

    result = sync_checkpoint_to_dvc(checkpoint, storage)

    assert result["status"] == "synced"
    assert result["md5"] == md5
    storage.upload_file.assert_called_once()
    sidecar = read_dvc_sidecar(checkpoint)
    assert sidecar is not None
    assert sidecar["md5"] == md5
