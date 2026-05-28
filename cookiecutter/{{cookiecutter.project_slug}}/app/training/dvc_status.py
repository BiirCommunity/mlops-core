"""DVC-compatible checkpoint tracking via MinIO (without requiring dvc CLI in runtime)."""

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

from app.training.minio_storage import MinioStorage

_DVC_MD5_RE = re.compile(r"^\s*-?\s*md5:\s*([0-9a-f]+)\s*$", re.MULTILINE)
_DVC_SIZE_RE = re.compile(r"^\s*size:\s*(\d+)\s*$", re.MULTILINE)
_DVC_PATH_RE = re.compile(r"^\s*path:\s*(\S+)\s*$", re.MULTILINE)


def dvc_remote_settings() -> tuple[str, str]:
    bucket = os.environ.get("DVC_BUCKET") or os.environ.get(
        "MINIO_BUCKET_MODELS", "{{ cookiecutter.minio_bucket_models }}"
    )
    prefix = (os.environ.get("DVC_PREFIX") or "dvc").strip("/")
    return bucket, prefix


def sidecar_path_for(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    return path.parent / f"{path.name}.dvc"


def compute_file_md5(
    checkpoint_path: str | Path, *, chunk_size: int = 8 * 1024 * 1024
) -> str:
    digest = hashlib.md5()
    with Path(checkpoint_path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_dvc_sidecar(checkpoint_path: str | Path) -> dict[str, Any] | None:
    sidecar = sidecar_path_for(checkpoint_path)
    if not sidecar.is_file():
        return None
    try:
        text = sidecar.read_text(encoding="utf-8")
    except OSError:
        return None
    md5_match = _DVC_MD5_RE.search(text)
    size_match = _DVC_SIZE_RE.search(text)
    path_match = _DVC_PATH_RE.search(text)
    if md5_match is None or size_match is None:
        return None
    return {
        "sidecar_path": str(sidecar),
        "md5": md5_match.group(1),
        "size": int(size_match.group(1)),
        "path": path_match.group(1) if path_match else Path(checkpoint_path).name,
    }


def write_dvc_sidecar(
    checkpoint_path: str | Path,
    *,
    md5: str,
    size: int,
) -> Path:
    path = Path(checkpoint_path)
    sidecar = sidecar_path_for(path)
    sidecar.write_text(
        "outs:\n"
        f"- md5: {md5}\n"
        f"  size: {size}\n"
        "  hash: md5\n"
        f"  path: {path.name}\n",
        encoding="utf-8",
    )
    return sidecar


def dvc_object_name(md5: str) -> str:
    bucket, prefix = dvc_remote_settings()
    del bucket
    return f"{prefix}/files/md5/{md5[:2]}/{md5[2:]}"


def remote_object_exists(storage: MinioStorage, md5: str) -> bool:
    bucket, _ = dvc_remote_settings()
    object_name = dvc_object_name(md5)
    try:
        storage.client.stat_object(bucket, object_name)
        return True
    except Exception:
        return False


def get_dvc_status(
    checkpoint_path: str | Path,
    storage: MinioStorage | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    bucket, prefix = dvc_remote_settings()
    sidecar = read_dvc_sidecar(path)

    if not path.is_file():
        tracked_md5 = sidecar["md5"] if sidecar else None
        remote_present: bool | None = None
        if storage is not None and tracked_md5:
            remote_present = remote_object_exists(storage, tracked_md5)
        return {
            "checkpoint_exists": False,
            "remote_uri": f"s3://{bucket}/{prefix}",
            "sidecar_path": (
                sidecar["sidecar_path"] if sidecar else str(sidecar_path_for(path))
            ),
            "sidecar": sidecar,
            "disk_md5": None,
            "disk_size": None,
            "tracked_md5": tracked_md5,
            "remote_present": remote_present,
            "remote_object": (
                f"s3://{bucket}/{dvc_object_name(tracked_md5)}" if tracked_md5 else None
            ),
            "in_sync": False,
            "status": "missing_checkpoint",
        }

    disk_md5 = compute_file_md5(path)
    disk_size = path.stat().st_size
    tracked_md5 = sidecar["md5"] if sidecar else None
    remote_present = None
    if storage is not None:
        check_md5 = tracked_md5 or disk_md5
        remote_present = remote_object_exists(storage, check_md5)

    disk_matches_sidecar = (
        sidecar is not None and tracked_md5 == disk_md5 and sidecar["size"] == disk_size
    )
    remote_ok = remote_present is True

    if sidecar is None:
        status = "untracked" if not remote_ok else "out_of_sync"
    elif disk_matches_sidecar and remote_ok:
        status = "in_sync"
    elif disk_matches_sidecar and remote_present is False:
        status = "remote_missing"
    elif not disk_matches_sidecar:
        status = "out_of_sync"
    else:
        status = "unknown"

    return {
        "checkpoint_exists": True,
        "remote_uri": f"s3://{bucket}/{prefix}",
        "sidecar_path": (
            sidecar["sidecar_path"] if sidecar else str(sidecar_path_for(path))
        ),
        "sidecar": sidecar,
        "disk_md5": disk_md5,
        "disk_size": disk_size,
        "tracked_md5": tracked_md5,
        "remote_present": remote_present,
        "remote_object": (
            f"s3://{bucket}/{dvc_object_name(tracked_md5)}" if tracked_md5 else None
        ),
        "in_sync": status == "in_sync",
        "status": status,
    }


def sync_checkpoint_to_dvc(
    checkpoint_path: str | Path,
    storage: MinioStorage,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    disk_md5 = compute_file_md5(path)
    disk_size = path.stat().st_size
    bucket, _ = dvc_remote_settings()
    object_name = dvc_object_name(disk_md5)

    if not remote_object_exists(storage, disk_md5):
        storage.upload_file(bucket, object_name, path)

    sidecar = write_dvc_sidecar(path, md5=disk_md5, size=disk_size)
    return {
        "status": "synced",
        "md5": disk_md5,
        "size": disk_size,
        "sidecar_path": str(sidecar),
        "remote_uri": f"s3://{bucket}/{object_name}",
        "synced_at": time.time(),
    }


def get_unified_model_status(
    *,
    checkpoint_path: str | Path,
    registry_name: str,
    storage: MinioStorage | None = None,
) -> dict[str, Any]:
    from app.training.inference_model import get_inference_model_status

    inference = get_inference_model_status(
        checkpoint_path=checkpoint_path,
        registry_name=registry_name,
    )
    dvc = get_dvc_status(checkpoint_path, storage=storage)

    needs_dvc_sync = dvc["status"] in {"out_of_sync", "untracked", "remote_missing"}
    needs_restart = inference["pending_reload"]
    pipeline_ok = (
        inference["status"] == "loaded"
        and dvc["status"] == "in_sync"
        and not needs_restart
    )

    if pipeline_ok:
        pipeline_status = "ready"
    elif needs_restart:
        pipeline_status = "restart_required"
    elif needs_dvc_sync:
        pipeline_status = "dvc_sync_required"
    elif inference["status"] == "missing_checkpoint":
        pipeline_status = "missing_checkpoint"
    else:
        pipeline_status = "attention"

    return {
        "pipeline_status": pipeline_status,
        "inference": inference,
        "dvc": dvc,
        "actions": {
            "restart_app": needs_restart,
            "sync_dvc": needs_dvc_sync,
        },
    }
