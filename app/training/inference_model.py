import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "active_model.json"


@dataclass(frozen=True)
class CheckpointFingerprint:
    size_bytes: int
    mtime: float

    def to_dict(self) -> dict[str, int | float]:
        return {"size_bytes": self.size_bytes, "mtime": self.mtime}

    def matches(self, other: "CheckpointFingerprint | None") -> bool:
        if other is None:
            return False
        return (
            self.size_bytes == other.size_bytes
            and abs(self.mtime - other.mtime) < 0.001
        )


_loaded_fingerprint: CheckpointFingerprint | None = None
_loaded_version: str | None = None
_loaded_registry_name: str | None = None
_model_revision: str = ""


def manifest_path_for(checkpoint_path: str | Path) -> Path:
    return Path(checkpoint_path).resolve().parent / MANIFEST_FILENAME


def checkpoint_fingerprint(checkpoint_path: str | Path) -> CheckpointFingerprint | None:
    path = Path(checkpoint_path)
    if not path.is_file():
        return None
    stat = path.stat()
    return CheckpointFingerprint(size_bytes=stat.st_size, mtime=stat.st_mtime)


def read_deploy_manifest(checkpoint_path: str | Path) -> dict[str, Any] | None:
    manifest_path = manifest_path_for(checkpoint_path)
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_deploy_manifest(
    *,
    checkpoint_path: str | Path,
    registry_name: str,
    version: str,
    object_name: str,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    fingerprint = checkpoint_fingerprint(path)
    payload: dict[str, Any] = {
        "registry_name": registry_name,
        "version": str(version),
        "checkpoint_path": str(path),
        "object_name": object_name,
        "deployed_at": time.time(),
    }
    if fingerprint is not None:
        payload["checkpoint"] = fingerprint.to_dict()
    manifest_path = manifest_path_for(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def register_inference_startup(
    *,
    checkpoint_path: str | Path,
    model_revision: str,
) -> None:
    global _loaded_fingerprint, _loaded_version, _loaded_registry_name, _model_revision

    _model_revision = model_revision
    _loaded_fingerprint = checkpoint_fingerprint(checkpoint_path)
    manifest = read_deploy_manifest(checkpoint_path)
    if manifest:
        _loaded_registry_name = str(manifest.get("registry_name") or "")
        manifest_fp = _fingerprint_from_manifest(manifest)
        if _loaded_fingerprint and _loaded_fingerprint.matches(manifest_fp):
            _loaded_version = str(manifest.get("version") or "") or None
        else:
            _loaded_version = None
    else:
        _loaded_registry_name = None
        _loaded_version = None


def get_inference_model_status(
    *,
    checkpoint_path: str | Path | None = None,
    registry_name: str | None = None,
) -> dict[str, Any]:
    path = Path(
        checkpoint_path or os.environ.get("CHECKPOINT_PATH", "/models/model.pt")
    )
    manifest = read_deploy_manifest(path)
    disk_fp = checkpoint_fingerprint(path)
    manifest_fp = _fingerprint_from_manifest(manifest) if manifest else None

    deployed_version = (
        str(manifest.get("version"))
        if manifest and manifest.get("version") is not None
        else None
    )
    deployed_at = manifest.get("deployed_at") if manifest else None
    resolved_registry = (
        registry_name
        or (str(manifest.get("registry_name")) if manifest else None)
        or _loaded_registry_name
        or os.environ.get("MLFLOW_MODEL_NAME", "llm-lora")
    )

    pending_reload = bool(
        disk_fp and _loaded_fingerprint and not disk_fp.matches(_loaded_fingerprint)
    )
    manifest_matches_disk = bool(
        disk_fp and manifest_fp and disk_fp.matches(manifest_fp)
    )

    loaded_version = _loaded_version
    if loaded_version is None and manifest_matches_disk and not pending_reload:
        loaded_version = deployed_version

    if pending_reload:
        status = "reload_required"
    elif loaded_version:
        status = "loaded"
    elif path.is_file():
        status = "unknown"
    else:
        status = "missing_checkpoint"

    return {
        "registry_name": resolved_registry,
        "checkpoint_path": str(path),
        "model_revision": _model_revision,
        "loaded_version": loaded_version,
        "deployed_version": deployed_version if manifest_matches_disk else None,
        "pending_reload": pending_reload,
        "deployed_at": deployed_at,
        "object_name": manifest.get("object_name") if manifest else None,
        "checkpoint_on_disk": disk_fp.to_dict() if disk_fp else None,
        "checkpoint_loaded_at_startup": (
            _loaded_fingerprint.to_dict() if _loaded_fingerprint else None
        ),
        "status": status,
    }


def _fingerprint_from_manifest(
    manifest: dict[str, Any],
) -> CheckpointFingerprint | None:
    raw = manifest.get("checkpoint")
    if not isinstance(raw, dict):
        return None
    size_bytes = raw.get("size_bytes")
    mtime = raw.get("mtime")
    if not isinstance(size_bytes, int) or not isinstance(mtime, (int, float)):
        return None
    return CheckpointFingerprint(size_bytes=size_bytes, mtime=float(mtime))
