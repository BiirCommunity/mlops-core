import time
from pathlib import Path
from typing import Any

from app.metrics import publish_dvc_status, record_dvc_sync
from app.training.config import TrainingSettings
from app.training.dvc_status import get_unified_model_status, sync_checkpoint_to_dvc
from app.training.minio_storage import MinioStorage

CACHE_TTL_SEC = 60.0

_cache: dict[str, Any] | None = None
_cache_expires_at: float = 0.0


def _invalidate_cache() -> None:
    global _cache, _cache_expires_at
    _cache = None
    _cache_expires_at = 0.0


def build_model_status(
    settings: TrainingSettings | None = None,
    *,
    quick: bool = True,
    use_cache: bool = True,
) -> dict[str, Any]:
    global _cache, _cache_expires_at

    now = time.time()
    if use_cache and _cache is not None and now < _cache_expires_at:
        publish_dvc_status(_cache)
        return _cache

    cfg = settings or TrainingSettings.from_env()
    storage = MinioStorage(cfg)
    result = get_unified_model_status(
        checkpoint_path=Path(cfg.base_checkpoint_path),
        registry_name=cfg.mlflow_model_name,
        storage=storage,
        quick=quick,
    )
    publish_dvc_status(result)
    _cache = result
    _cache_expires_at = now + CACHE_TTL_SEC
    return result


def sync_model_to_dvc(settings: TrainingSettings | None = None) -> dict[str, Any]:
    cfg = settings or TrainingSettings.from_env()
    storage = MinioStorage(cfg)
    try:
        dvc_result = sync_checkpoint_to_dvc(cfg.base_checkpoint_path, storage)
        record_dvc_sync(success=True)
    except Exception:
        record_dvc_sync(success=False)
        raise

    _invalidate_cache()
    unified = build_model_status(cfg, quick=False, use_cache=False)
    return {"dvc_sync": dvc_result, **unified}
