"""Unified model pipeline status (MLflow deploy + DVC + inference)."""

from pathlib import Path
from typing import Any

from app.training.config import TrainingSettings
from app.training.dvc_status import get_unified_model_status, sync_checkpoint_to_dvc
from app.training.minio_storage import MinioStorage


def build_model_status(settings: TrainingSettings | None = None) -> dict[str, Any]:
    cfg = settings or TrainingSettings.from_env()
    storage = MinioStorage(cfg)
    return get_unified_model_status(
        checkpoint_path=Path(cfg.base_checkpoint_path),
        registry_name=cfg.mlflow_model_name,
        storage=storage,
    )


def sync_model_to_dvc(settings: TrainingSettings | None = None) -> dict[str, Any]:
    cfg = settings or TrainingSettings.from_env()
    storage = MinioStorage(cfg)
    dvc_result = sync_checkpoint_to_dvc(cfg.base_checkpoint_path, storage)
    unified = get_unified_model_status(
        checkpoint_path=cfg.base_checkpoint_path,
        registry_name=cfg.mlflow_model_name,
        storage=storage,
    )
    return {"dvc_sync": dvc_result, **unified}
