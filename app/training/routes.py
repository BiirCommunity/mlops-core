import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.training.auth import (
    access_token_configured,
    extract_bearer_token,
    require_access_token,
    verify_token_value,
)
from app.training.checkpoints import (
    list_deployable_checkpoints,
    register_job_checkpoint,
)
from app.training.config import TrainingSettings
from app.training.dataset import dataset_summary
from app.training.model_status import build_model_status, sync_model_to_dvc
from app.training.jobs import TrainingJobManager
from app.training.mlflow_registry import MLflowRegistry
from app.training.minio_storage import MinioStorage
from app.training.trainer import LoRATrainConfig

auth_router = APIRouter(prefix="/training/auth", tags=["training-auth"])
router = APIRouter(
    prefix="/training",
    tags=["training"],
    dependencies=[Depends(require_access_token)],
)
_settings = TrainingSettings.from_env()
_jobs: TrainingJobManager | None = None
_storage: MinioStorage | None = None
_registry: MLflowRegistry | None = None


def _job_manager() -> TrainingJobManager:
    global _jobs
    if _jobs is None:
        _jobs = TrainingJobManager(_settings)
    return _jobs


def _minio_storage() -> MinioStorage:
    global _storage
    if _storage is None:
        _storage = MinioStorage(_settings)
    return _storage


def _mlflow_registry() -> MLflowRegistry:
    global _registry
    if _registry is None:
        _registry = MLflowRegistry(_settings, _minio_storage())
    return _registry


class StartTrainingRequest(BaseModel):
    dataset_path: str | None = None
    run_name: str = Field(default="lora-posttrain")
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=2, ge=1, le=128)
    learning_rate: float = Field(default=2e-4, gt=0.0, le=1.0)
    max_seq_len: int = Field(default=512, ge=64, le=4096)
    lora_rank: int = Field(default=8, ge=1, le=128)
    lora_alpha: float = Field(default=16.0, gt=0.0, le=256.0)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=32)
    seed: int = Field(default=42, ge=0)


class DeployModelRequest(BaseModel):
    version: str
    target_path: str | None = None


class RegisterCheckpointRequest(BaseModel):
    job_id: str
    epoch: int = Field(ge=1)


class VerifyTokenRequest(BaseModel):
    token: str | None = None


@auth_router.get("/status")
async def auth_status() -> dict[str, bool]:
    return {"token_required": access_token_configured()}


@auth_router.post("/verify")
async def verify_token(req: VerifyTokenRequest, request: Request) -> dict[str, str]:
    token = req.token or extract_bearer_token(request)
    verify_token_value(token)
    return {"status": "ok"}


@router.get("/health")
async def training_health() -> dict[str, Any]:
    minio_ok, minio_msg = _minio_storage().ping()
    mlflow_ok, mlflow_msg = _mlflow_registry().ping()
    return {
        "status": "ok" if minio_ok and mlflow_ok else "degraded",
        "minio": {"ok": minio_ok, "message": minio_msg},
        "mlflow": {"ok": mlflow_ok, "message": mlflow_msg},
        "mlflow_tracking_uri": _settings.mlflow_tracking_uri,
        "mlflow_ui": os.environ.get("MLFLOW_UI_URL", _settings.mlflow_tracking_uri),
        "minio_console": os.environ.get("MINIO_CONSOLE_URL", ""),
        "admin_ui": os.environ.get("ADMIN_UI_URL", "http://localhost:3000"),
        "token_required": access_token_configured(),
    }


@router.get("/jobs")
async def list_training_jobs(limit: int = 50) -> dict[str, Any]:
    jobs = _job_manager().list_jobs(limit=max(1, min(limit, 100)))
    return {"count": len(jobs), "jobs": jobs}


@router.get("/jobs/{job_id}")
async def get_training_job(job_id: str) -> dict[str, Any]:
    job = _job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="training job not found")
    return job.to_dict()


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str) -> dict[str, Any]:
    try:
        job = _job_manager().cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="training job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.to_dict()


@router.delete("/jobs/{job_id}")
async def delete_training_job(job_id: str) -> dict[str, str]:
    try:
        _job_manager().delete_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="training job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "deleted", "id": job_id}


def _example_dataset_candidates() -> list[Path]:
    return [
        Path(_settings.training_data_dir) / "lora_posttrain_sample.jsonl",
        Path("/app/data/examples/lora_posttrain_sample.jsonl"),
        Path("data/examples/lora_posttrain_sample.jsonl"),
    ]


def _resolve_default_dataset_path() -> Path | None:
    for candidate in _example_dataset_candidates():
        if candidate.exists():
            return candidate
    return None


def _resolve_dataset_path(dataset_path: str | None) -> Path:
    if dataset_path is None or not dataset_path.strip():
        default_path = _resolve_default_dataset_path()
        if default_path is None:
            raise HTTPException(
                status_code=400,
                detail="dataset_path is required (example file not found)",
            )
        return default_path

    value = dataset_path.strip()
    if value.startswith("s3://"):
        storage = _minio_storage()
        try:
            return storage.download_dataset_uri(value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(
                status_code=400,
                detail=f"failed to download dataset from MinIO: {exc}",
            ) from exc

    local_path = Path(value)
    if not local_path.exists():
        raise HTTPException(status_code=400, detail=f"dataset not found: {value}")
    return local_path


@router.get("/datasets")
async def list_datasets() -> dict[str, Any]:
    storage = _minio_storage()
    try:
        datasets = storage.list_dataset_objects()
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=503,
            detail=f"failed to list datasets from MinIO: {exc}",
        ) from exc
    default_path = _resolve_default_dataset_path()
    return {
        "count": len(datasets),
        "datasets": datasets,
        "default_dataset": {
            "label": (
                default_path.name if default_path else "lora_posttrain_sample.jsonl"
            ),
            "local_path": str(default_path) if default_path else None,
        },
    }


@router.post("/jobs")
async def start_training_job(req: StartTrainingRequest) -> dict[str, Any]:
    dataset_path = str(_resolve_dataset_path(req.dataset_path))

    config = LoRATrainConfig(
        dataset_path=dataset_path,
        run_name=req.run_name,
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        max_seq_len=req.max_seq_len,
        lora_rank=req.lora_rank,
        lora_alpha=req.lora_alpha,
        gradient_accumulation_steps=req.gradient_accumulation_steps,
        seed=req.seed,
    )
    job = _job_manager().submit(config)
    return job.to_dict()


@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="only .jsonl files are supported")

    local_dir = Path(_settings.training_data_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / file.filename
    content = await file.read()
    local_path.write_bytes(content)

    object_name = f"datasets/{file.filename}"
    _minio_storage().ensure_buckets()
    minio_uri = _minio_storage().upload_bytes(
        _settings.minio_bucket_datasets,
        object_name,
        content,
        content_type="application/jsonl",
    )
    summary = dataset_summary(local_path)
    return {
        "local_path": str(local_path),
        "minio_uri": minio_uri,
        "summary": summary,
    }


@router.get("/datasets/example")
async def download_example_dataset() -> FileResponse:
    candidate = _resolve_default_dataset_path()
    if candidate is None:
        raise HTTPException(status_code=404, detail="example dataset not found")
    return FileResponse(
        candidate,
        media_type="application/jsonl",
        filename=candidate.name,
    )


@router.get("/models/status")
async def get_model_pipeline_status() -> dict[str, Any]:
    return build_model_status(_settings)


@router.get("/models/active")
async def get_active_inference_model() -> dict[str, Any]:
    return build_model_status(_settings)


@router.post("/models/dvc/sync")
async def sync_dvc_checkpoint() -> dict[str, Any]:
    try:
        return sync_model_to_dvc(_settings)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/models")
async def list_registered_models(limit: int = 20) -> dict[str, Any]:
    versions = _mlflow_registry().list_model_versions(limit=max(1, min(limit, 100)))
    return {"count": len(versions), "versions": versions}


@router.get("/models/checkpoints")
async def list_training_checkpoints() -> dict[str, Any]:
    checkpoints = list_deployable_checkpoints(_job_manager())
    return {"count": len(checkpoints), "checkpoints": checkpoints}


@router.post("/models/checkpoints/register")
async def register_training_checkpoint(
    req: RegisterCheckpointRequest,
) -> dict[str, Any]:
    try:
        return register_job_checkpoint(
            job_manager=_job_manager(),
            registry=_mlflow_registry(),
            settings=_settings,
            job_id=req.job_id,
            epoch=req.epoch,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="training job not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/models/register-base")
async def register_base_model() -> dict[str, Any]:
    registry = _mlflow_registry()
    checkpoint = Path(_settings.base_checkpoint_path)
    if not checkpoint.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"checkpoint not found: {checkpoint}",
        )
    try:
        result = registry.register_base_checkpoint_if_needed(
            checkpoint_path=checkpoint,
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result is None:
        versions = registry.list_model_versions(limit=5)
        return {
            "status": "skipped",
            "reason": "already_registered",
            "versions": versions,
        }
    return {"status": "registered", **result}


@router.post("/models/deploy")
async def deploy_registered_model(req: DeployModelRequest) -> dict[str, Any]:
    try:
        return _job_manager().deploy_model_version(
            version=req.version,
            target_path=req.target_path,
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=str(exc)) from exc
