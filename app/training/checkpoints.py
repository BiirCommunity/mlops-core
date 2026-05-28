from pathlib import Path
from typing import Any

from app.training.config import TrainingSettings
from app.training.jobs import TrainingJob, TrainingJobManager
from app.training.mlflow_registry import MLflowRegistry


def _collect_job_checkpoints(job: TrainingJob) -> list[dict[str, Any]]:
    for source in (job.result, job.progress):
        if not isinstance(source, dict):
            continue
        checkpoints = source.get("checkpoints")
        if isinstance(checkpoints, list) and checkpoints:
            return [
                cp for cp in checkpoints if isinstance(cp, dict) and cp.get("epoch")
            ]
    return []


def list_deployable_checkpoints(
    job_manager: TrainingJobManager,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in job_manager._jobs.values():  # pylint: disable=protected-access
        if job.status not in {"cancelled", "failed", "completed"}:
            continue
        registered = {}
        if isinstance(job.result, dict):
            raw = job.result.get("registered_versions")
            if isinstance(raw, dict):
                registered = {str(k): str(v) for k, v in raw.items()}
        for checkpoint in _collect_job_checkpoints(job):
            epoch = int(checkpoint["epoch"])
            model_path = Path(str(checkpoint.get("model_path", "")))
            adapter_path = Path(str(checkpoint.get("adapter_path", "")))
            rows.append(
                {
                    "job_id": job.id,
                    "epoch": epoch,
                    "run_name": job.config.get("run_name", ""),
                    "job_status": job.status,
                    "model_path": str(model_path),
                    "adapter_path": str(adapter_path),
                    "minio_model_uri": checkpoint.get("minio_model_uri"),
                    "minio_adapter_uri": checkpoint.get("minio_adapter_uri"),
                    "available_locally": model_path.is_file()
                    and adapter_path.is_file(),
                    "registered_version": registered.get(str(epoch)),
                }
            )
    rows.sort(key=lambda row: (row["job_id"], row["epoch"]), reverse=True)
    return rows


def register_job_checkpoint(
    *,
    job_manager: TrainingJobManager,
    registry: MLflowRegistry,
    settings: TrainingSettings,
    job_id: str,
    epoch: int,
) -> dict[str, Any]:
    job = job_manager.get_job(job_id)
    if job is None:
        raise KeyError(job_id)
    if job.status not in {"cancelled", "failed", "completed"}:
        raise ValueError(f"job {job_id} is not finished (status={job.status})")

    checkpoint = next(
        (cp for cp in _collect_job_checkpoints(job) if int(cp["epoch"]) == epoch),
        None,
    )
    if checkpoint is None:
        raise ValueError(f"checkpoint epoch {epoch} not found for job {job_id}")

    registered = {}
    if isinstance(job.result, dict):
        raw = job.result.get("registered_versions")
        if isinstance(raw, dict):
            registered = dict(raw)
    if str(epoch) in registered:
        return {
            "status": "already_registered",
            "model_version": registered[str(epoch)],
            "job_id": job_id,
            "epoch": epoch,
        }

    model_path = Path(str(checkpoint["model_path"]))
    adapter_path = Path(str(checkpoint.get("adapter_path", "")))
    if not model_path.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {model_path}")
    if not adapter_path.is_file():
        raise FileNotFoundError(f"adapter checkpoint not found: {adapter_path}")

    metrics: dict[str, float] = {}
    if isinstance(job.result, dict) and isinstance(
        job.result.get("final_loss"), (int, float)
    ):
        metrics["final_loss"] = float(job.result["final_loss"])

    publish = registry.register_training_checkpoint(
        job_id=job_id,
        epoch=epoch,
        run_name=str(job.config.get("run_name", "lora-posttrain")),
        job_status=job.status,
        model_path=model_path,
        adapter_path=adapter_path,
        params={
            "job_id": job_id,
            "epoch": epoch,
            "run_name": job.config.get("run_name"),
            "dataset_path": job.config.get("dataset_path"),
            "job_status": job.status,
        },
        metrics=metrics,
    )

    if not isinstance(job.result, dict):
        job.result = {}
    job.result.setdefault("registered_versions", {})
    job.result["registered_versions"][str(epoch)] = publish["model_version"]
    job_manager._save_job(job)  # pylint: disable=protected-access
    return {
        "status": "registered",
        "job_id": job_id,
        "epoch": epoch,
        **publish,
    }
