import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.training.config import TrainingSettings
from app.training.trainer import LoRATrainConfig, LoRATrainer, TrainingCancelledError


@dataclass
class TrainingJob:
    id: str
    status: str
    config: dict[str, Any]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    cancel_requested: bool = False
    progress: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainingJobManager:
    def __init__(self, settings: TrainingSettings | None = None) -> None:
        self.settings = settings or TrainingSettings.from_env()
        self.jobs_dir = Path(self.settings.training_jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.jobs_dir / "jobs.json"
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lora-train"
        )
        self._jobs: dict[str, TrainingJob] = self._load_state()

    def _load_state(self) -> dict[str, TrainingJob]:
        if not self.state_path.exists():
            return {}
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        jobs: dict[str, TrainingJob] = {}
        for item in payload.get("jobs", []):
            item.setdefault("cancel_requested", False)
            if item.get("status") == "running":
                item["status"] = "failed"
                item["error"] = "interrupted by service restart"
                item["finished_at"] = time.time()
            jobs[item["id"]] = TrainingJob(**item)
        return jobs

    def _save_state(self) -> None:
        payload = {"jobs": [job.to_dict() for job in self._jobs.values()]}
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_job(self, job: TrainingJob) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._save_state()

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = sorted(
                self._jobs.values(),
                key=lambda job: job.created_at,
                reverse=True,
            )
            return [job.to_dict() for job in rows[:limit]]

    def get_job(self, job_id: str) -> TrainingJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, config: LoRATrainConfig) -> TrainingJob:
        job = TrainingJob(
            id=f"train-{uuid.uuid4().hex[:12]}",
            status="queued",
            config=asdict(config),
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._save_state()
        self._executor.submit(self._run_job, job.id)
        return job

    def cancel_job(self, job_id: str) -> TrainingJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status not in {"queued", "running"}:
                raise ValueError(f"cannot cancel job in status {job.status}")
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = time.time()
                job.error = "cancelled before start"
            self._save_state()
            return job

    def delete_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status == "running":
                raise ValueError(
                    "cannot delete running job; cancel it or restart the service"
                )
            del self._jobs[job_id]
            self._save_state()

    def _should_stop(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return job is not None and job.cancel_requested

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested:
                job.status = "cancelled"
                job.finished_at = time.time()
                self._save_state()
                return
            job.status = "running"
            job.started_at = time.time()
            self._save_state()

        config_payload = dict(job.config)
        config_payload["job_id"] = job_id
        config = LoRATrainConfig(**config_payload)

        def progress_callback(payload: dict[str, Any]) -> None:
            with self._lock:
                current = self._jobs[job_id]
                current.progress = payload
                self._save_state()

        try:
            result = LoRATrainer(self.settings).train(
                config,
                progress_callback=progress_callback,
                should_stop=lambda: self._should_stop(job_id),
            )
            with self._lock:
                current = self._jobs[job_id]
                current.status = "completed"
                current.finished_at = time.time()
                current.result = asdict(result)
                checkpoints = current.progress.get("checkpoints")
                if isinstance(checkpoints, list):
                    current.result["checkpoints"] = checkpoints
                current.progress = {
                    "final_loss": result.final_loss,
                    "epochs_completed": result.epochs_completed,
                    "checkpoints": current.result.get("checkpoints", []),
                }
                self._save_state()
        except TrainingCancelledError as exc:
            with self._lock:
                current = self._jobs[job_id]
                current.status = "cancelled"
                current.finished_at = time.time()
                current.error = "stopped by user"
                current.result = exc.partial_result
                current.progress = {
                    "epochs_completed": exc.partial_result.get("epochs_completed", 0),
                    "checkpoints": exc.partial_result.get("checkpoints", []),
                    "latest_checkpoint": (
                        exc.partial_result.get("checkpoints", [None])[-1]
                        if exc.partial_result.get("checkpoints")
                        else None
                    ),
                }
                self._save_state()
        except Exception as exc:  # pylint: disable=broad-except
            with self._lock:
                current = self._jobs[job_id]
                current.status = "failed"
                current.finished_at = time.time()
                current.error = str(exc)
                checkpoints = current.progress.get("checkpoints")
                if isinstance(checkpoints, list) and checkpoints:
                    current.result = {
                        "epochs_completed": current.progress.get("epoch", 0),
                        "checkpoints": checkpoints,
                        "latest_checkpoint": checkpoints[-1],
                        "failed": True,
                    }
                self._save_state()

    def deploy_model_version(
        self,
        *,
        version: str,
        target_path: str | None = None,
    ) -> dict[str, str]:
        from app.training.mlflow_registry import MLflowRegistry

        registry = MLflowRegistry(self.settings)
        object_name = f"models/{registry.settings.mlflow_model_name}/{version}/model.pt"
        destination = Path(
            target_path or self.settings.base_checkpoint_path,
        )
        registry.storage.download_file(
            registry.settings.minio_bucket_models,
            object_name,
            destination,
        )
        from app.training.inference_model import (
            get_inference_model_status,
            write_deploy_manifest,
        )

        write_deploy_manifest(
            checkpoint_path=destination,
            registry_name=registry.settings.mlflow_model_name,
            version=version,
            object_name=object_name,
        )
        status = get_inference_model_status(
            checkpoint_path=destination,
            registry_name=registry.settings.mlflow_model_name,
        )
        dvc_sync: dict[str, Any] = {"status": "skipped"}
        try:
            from app.training.dvc_status import sync_checkpoint_to_dvc

            dvc_sync = sync_checkpoint_to_dvc(destination, registry.storage)
            from app.metrics import record_dvc_sync

            record_dvc_sync(success=True)
        except Exception as exc:  # pylint: disable=broad-except
            from app.metrics import record_dvc_sync

            record_dvc_sync(success=False)
            dvc_sync = {"status": "error", "message": str(exc)}

        from app.training.model_status import build_model_status

        return {
            "version": version,
            "local_path": str(destination),
            "object_name": object_name,
            "pending_reload": status["pending_reload"],
            "inference_status": status["status"],
            "dvc_sync": dvc_sync,
            "pipeline": build_model_status(self.settings, quick=False, use_cache=False),
        }
