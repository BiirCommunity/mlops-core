"""MLflow experiment tracking and model registry integration."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from app.training.config import TrainingSettings
from app.training.minio_storage import MinioStorage


class MLflowRegistry:
    def __init__(
        self,
        settings: TrainingSettings | None = None,
        storage: MinioStorage | None = None,
    ) -> None:
        self.settings = settings or TrainingSettings.from_env()
        self.storage = storage or MinioStorage(self.settings)
        self._configure_env()
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        self.client = MlflowClient()

    def _configure_env(self) -> None:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", self.settings.minio_access_key)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", self.settings.minio_secret_key)
        os.environ.setdefault(
            "MLFLOW_S3_ENDPOINT_URL",
            f"{'https' if self.settings.minio_secure else 'http'}://{self.settings.minio_endpoint}",
        )

    def ensure_experiment(self) -> str:
        experiment = mlflow.get_experiment_by_name(self.settings.mlflow_experiment)
        if experiment is None:
            return mlflow.create_experiment(
                self.settings.mlflow_experiment,
                artifact_location=(
                    f"s3://{self.settings.minio_bucket_models}/mlflow-artifacts"
                ),
            )
        return experiment.experiment_id

    def start_run(self, *, run_name: str, tags: dict[str, str] | None = None):
        self.ensure_experiment()
        mlflow.set_experiment(self.settings.mlflow_experiment)
        return mlflow.start_run(run_name=run_name, tags=tags or {})

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params({key: str(value) for key, value in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        mlflow.log_artifact(str(local_path))

    def register_model_version(
        self,
        *,
        run_id: str,
        artifact_path: str,
        description: str,
        tags: dict[str, str] | None = None,
    ) -> ModelVersion:
        source = f"runs:/{run_id}/{artifact_path}"
        registered = mlflow.register_model(
            source,
            self.settings.mlflow_model_name,
        )
        if description:
            self.client.update_model_version(
                name=self.settings.mlflow_model_name,
                version=registered.version,
                description=description,
            )
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    self.settings.mlflow_model_name,
                    registered.version,
                    key,
                    value,
                )
        return registered

    def publish_model_to_minio(
        self,
        *,
        model_path: Path,
        model_name: str,
        version: str,
    ) -> str:
        object_name = f"models/{model_name}/{version}/model.pt"
        self.storage.ensure_buckets()
        return self.storage.upload_file(
            self.settings.minio_bucket_models,
            object_name,
            model_path,
            content_type="application/octet-stream",
        )

    def publish_adapter_to_minio(
        self,
        *,
        adapter_path: Path,
        model_name: str,
        version: str,
    ) -> str:
        object_name = f"models/{model_name}/{version}/lora_adapter.pt"
        self.storage.ensure_buckets()
        return self.storage.upload_file(
            self.settings.minio_bucket_models,
            object_name,
            adapter_path,
            content_type="application/octet-stream",
        )

    def list_model_versions(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            versions = self.client.search_model_versions(
                f"name='{self.settings.mlflow_model_name}'"
            )
        except Exception:  # pylint: disable=broad-except
            return []
        rows: list[dict[str, Any]] = []
        for item in sorted(versions, key=lambda row: int(row.version), reverse=True)[
            :limit
        ]:
            rows.append(
                {
                    "name": item.name,
                    "version": item.version,
                    "stage": item.current_stage,
                    "run_id": item.run_id,
                    "status": item.status,
                    "description": item.description or "",
                    "tags": dict(item.tags or {}),
                }
            )
        return rows

    def list_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            experiments = self.client.search_experiments()
        except Exception:  # pylint: disable=broad-except
            return []
        rows: list[dict[str, Any]] = []
        for item in experiments[:limit]:
            rows.append(
                {
                    "experiment_id": item.experiment_id,
                    "name": item.name,
                    "lifecycle_stage": item.lifecycle_stage,
                    "artifact_location": item.artifact_location,
                }
            )
        return rows

    def list_runs(
        self,
        *,
        experiment_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []
        try:
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=limit,
                order_by=["start_time DESC"],
            )
        except Exception:  # pylint: disable=broad-except
            return []
        rows: list[dict[str, Any]] = []
        for run in runs:
            rows.append(
                {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params),
                    "tags": dict(run.data.tags),
                }
            )
        return rows

    def ping(self) -> tuple[bool, str]:
        try:
            self.ensure_experiment()
            return True, "ok"
        except Exception as exc:  # pylint: disable=broad-except
            return False, str(exc)

    def save_and_log_model_bundle(
        self,
        *,
        run_id: str,
        model_path: Path,
        adapter_path: Path | None,
        params: dict[str, Any],
        final_metrics: dict[str, float],
        description: str = "LoRA post-trained CausalLM checkpoint",
        tags: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="mlflow-artifacts-") as tmp_dir:
            bundle_dir = Path(tmp_dir) / "model_bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            target_model = bundle_dir / "model.pt"
            target_model.write_bytes(model_path.read_bytes())
            if adapter_path is not None and adapter_path.exists():
                (bundle_dir / "lora_adapter.pt").write_bytes(adapter_path.read_bytes())
            metadata = {
                "params": params,
                "metrics": final_metrics,
            }
            (bundle_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifacts(str(bundle_dir), artifact_path="model")
        version = self.register_model_version(
            run_id=run_id,
            artifact_path="model",
            description=description,
            tags=tags or {"training_type": "lora_posttrain"},
        )
        minio_uri = self.publish_model_to_minio(
            model_path=model_path,
            model_name=self.settings.mlflow_model_name,
            version=str(version.version),
        )
        adapter_uri = None
        if adapter_path is not None and adapter_path.exists():
            adapter_uri = self.publish_adapter_to_minio(
                adapter_path=adapter_path,
                model_name=self.settings.mlflow_model_name,
                version=str(version.version),
            )
        return {
            "model_version": str(version.version),
            "registry_name": self.settings.mlflow_model_name,
            "minio_model_uri": minio_uri,
            "minio_adapter_uri": adapter_uri,
        }

    def register_base_checkpoint(
        self,
        *,
        checkpoint_path: Path | str | None = None,
        description: str = "Base CausalLM checkpoint loaded at startup",
    ) -> dict[str, Any]:
        path = Path(checkpoint_path or self.settings.base_checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")

        revision = os.environ.get("MODEL_REVISION") or path.name
        with self.start_run(
            run_name=f"base-{revision}",
            tags={"source": "base_checkpoint"},
        ):
            run_id = mlflow.active_run().info.run_id
            params = {
                "checkpoint_path": str(path),
                "model_revision": revision,
                "source": "base_checkpoint",
            }
            self.log_params(params)
            return self.save_and_log_model_bundle(
                run_id=run_id,
                model_path=path,
                adapter_path=None,
                params=params,
                final_metrics={"bootstrap": 1.0},
            )

    def register_base_checkpoint_if_needed(
        self,
        *,
        checkpoint_path: Path | str | None = None,
    ) -> dict[str, Any] | None:
        if self.list_model_versions(limit=1):
            return None
        return self.register_base_checkpoint(checkpoint_path=checkpoint_path)

    def register_training_checkpoint(
        self,
        *,
        job_id: str,
        epoch: int,
        run_name: str,
        job_status: str,
        model_path: Path,
        adapter_path: Path,
        params: dict[str, Any],
        metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        description = f"LoRA checkpoint from job {job_id}, epoch {epoch} ({job_status})"
        with self.start_run(
            run_name=f"{run_name}-checkpoint-e{epoch}",
            tags={
                "source": "training_checkpoint",
                "job_id": job_id,
                "epoch": str(epoch),
            },
        ) as run:
            run_id = run.info.run_id
            self.log_params(params)
            if metrics:
                self.log_metrics(metrics)
            return self.save_and_log_model_bundle(
                run_id=run_id,
                model_path=model_path,
                adapter_path=adapter_path,
                params=params,
                final_metrics=metrics or {"epoch": float(epoch)},
                description=description,
                tags={
                    "training_type": "lora_posttrain",
                    "source": "training_checkpoint",
                    "job_id": job_id,
                    "epoch": str(epoch),
                    "job_status": job_status,
                },
            )
