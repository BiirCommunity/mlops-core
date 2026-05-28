"""Training / MLflow / MinIO configuration from environment."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSettings:
    mlflow_tracking_uri: str
    mlflow_experiment: str
    mlflow_model_name: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_bucket_models: str
    minio_bucket_datasets: str
    training_data_dir: str
    training_jobs_dir: str
    base_checkpoint_path: str
    tokenizer_name: str
    device: str
    register_base_model_on_startup: bool

    @classmethod
    def from_env(cls) -> "TrainingSettings":
        return cls(
            mlflow_tracking_uri=os.environ.get(
                "MLFLOW_TRACKING_URI", "http://{{ cookiecutter.mlops_registry }}"
            ),
            mlflow_experiment=os.environ.get(
                "MLFLOW_EXPERIMENT_NAME", "llm-lora-posttrain"
            ),
            mlflow_model_name=os.environ.get("MLFLOW_MODEL_NAME", "llm-lora"),
            minio_endpoint=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
            minio_access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            minio_secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            minio_secure=os.environ.get("MINIO_SECURE", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            minio_bucket_models=os.environ.get("MINIO_BUCKET_MODELS", "{{ cookiecutter.minio_bucket_models }}"),
            minio_bucket_datasets=os.environ.get(
                "MINIO_BUCKET_DATASETS", "{{ cookiecutter.minio_bucket_datasets }}"
            ),
            training_data_dir=os.environ.get("TRAINING_DATA_DIR", "/app/data/training"),
            training_jobs_dir=os.environ.get(
                "TRAINING_JOBS_DIR", "/app/reports/training"
            ),
            base_checkpoint_path=os.environ.get("CHECKPOINT_PATH", "/models/model.pt"),
            tokenizer_name=os.environ.get(
                "TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B"
            ),
            device=os.environ.get("DEVICE", "cuda"),
            register_base_model_on_startup=os.environ.get(
                "REGISTER_BASE_MODEL_ON_STARTUP", "1"
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
        )
