import json
from pathlib import Path

import pytest

from app.training.checkpoints import (
    list_deployable_checkpoints,
    register_job_checkpoint,
)
from app.training.jobs import TrainingJob, TrainingJobManager


@pytest.fixture
def job_manager(tmp_path: Path, monkeypatch) -> TrainingJobManager:
    monkeypatch.setenv("TRAINING_JOBS_DIR", str(tmp_path))
    manager = TrainingJobManager()
    job = TrainingJob(
        id="train-demo",
        status="cancelled",
        config={"run_name": "lora-posttrain", "dataset_path": "/tmp/x.jsonl"},
        created_at=1.0,
        result={
            "epochs_completed": 2,
            "checkpoints": [
                {
                    "epoch": 1,
                    "model_path": str(tmp_path / "epoch_1" / "model.pt"),
                    "adapter_path": str(tmp_path / "epoch_1" / "lora_adapter.pt"),
                },
                {
                    "epoch": 2,
                    "model_path": str(tmp_path / "epoch_2" / "model.pt"),
                    "adapter_path": str(tmp_path / "epoch_2" / "lora_adapter.pt"),
                },
            ],
        },
    )
    manager._jobs[job.id] = job  # pylint: disable=protected-access
    for epoch in (1, 2):
        epoch_dir = tmp_path / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        (epoch_dir / "model.pt").write_bytes(b"model")
        (epoch_dir / "lora_adapter.pt").write_bytes(b"adapter")
    return manager


def test_list_deployable_checkpoints(job_manager: TrainingJobManager) -> None:
    rows = list_deployable_checkpoints(job_manager)
    assert len(rows) == 2
    assert rows[0]["epoch"] == 2
    assert rows[0]["available_locally"] is True
    assert rows[0]["registered_version"] is None


def test_register_job_checkpoint(
    job_manager: TrainingJobManager, tmp_path: Path
) -> None:
    class _FakeRegistry:
        def register_training_checkpoint(self, **kwargs):
            return {
                "model_version": "7",
                "registry_name": "llm-lora",
                "minio_model_uri": "s3://{{ cookiecutter.minio_bucket_models }}/models/llm-lora/7/model.pt",
                "minio_adapter_uri": "s3://{{ cookiecutter.minio_bucket_models }}/models/llm-lora/7/lora_adapter.pt",
            }

    result = register_job_checkpoint(
        job_manager=job_manager,
        registry=_FakeRegistry(),
        settings=job_manager.settings,
        job_id="train-demo",
        epoch=2,
    )
    assert result["status"] == "registered"
    assert result["model_version"] == "7"
    saved = json.loads((tmp_path / "jobs.json").read_text(encoding="utf-8"))
    job = saved["jobs"][0]
    assert job["result"]["registered_versions"]["2"] == "7"
