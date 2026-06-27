import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.training import model_status as model_status_module
from app.training.model_status import build_model_status


@pytest.fixture(autouse=True)
def clear_model_status_cache() -> None:
    model_status_module._invalidate_cache()
    yield
    model_status_module._invalidate_cache()


def test_build_model_status_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_unified(**_kwargs):
        calls["count"] += 1
        return {
            "pipeline_status": "ready",
            "dvc": {
                "status": "in_sync",
                "in_sync": True,
                "checkpoint_exists": True,
                "remote_present": True,
                "sidecar": {"md5": "abc", "size": 1},
                "disk_size": 1,
            },
            "inference": {"status": "loaded"},
            "actions": {"restart_app": False, "sync_dvc": False},
        }

    monkeypatch.setattr(
        "app.training.model_status.get_unified_model_status",
        fake_unified,
    )
    monkeypatch.setattr(
        "app.training.model_status.MinioStorage",
        lambda _cfg: MagicMock(),
    )
    monkeypatch.setattr(
        "app.training.model_status.TrainingSettings.from_env",
        lambda: MagicMock(
            base_checkpoint_path=Path("/models/model.pt"),
            mlflow_model_name="test-model",
        ),
    )

    first = build_model_status(quick=True)
    second = build_model_status(quick=True)

    assert first["pipeline_status"] == "ready"
    assert second["pipeline_status"] == "ready"
    assert calls["count"] == 1


def test_build_model_status_cache_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_unified(**_kwargs):
        calls["count"] += 1
        return {
            "pipeline_status": "ready",
            "dvc": {
                "status": "in_sync",
                "in_sync": True,
                "checkpoint_exists": True,
                "remote_present": True,
                "sidecar": {"md5": "abc", "size": 1},
                "disk_size": 1,
            },
            "inference": {"status": "loaded"},
            "actions": {"restart_app": False, "sync_dvc": False},
        }

    monkeypatch.setattr(
        "app.training.model_status.get_unified_model_status",
        fake_unified,
    )
    monkeypatch.setattr(
        "app.training.model_status.MinioStorage",
        lambda _cfg: MagicMock(),
    )
    monkeypatch.setattr(
        "app.training.model_status.TrainingSettings.from_env",
        lambda: MagicMock(
            base_checkpoint_path=Path("/models/model.pt"),
            mlflow_model_name="test-model",
        ),
    )
    monkeypatch.setattr("app.training.model_status.CACHE_TTL_SEC", 0.01)

    build_model_status(quick=True)
    time.sleep(0.02)
    build_model_status(quick=True)

    assert calls["count"] == 2
