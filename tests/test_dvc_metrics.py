from prometheus_client import REGISTRY

from app.metrics import (
    DVC_IN_SYNC,
    MODEL_PIPELINE_READY,
    publish_dvc_status,
    record_dvc_sync,
)


def _sample_value(name: str, labels: dict[str, str] | None = None) -> float:
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name != name:
                continue
            if labels is None:
                if not sample.labels:
                    return sample.value
            elif sample.labels == labels:
                return sample.value
    raise AssertionError(f"metric {name} labels={labels} not found")


def test_publish_dvc_status_in_sync() -> None:
    publish_dvc_status(
        {
            "pipeline_status": "ready",
            "dvc": {
                "status": "in_sync",
                "in_sync": True,
                "checkpoint_exists": True,
                "remote_present": True,
                "sidecar": {"md5": "abc", "size": 100},
                "disk_size": 100,
            },
        }
    )

    assert _sample_value("dvc_in_sync") == 1.0
    assert _sample_value("model_pipeline_ready") == 1.0
    assert _sample_value("dvc_checkpoint_status", {"status": "in_sync"}) == 1.0
    assert _sample_value("dvc_checkpoint_status", {"status": "out_of_sync"}) == 0.0
    assert _sample_value("dvc_remote_present") == 1.0


def test_publish_dvc_status_remote_unknown() -> None:
    publish_dvc_status(
        {
            "pipeline_status": "dvc_sync_required",
            "dvc": {
                "status": "out_of_sync",
                "in_sync": False,
                "checkpoint_exists": True,
                "remote_present": None,
                "sidecar": None,
                "disk_size": 42,
            },
        }
    )

    assert _sample_value("dvc_remote_present") == -1.0
    assert _sample_value("model_pipeline_ready") == 0.0
    assert (
        _sample_value("model_pipeline_status", {"status": "dvc_sync_required"}) == 1.0
    )


def test_record_dvc_sync_increments_counter() -> None:
    record_dvc_sync(success=True)
    value = _sample_value("dvc_sync_total", {"result": "success"})
    record_dvc_sync(success=True)
    assert _sample_value("dvc_sync_total", {"result": "success"}) == value + 1.0
