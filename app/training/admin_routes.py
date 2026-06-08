from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.core.interaction_log import InteractionLog
from app.metrics import get_drift_report_writer, get_drift_snapshot, get_interaction_log
from app.training.auth import require_access_token
from app.training.config import TrainingSettings
from app.training.jobs import TrainingJobManager
from app.training.mlflow_registry import MLflowRegistry
from app.training.model_status import build_model_status
from app.training.minio_storage import MinioStorage

admin_router = APIRouter(
    prefix="/training/admin",
    tags=["training-admin"],
    dependencies=[Depends(require_access_token)],
)
_settings = TrainingSettings.from_env()
_jobs: TrainingJobManager | None = None
_registry: MLflowRegistry | None = None


def _job_manager() -> TrainingJobManager:
    global _jobs
    if _jobs is None:
        _jobs = TrainingJobManager(_settings)
    return _jobs


def _mlflow_registry() -> MLflowRegistry:
    global _registry
    if _registry is None:
        _registry = MLflowRegistry(_settings, MinioStorage(_settings))
    return _registry


def _interaction_log() -> InteractionLog:
    log = get_interaction_log()
    if log is None:
        raise HTTPException(status_code=503, detail="interaction log disabled")
    return log


@admin_router.get("/overview")
async def admin_overview() -> dict[str, Any]:
    log = _interaction_log()
    snapshot = get_drift_snapshot()
    writer = get_drift_report_writer()
    latest_report = writer.load_latest() if writer is not None else None
    jobs = _job_manager().list_jobs(limit=5)
    return {
        "interactions": log.stats(),
        "drift": snapshot,
        "latest_drift_report": latest_report,
        "recent_jobs": jobs,
        "inference_model": build_model_status(_settings),
    }


@admin_router.get("/interactions")
async def list_interactions(
    limit: int = 100,
    anomalies_only: bool = False,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    log = _interaction_log()
    rows = log.list_records(
        limit=max(1, min(limit, 500)),
        anomalies_only=anomalies_only,
        conversation_id=conversation_id,
    )
    return {
        "count": len(rows),
        "stats": log.stats(),
        "interactions": [row.to_dict() for row in rows],
    }


@admin_router.get("/interactions/{record_id}")
async def get_interaction(record_id: str) -> dict[str, Any]:
    log = _interaction_log()
    record = log.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="interaction not found")
    return record.to_dict()


@admin_router.get("/drift/alerts")
async def drift_alerts(limit: int = 20) -> dict[str, Any]:
    snapshot = get_drift_snapshot()
    writer = get_drift_report_writer()
    reports = writer.list_reports(limit=max(1, min(limit, 100))) if writer else []
    alerts: list[dict[str, Any]] = []

    if snapshot:
        for kind, score in (
            ("data", snapshot.get("data_drift_score", 0.0)),
            ("concept", snapshot.get("concept_drift_score", 0.0)),
            ("target", snapshot.get("target_drift_score", 0.0)),
        ):
            severity = _score_severity(float(score))
            if severity != "green":
                alerts.append(
                    {
                        "source": "live",
                        "kind": kind,
                        "severity": severity,
                        "score": float(score),
                        "message": f"Live {kind} drift score is {float(score):.3f}",
                    }
                )

    if writer is not None:
        for item in reports:
            report = writer.load(item["report_id"])
            if report is None:
                continue
            severity = report.get("severity", "green")
            if severity in {"yellow", "red"}:
                alerts.append(
                    {
                        "source": "report",
                        "report_id": report["report_id"],
                        "severity": severity,
                        "generated_at": report.get("generated_at"),
                        "summary": report.get("summary"),
                        "scores": {
                            "data": report.get("data_drift", {}).get("score"),
                            "concept": report.get("concept_drift", {}).get("score"),
                            "target": report.get("target_drift", {}).get("score"),
                        },
                    }
                )

    return {
        "count": len(alerts),
        "live_snapshot": snapshot,
        "alerts": alerts,
    }


@admin_router.get("/drift/reports")
async def list_drift_reports(limit: int = 50) -> dict[str, Any]:
    writer = get_drift_report_writer()
    if writer is None:
        raise HTTPException(status_code=503, detail="drift reporting disabled")
    rows: list[dict[str, Any]] = []
    for item in writer.list_reports(limit=max(1, min(limit, 200))):
        report = writer.load(item["report_id"])
        if report is None:
            continue
        rows.append(report)
    return {"count": len(rows), "reports": rows}


@admin_router.get("/drift/reports/latest")
async def get_latest_drift_report_admin() -> dict[str, Any]:
    writer = get_drift_report_writer()
    if writer is None:
        raise HTTPException(status_code=503, detail="drift reporting disabled")
    report = writer.load_latest()
    if report is None:
        raise HTTPException(status_code=404, detail="drift report ещё не сгенерирован")
    return report


@admin_router.get("/drift/reports/{report_id}")
async def get_drift_report_admin(report_id: str) -> dict[str, Any]:
    writer = get_drift_report_writer()
    if writer is None:
        raise HTTPException(status_code=503, detail="drift reporting disabled")
    report = writer.load(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="drift report not found")
    return report


@admin_router.get("/experiments")
async def list_experiments(limit: int = 50) -> dict[str, Any]:
    rows = _mlflow_registry().list_experiments(limit=max(1, min(limit, 200)))
    return {"count": len(rows), "experiments": rows}


@admin_router.get("/experiments/runs")
async def list_experiment_runs(
    experiment_name: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    name = experiment_name or _settings.mlflow_experiment
    rows = _mlflow_registry().list_runs(
        experiment_name=name,
        limit=max(1, min(limit, 200)),
    )
    return {"count": len(rows), "experiment_name": name, "runs": rows}


def _score_severity(score: float) -> str:
    if score >= 0.7:
        return "red"
    if score >= 0.4:
        return "yellow"
    return "green"
