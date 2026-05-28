"""Drift report generation and persistence."""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.drift import PSI_CRITICAL, PSI_WARN, DriftMonitor, DriftSnapshot

SEVERITY_GREEN = "green"
SEVERITY_YELLOW = "yellow"
SEVERITY_RED = "red"


def _severity_from_scores(*scores: float) -> str:
    peak = max(scores, default=0.0)
    if peak >= 0.7:
        return SEVERITY_RED
    if peak >= 0.4:
        return SEVERITY_YELLOW
    return SEVERITY_GREEN


def _window_distributions(window) -> dict[str, dict[str, int]]:
    return {
        "prompt_languages": dict(window.prompt_languages),
        "response_languages": dict(window.response_languages),
        "prompt_lengths": dict(window.prompt_lengths),
        "response_lengths": dict(window.response_lengths),
        "toxicity_tiers": dict(window.toxicity_tiers),
        "json_validity": dict(window.json_validity),
        "user_ratings": dict(window.user_ratings),
    }


def _build_summary(snapshot: DriftSnapshot, severity: str) -> str:
    if snapshot.baseline_locked < 1.0:
        return (
            f"Baseline ещё не зафиксирован "
            f"({snapshot.baseline_samples} samples). Отчёт предварительный."
        )

    parts = [
        f"Severity: {severity}.",
        f"Data drift score={snapshot.data_drift_score:.3f}, "
        f"concept={snapshot.concept_drift_score:.3f}, "
        f"target={snapshot.target_drift_score:.3f}.",
    ]
    if snapshot.target_user_rating_psi > PSI_WARN:
        parts.append(
            f"Сдвиг пользовательских оценок PSI={snapshot.target_user_rating_psi:.3f}."
        )
    if snapshot.data_embedding_distance > 0.1:
        parts.append(
            "Промпты сместились в embedding-пространстве "
            f"(distance={snapshot.data_embedding_distance:.3f})."
        )
    if snapshot.concept_response_embedding_distance > 0.1:
        parts.append(
            "Ответы модели сместились "
            f"(distance={snapshot.concept_response_embedding_distance:.3f})."
        )
    return " ".join(parts)


@dataclass(frozen=True)
class DriftReport:
    report_id: str
    generated_at: str
    status: str
    severity: str
    summary: str
    windows: dict[str, int | float | bool]
    data_drift: dict[str, float]
    concept_drift: dict[str, float]
    target_drift: dict[str, float]
    distributions: dict[str, dict[str, dict[str, int]]]
    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_drift_report(monitor: DriftMonitor) -> DriftReport:
    snapshot = monitor.snapshot
    generated_at = (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    report_id = f"drift-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{time.time_ns()}"

    if snapshot.baseline_locked < 1.0:
        status = "collecting"
    elif snapshot.window_samples == 0:
        status = "empty"
    else:
        status = "ready"

    severity = _severity_from_scores(
        snapshot.data_drift_score,
        snapshot.concept_drift_score,
        snapshot.target_drift_score,
    )
    has_ratings = (
        sum(monitor.baseline.user_ratings.values()) > 0
        and sum(monitor.current.user_ratings.values()) > 0
    )

    return DriftReport(
        report_id=report_id,
        generated_at=generated_at,
        status=status,
        severity=severity,
        summary=_build_summary(snapshot, severity),
        windows={
            "baseline_samples": snapshot.baseline_samples,
            "window_samples": snapshot.window_samples,
            "baseline_locked": snapshot.baseline_locked >= 1.0,
            "baseline_size": monitor.baseline_size,
            "window_size": monitor.window_size,
        },
        data_drift={
            "score": snapshot.data_drift_score,
            "embedding_distance": snapshot.data_embedding_distance,
            "prompt_length_psi": snapshot.data_prompt_length_psi,
            "language_psi": snapshot.data_language_psi,
        },
        concept_drift={
            "score": snapshot.concept_drift_score,
            "response_embedding_distance": snapshot.concept_response_embedding_distance,
            "toxicity_psi": snapshot.concept_toxicity_psi,
            "response_length_psi": snapshot.concept_response_length_psi,
        },
        target_drift={
            "score": snapshot.target_drift_score,
            "user_rating_psi": snapshot.target_user_rating_psi,
            "uses_user_ratings": has_ratings,
            "toxicity_tier_psi": snapshot.target_toxicity_tier_psi,
            "json_valid_psi": snapshot.target_json_valid_psi,
            "response_language_psi": snapshot.target_response_language_psi,
        },
        distributions={
            "baseline": _window_distributions(monitor.baseline),
            "current": _window_distributions(monitor.current),
        },
        thresholds={"psi_warn": PSI_WARN, "psi_critical": PSI_CRITICAL},
    )


def render_markdown(report: DriftReport) -> str:
    lines = [
        f"# Drift report `{report.report_id}`",
        "",
        f"- **Generated at:** {report.generated_at}",
        f"- **Status:** {report.status}",
        f"- **Severity:** {report.severity}",
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "## Scores",
        "",
        "| Type | Score |",
        "|------|-------|",
        f"| Data | {report.data_drift['score']:.4f} |",
        f"| Concept | {report.concept_drift['score']:.4f} |",
        f"| Target | {report.target_drift['score']:.4f} |",
        "",
        "## Windows",
        "",
        f"- Baseline samples: {report.windows['baseline_samples']}",
        f"- Current window samples: {report.windows['window_samples']}",
        f"- Baseline locked: {report.windows['baseline_locked']}",
        "",
    ]
    if report.distributions["current"].get("user_ratings"):
        lines.extend(
            [
                "## User ratings (current window)",
                "",
                str(report.distributions["current"]["user_ratings"]),
                "",
            ]
        )
    return "\n".join(lines)


class DriftReportWriter:
    def __init__(
        self,
        report_dir: str | None = None,
        *,
        interval_sec: float | None = None,
        min_window_samples: int | None = None,
        keep_reports: int | None = None,
    ) -> None:
        self.report_dir = Path(
            report_dir or os.environ.get("DRIFT_REPORT_DIR", "reports/drift")
        )
        self.interval_sec = interval_sec or float(
            os.environ.get("DRIFT_REPORT_INTERVAL_SEC", "300")
        )
        self.min_window_samples = min_window_samples or int(
            os.environ.get("DRIFT_REPORT_MIN_WINDOW_SAMPLES", "10")
        )
        self.keep_reports = keep_reports or int(
            os.environ.get("DRIFT_REPORT_KEEP", "100")
        )
        self._last_written_at = 0.0
        self._last_severity = SEVERITY_GREEN
        self._generations_since_report = 0
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def maybe_write(self, monitor: DriftMonitor) -> DriftReport | None:
        self._generations_since_report += 1
        snapshot = monitor.snapshot

        if snapshot.baseline_locked < 1.0:
            return None
        if snapshot.window_samples < self.min_window_samples:
            return None

        if not (self.report_dir / "latest.json").is_file():
            report = self.write(monitor)
            self._last_written_at = time.time()
            self._last_severity = _severity_from_scores(
                snapshot.data_drift_score,
                snapshot.concept_drift_score,
                snapshot.target_drift_score,
            )
            self._generations_since_report = 0
            return report

        severity = _severity_from_scores(
            snapshot.data_drift_score,
            snapshot.concept_drift_score,
            snapshot.target_drift_score,
        )
        now = time.time()
        elapsed = now - self._last_written_at
        severity_escalated = (
            severity == SEVERITY_RED and self._last_severity != SEVERITY_RED
        )
        interval_passed = elapsed >= self.interval_sec

        if not severity_escalated and not interval_passed:
            return None

        report = self.write(monitor)
        self._last_written_at = now
        self._last_severity = severity
        self._generations_since_report = 0
        return report

    def write(self, monitor: DriftMonitor) -> DriftReport:
        report = build_drift_report(monitor)
        json_path = self.report_dir / f"{report.report_id}.json"
        md_path = self.report_dir / f"{report.report_id}.md"
        latest_path = self.report_dir / "latest.json"

        payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        json_path.write_text(payload, encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        latest_path.write_text(payload, encoding="utf-8")
        self._rotate_old_reports()
        return report

    def load_latest(self) -> dict[str, Any] | None:
        latest_path = self.report_dir / "latest.json"
        if not latest_path.is_file():
            return None
        return json.loads(latest_path.read_text(encoding="utf-8"))

    def list_reports(self, *, limit: int = 20) -> list[dict[str, str]]:
        files = sorted(
            self.report_dir.glob("drift-*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        items: list[dict[str, str]] = []
        for path in files[:limit]:
            items.append(
                {
                    "report_id": path.stem,
                    "path": str(path),
                    "generated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
            )
        return items

    def load(self, report_id: str) -> dict[str, Any] | None:
        path = self.report_dir / f"{report_id}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _rotate_old_reports(self) -> None:
        files = sorted(
            self.report_dir.glob("drift-*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in files[self.keep_reports :]:
            path.unlink(missing_ok=True)
            md_path = path.with_suffix(".md")
            md_path.unlink(missing_ok=True)
