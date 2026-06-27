import json
from typing import Any, Iterable

from langdetect import LangDetectException, detect
from prometheus_client import Counter, Gauge, Histogram, Summary

from app.core.drift import DriftMonitor, DriftSnapshot
from app.core.drift_report import DriftReportWriter
from app.core.toxicity import ToxicityScorer

REQUEST_LATENCY_SECONDS = Histogram(
    "request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint", "method", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)

PROMPT_LENGTH_CHARS = Summary(
    "prompt_length_chars",
    "Prompt length in characters",
)
RESPONSE_LENGTH_CHARS = Summary(
    "response_length_chars",
    "Response length in characters",
)
RESPONSE_TOXICITY_SCORE = Summary(
    "response_toxicity_score",
    "Model-based toxicity score in [0, 1] (multilingual XLM-R)",
)
RESPONSE_JSON_VALID_TOTAL = Counter(
    "response_json_valid_total",
    "JSON validity of model responses",
    ["status"],
)
PROMPT_LANGUAGE_TOTAL = Counter(
    "prompt_language_total",
    "Detected prompt language",
    ["lang"],
)

CHAT_COMPLETIONS_TOTAL = Counter(
    "chat_completions_total",
    "Chat completion requests",
    ["status"],
)
USER_RATING_TOTAL = Counter(
    "user_rating_total",
    "User ratings submitted for completions",
    ["rating"],
)
USER_RATING_SCORE = Summary(
    "user_rating_score",
    "User rating values from 1 (bad) to 5 (excellent)",
)

HEALTH_UP = Gauge(
    "health_up",
    "1 if the service passes health checks, 0 otherwise",
)
HEALTH_CHECK_TOTAL = Counter(
    "health_check_total",
    "Health endpoint invocations",
    ["status"],
)
REDIS_UP = Gauge(
    "redis_up",
    "1 if Redis is reachable, 0 if configured but unreachable",
)
MODEL_LOADED = Gauge(
    "model_loaded",
    "1 if the model checkpoint is loaded",
)
TOXICITY_MODEL_LOADED = Gauge(
    "toxicity_model_loaded",
    "1 if the toxicity classifier is loaded",
)
DRIFT_MODEL_LOADED = Gauge(
    "drift_model_loaded",
    "1 if the drift embedding model is loaded",
)

DRIFT_DATA_EMBEDDING_DISTANCE = Gauge(
    "drift_data_embedding_distance",
    "Cosine distance between baseline and current prompt embedding centroids",
)
DRIFT_DATA_PROMPT_LENGTH_PSI = Gauge(
    "drift_data_prompt_length_psi",
    "PSI of prompt length buckets (baseline vs current window)",
)
DRIFT_DATA_LANGUAGE_PSI = Gauge(
    "drift_data_language_psi",
    "PSI of prompt language distribution (baseline vs current window)",
)
DRIFT_DATA_SCORE = Gauge(
    "drift_data_score",
    "Composite data drift score in [0, 1]",
)

DRIFT_CONCEPT_RESPONSE_EMBEDDING_DISTANCE = Gauge(
    "drift_concept_response_embedding_distance",
    "Cosine distance between baseline and current response embedding centroids",
)
DRIFT_CONCEPT_TOXICITY_PSI = Gauge(
    "drift_concept_toxicity_psi",
    "PSI of model toxicity tiers on outputs (concept drift proxy)",
)
DRIFT_CONCEPT_RESPONSE_LENGTH_PSI = Gauge(
    "drift_concept_response_length_psi",
    "PSI of response length buckets (concept drift proxy)",
)
DRIFT_CONCEPT_SCORE = Gauge(
    "drift_concept_score",
    "Composite concept drift score in [0, 1]",
)

DRIFT_TARGET_TOXICITY_TIER_PSI = Gauge(
    "drift_target_toxicity_tier_psi",
    "PSI of toxicity tier labels (target drift proxy)",
)
DRIFT_TARGET_JSON_VALID_PSI = Gauge(
    "drift_target_json_valid_psi",
    "PSI of JSON validity labels (target drift proxy)",
)
DRIFT_TARGET_RESPONSE_LANGUAGE_PSI = Gauge(
    "drift_target_response_language_psi",
    "PSI of response language labels (target drift proxy)",
)
DRIFT_TARGET_USER_RATING_PSI = Gauge(
    "drift_target_user_rating_psi",
    "PSI of user ratings 1-5 (baseline vs current window)",
)
DRIFT_TARGET_SCORE = Gauge(
    "drift_target_score",
    "Composite target drift score in [0, 1]",
)

DRIFT_BASELINE_SAMPLES = Gauge(
    "drift_baseline_samples",
    "Number of samples in frozen baseline window",
)
DRIFT_WINDOW_SAMPLES = Gauge(
    "drift_window_samples",
    "Number of samples in rolling current window",
)
DRIFT_BASELINE_LOCKED = Gauge(
    "drift_baseline_locked",
    "1 once baseline window is full and frozen",
)

DVC_STATUSES = (
    "in_sync",
    "out_of_sync",
    "untracked",
    "remote_missing",
    "missing_checkpoint",
    "unknown",
)
PIPELINE_STATUSES = (
    "ready",
    "restart_required",
    "dvc_sync_required",
    "missing_checkpoint",
    "attention",
)

DVC_CHECKPOINT_STATUS = Gauge(
    "dvc_checkpoint_status",
    "DVC checkpoint status (one-hot by status label)",
    ["status"],
)
MODEL_PIPELINE_STATUS = Gauge(
    "model_pipeline_status",
    "Unified model pipeline status (one-hot by status label)",
    ["status"],
)
DVC_IN_SYNC = Gauge(
    "dvc_in_sync",
    "1 if DVC checkpoint is in sync with sidecar and remote",
)
DVC_CHECKPOINT_EXISTS = Gauge(
    "dvc_checkpoint_exists",
    "1 if the model checkpoint file exists on disk",
)
DVC_REMOTE_PRESENT = Gauge(
    "dvc_remote_present",
    "1 if remote DVC object exists, 0 if missing, -1 if unknown",
)
DVC_SIDECAR_PRESENT = Gauge(
    "dvc_sidecar_present",
    "1 if the DVC sidecar file exists",
)
MODEL_PIPELINE_READY = Gauge(
    "model_pipeline_ready",
    "1 if inference is loaded, DVC is in sync, and no reload is pending",
)
DVC_CHECKPOINT_SIZE_BYTES = Gauge(
    "dvc_checkpoint_size_bytes",
    "Size of the model checkpoint on disk in bytes",
)
DVC_SYNC_TOTAL = Counter(
    "dvc_sync",
    "DVC sync operations",
    ["result"],
)

_toxicity_scorer: ToxicityScorer | None = None
_drift_monitor: DriftMonitor | None = None
_drift_report_writer: DriftReportWriter | None = None
_interaction_log = None


def set_toxicity_scorer(scorer: ToxicityScorer | None) -> None:
    global _toxicity_scorer
    _toxicity_scorer = scorer
    TOXICITY_MODEL_LOADED.set(1 if scorer is not None else 0)


def set_drift_monitor(monitor: DriftMonitor | None) -> None:
    global _drift_monitor
    _drift_monitor = monitor
    DRIFT_MODEL_LOADED.set(1 if monitor is not None else 0)
    if monitor is not None:
        publish_drift_snapshot(monitor.snapshot)


def set_drift_report_writer(writer: DriftReportWriter | None) -> None:
    global _drift_report_writer
    _drift_report_writer = writer


def get_drift_report_writer() -> DriftReportWriter | None:
    return _drift_report_writer


def set_interaction_log(log) -> None:
    global _interaction_log
    _interaction_log = log


def get_interaction_log():
    return _interaction_log


def get_drift_snapshot() -> dict[str, float | int | bool] | None:
    if _drift_monitor is None:
        return None
    snapshot = _drift_monitor.snapshot
    return {
        "data_drift_score": snapshot.data_drift_score,
        "concept_drift_score": snapshot.concept_drift_score,
        "target_drift_score": snapshot.target_drift_score,
        "data_embedding_distance": snapshot.data_embedding_distance,
        "concept_response_embedding_distance": (
            snapshot.concept_response_embedding_distance
        ),
        "data_prompt_length_psi": snapshot.data_prompt_length_psi,
        "data_language_psi": snapshot.data_language_psi,
        "baseline_samples": snapshot.baseline_samples,
        "window_samples": snapshot.window_samples,
        "baseline_locked": snapshot.baseline_locked >= 1.0,
    }


def publish_drift_snapshot(snapshot: DriftSnapshot) -> None:
    DRIFT_DATA_EMBEDDING_DISTANCE.set(snapshot.data_embedding_distance)
    DRIFT_DATA_PROMPT_LENGTH_PSI.set(snapshot.data_prompt_length_psi)
    DRIFT_DATA_LANGUAGE_PSI.set(snapshot.data_language_psi)
    DRIFT_DATA_SCORE.set(snapshot.data_drift_score)
    DRIFT_CONCEPT_RESPONSE_EMBEDDING_DISTANCE.set(
        snapshot.concept_response_embedding_distance
    )
    DRIFT_CONCEPT_TOXICITY_PSI.set(snapshot.concept_toxicity_psi)
    DRIFT_CONCEPT_RESPONSE_LENGTH_PSI.set(snapshot.concept_response_length_psi)
    DRIFT_CONCEPT_SCORE.set(snapshot.concept_drift_score)
    DRIFT_TARGET_TOXICITY_TIER_PSI.set(snapshot.target_toxicity_tier_psi)
    DRIFT_TARGET_JSON_VALID_PSI.set(snapshot.target_json_valid_psi)
    DRIFT_TARGET_RESPONSE_LANGUAGE_PSI.set(snapshot.target_response_language_psi)
    DRIFT_TARGET_USER_RATING_PSI.set(snapshot.target_user_rating_psi)
    DRIFT_TARGET_SCORE.set(snapshot.target_drift_score)
    DRIFT_BASELINE_SAMPLES.set(snapshot.baseline_samples)
    DRIFT_WINDOW_SAMPLES.set(snapshot.window_samples)
    DRIFT_BASELINE_LOCKED.set(snapshot.baseline_locked)

    if _drift_monitor is not None and _drift_report_writer is not None:
        _drift_report_writer.maybe_write(_drift_monitor)


def score_turn_toxicity(*, prompt_text: str, response_text: str) -> float:
    if _toxicity_scorer is None:
        return 0.0
    return _toxicity_scorer.score_turn(
        prompt_text=prompt_text,
        response_text=response_text,
    )


def detect_prompt_language(text: str) -> str:
    stripped = text.strip()
    if len(stripped) < 3:
        return "unknown"
    try:
        return detect(stripped)
    except LangDetectException:
        return "unknown"


def is_valid_json_response(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError:
        return False


def detect_response_language(text: str) -> str:
    return detect_prompt_language(text)


def _observe_drift(
    *,
    prompt_text: str,
    response_text: str,
    prompt_lang: str,
    response_lang: str,
    toxicity: float,
    json_valid: bool,
) -> bool:
    if _drift_monitor is None:
        return False
    include_baseline = _drift_monitor.baseline_is_open()
    snapshot = _drift_monitor.observe(
        prompt_text=prompt_text,
        response_text=response_text,
        prompt_lang=prompt_lang,
        response_lang=response_lang,
        toxicity=toxicity,
        json_valid=json_valid,
    )
    publish_drift_snapshot(snapshot)
    return include_baseline


def record_chat_completion_metrics(
    *,
    prompt_text: str,
    response_text: str,
    status: str,
) -> tuple[bool, dict[str, float | str | bool]]:
    prompt_lang = detect_prompt_language(prompt_text)
    response_lang = detect_response_language(response_text)
    json_valid = is_valid_json_response(response_text)
    toxicity = score_turn_toxicity(
        prompt_text=prompt_text,
        response_text=response_text,
    )

    PROMPT_LENGTH_CHARS.observe(len(prompt_text))
    RESPONSE_LENGTH_CHARS.observe(len(response_text))
    RESPONSE_TOXICITY_SCORE.observe(toxicity)
    PROMPT_LANGUAGE_TOTAL.labels(lang=prompt_lang).inc()
    RESPONSE_JSON_VALID_TOTAL.labels(status="valid" if json_valid else "invalid").inc()
    CHAT_COMPLETIONS_TOTAL.labels(status=status).inc()

    include_baseline = _observe_drift(
        prompt_text=prompt_text,
        response_text=response_text,
        prompt_lang=prompt_lang,
        response_lang=response_lang,
        toxicity=toxicity,
        json_valid=json_valid,
    )
    turn_metrics = {
        "prompt_lang": prompt_lang,
        "response_lang": response_lang,
        "json_valid": json_valid,
        "toxicity": toxicity,
        "status": status,
    }
    return include_baseline, turn_metrics


def record_user_rating(*, rating: int, include_baseline: bool) -> None:
    if rating < 1 or rating > 5:
        raise ValueError("rating must be between 1 and 5")

    label = str(rating)
    USER_RATING_TOTAL.labels(rating=label).inc()
    USER_RATING_SCORE.observe(float(rating))

    if _drift_monitor is not None:
        snapshot = _drift_monitor.record_user_rating(
            rating,
            include_baseline=include_baseline,
        )
        publish_drift_snapshot(snapshot)


def record_request_latency(
    *,
    endpoint: str,
    method: str,
    status: int | str,
    duration_sec: float,
) -> None:
    REQUEST_LATENCY_SECONDS.labels(
        endpoint=endpoint,
        method=method,
        status=str(status),
    ).observe(duration_sec)


def set_model_loaded(loaded: bool) -> None:
    MODEL_LOADED.set(1 if loaded else 0)


def _set_one_hot_gauge(gauge: Gauge, labels: tuple[str, ...], active: str) -> None:
    for label in labels:
        gauge.labels(status=label).set(1 if label == active else 0)


def publish_dvc_status(unified: dict[str, Any]) -> None:
    dvc = unified["dvc"]
    pipeline_status = unified["pipeline_status"]
    dvc_status = dvc["status"]

    _set_one_hot_gauge(DVC_CHECKPOINT_STATUS, DVC_STATUSES, dvc_status)
    _set_one_hot_gauge(MODEL_PIPELINE_STATUS, PIPELINE_STATUSES, pipeline_status)

    DVC_IN_SYNC.set(1 if dvc.get("in_sync") else 0)
    DVC_CHECKPOINT_EXISTS.set(1 if dvc.get("checkpoint_exists") else 0)
    DVC_SIDECAR_PRESENT.set(1 if dvc.get("sidecar") is not None else 0)
    MODEL_PIPELINE_READY.set(1 if pipeline_status == "ready" else 0)

    remote_present = dvc.get("remote_present")
    if remote_present is True:
        DVC_REMOTE_PRESENT.set(1)
    elif remote_present is False:
        DVC_REMOTE_PRESENT.set(0)
    else:
        DVC_REMOTE_PRESENT.set(-1)

    disk_size = dvc.get("disk_size")
    DVC_CHECKPOINT_SIZE_BYTES.set(float(disk_size) if disk_size is not None else 0)


def record_dvc_sync(*, success: bool) -> None:
    DVC_SYNC_TOTAL.labels(result="success" if success else "error").inc()


def refresh_dependency_gauges(*, redis_client) -> tuple[bool, dict[str, str]]:
    checks: dict[str, str] = {"model": "ok"}
    set_model_loaded(True)

    if redis_client is None:
        checks["redis"] = "disabled"
        REDIS_UP.set(1)
    else:
        try:
            redis_client.ping()
            checks["redis"] = "ok"
            REDIS_UP.set(1)
        except Exception:  # pylint: disable=broad-except
            checks["redis"] = "error"
            REDIS_UP.set(0)

    healthy = checks["redis"] != "error"
    HEALTH_UP.set(1 if healthy else 0)
    return healthy, checks


def record_health_check(*, healthy: bool) -> None:
    HEALTH_CHECK_TOTAL.labels(status="ok" if healthy else "degraded").inc()


def excluded_latency_paths() -> Iterable[str]:
    return ("/metrics",)
