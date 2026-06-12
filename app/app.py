import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import List, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from app.conf.model import get_device
from app.core.anomalies import detect_anomaly_flags
from app.core.architecture import GenerationConfig, build_model, generate
from app.core.completion_registry import CompletionRecord, CompletionRegistry
from app.core.drift import DriftMonitor
from app.core.drift_report import DriftReportWriter
from app.core.embeddings import PromptEmbedder
from app.core.inference_auth import (
    inference_api_key_configured,
    require_inference_api_key,
)
from app.core.interaction_log import InteractionLog
from app.core.session_cache import RedisTTTSessionCache
from app.core.toxicity import ToxicityScorer
from app.core.ttt import extract_inner_state_dict, load_inner_state_dict, ttt_adapt
from app.api_catalog import API_VERSION, build_api_index
from app.openapi import configure_openapi, register_docs_routes
from app.training.admin_routes import admin_router
from app.training.config import TrainingSettings
from app.training.inference_model import register_inference_startup
from app.training.mlflow_registry import MLflowRegistry
from app.training.routes import auth_router, router as training_router
from app.metrics import (
    excluded_latency_paths,
    get_drift_report_writer,
    record_chat_completion_metrics,
    record_health_check,
    record_request_latency,
    record_user_rating,
    refresh_dependency_gauges,
    set_drift_monitor,
    set_drift_report_writer,
    set_interaction_log,
    get_interaction_log,
    set_model_loaded,
    set_toxicity_scorer,
)

if torch.cuda.is_available():
    _allow_tf32 = os.environ.get("ALLOW_TF32", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    torch.backends.cuda.matmul.allow_tf32 = _allow_tf32
    torch.backends.cudnn.allow_tf32 = _allow_tf32


class ChatMessage(BaseModel):
    role: str = Field(description="Роль: user, assistant или system")
    content: str = Field(description="Текст сообщения")
    name: Optional[str] = Field(default=None, description="Опциональное имя участника")


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "local",
                "messages": [{"role": "user", "content": "Привет!"}],
                "max_tokens": 100,
                "session_id": "dialog-1",
            }
        }
    )

    model: str = Field(default="local", description="Идентификатор модели (локальная)")
    messages: List[ChatMessage] = Field(description="История диалога")
    max_tokens: int = Field(default=200, ge=1, description="Максимум токенов ответа")
    temperature: float = Field(default=0.8, ge=0, description="Температура sampling")
    top_p: float = Field(default=0.9, ge=0, le=1, description="Nucleus sampling top-p")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling (0 = выключено)")
    repetition_penalty: float = Field(default=1.1, ge=1, description="Штраф за повторы")
    stream: bool = Field(
        default=False, description="Потоковый ответ (не поддерживается)"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="ID сессии TTT — состояние адаптации сохраняется в Redis",
    )
    conversation_id: Optional[str] = Field(
        default=None, description="ID диалога для логирования"
    )
    user_rating: Optional[int] = Field(
        default=None, ge=1, le=5, description="Оценка пользователя 1–5"
    )


class FeedbackRequest(BaseModel):
    completion_id: str = Field(description="ID ответа из chat/completions")
    rating: int = Field(ge=1, le=5, description="Оценка 1–5")


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


device = get_device(os.environ.get("DEVICE"))
checkpoint_path = os.environ.get("CHECKPOINT_PATH", "model.pt")
tokenizer_name = os.environ.get("TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B")

print(f"[startup] device: {device}")
print(f"[startup] checkpoint: {checkpoint_path}")
print(f"[startup] tokenizer: {tokenizer_name}")

_hf_token = (os.environ.get("HF_TOKEN") or "").strip() or None
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=_hf_token)
except OSError as e:
    raise RuntimeError(
        "Не удалось загрузить токенизатор с Hugging Face. "
        "Для meta-llama/*: на https://huggingface.co примите условия модели, "
        "задайте HF_TOKEN в окружении. Если токен fine-grained — в настройках токена "
        "включите «Access to public gated repositories». "
        "Либо укажите TOKENIZER_NAME как локальный путь к уже скачанным файлам."
    ) from e
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = build_model(device=device, checkpoint_path=checkpoint_path)
base_model.eval()
for p in base_model.parameters():
    p.requires_grad_(False)

TTT_STEPS = int(os.environ.get("TTT_STEPS", 5))
TTT_LR = float(os.environ.get("TTT_LR", 1e-3))
TTT_SAVE_EACH_STEP = os.environ.get("TTT_SAVE_EACH_STEP", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

SESSION_TTL_SEC = int(os.environ.get("SESSION_TTL_SEC", 3600))
SESSION_LOCK_TTL_SEC = int(os.environ.get("SESSION_LOCK_TTL_SEC", 30))
SESSION_LOCK_BLOCKING_TIMEOUT_SEC = float(
    os.environ.get("SESSION_LOCK_BLOCKING_TIMEOUT_SEC", "10.0")
)
MODEL_REVISION = os.environ.get("MODEL_REVISION") or os.path.basename(checkpoint_path)
register_inference_startup(
    checkpoint_path=checkpoint_path,
    model_revision=MODEL_REVISION,
)

REDIS_URL = os.environ.get("REDIS_URL", "").strip()
SESSION_CACHE: RedisTTTSessionCache | None = None
if REDIS_URL:
    SESSION_CACHE = RedisTTTSessionCache(
        redis_url=REDIS_URL,
        ttl_sec=SESSION_TTL_SEC,
        lock_ttl_sec=SESSION_LOCK_TTL_SEC,
        lock_blocking_timeout_sec=SESSION_LOCK_BLOCKING_TIMEOUT_SEC,
    )
    print(
        f"[startup] Redis session cache enabled: "
        f"redis_url={REDIS_URL}, checkpoint_id={MODEL_REVISION}"
    )
else:
    print("[startup] Redis session cache disabled")

MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "4096"))
COMPLETION_REGISTRY = CompletionRegistry()


def tokenize(text: str) -> torch.Tensor:
    return tokenizer.encode(text, return_tensors="pt").squeeze(0)


def prepare_user_prompt_ids(user_text: str) -> torch.Tensor:
    ids = tokenize(user_text).to(device)
    n = int(ids.shape[0])
    if n == 0:
        raise HTTPException(
            status_code=400,
            detail="Токенизация дала пустую последовательность.",
        )
    if n > MAX_CONTEXT_TOKENS:
        ids = ids[-MAX_CONTEXT_TOKENS:]
        n = int(ids.shape[0])
    if n < 2:
        ids = torch.cat([ids, ids[-1:].clone()], dim=0)
    return ids


def eos_token_ids_for_generation() -> tuple[int, ...]:
    out: list[int] = []
    e = tokenizer.eos_token_id
    if e is not None:
        out.append(int(e))
    eot = getattr(tokenizer, "eot_id", None)
    if isinstance(eot, int) and eot not in out:
        out.append(eot)
    vs = getattr(tokenizer, "vocab_size", None)
    if isinstance(vs, int):
        for extra in (128009, 128008):
            if extra < vs and extra not in out:
                out.append(extra)
    if not out:
        out = [128001]
    seen: set[int] = set()
    uniq: list[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return tuple(uniq)


def resolve_dialog_id(req: ChatCompletionRequest) -> str | None:
    for raw in (req.session_id, req.conversation_id):
        if raw is None:
            continue
        s = str(raw).strip()
        if s:
            return s
    return None


def _cache_segment(raw: str | None, *, max_plain: int, hash_len: int) -> str:
    if not raw or not (s := str(raw).strip()):
        return "_"
    if len(s) > max_plain or any(c in s for c in "|\n\r\x00"):
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        return f"h{hash_len}:{h[:hash_len]}"
    return s[:max_plain]


def resolve_ttt_cache_key(req: ChatCompletionRequest) -> str | None:
    dialog = resolve_dialog_id(req)
    if not dialog:
        return None
    dlg_seg = _cache_segment(dialog, max_plain=160, hash_len=32)
    rev_seg = _cache_segment(MODEL_REVISION, max_plain=120, hash_len=24)
    return f"{rev_seg}|d={dlg_seg}"


def extract_last_user_text_stripped(messages: List[ChatMessage]) -> str | None:
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        text = msg.content.strip()
        if text:
            return text
    return None


def build_prompt_ids(req: ChatCompletionRequest) -> torch.Tensor:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages пусты")
    user_text = extract_last_user_text_stripped(req.messages)
    if not user_text:
        raise HTTPException(
            status_code=400,
            detail="Нужно непустое сообщение role=user.",
        )
    return prepare_user_prompt_ids(user_text)


def build_generation_config(req: ChatCompletionRequest) -> GenerationConfig:
    eos_all = eos_token_ids_for_generation()
    return GenerationConfig(
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        eos_token_id=eos_all[0],
        eos_token_ids=eos_all,
    )


def _build_one_shot_ttt_model(user_text: str):
    if TTT_STEPS <= 0:
        return base_model
    context_ids = prepare_user_prompt_ids(user_text)
    return ttt_adapt(
        base_model,
        context_ids=context_ids,
        device=device,
        n_steps=TTT_STEPS,
        lr=TTT_LR,
        verbose=False,
        clone_model=True,
    )


def build_inference_model_for_request(req: ChatCompletionRequest):
    user_text = extract_last_user_text_stripped(req.messages)
    if not user_text or TTT_STEPS <= 0:
        return base_model

    context_ids = prepare_user_prompt_ids(user_text)
    cache_key = resolve_ttt_cache_key(req)

    if SESSION_CACHE is None or not cache_key:
        return ttt_adapt(
            base_model,
            context_ids=context_ids,
            device=device,
            n_steps=TTT_STEPS,
            lr=TTT_LR,
            verbose=False,
            clone_model=True,
        )

    assert SESSION_CACHE is not None

    def save_step_callback(
        step_idx: int,
        inner_state: dict[str, torch.Tensor],
        loss_value: float,
    ) -> None:
        if not TTT_SAVE_EACH_STEP:
            return
        try:
            SESSION_CACHE.save_inner_state(
                cache_key,
                inner_state,
                checkpoint_id=MODEL_REVISION,
                extra_meta={
                    "mode": "intermediate",
                    "step": step_idx,
                    "loss": loss_value,
                },
            )
        except Exception as exc:  # pylint: disable=broad-except,broad-exception-caught
            print(
                f"[warn] failed to save intermediate inner_state for "
                f"ttt_cache_key={cache_key!r}: {exc}"
            )

    try:
        with SESSION_CACHE.session_lock(
            cache_key,
            timeout_sec=SESSION_LOCK_TTL_SEC,
            blocking_timeout_sec=SESSION_LOCK_BLOCKING_TIMEOUT_SEC,
        ):
            session_model = deepcopy(base_model)

            cached_inner_state = SESSION_CACHE.load_inner_state(
                cache_key,
                checkpoint_id=MODEL_REVISION,
                device=device,
            )
            if cached_inner_state is not None:
                try:
                    load_inner_state_dict(
                        session_model, cached_inner_state, strict=True
                    )
                except (  # pylint: disable=broad-except,broad-exception-caught
                    Exception
                ) as exc:  # pylint: disable=broad-except,broad-exception-caught
                    print(
                        f"[warn] cached inner_state несовместим с моделью, "
                        f"cache reset: {exc}"
                    )
                    SESSION_CACHE.delete_session(cache_key)

            adapted_model = ttt_adapt(
                session_model,
                context_ids=context_ids,
                device=device,
                n_steps=TTT_STEPS,
                lr=TTT_LR,
                verbose=False,
                clone_model=False,
                step_callback=save_step_callback if TTT_SAVE_EACH_STEP else None,
            )
            SESSION_CACHE.save_inner_state(
                cache_key,
                extract_inner_state_dict(adapted_model),
                checkpoint_id=MODEL_REVISION,
                extra_meta={
                    "mode": "final",
                    "prompt_tokens": int(context_ids.shape[0]),
                },
            )
            return adapted_model

    except TimeoutError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"TTT cache busy for this dialog (key suffix …{cache_key[-48:]})",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-except,broad-exception-caught
        print(
            f"[warn] session cache path failed for ttt_cache_key={cache_key!r}, "
            f"fallback to one-shot TTT/base model: {exc}"
        )
        return _build_one_shot_ttt_model(user_text)


def run_inference(req: ChatCompletionRequest) -> ChatCompletionResponse:
    if req.stream:
        raise HTTPException(
            status_code=501,
            detail="stream не поддерживается.",
        )

    user_text = extract_last_user_text_stripped(req.messages)
    if not user_text:
        raise HTTPException(
            status_code=400, detail="Нужно непустое сообщение role=user."
        )

    prompt_ids = prepare_user_prompt_ids(user_text)
    gen_cfg = build_generation_config(req)
    adapted_model = build_inference_model_for_request(req)

    with torch.no_grad():
        new_ids = generate(
            adapted_model,
            prompt_ids=prompt_ids,
            device=device,
            gen_cfg=gen_cfg,
        )

    reply_text = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
    prompt_tokens = int(prompt_ids.shape[0])
    completion_tokens = int(new_ids.shape[0])
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    include_baseline, turn_metrics = record_chat_completion_metrics(
        prompt_text=user_text,
        response_text=reply_text,
        status="success",
    )
    anomaly_flags = detect_anomaly_flags(
        prompt=user_text,
        response=reply_text,
        prompt_lang=str(turn_metrics["prompt_lang"]),
        response_lang=str(turn_metrics["response_lang"]),
        toxicity=float(turn_metrics["toxicity"]),
        json_valid=bool(turn_metrics["json_valid"]),
        status=str(turn_metrics["status"]),
        user_rating=req.user_rating,
    )
    interaction_log = get_interaction_log()
    if interaction_log is not None:
        interaction_log.create(
            completion_id=completion_id,
            prompt=user_text,
            response=reply_text,
            prompt_lang=str(turn_metrics["prompt_lang"]),
            response_lang=str(turn_metrics["response_lang"]),
            toxicity=float(turn_metrics["toxicity"]),
            json_valid=bool(turn_metrics["json_valid"]),
            status="success",
            anomaly_flags=anomaly_flags,
            session_id=req.session_id,
            conversation_id=req.conversation_id,
            user_rating=req.user_rating,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=req.model,
        )
    COMPLETION_REGISTRY.put(
        completion_id,
        CompletionRecord(in_baseline=include_baseline, created_at=time.time()),
    )
    if req.user_rating is not None:
        record_user_rating(rating=req.user_rating, include_baseline=include_baseline)

    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=reply_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _bootstrap_model_registry() -> None:
    settings = TrainingSettings.from_env()
    if not settings.register_base_model_on_startup:
        print("[startup] model registry bootstrap disabled")
        return
    try:
        registry = MLflowRegistry(settings)
        result = registry.register_base_checkpoint_if_needed()
        if result is None:
            print("[startup] model registry bootstrap skipped (already registered)")
            return
        print(
            "[startup] base checkpoint registered in MLflow/MinIO: "
            f"v{result['model_version']} -> {result['minio_model_uri']}"
        )
    except Exception as exc:  # pylint: disable=broad-except,broad-exception-caught
        print(f"[warn] model registry bootstrap failed: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    set_model_loaded(True)
    try:
        toxicity_scorer = ToxicityScorer(hf_token=_hf_token)
        set_toxicity_scorer(toxicity_scorer)
        print(
            f"[startup] toxicity model loaded: {toxicity_scorer.model_name} "
            f"on {toxicity_scorer.device}"
        )
    except Exception as exc:  # pylint: disable=broad-except,broad-exception-caught
        set_toxicity_scorer(None)
        print(f"[warn] toxicity model failed to load, metric will be 0: {exc}")

    try:
        embedder = PromptEmbedder()
        drift_monitor = DriftMonitor(
            embedder,
            redis_client=SESSION_CACHE.client if SESSION_CACHE is not None else None,
        )
        set_drift_monitor(drift_monitor)
        report_writer = DriftReportWriter()
        set_drift_report_writer(report_writer)
        interaction_log = InteractionLog()
        set_interaction_log(interaction_log)
        print(
            f"[startup] drift embedder loaded: {embedder.model_name} "
            f"baseline={drift_monitor.baseline_size} "
            f"window={drift_monitor.window_size} "
            f"reports={report_writer.report_dir} "
            f"interactions={interaction_log.log_dir}"
        )
    except Exception as exc:  # pylint: disable=broad-except,broad-exception-caught
        set_drift_monitor(None)
        set_drift_report_writer(None)
        set_interaction_log(None)
        print(f"[warn] drift monitor failed to load: {exc}")

    healthy, _ = refresh_dependency_gauges(
        redis_client=SESSION_CACHE.client if SESSION_CACHE is not None else None
    )
    record_health_check(healthy=healthy)
    if inference_api_key_configured():
        print("[startup] inference API key auth enabled for /v1/chat/completions")
    else:
        print(
            "[warn] INFERENCE_API_KEY is not set; "
            "/v1/chat/completions and /v1/feedback are disabled"
        )
    threading.Thread(
        target=_bootstrap_model_registry,
        name="registry-bootstrap",
        daemon=True,
    ).start()
    yield


API_DESCRIPTION = (
    "## MLOps Core API\n\n"
    f"Версия публичного API: **{API_VERSION}**.\n\n"
    "### Маршруты через Ingress (`https://adaptive-llm.ru`)\n"
    "- **Inference:** `POST /api/chat/completions`, `POST /api/chat/feedback` "
    "— нужен **inference API key** (`X-API-Key` или `Authorization: Bearer`).\n"
    "- **Training / Admin:** `/api/*` → jobs, datasets, models, drift admin "
    "— нужен **admin token** (`Authorization: Bearer` после `/api/auth/login`).\n"
    "- **Auth:** `/api/auth`, `/api/users`, `/api/api-keys` — сервис auth-service.\n"
    "- **Swagger:** `/api/docs` (этот UI).\n\n"
    "### Прямой доступ к app (NodePort :30800)\n"
    f"- Inference: `/{API_VERSION}/chat/completions`, `/{API_VERSION}/feedback`\n"
    f"- Training: `/{API_VERSION}/training/*`\n"
    "- Probes: `/health`, `/metrics` (без авторизации)\n\n"
    "В Swagger нажмите **Authorize** и укажите ключ или Bearer-токен."
)

OPENAPI_TAGS = [
    {
        "name": "v1-inference",
        "description": (
            "Генерация ответов LLM + TTT-сессии. "
            "Требуется inference API key (X-API-Key или Bearer)."
        ),
    },
    {
        "name": "v1-monitoring",
        "description": (
            "Health, Prometheus metrics, отчёты drift. "
            "/health и /metrics — без авторизации."
        ),
    },
    {
        "name": "v1-training-auth",
        "description": (
            "Legacy-проверка ACCESS_TOKEN. "
            "Предпочтительно: auth-service через /api/auth."
        ),
    },
    {
        "name": "v1-training",
        "description": (
            "LoRA post-train: jobs, datasets, models. " "Требуется admin Bearer-токен."
        ),
    },
    {
        "name": "v1-training-admin",
        "description": (
            "Admin: overview, interactions, drift alerts, MLflow experiments. "
            "Требуется admin Bearer-токен."
        ),
    },
]

app = FastAPI(
    title="MLOps Core API",
    description=API_DESCRIPTION,
    version="0.1.0",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
_admin_ui_origins = [
    origin.strip()
    for origin in os.environ.get(
        "ADMIN_UI_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_admin_ui_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(training_router)
app.include_router(admin_router)
configure_openapi(app)
register_docs_routes(app)


@app.middleware("http")
async def prometheus_request_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    if request.url.path not in excluded_latency_paths():
        record_request_latency(
            endpoint=request.url.path,
            method=request.method,
            status=response.status_code,
            duration_sec=time.perf_counter() - start,
        )
    return response


@app.get("/", tags=["v1-monitoring"], include_in_schema=True, summary="Каталог API")
async def api_root() -> dict[str, object]:
    return build_api_index(via_ingress=False)


@app.get(
    f"/{API_VERSION}",
    tags=["v1-monitoring"],
    include_in_schema=True,
    summary="Каталог API v1",
)
async def api_version_root() -> dict[str, object]:
    return build_api_index(via_ingress=False)


@app.get("/health", tags=["v1-monitoring"], summary="Health check (k8s probe)")
async def health() -> Response:
    healthy, checks = refresh_dependency_gauges(
        redis_client=SESSION_CACHE.client if SESSION_CACHE is not None else None
    )
    record_health_check(healthy=healthy)
    payload = {
        "status": "ok" if healthy else "degraded",
        "checks": checks,
    }
    return Response(
        content=json.dumps(payload),
        media_type="application/json",
        status_code=200 if healthy else 503,
    )


@app.get("/metrics", tags=["v1-monitoring"], summary="Prometheus metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get(
    "/v1/drift/reports",
    tags=["v1-monitoring"],
    summary="Список JSON-отчётов drift",
)
async def list_drift_reports(limit: int = 20) -> dict[str, list[dict[str, str]] | int]:
    writer = get_drift_report_writer()
    if writer is None:
        raise HTTPException(status_code=503, detail="drift reporting disabled")
    bounded_limit = max(1, min(limit, 100))
    reports = writer.list_reports(limit=bounded_limit)
    return {"count": len(reports), "reports": reports}


@app.get(
    "/v1/drift/reports/latest",
    tags=["v1-monitoring"],
    summary="Последний отчёт drift",
)
async def get_latest_drift_report() -> dict:
    writer = get_drift_report_writer()
    if writer is None:
        raise HTTPException(status_code=503, detail="drift reporting disabled")
    report = writer.load_latest()
    if report is None:
        raise HTTPException(status_code=404, detail="drift report ещё не сгенерирован")
    return report


@app.get(
    "/v1/drift/reports/{report_id}",
    tags=["v1-monitoring"],
    summary="Отчёт drift по ID",
)
async def get_drift_report(report_id: str) -> dict:
    writer = get_drift_report_writer()
    if writer is None:
        raise HTTPException(status_code=503, detail="drift reporting disabled")
    report = writer.load(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="drift report not found")
    return report


@app.post(
    "/v1/feedback",
    tags=["v1-inference"],
    dependencies=[Depends(require_inference_api_key)],
    summary="Оценка ответа модели",
    description=(
        "Отправляет user rating для completion_id из chat/completions. "
        "Требуется inference API key."
    ),
)
async def submit_feedback(req: FeedbackRequest) -> dict[str, str | int]:
    record = COMPLETION_REGISTRY.get(req.completion_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="completion_id не найден или срок хранения истёк",
        )
    record_user_rating(rating=req.rating, include_baseline=record.in_baseline)
    interaction_log = get_interaction_log()
    if interaction_log is not None:
        interaction_log.update_rating(req.completion_id, req.rating)
    return {
        "status": "ok",
        "completion_id": req.completion_id,
        "rating": req.rating,
    }


@app.post(
    "/v1/chat/completions",
    tags=["v1-inference"],
    response_model=ChatCompletionResponse,
    dependencies=[Depends(require_inference_api_key)],
    summary="Chat completions (inference + TTT)",
    description=(
        "OpenAI-совместимый endpoint. Поддерживает TTT через session_id. "
        "Через Ingress: POST /api/chat/completions. "
        "Требуется inference API key (X-API-Key или Authorization: Bearer)."
    ),
)
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    try:
        return run_inference(req)
    except HTTPException as exc:
        user_text = extract_last_user_text_stripped(req.messages) or ""
        record_chat_completion_metrics(
            prompt_text=user_text,
            response_text="",
            status=f"error_{exc.status_code}",
        )
        raise
