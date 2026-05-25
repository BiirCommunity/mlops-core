import hashlib
import os
import time
import uuid
from copy import deepcopy
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from transformers import AutoTokenizer

from app.conf.model import get_device
from app.core.architecture import GenerationConfig, build_model, generate
from app.core.session_cache import RedisTTTSessionCache
from app.core.ttt import extract_inner_state_dict, load_inner_state_dict, ttt_adapt

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
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: List[ChatMessage]
    max_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.1
    stream: bool = False
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None


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


def tokenize(text: str) -> torch.Tensor:
    """Как `chat_ttt.tokenize`."""
    return tokenizer.encode(text, return_tensors="pt").squeeze(0)


def prepare_user_prompt_ids(user_text: str) -> torch.Tensor:
    """
    Токены последнего user-сообщения для TTT и генерации.

    Как в chat_ttt на непустом тексте, плюс:
    - обрезка по MAX_CONTEXT_TOKENS;
    - если в последовательности < 2 токенов, дублируем последний (иначе make_batch/TTT
      не определены, а раньше TTT молча пропускался и оставалась «холодная» база).
    """
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
    """Набор стоп-токенов для Llama 3 (eos + eot и т.д.), как ожидает чекпоинт."""
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
    """
    Идентификатор диалога (ветки чата): один стабильный UUID/строка на тред.

    Приоритет: `session_id`, иначе `conversation_id` (удобное имя под UI).
    Без диалога Redis-кэш TTT не используется (одноразовая адаптация на запрос).
    """
    for raw in (req.session_id, req.conversation_id):
        if raw is None:
            continue
        s = str(raw).strip()
        if s:
            return s
    return None


def _cache_segment(raw: str | None, *, max_plain: int, hash_len: int) -> str:
    """длинные/странные строки → sha256-префикс."""
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
    """Последний непустой user после strip"""
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
            detail="Нужно непустое сообщение role=user (как ввод в chat_ttt).",
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
    """
    Как тело цикла в `chat_ttt.main`: без Redis — одноразовый TTT;
    с Redis — при непустом идентификаторе диалога (`session_id` или `conversation_id`);
    ключ кэша — ревизия чекпоинта + диалог (см. `resolve_ttt_cache_key`).
    """
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

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
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


app = FastAPI(title="LM API (логика как e2e/pytorch_model/chat_ttt)")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    return run_inference(req)
