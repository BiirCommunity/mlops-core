import os
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent))

from app.conf.model import get_device
from app.core.architecture import build_model
from app.core.session_cache import RedisTTTSessionCache
from app.core.transformer import Batch
from app.core.ttt import extract_inner_state_dict, load_inner_state_dict, ttt_adapt

# -------------------------
# TF32
# -------------------------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# -------------------------
# API MODELS
# -------------------------
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: List[ChatMessage]
    max_tokens: int = Field(200, alias="max_tokens")
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.1
    stream: bool = False
    user: Optional[str] = None  # session_id


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


# -------------------------
# SESSION CACHE CONFIG
# -------------------------
device = get_device(os.environ.get("DEVICE"))

checkpoint_path = os.environ.get("CHECKPOINT_PATH", "app/models/finetuned2.pt")
tokenizer_name = os.environ.get("TOKENIZER_NAME", "meta-llama/Meta-Llama-3-8B")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
base_model = build_model(device=device, checkpoint_path=checkpoint_path)

base_model.eval()
for p in base_model.parameters():
    p.requires_grad_(False)

TTT_STEPS = int(os.environ.get("TTT_STEPS", 5))
TTT_LR = float(os.environ.get("TTT_LR", 1e-3))

MODEL_REVISION = os.path.basename(checkpoint_path)

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis-llm:6379").strip()

SESSION_CACHE = (
    RedisTTTSessionCache(
        redis_url=REDIS_URL,
        ttl_sec=3600,
        lock_ttl_sec=30,
        lock_blocking_timeout_sec=10,
    )
    if REDIS_URL
    else None
)


# -------------------------
# HELPERS
# -------------------------
def messages_to_chatml(messages: List[ChatMessage]) -> str:
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def extract_session_id(req: ChatCompletionRequest) -> str | None:
    return (req.user or "").strip() or None


def extract_last_user(messages: List[ChatMessage]) -> str | None:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return None


# -------------------------
# TTT SESSION MODEL BUILDER
# -------------------------
def build_session_model(req: ChatCompletionRequest, prompt_ids: torch.Tensor):
    session_id = extract_session_id(req)

    # NO SESSION
    if SESSION_CACHE is None or not session_id:
        return ttt_adapt(
            base_model,
            context_ids=prompt_ids,
            device=device,
            n_steps=TTT_STEPS,
            lr=TTT_LR,
            clone_model=True,
        )

    # SESSION MODE
    with SESSION_CACHE.session_lock(session_id):
        session_model = deepcopy(base_model)

        cached = SESSION_CACHE.load_inner_state(
            session_id,
            checkpoint_id=MODEL_REVISION,
            device=device,
        )

        if cached:
            load_inner_state_dict(session_model, cached, strict=True)

        adapted = ttt_adapt(
            session_model,
            context_ids=prompt_ids,
            device=device,
            n_steps=TTT_STEPS,
            lr=TTT_LR,
            clone_model=False,
        )

        SESSION_CACHE.save_inner_state(
            session_id,
            extract_inner_state_dict(adapted),
            checkpoint_id=MODEL_REVISION,
        )

        return adapted


# -------------------------
# GENERATION (simplified)
# -------------------------
def _initialize_generation_state(
    model_instance: torch.nn.Module, prompt_ids: torch.Tensor
) -> tuple:
    """Инициализирует состояние для генерации."""
    T = prompt_ids.shape[0]
    position_ids = torch.arange(T, device=device).unsqueeze(0)

    batch = Batch(
        input_ids=prompt_ids.unsqueeze(0),
        target_tokens=torch.zeros(1, T, dtype=torch.long, device=device),
        loss_masks=torch.zeros(1, T, dtype=torch.float32, device=device),
        position_ids=position_ids,
    )

    state = [None] * model_instance.config.num_hidden_layers
    out = model_instance(state=state, seq=batch)

    return out.new_state, out.logits[0, -1], T


def _sample_next_token(logits: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Семплирует следующий токен из логитов."""
    probs = torch.softmax(logits, dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1).squeeze(0)
    return next_tok.item(), next_tok


def _create_single_token_batch(token: torch.Tensor, position: int) -> Batch:
    """Создает батч для одного токена."""
    pos = torch.tensor([[position]], device=device)
    return Batch(
        input_ids=token.unsqueeze(0).unsqueeze(0),
        target_tokens=torch.zeros(1, 1, device=device),
        loss_masks=torch.zeros(1, 1, device=device),
        position_ids=pos,
    )


def generate(
    model_instance: torch.nn.Module, prompt_ids: torch.Tensor, max_new_tokens: int = 200
) -> torch.Tensor:
    """Генерирует новые токены на основе промпта."""
    generated = prompt_ids.clone().to(device)
    state, logits, T = _initialize_generation_state(model_instance, prompt_ids)

    new_tokens = []

    for i in range(max_new_tokens):
        tok_id, next_tok = _sample_next_token(logits)
        new_tokens.append(tok_id)

        generated = torch.cat(
            [generated, torch.tensor([tok_id], device=generated.device)],
            dim=0,
        )

        if tok_id == tokenizer.eos_token_id:
            break

        batch = _create_single_token_batch(next_tok, T + i)
        out = model_instance(state=state, seq=batch)
        state = out.new_state
        logits = out.logits[0, -1]

    return torch.tensor(new_tokens, device=device)


# -------------------------
# INFERENCE
# -------------------------
def run_inference(req: ChatCompletionRequest) -> ChatCompletionResponse:
    prompt_text = messages_to_chatml(req.messages)
    prompt_ids = (
        tokenizer.encode(prompt_text, return_tensors="pt").squeeze(0).to(device)
    )

    session_model = build_session_model(req, prompt_ids)

    with torch.no_grad():
        new_tokens = generate(session_model, prompt_ids, req.max_tokens)

    text = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(new_tokens),
            total_tokens=len(prompt_ids) + len(new_tokens),
        ),
    )


# -------------------------
# FASTAPI
# -------------------------
app = FastAPI(title="TTT + Session Cache LM")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat(req: ChatCompletionRequest):
    return run_inference(req)
