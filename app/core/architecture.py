from dataclasses import dataclass

import torch
import torch.nn.functional as F

from app.conf.model import MODEL_CFG
from app.core.transformer import Batch, CausalLM, cross_entropy_loss_and_accuracy


@dataclass
class GenerationConfig:
    """
    Параметры, управляющие стратегией генерации текста.

    Важно:
    - max_new_tokens: сколько новых токенов добавляем поверх промпта;
    - temperature: сглаживание распределения (1.0 — без изменений);
    - top_p / top_k: nucleus / top-k отбор кандидатов;
    - repetition_penalty: штраф за повторяющиеся токены;
    - eos_token_id: специальный токен завершения (если встречен — стоп).
    """

    max_new_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.1
    eos_token_id: int = 128001


def build_model(device: torch.device, checkpoint_path: str | None = None) -> CausalLM:
    """
    Создать экземпляр `CausalLM` с конфигом `MODEL_CFG` и,
    при необходимости, загрузить веса из PyTorch‑чекпоинта.

    Ожидаемый формат чекпоинта:
        {"model_weights": <state_dict>}
    совместим с исходным `125m_pytorch.pt`.
    """
    model = CausalLM(MODEL_CFG)

    if checkpoint_path is not None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        weights = payload["model_weights"]

        if "lm_head.weight" not in weights and "model.wte.weight" in weights:
            weights["lm_head.weight"] = (
                weights["model.wte.weight"].clone().t().contiguous()
            )

        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing:
            print(f"[warn] Missing keys: {missing}")
        if unexpected:
            print(f"[warn] Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


def make_batch(input_ids: torch.Tensor, device: torch.device) -> Batch:
    """
    Построить `Batch` для задачи language modeling из ОДНОЙ последовательности токенов.

    Вход:
        input_ids: [T] — последовательность токенов (например, токенизированный текст).

    Выход:
        Batch c:
        - input_ids: [1, T-1]  — все токены, кроме последнего;
        - target_tokens: [1, T-1] — все токены, кроме первого (сдвиг на 1);
        - loss_masks: [1, T-1] — единицы (считать loss на каждом токене).
    """
    if input_ids.ndim != 1:
        raise ValueError(
            f"input_ids должен быть 1D, получили shape={tuple(input_ids.shape)}"
        )

    T = int(input_ids.shape[0])
    if T < 2:
        raise ValueError(
            "Нужно минимум 2 токена: иначе нельзя построить (src, tgt) со сдвигом на 1."
        )

    src = input_ids[:-1]
    tgt = input_ids[1:]
    mask = torch.ones(T - 1, dtype=torch.float32, device=device)

    return Batch(
        input_ids=src.unsqueeze(0).to(device),
        target_tokens=tgt.unsqueeze(0).to(device),
        loss_masks=mask.unsqueeze(0).to(device),
    )


def _apply_repetition_penalty(
    logits: torch.Tensor, generated_tokens: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Применить штраф за повторение токенов."""
    if penalty == 1.0:
        return logits

    for tok in generated_tokens:
        val = logits[tok]
        if val > 0:
            logits[tok] = val / penalty
        else:
            logits[tok] = val * penalty
    return logits


def _apply_top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Оставить только top-k токенов."""
    if k <= 0:
        return logits

    top_k_vals, _ = torch.topk(logits, k)
    logits[logits < top_k_vals[-1]] = float("-inf")
    return logits


def _apply_top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Применить nucleus sampling (top-p)."""
    if p >= 1.0:
        return logits

    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs_sorted = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs_sorted, dim=-1)
    remove = cum_probs - probs_sorted > p
    sorted_logits[remove] = float("-inf")

    return torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)


def _get_next_token_logits(
    model: CausalLM, generated: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Получить логиты для следующего токена."""
    T = generated.shape[0]
    batch = Batch(
        input_ids=generated.unsqueeze(0),
        target_tokens=torch.zeros(1, T, dtype=torch.long, device=device),
        loss_masks=torch.zeros(1, T, dtype=torch.float32, device=device),
    )

    state = [None] * MODEL_CFG.num_hidden_layers
    out = model(state=state, seq=batch)
    return out.logits[0, -1, :]


@torch.no_grad()
def generate(
    model: CausalLM,
    prompt_ids: torch.Tensor,
    device: torch.device,
    gen_cfg: GenerationConfig | None = None,
) -> torch.Tensor:
    """
    Авторегрессивно сгенерировать продолжение последовательности токенов.
    """
    if gen_cfg is None:
        gen_cfg = GenerationConfig()

    model.eval()
    generated = prompt_ids.clone().to(device)

    for _ in range(gen_cfg.max_new_tokens):
        # Получаем логиты для следующего токена
        logits = _get_next_token_logits(model, generated, device)

        # Применяем фильтры и penalties
        logits = _apply_repetition_penalty(
            logits, generated, gen_cfg.repetition_penalty
        )

        if gen_cfg.temperature != 1.0:
            logits = logits / gen_cfg.temperature

        logits = _apply_top_k_filter(logits, gen_cfg.top_k)
        logits = _apply_top_p_filter(logits, gen_cfg.top_p)

        # Семплируем следующий токен
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_tok], dim=0)

        if next_tok.item() == gen_cfg.eos_token_id:
            break

    return generated[prompt_ids.shape[0] :]


def lm_loss(
    model: CausalLM,
    input_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Удобная обёртка: посчитать loss для одной токенизированной последовательности.
    """
    batch = make_batch(input_ids.to(device), device)
    state = [None] * MODEL_CFG.num_hidden_layers
    out = model(state=state, seq=batch)
    loss, _ = cross_entropy_loss_and_accuracy(
        out.logits, batch.target_tokens, batch.loss_masks
    )
    return loss
