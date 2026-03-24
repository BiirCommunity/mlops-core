from __future__ import annotations

import torch

try:
    from .transformer import ModelConfig
except ImportError:  # script-mode fallback
    from transformer import ModelConfig  # type: ignore


MODEL_CFG = ModelConfig(
    vocab_size=128256,
    output_size=128256,
    hidden_size=768,
    intermediate_size=1664,
    num_hidden_layers=12,
    num_attention_heads=12,
    suffix_len=3,
    prime=True,
    qk_norm=True,
    pre_norm=True,
    post_norm=True,
    compute_dtype="float32",
    param_dtype="float32",
    state_dtype="float32",
)


def get_device(device_str: str | None = None) -> torch.device:
    """
    Удобная функция выбора устройства.

    Приоритет:
      1. Если явно передана строка (например, "cuda" или "cpu") — используем её.
      2. Иначе берём "cuda", если она доступна, иначе "cpu".
    """
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
