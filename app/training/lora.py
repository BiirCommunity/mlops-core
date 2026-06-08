import math
from typing import Iterable

import torch
from torch import nn

from app.core.transformer import CausalLM, NormalLinear, promote_dtype

DEFAULT_LORA_TARGETS = ("wq", "wk", "wv", "wo", "w1", "w2", "w3")


class LoRALinear(nn.Module):
    """Low-rank adapter around a frozen NormalLinear layer."""

    def __init__(
        self,
        base: NormalLinear,
        *,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        in_features = base.in_features
        out_features = base.out_features
        device = base.weight.device
        self.lora_a = nn.Parameter(torch.zeros(in_features, rank, device=device))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a @ self.lora_b) * self.scaling
        _, lora_out = promote_dtype(base_out, lora_out, dtype=self.base.compute_dtype)
        return base_out + lora_out

    def merged_weight(self) -> torch.Tensor:
        delta = (self.lora_a @ self.lora_b) * self.scaling
        return self.base.weight.data + delta.to(self.base.weight.dtype)

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "lora_a": self.lora_a.detach().cpu(),
            "lora_b": self.lora_b.detach().cpu(),
            "rank": torch.tensor(self.rank),
            "alpha": torch.tensor(self.scaling * self.rank),
            "base_weight": self.base.weight.detach().cpu(),
        }


def _replace_child(parent: nn.Module, name: str, module: nn.Module) -> None:
    setattr(parent, name, module)


def apply_lora_to_model(
    model: CausalLM,
    *,
    rank: int,
    alpha: float,
    target_modules: Iterable[str] | None = None,
) -> list[str]:
    """Inject LoRA into attention and FFN projections. Returns adapted param names."""
    targets = set(target_modules or DEFAULT_LORA_TARGETS)
    adapted: list[str] = []

    block_list = model.model.h.blocks
    for block_idx, block in enumerate(block_list):
        attention = block.seq_modeling_block
        for attr in ("wq", "wk", "wv", "wo"):
            if attr not in targets:
                continue
            base = getattr(attention, attr)
            if not isinstance(base, NormalLinear):
                continue
            lora = LoRALinear(base, rank=rank, alpha=alpha)
            _replace_child(attention, attr, lora)
            adapted.append(f"model.h.blocks.{block_idx}.seq_modeling_block.{attr}")

        for module_name, ffn in (
            ("feed_forward", block.feed_forward),
            ("feed_forward_prime", block.feed_forward_prime),
        ):
            if ffn is None:
                continue
            for attr in ("w1", "w2", "w3"):
                if attr not in targets:
                    continue
                base = getattr(ffn, attr, None)
                if base is None or not isinstance(base, NormalLinear):
                    continue
                lora = LoRALinear(base, rank=rank, alpha=alpha)
                _replace_child(ffn, attr, lora)
                adapted.append(f"model.h.blocks.{block_idx}.{module_name}.{attr}")

    if not adapted:
        raise RuntimeError("LoRA targets not found in model")
    return adapted


def iter_lora_modules(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    found: list[tuple[str, LoRALinear]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            found.append((name, module))
    return found


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for _, module in iter_lora_modules(model):
        params.extend([module.lora_a, module.lora_b])
    return params


def merge_lora_weights(model: CausalLM) -> None:
    """Merge LoRA deltas into base weights and remove adapters in-place."""
    for name, module in list(iter_lora_modules(model)):
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        module.base.weight.data.copy_(module.merged_weight())
        _replace_child(parent, child_name, module.base)


def export_lora_adapter(model: CausalLM) -> dict[str, torch.Tensor | float | int]:
    payload: dict[str, torch.Tensor | float | int] = {}
    modules = iter_lora_modules(model)
    if not modules:
        return payload
    first = modules[0][1]
    payload["rank"] = first.rank
    payload["alpha"] = float(first.scaling * first.rank)
    for name, module in modules:
        for key, value in module.adapter_state_dict().items():
            payload[f"{name}.{key}"] = value
    return payload


def count_lora_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in lora_parameters(model))


def describe_lora_layers(model: nn.Module) -> list[str]:
    return [name for name, _ in iter_lora_modules(model)]
