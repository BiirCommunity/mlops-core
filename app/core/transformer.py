from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F


# Configuration
class SeqModelingBlockType(StrEnum):
    """Type of sequence modeling block."""

    self_attention = "self_attention"


# pylint: disable=too-many-instance-attributes
@dataclass(unsafe_hash=True, eq=True)
class ModelConfig:
    """
    Minimal model configuration compatible with original architecture.
    """

    # Core
    name: str = "unnamed"
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    output_size: int = 32000

    # Sequence
    mini_batch_size: int = 1024
    sliding_window_size: int = 1024
    seq_len: int = 131072

    # Normalization
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Dropout
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0

    # Architecture
    tie_word_embeddings: bool = False
    seq_modeling_block: str = "self_attention"
    rope_theta: float = 10000.0

    # Dtypes
    compute_dtype: str = "bf16"
    param_dtype: str = "fp32"
    state_dtype: str = "fp32"

    # TTT
    suffix_len: int = 0
    prime: bool = False
    qk_norm: bool = True
    pre_norm: bool = True
    post_norm: bool = True
    feed_forward_prime: str = "swiglu"

    # Derived
    seq_modeling_block_type: SeqModelingBlockType = field(
        default=SeqModelingBlockType.self_attention,
        init=False,
    )

    def __post_init__(self) -> None:
        self.seq_modeling_block_type = SeqModelingBlockType(self.seq_modeling_block)


# Batch and helpers
@dataclass
class Batch:
    """Minimal batch container."""

    input_ids: torch.Tensor
    target_tokens: torch.Tensor
    loss_masks: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    index: int | slice | None = None

    @property
    def shape(self):
        return self.input_ids.shape

    def slice_index(self, index: int | slice) -> "Batch":
        return Batch(
            input_ids=self.input_ids[index],
            target_tokens=self.target_tokens[index],
            loss_masks=self.loss_masks[index],
            attention_mask=(
                self.attention_mask[index] if self.attention_mask is not None else None
            ),
            position_ids=(
                self.position_ids[index] if self.position_ids is not None else None
            ),
            index=index,
        )


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "float64": torch.float64,
}


def get_torch_dtype(name: str) -> torch.dtype:
    """Convert string dtype name to torch.dtype."""
    return _DTYPE_MAP[name]


def promote_dtype(*tensors: torch.Tensor, dtype: torch.dtype) -> list[torch.Tensor]:
    """Cast all tensors to the given dtype."""
    return [t.to(dtype) for t in tensors]


# RoPE utilities
def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Precompute RoPE complex frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype)[: dim // 2] / dim))
    t = torch.arange(end, dtype=dtype)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings."""
    B, T, H, _ = x.shape
    x = x.reshape(B, T, H, -1, 2)
    x_c = torch.view_as_complex(x)
    if freqs_cis.ndim == 3:
        freqs_cis = freqs_cis.unsqueeze(2)
    return torch.view_as_real(x_c * freqs_cis).reshape(B, T, H, -1)


# Linear layer with normal init
class NormalLinear(nn.Module):
    """Linear layer with normal initialization and no bias."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        config: ModelConfig,
        in_features: int,
        out_features: int,
        *,
        name: str = "",
        std: float,
    ) -> None:
        super().__init__()
        self.compute_dtype = get_torch_dtype(config.compute_dtype)
        self.param_dtype = get_torch_dtype(config.param_dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.layer_name = name

        weight = torch.empty(in_features, out_features, dtype=self.param_dtype)
        nn.init.normal_(weight, std=std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, weight = promote_dtype(x, self.weight, dtype=self.compute_dtype)
        return x @ weight


# Attention
# pylint: disable=too-many-instance-attributes
class AttentionBase(nn.Module):
    """Base class for attention variants."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = get_torch_dtype(config.compute_dtype)
        self.param_dtype = get_torch_dtype(config.param_dtype)

        embed_dim = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = embed_dim // self.num_heads

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self._init_linear_layers(config, embed_dim)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

        self.register_buffer(
            "_freqs_cis",
            precompute_freqs_cis(
                self.head_dim, 2 * config.seq_len, theta=config.rope_theta
            ),
            persistent=False,
        )

    def _init_linear_layers(self, config: ModelConfig, embed_dim: int) -> None:
        """Initialize Q, K, V, O projections."""
        for attr in ("wq", "wk", "wv", "wo"):
            setattr(
                self,
                attr,
                NormalLinear(
                    config,
                    in_features=embed_dim,
                    out_features=embed_dim,
                    std=config.initializer_range,
                    name=attr,
                ),
            )

    @property
    def freqs_cis(self) -> torch.Tensor:
        return self._freqs_cis

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, D = x.shape
        return x.reshape(B, T, H * D)

    def project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

    # pylint: disable=too-many-arguments
    def apply_rope(
        self, xis: tuple[torch.Tensor, ...], position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        freqs = self.freqs_cis[position_ids].unsqueeze(2)
        return tuple(apply_rotary_emb(x, freqs) for x in xis)

    def get_attention_input(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xq, xk, xv = self.project_qkv(hidden_states)
        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)
        if self.config.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        xq, xk = self.apply_rope((xq, xk), position_ids=position_ids)
        return xq, xk, xv

    def get_attention_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.resid_dropout(self.wo(attn_output))

    # pylint: disable=not-callable
    def core_attention_op(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Core scaled dot-product attention."""
        if self.config.attn_pdrop > 0.0:
            raise ValueError("attn_pdrop > 0 not implemented")

        xq_ = xq.permute(0, 2, 1, 3)
        xk_ = xk.permute(0, 2, 1, 3)
        xv_ = xv.permute(0, 2, 1, 3)

        if attention_mask is not None:
            attn_bias = torch.zeros(
                1, 1, xq_.shape[2], xk_.shape[2], dtype=xq_.dtype, device=xq_.device
            )
            attn_bias = attn_bias.masked_fill(
                ~attention_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            out = F.scaled_dot_product_attention(xq_, xk_, xv_, attn_mask=attn_bias)
        else:
            out = F.scaled_dot_product_attention(xq_, xk_, xv_, is_causal=True)

        return self._merge_heads(out.permute(0, 2, 1, 3))

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError


# pylint: disable=arguments-differ,too-many-locals
class Attention(AttentionBase):
    """Full causal attention with KV-cache support."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq: Batch,
        state: Optional[tuple],
        **_kwargs,
    ) -> tuple[torch.Tensor, Any]:
        if hidden_states.dim() != 3:
            hidden_states = hidden_states.view(
                hidden_states.shape[0], hidden_states.shape[1], -1
            )
        B, T, _ = hidden_states.shape

        position_ids = self._get_position_ids(seq, B, T, hidden_states.device)
        xq, xk, xv = self.get_attention_input(hidden_states, position_ids)

        k_cache, v_cache = state or (None, None)
        if k_cache is not None:
            xk = torch.cat([k_cache, xk], dim=1)
            xv = torch.cat([v_cache, xv], dim=1)

        attn_out = self._compute_attention(xq, xk, xv)
        k_new, v_new = self._update_cache(xk, xv)

        return self.get_attention_output(attn_out), (k_new, v_new)

    def _get_position_ids(
        self, seq: Batch, B: int, T: int, device: torch.device
    ) -> torch.Tensor:
        if seq.position_ids is None:
            return torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        return seq.position_ids

    def _compute_attention(
        self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor
    ) -> torch.Tensor:
        xq_ = xq.permute(0, 2, 1, 3)
        xk_ = xk.permute(0, 2, 1, 3)
        xv_ = xv.permute(0, 2, 1, 3)
        attn_out = F.scaled_dot_product_attention(  # pylint: disable=not-callable
            xq_, xk_, xv_, is_causal=True
        )
        return self._merge_heads(attn_out.permute(0, 2, 1, 3))

    def _update_cache(
        self, xk: torch.Tensor, xv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_new, v_new = xk, xv
        if (
            self.config.sliding_window_size
            and k_new.shape[1] > self.config.sliding_window_size
        ):
            k_new = k_new[:, -self.config.sliding_window_size :, :, :]
            v_new = v_new[:, -self.config.sliding_window_size :, :, :]
        return k_new, v_new


# Feed‑forward (SwiGLU)
class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.w1 = NormalLinear(
            config,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            std=config.initializer_range,
            name="w1",
        )
        self.w2 = NormalLinear(
            config,
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            std=config.initializer_range,
            name="w2",
        )
        self.w3 = NormalLinear(
            config,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            std=config.initializer_range,
            name="w3",
        )
        self.dropout = nn.Dropout(p=config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# Prime storage (TTT support)
# pylint: disable=abstract-method
class PrimeStorage(nn.Module):
    """Holds prime FFN layers for TTT suffix blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.feed_forward_prime != "swiglu":
            raise NotImplementedError("Only feed_forward_prime='swiglu' is supported.")

        suffix_len = config.suffix_len
        self.feed_forward_prime = nn.ModuleList(
            [SwiGLUMLP(config) for _ in range(suffix_len)]
        )
        self.ffn_prime_norm = nn.ModuleList(
            [
                nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                for _ in range(suffix_len)
            ]
        )
        self.ffn_prime_post_norm = nn.ModuleList(
            [
                nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                for _ in range(suffix_len)
            ]
        )


# Transformer Block
# pylint: disable=too-many-instance-attributes
class Block(nn.Module):
    """Single transformer block with optional prime FFN."""

    def __init__(
        self,
        config: ModelConfig,
        feed_forward_prime: Optional[SwiGLUMLP] = None,
        ffn_prime_norm: Optional[nn.RMSNorm] = None,
        ffn_prime_post_norm: Optional[nn.RMSNorm] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = get_torch_dtype(config.compute_dtype)

        if config.seq_modeling_block != "self_attention":
            raise NotImplementedError(
                f"Sequence Modeling Layer {config.seq_modeling_block} Not Implemented."
            )

        self.seq_modeling_block: AttentionBase = Attention(config)
        self.feed_forward = SwiGLUMLP(config)

        self.seq_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.seq_post_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_post_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.ffn_prime_norm = ffn_prime_norm
        self.ffn_prime_post_norm = ffn_prime_post_norm
        self.feed_forward_prime = feed_forward_prime

    def _seq_forward(
        self,
        hidden_states: torch.Tensor,
        state: Any,
        seq: Batch,
        is_prefix: bool,
    ) -> tuple[torch.Tensor, Any]:
        inp = self.seq_norm(hidden_states) if self.config.pre_norm else hidden_states
        out, new_state = self.seq_modeling_block(inp, seq, state, is_prefix=is_prefix)
        if self.config.post_norm:
            out = self.seq_post_norm(out)
        return out, new_state

    def _ffn_forward(
        self,
        hidden_states: torch.Tensor,
        ffn_norm: nn.RMSNorm,
        feed_forward: SwiGLUMLP,
        ffn_post_norm: nn.RMSNorm,
    ) -> torch.Tensor:
        inp = ffn_norm(hidden_states) if self.config.pre_norm else hidden_states
        out = feed_forward(inp)
        if self.config.post_norm:
            out = ffn_post_norm(out)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Any,
        seq: Batch,
        is_prefix: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        seq_out, new_state = self._seq_forward(
            hidden_states, state, seq, is_prefix=is_prefix
        )
        hidden_states = hidden_states + seq_out

        if self.feed_forward_prime is not None:
            prime_out = self._ffn_forward(
                hidden_states,
                self.ffn_prime_norm,
                self.feed_forward_prime,
                self.ffn_prime_post_norm,
            )
            hidden_states = hidden_states + prime_out

        ffn_out = self._ffn_forward(
            hidden_states,
            self.ffn_norm,
            self.feed_forward,
            self.ffn_post_norm,
        )
        hidden_states = hidden_states + ffn_out

        return hidden_states, new_state


# Block collection and Transformer model
@dataclass
class BaseModelOutput:
    last_hidden_state: torch.Tensor
    state: Optional[list] = None


class BlockCollection(nn.Module):
    """Flat stack of transformer blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.num_hidden_layers)]
        )
        self.prime_storage: Optional[PrimeStorage] = (
            PrimeStorage(config) if config.prime else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[list],
        seq: Batch,
    ) -> BaseModelOutput:
        new_states = []
        for i, block in enumerate(self.blocks):
            sub = state[i] if (state is not None and i < len(state)) else None
            hidden_states, sub = block(hidden_states, sub, seq)
            new_states.append(sub)
        return BaseModelOutput(last_hidden_state=hidden_states, state=new_states)


class TransformerModel(nn.Module):
    """Transformer backbone: embeddings + blocks + final norm."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = get_torch_dtype(config.compute_dtype)

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.wte.weight, std=config.initializer_range)

        self.dropout = nn.Dropout(p=config.embd_pdrop)
        self.h: BlockCollection = BlockCollection(config)
        self.ln_f = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def wte_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.wte(input_ids.long()).to(self.compute_dtype)
        return self.dropout(emb)

    def forward(self, state: Optional[list], seq: Batch) -> BaseModelOutput:
        hidden_states = self.wte_call(seq.input_ids)
        outputs: BaseModelOutput = self.h(hidden_states, state=state, seq=seq)
        return BaseModelOutput(
            last_hidden_state=self.ln_f(outputs.last_hidden_state),
            state=outputs.state,
        )


# Causal LM wrapper
@dataclass
class CausalLMOutput:
    last_hidden_states: torch.Tensor
    logits: torch.Tensor
    new_state: Any


class CausalLM(nn.Module):
    """Language model with optional tied embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = get_torch_dtype(config.compute_dtype)
        self.model = TransformerModel(config)
        self.lm_head = self._init_lm_head(config)

    def _init_lm_head(self, config: ModelConfig) -> Optional[NormalLinear]:
        if config.tie_word_embeddings:
            return None
        return NormalLinear(
            config,
            in_features=config.hidden_size,
            out_features=config.output_size,
            std=config.initializer_range,
            name="lm_head",
        )

    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.tie_word_embeddings:
            hs, kernel = promote_dtype(
                hidden_states, self.model.wte.weight.T, dtype=self.compute_dtype
            )
            return hs @ kernel
        return self.lm_head(hidden_states)

    def wte_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.wte_call(input_ids)

    def forward(self, state: Optional[list], seq: Batch) -> CausalLMOutput:
        outputs = self.model(state, seq)
        hs = outputs.last_hidden_state
        assert hs.dtype == self.compute_dtype
        return CausalLMOutput(
            last_hidden_states=hs,
            logits=self._compute_logits(hs),
            new_state=outputs.state,
        )


# Loss functions
def cross_entropy_loss_and_accuracy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy loss with mask support."""
    if valid is None:
        valid = torch.ones(tokens.shape, dtype=torch.float32, device=tokens.device)
    valid = valid.float()
    valid_text_length = torch.clamp(valid.sum(dim=-1), min=1e-10)

    logits = logits.float()
    log_prob = F.log_softmax(logits, dim=-1)
    token_log_prob = log_prob.gather(-1, tokens.long().unsqueeze(-1)).squeeze(-1)
    token_log_prob = torch.where(
        valid > 0.0, token_log_prob, torch.zeros_like(token_log_prob)
    )

    token_wise_loss = -token_log_prob
    loss = (token_wise_loss.sum(dim=-1) / valid_text_length).mean()
    return loss, loss


def token_log_probs(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute log probabilities of target tokens."""
    return (
        F.log_softmax(logits, dim=-1)
        .gather(-1, targets.long().unsqueeze(-1))
        .squeeze(-1)
    )
