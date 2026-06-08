import torch

from app.core.transformer import cross_entropy_loss_and_accuracy  # type: ignore


def language_modeling_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    loss_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    loss, _ = cross_entropy_loss_and_accuracy(logits, target_tokens, loss_masks)
    return loss
