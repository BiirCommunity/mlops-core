"""Chat JSONL dataset for LoRA post-training with assistant-only loss mask."""

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from app.core.transformer import Batch
from app.training.tokenizer_utils import has_chat_template


class ChatJSONLDataset(Dataset):
    """
    JSONL with records:
      {"messages": [{"role": "user"|"assistant"|"system", "content": "..."}]}

    Loss is computed only on assistant tokens.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        *,
        max_seq_len: int = 512,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = self._load_examples()

    def _load_examples(self) -> list[list[dict[str, str]]]:
        examples: list[list[dict[str, str]]] = []
        for line_no, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            payload = json.loads(stripped)
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"{self.path}:{line_no}: expected non-empty 'messages' list"
                )
            if messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"{self.path}:{line_no}: last message must be role=assistant"
                )
            examples.append(messages)
        if not examples:
            raise ValueError(f"{self.path}: dataset is empty")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _render_chat(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool
    ) -> str:
        if has_chat_template(self.tokenizer):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        parts: list[str] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n".join(parts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        messages = self.examples[idx]
        prompt_messages = messages[:-1]
        full_text = self._render_chat(messages, add_generation_prompt=False)
        prompt_text = self._render_chat(
            prompt_messages,
            add_generation_prompt=True,
        )

        full_ids = self.tokenizer.encode(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len + 1,
        )
        prompt_ids = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len + 1,
        )

        tokens = torch.tensor(full_ids, dtype=torch.long)
        prompt_len = min(len(prompt_ids), max(len(tokens) - 1, 1))
        loss_mask = torch.zeros(tokens.shape[0], dtype=torch.float32)
        loss_mask[prompt_len:] = 1.0
        if loss_mask.sum() <= 0:
            loss_mask[-1] = 1.0
        return tokens, loss_mask


def collate_chat_batch(
    batch_items: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> Batch:
    max_len = max(item[0].shape[0] for item in batch_items)
    input_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []

    for tokens, loss_mask in batch_items:
        if tokens.shape[0] < 2:
            tokens = torch.cat([tokens, tokens[-1:].clone()])
            loss_mask = torch.cat([loss_mask, loss_mask[-1:].clone()])
        pad_len = max_len - tokens.shape[0]
        if pad_len > 0:
            pad = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            tokens = torch.cat([tokens, pad])
            loss_mask = torch.cat([loss_mask, torch.zeros(pad_len)])
        input_rows.append(tokens[:-1])
        target_rows.append(tokens[1:])
        mask_rows.append(loss_mask[1:])

    input_ids = torch.stack(input_rows, dim=0).to(device)
    target_tokens = torch.stack(target_rows, dim=0).to(device)
    loss_masks = torch.stack(mask_rows, dim=0).to(device)
    return Batch(
        input_ids=input_ids,
        target_tokens=target_tokens,
        loss_masks=loss_masks,
    )


def create_chat_dataloader(
    path: str | Path,
    tokenizer,
    *,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
) -> DataLoader:
    dataset = ChatJSONLDataset(path, tokenizer, max_seq_len=max_seq_len)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0

    def _collate(items: list[tuple[torch.Tensor, torch.Tensor]]) -> Batch:
        return collate_chat_batch(
            items,
            pad_token_id=pad_token_id,
            device=device,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
    )


def dataset_summary(path: str | Path) -> dict[str, Any]:
    dataset = Path(path)
    examples = 0
    roles: dict[str, int] = {}
    for line in dataset.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        payload = json.loads(stripped)
        examples += 1
        for message in payload.get("messages", []):
            role = str(message.get("role", "unknown"))
            roles[role] = roles.get(role, 0) + 1
    return {
        "path": str(dataset),
        "examples": examples,
        "roles": roles,
    }
