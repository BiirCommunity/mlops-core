from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from app.core.transformer import Batch


class TextDataset(Dataset):
    """Текстовый датасет для language modeling: файл → токены → чанки длиной seq_len+1."""

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        seq_len: int,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        text = self.path.read_text(encoding="utf-8")
        input_ids = tokenizer.encode(text, return_tensors="pt").squeeze(0)

        n_full = (input_ids.shape[0] - 1) // self.seq_len
        self.chunks: List[torch.Tensor] = []
        for i in range(n_full):
            start = i * self.seq_len
            end = start + self.seq_len + 1
            self.chunks.append(input_ids[start:end])

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]


def collate_lm(
    batch_tensors: Iterable[torch.Tensor],
    device: torch.device,
) -> Batch:
    tensors = list(batch_tensors)
    assert len(tensors) > 0

    src_list = [t[:-1] for t in tensors]
    tgt_list = [t[1:] for t in tensors]

    src = torch.stack(src_list, dim=0).to(device)
    tgt = torch.stack(tgt_list, dim=0).to(device)
    mask = torch.ones_like(tgt, dtype=torch.float32, device=device)

    return Batch(
        input_ids=src,
        target_tokens=tgt,
        loss_masks=mask,
    )


def create_dataloader(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    path: str | Path,
    tokenizer,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
    dataset_fraction: float = 1.0,
) -> DataLoader:
    dataset = TextDataset(path, tokenizer=tokenizer, seq_len=seq_len)

    if dataset_fraction < 1.0:
        n_samples = int(len(dataset) * dataset_fraction)
        indices = torch.randperm(len(dataset))[:n_samples]
        dataset = Subset(dataset, indices)

    def _collate(batch: List[torch.Tensor]) -> Batch:
        return collate_lm(batch, device=device)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
    )
