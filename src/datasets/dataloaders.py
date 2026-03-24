"""Train/val DataLoader pair (shuffle on train only)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def train_val_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
