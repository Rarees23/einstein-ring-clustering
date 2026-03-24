"""Reconstruction losses; names map to :attr:`RuntimeConfig.loss_kind`."""

from __future__ import annotations

from collections.abc import Callable

import torch

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Up-weight pixels above a small threshold (ring signal vs background).
    mask = 0.1 + 0.9 * (target > 0.01).float()
    return ((pred - target) ** 2 * mask).mean()


_REGISTRY: dict[str, LossFn] = {
    "mse": mse_loss,
    "masked_mse": masked_mse_loss,
}


def register_loss_kind(kind: str, fn: LossFn) -> None:
    _REGISTRY[kind] = fn


def get_loss_fn(kind: str) -> LossFn:
    try:
        return _REGISTRY[kind]
    except KeyError as e:
        raise ValueError(f"Unknown loss_kind {kind!r}. Known: {sorted(_REGISTRY)}") from e
