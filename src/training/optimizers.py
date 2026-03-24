"""Optimizer factory; names map to :attr:`RuntimeConfig.optimizer_kind`."""

from __future__ import annotations

from collections.abc import Iterable

import torch.optim as optim


def build_optimizer(
    kind: str,
    params: Iterable,
    *,
    lr: float,
) -> optim.Optimizer:
    k = kind.lower()
    if k == "adam":
        return optim.Adam(params, lr=lr)
    if k == "adamw":
        return optim.AdamW(params, lr=lr)
    if k == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer_kind {kind!r}. Known: adam, adamw, sgd")
