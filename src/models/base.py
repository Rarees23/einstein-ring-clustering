"""Abstract AE contract: ``encoder`` for latents, ``forward`` for reconstruction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class LatentReconstructionModel(nn.Module, ABC):
    encoder: nn.Module

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction; same shape as ``x`` (NCHW)."""
        pass
