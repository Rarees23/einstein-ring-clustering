from __future__ import annotations

import torch

from src.models.conv_autoencoder import ConvAutoencoder


def build_autoencoder(in_channels: int, latent_dim: int, device: str) -> torch.nn.Module:
    return ConvAutoencoder(in_channels=in_channels, latent_dim=latent_dim).to(device)

