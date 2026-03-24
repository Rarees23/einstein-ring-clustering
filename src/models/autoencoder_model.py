from __future__ import annotations

from src.core.runtime import RuntimeConfig
from src.models.base import LatentReconstructionModel
from src.models.conv_autoencoder import ConvAutoencoder

_MODEL_REGISTRY: dict[str, type[LatentReconstructionModel]] = {
    "conv_ae": ConvAutoencoder,
}


def register_model_kind(kind: str, cls: type[LatentReconstructionModel]) -> None:
    _MODEL_REGISTRY[kind] = cls


def build_autoencoder(cfg: RuntimeConfig) -> LatentReconstructionModel:
    try:
        cls = _MODEL_REGISTRY[cfg.model_kind]
    except KeyError as e:
        raise ValueError(
            f"Unknown model_kind {cfg.model_kind!r}. Known kinds: {sorted(_MODEL_REGISTRY)}",
        ) from e
    return cls(in_channels=cfg.in_channels, latent_dim=cfg.latent_dim).to(cfg.device)
