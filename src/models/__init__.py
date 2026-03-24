"""Encoder/decoder models; register with ``register_model_kind``, pick via ``RuntimeConfig.model_kind``."""

from src.models.autoencoder_model import build_autoencoder, register_model_kind
from src.models.base import LatentReconstructionModel
from src.models.conv_autoencoder import ConvAutoencoder

__all__ = [
    "ConvAutoencoder",
    "LatentReconstructionModel",
    "build_autoencoder",
    "register_model_kind",
]
