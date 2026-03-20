from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


LATENT_SCHEMA_VERSION = "pooled-v1"


def extract_latents(
    model: torch.nn.Module,
    images: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    ds = TensorDataset(torch.tensor(images, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            z = model.encoder(x)
            # Canonical latent contract: global average pooled latent tensor.
            z = z.mean(dim=(2, 3)) if z.ndim == 4 else z
            out.append(z.detach().cpu().numpy())
    return np.vstack(out) if out else np.empty((0, 0), dtype=np.float32)

