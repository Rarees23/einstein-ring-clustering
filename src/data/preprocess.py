"""
Shared FITS loading and image preprocessing for the Einstein ring pipeline.
Single convention: 99.5 percentile clip, normalize by that value, resize to config size.
"""

import glob
import os

import numpy as np
from astropy.io import fits
from skimage.transform import resize

from src.core.runtime import RuntimeConfig


def find_fits(root: str) -> list[str]:
    """Return sorted list of all .fits paths under root."""
    return sorted(glob.glob(os.path.join(root, "**/*.fits"), recursive=True))


def load_fits_image(path: str) -> np.ndarray:
    """Load first 2D slice from a FITS file as float32."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data.ndim > 2:
            idx = tuple(0 for _ in range(data.ndim - 2))
            data = data[idx + (slice(None), slice(None))]
        return np.squeeze(data).astype(np.float32)


def preprocess_image(img2d: np.ndarray, out_h: int | None = None, out_w: int | None = None) -> np.ndarray:
    """Normalize and resize a 2D image for model input. Returns shape (1, H, W)."""
    cfg = RuntimeConfig.default()
    out_h = out_h if out_h is not None else cfg.image_h
    out_w = out_w if out_w is not None else cfg.image_w

    img = np.nan_to_num(img2d, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.abs(img)
    p99 = np.percentile(img, 99.5)
    img = np.clip(img, 0, p99)
    if p99 > 0:
        img = img / p99
    if (img.shape[0], img.shape[1]) != (out_h, out_w):
        img = resize(img, (out_h, out_w), preserve_range=True, anti_aliasing=True)
    return np.expand_dims(img.astype(np.float32), axis=0)

