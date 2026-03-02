"""
Shared FITS loading and image preprocessing for the Einstein ring pipeline.
Single convention: 99.5 percentile clip, normalize by that value, resize to config size.
"""
import os
import glob
import numpy as np
from astropy.io import fits
from skimage.transform import resize

from config import IMG_H, IMG_W

def find_fits(root):
    """Return sorted list of all .fits paths under root."""
    return sorted(glob.glob(os.path.join(root, "**/*.fits"), recursive=True))

def load_fits_image(path):
    """Load first 2D slice from a FITS file as float32."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data.ndim > 2:
            idx = tuple(0 for _ in range(data.ndim - 2))
            data = data[idx + (slice(None), slice(None))]
        return np.squeeze(data).astype(np.float32)

def preprocess_image(img2d, out_h=IMG_H, out_w=IMG_W):
    """
    Normalize and resize a 2D image for the autoencoder.
    - Replace nan/inf with 0, take abs
    - Clip values above 99.5 percentile, then normalize by that value
    - Resize to (out_h, out_w) if needed
    Returns shape (1, H, W) float32.
    """
    img = np.nan_to_num(img2d, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.abs(img)
    p99 = np.percentile(img, 99.5)
    img = np.clip(img, 0, p99)
    if p99 > 0:
        img = img / p99
    if (img.shape[0], img.shape[1]) != (out_h, out_w):
        img = resize(img, (out_h, out_w), preserve_range=True, anti_aliasing=True)
    return np.expand_dims(img.astype(np.float32), axis=0)
