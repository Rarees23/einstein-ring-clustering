"""
Open and display all FITS images in a folder (and all subfolders).
Set FOLDER at the start of the code, then run:  python view_fits.py
"""
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# ----- Set your folder here (all .fits in this folder and subfolders will be shown) -----
FOLDER = Path(__file__).resolve().parent / "Euclid_Lens_Dataset"

# Optional: set to a number to show only the first N images (e.g. 10); None = show all
MAX_IMAGES = None


def load_fits_image(path):
    """Return first 2D image from a FITS file (tries all HDUs if primary is empty)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with fits.open(path, memmap=False) as hdul:
        for i, hdu in enumerate(hdul):
            data = hdu.data
            if data is None or not getattr(data, "ndim", 0) >= 2:
                continue
            if data.ndim > 2:
                data = data[0, ...]
            img = np.squeeze(data).astype(np.float32)
            return img
        raise ValueError(f"No image HDU found in {path}")


def main():
    folder = Path(FOLDER)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    fits_paths = sorted(folder.rglob("*.fits"))
    if not fits_paths:
        raise SystemExit(f"No .fits files found under {folder}")

    if MAX_IMAGES is not None:
        fits_paths = fits_paths[:MAX_IMAGES]

    print(f"Showing {len(fits_paths)} FITS from {folder}")

    for n, path in enumerate(fits_paths, 1):
        try:
            img = load_fits_image(path)
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
            continue

        # Brighten / enhance contrast for visualization only
        lo, hi = np.percentile(img, [1.0, 99.5])
        if hi <= lo:
            lo, hi = float(np.min(img)), float(np.max(img))
        if hi > lo:
            img_disp = (img - lo) / (hi - lo)
        else:
            img_disp = img - lo

        gamma = 0.7  # < 1 brightens faint structures
        img_disp = np.clip(img_disp, 0.0, 1.0) ** gamma

        rel = path.relative_to(folder)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_disp, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
        plt.colorbar(label="Scaled flux")
        plt.title(f"{n}/{len(fits_paths)}  {rel}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
