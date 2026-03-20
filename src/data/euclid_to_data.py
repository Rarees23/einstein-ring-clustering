"""Transform Euclid FITS into ring-focused data folders."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil

import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi
from tqdm import tqdm

from src.core.runtime import RuntimeConfig
from src.data.preprocess import find_fits


def load_fits_image_any_hdu(path: str) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            data = hdu.data
            if data is None or not getattr(data, "ndim", 0) >= 2:
                continue
            if data.ndim > 2:
                data = data[0, ...]
            return np.squeeze(data).astype(np.float32)
        raise ValueError(f"No image HDU found in {path}")


def _ensure_2d(data: np.ndarray) -> np.ndarray:
    data = np.squeeze(data)
    if data.ndim > 2:
        data = data[tuple(0 for _ in range(data.ndim - 2)) + (slice(None), slice(None))]
    return np.squeeze(data).astype(np.float32)


def _isolate_ring_annulus(img: np.ndarray, inner_frac: float = 0.02, outer_frac: float = 0.12) -> np.ndarray:
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r_max = min(h, w) / 2.0
    r_inner = inner_frac * r_max
    r_outer = outer_frac * r_max
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring_mask = (rr >= r_inner) & (rr <= r_outer)
    out = np.zeros_like(img, dtype=np.float32)
    out[ring_mask] = img[ring_mask]
    return out


def _crop_to_nonzero(img: np.ndarray, border: int = 2) -> np.ndarray:
    mask = img > 0
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    y_min = max(0, int(ys.min()) - border)
    y_max = min(img.shape[0] - 1, int(ys.max()) + border)
    x_min = max(0, int(xs.min()) - border)
    x_max = min(img.shape[1] - 1, int(xs.max()) + border)
    return img[y_min : y_max + 1, x_min : x_max + 1]


def isolate_einstein_ring(
    img: np.ndarray,
    smooth_sigma: float,
    center_mask_radius: float = 0.0,
    clip_percentile: float = 99.5,
) -> np.ndarray:
    cfg = RuntimeConfig.default()
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.abs(img)
    smoothed = ndi.gaussian_filter(img, sigma=smooth_sigma, mode="constant", cval=0.0)
    ring = np.clip(img - smoothed, 0.0, None)
    if center_mask_radius > 0:
        h, w = ring.shape
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.ogrid[:h, :w]
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        ring[r2 <= center_mask_radius**2] = 0.0
    p = np.percentile(ring, clip_percentile)
    if p > 0:
        ring = ring / p
    ring = np.clip(ring, 0.0, 1.0).astype(np.float32)
    ring = _isolate_ring_annulus(ring)
    ring = _crop_to_nonzero(ring, border=2)

    h, w = ring.shape
    if (h, w) != (cfg.image_h, cfg.image_w):
        zoom_y = cfg.image_h / float(h)
        zoom_x = cfg.image_w / float(w)
        ring = ndi.zoom(ring, (zoom_y, zoom_x), order=0)
        h2, w2 = ring.shape
        out = np.zeros((cfg.image_h, cfg.image_w), dtype=ring.dtype)
        y0 = max(0, (h2 - cfg.image_h) // 2)
        x0 = max(0, (w2 - cfg.image_w) // 2)
        y1 = min(h2, y0 + cfg.image_h)
        x1 = min(w2, x0 + cfg.image_w)
        ty0 = max(0, (cfg.image_h - (y1 - y0)) // 2)
        tx0 = max(0, (cfg.image_w - (x1 - x0)) // 2)
        ty1 = ty0 + (y1 - y0)
        tx1 = tx0 + (x1 - x0)
        out[ty0:ty1, tx0:tx1] = ring[y0:y1, x0:x1]
        ring = out
    return ring


def estimate_einstein_radius_pixels(ring: np.ndarray) -> float:
    h, w = ring.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    threshold = np.percentile(ring, 85.0)
    if threshold <= 0:
        return 0.0
    ring_pixels = r[ring >= threshold]
    if ring_pixels.size == 0:
        return 0.0
    return float(np.median(ring_pixels))


def object_id_from_path(root: str, fits_path: str) -> str:
    rel = os.path.relpath(fits_path, root).replace("\\", "/")
    return os.path.splitext(rel)[0].replace("/", "_") or "unknown"


def write_data_folder(out_root: str, object_id: str, ring_image: np.ndarray, einstein_radius_estimate: float) -> None:
    obj_dir = os.path.join(out_root, object_id)
    os.makedirs(obj_dir, exist_ok=True)
    main_fits = os.path.join(obj_dir, "mge_source_light.fits")
    ring_image = np.squeeze(ring_image).astype(np.float32)
    if ring_image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {ring_image.shape}")
    fits.PrimaryHDU(ring_image).writeto(main_fits, overwrite=True)
    with open(os.path.join(obj_dir, "einstein_radius.txt"), "w", encoding="utf-8") as f:
        f.write(f"{einstein_radius_estimate:.6f}\n")
    with open(os.path.join(obj_dir, "normalized_thickness.txt"), "w", encoding="utf-8") as f:
        f.write("0.0\n")
    with open(os.path.join(obj_dir, "result_lens_mass.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "einstein_radius_median_pdf": einstein_radius_estimate,
                "source": "euclid_to_data",
                "note": "Placeholder; run full lens model for real parameters.",
            },
            f,
            indent=2,
        )


def run_euclid_to_data(input_root: str, out_root: str, smooth_sigma: float = 8.0, center_mask_r: float = 5.0) -> None:
    fits_list = [p for p in find_fits(input_root) if not os.path.basename(p).startswith("._")]
    if not fits_list:
        raise SystemExit(f"No .fits files found under {input_root}")
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)
    for path in tqdm(fits_list, desc="Euclid -> data"):
        try:
            raw = load_fits_image_any_hdu(path)
            raw_2d = _ensure_2d(raw)
            ring = isolate_einstein_ring(raw_2d, smooth_sigma=smooth_sigma, center_mask_radius=center_mask_r)
            er_px = estimate_einstein_radius_pixels(ring)
            obj_id = re.sub(r'[<>:"/\\|?*]', "_", object_id_from_path(input_root, path)).strip(" .") or "unknown"
            write_data_folder(out_root, obj_id, ring, er_px)
        except Exception as e:
            tqdm.write(f"Skipping {path}: {e}")
    print(f"Done. Wrote {len(fits_list)} objects under {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform Euclid lens FITS to data-folder format (Einstein ring only)")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--smooth", type=float, default=8.0)
    parser.add_argument("--center-mask", type=float, default=5.0)
    args = parser.parse_args()
    cfg = RuntimeConfig.default()
    input_root = os.path.abspath(args.input or os.path.join(cfg.repo_root, "Euclid_Lens_Dataset"))
    out_root = os.path.abspath(args.output or os.path.join(cfg.repo_root, "data_euclid"))
    if not os.path.isdir(input_root):
        raise SystemExit(f"Input directory not found: {input_root}")
    run_euclid_to_data(input_root, out_root, smooth_sigma=args.smooth, center_mask_r=args.center_mask)


if __name__ == "__main__":
    main()

