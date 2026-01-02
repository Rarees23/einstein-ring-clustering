import os
import glob
import numpy as np
import torch
from astropy.io import fits
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm

from ConvolutionalAutoencoder import ConvAutoencoder
from config import DATA_DIR, SAVED_MODELS_DIR, IMG_H, IMG_W, IN_CHANNELS, LATENT_DIM, DEVICE, RECON_THRESHOLD, SHOW_FIRST_N

# ---------------- PATHS ----------------
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.pth")
TEST_DIR = DATA_DIR

# ---------------- Helpers ----------------
def find_fits_files(root):
    return sorted(glob.glob(os.path.join(root, "**/*.fits"), recursive=True))

def load_fits_image(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data.ndim > 2:
            idx = tuple(0 for _ in range(data.ndim - 2))
            data = data[idx + (slice(None), slice(None))]
        return np.squeeze(data).astype(np.float32)

def preprocess_image(img2d, out_h=IMG_H, out_w=IMG_W):
    img = np.nan_to_num(img2d, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.abs(img)
    threshold = np.percentile(img, 99)
    img[img < threshold] = 0
    max_val = img.max()
    if max_val > 0:
        img = img / max_val
    if (img.shape[0], img.shape[1]) != (out_h, out_w):
        img = resize(img, (out_h, out_w), preserve_range=True, anti_aliasing=True)
    return np.expand_dims(img.astype(np.float32), axis=0)

def masked_mse(output, target, mask_thresh=0.01):
    mask = (target > mask_thresh)
    if mask.sum() == 0:
        return 0.0, mask.astype(np.uint8)
    masked_sq = ((output - target) ** 2) * mask
    return float(masked_sq.sum() / mask.sum()), mask.astype(np.uint8)

def masked_psnr(mse_masked, max_val=1.0):
    return 10.0 * np.log10(max_val**2 / mse_masked) if mse_masked > 0 else float("inf")

def fraction_reconstructed(output, target, recon_thresh=RECON_THRESHOLD, mask_thresh=0.01):
    mask = (target > mask_thresh)
    if mask.sum() == 0:
        return 0.0
    recon_hits = ((output > recon_thresh) & mask).sum()
    return float(recon_hits) / float(mask.sum())

def show_comparison_grid(orig_list, recon_list, residual_list, mask_list, filenames):
    n = len(orig_list)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3*n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i in range(n):
        vmax = max(orig_list[i].max(), recon_list[i].max())
        axes[i,0].imshow(orig_list[i], origin='lower', vmin=0, vmax=vmax)
        axes[i,0].set_title(f"{filenames[i]}\nOriginal")
        axes[i,0].axis('off')
        
        axes[i,1].imshow(recon_list[i], origin='lower', vmin=0, vmax=vmax)
        axes[i,1].set_title("Reconstruction")
        axes[i,1].axis('off')
        
        axes[i,2].imshow(residual_list[i], origin='lower')
        axes[i,2].set_title("Residual")
        axes[i,2].axis('off')
        
        axes[i,3].imshow(orig_list[i], origin='lower', vmin=0, vmax=vmax)
        axes[i,3].imshow(mask_list[i], origin='lower', alpha=0.4, cmap='Reds')
        axes[i,3].set_title("Mask overlay")
        axes[i,3].axis('off')
    plt.tight_layout()
    plt.show()

# ---------------- Load Model ----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = ConvAutoencoder(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- Test Files ----------------
fits_files = find_fits_files(TEST_DIR)
if len(fits_files) == 0:
    raise RuntimeError(f"No FITS files found in {TEST_DIR}")

all_mse, all_psnr, all_frac = [], [], []
grid_orig, grid_recon, grid_resid, grid_mask, grid_fnames = [], [], [], [], []

print(f"{'Image':<30} {'Masked MSE':>10} {'PSNR(dB)':>10} {'Frac Reconst':>13} {'Latent Dim':>12}")
print("-"*80)

for idx, path in enumerate(tqdm(fits_files, desc="Testing FITS")):
    try:
        raw = load_fits_image(path)
    except Exception as e:
        print(f"Skipping {path}: {e}")
        continue

    proc = preprocess_image(raw)
    tensor = torch.tensor(proc[None], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        recon = model(tensor).squeeze(0).cpu().numpy()
        latent_tensor = model.encoder(tensor)
        latent_vector = torch.flatten(latent_tensor, start_dim=1).squeeze(0).detach().cpu().numpy()

    orig = proc[0]
    recon_img = recon[0]
    mse_masked, mask = masked_mse(recon_img, orig)
    psnr_masked = masked_psnr(mse_masked)
    frac = fraction_reconstructed(recon_img, orig)
    residual = np.abs(orig - recon_img)

    print(f"{os.path.basename(path):<30} {mse_masked:10.6f} {psnr_masked:10.2f} {frac:13.4f} {latent_vector.shape[0]:12}")

    all_mse.append(mse_masked)
    all_psnr.append(psnr_masked)
    all_frac.append(frac)

    if idx < SHOW_FIRST_N:
        grid_orig.append(orig)
        grid_recon.append(recon_img)
        grid_resid.append(residual)
        grid_mask.append(mask)
        grid_fnames.append(os.path.basename(path))

if grid_orig:
    show_comparison_grid(grid_orig, grid_recon, grid_resid, grid_mask, grid_fnames)

# ---------------- Summary ----------------
print("\nSummary Statistics for all images:")
print(f"Masked MSE: mean={np.mean(all_mse):.6f}, median={np.median(all_mse):.6f}, std={np.std(all_mse):.6f}")
print(f"Masked PSNR: mean={np.mean(all_psnr):.2f} dB, median={np.median(all_psnr):.2f} dB, std={np.std(all_psnr):.2f} dB")
print(f"Fraction reconstructed: mean={np.mean(all_frac):.4f}, median={np.median(all_frac):.4f}, std={np.std(all_frac):.4f}")
