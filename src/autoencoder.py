"""
Trains a convolutional autoencoder on FITS images where only Einstein rings matter.
Shows side-by-side comparisons only after the final epoch and saves the best model.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from astropy.io import fits
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm

from ConvolutionalAutoencoder import ConvAutoencoder
from config import DATA_DIR, SAVED_MODELS_DIR, BEST_MODEL_PATH, IMG_H, IMG_W, IN_CHANNELS, LATENT_DIM, BATCH_SIZE, NUM_EPOCHS, DEVICE

# --- Ensure directories exist ---
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# --- Dataset helpers ---
def find_fits_files(root):
    return sorted(glob.glob(os.path.join(root, "**/*.fits"), recursive=True))

def load_fits_image(path):
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"No image data in {path}")
        if data.ndim > 2:
            idx = tuple(0 for _ in range(data.ndim - 2))
            data = data[idx + (slice(None), slice(None))]
        data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"Cannot convert {path} to 2D, got shape {data.shape}")
        return data.astype(np.float32)

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

def show_comparison(original, reconstructed, vmax=None):
    plt.figure(figsize=(6, 3))
    vmax = vmax if vmax else max(np.nanmax(original), np.nanmax(reconstructed))
    plt.subplot(1, 2, 1)
    plt.imshow(original, origin='lower', vmin=0, vmax=vmax)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, origin='lower', vmin=0, vmax=vmax)
    plt.title("Reconstruction")
    plt.axis("off")
    plt.show()

# --- Load Dataset ---
fits_files = find_fits_files(DATA_DIR)
images = []
for f in tqdm(fits_files, desc="Loading FITS images"):
    try:
        img = load_fits_image(f)
        img = preprocess_image(img)
        images.append(img)
    except Exception as e:
        print(f"Skipping {f}: {e}")

if len(images) == 0:
    raise RuntimeError(f"No images found in {DATA_DIR} after preprocessing.")

images = np.array(images, dtype=np.float32)
dataset = TensorDataset(torch.tensor(images))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# --- Model ---
model = ConvAutoencoder(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Masked MSE to focus on rings
def mask_mse_loss(output, target):
    mask = (target > 0.01).float()
    return ((output - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)

# --- Training ---
best_loss = float("inf")
best_model_state = None
loss_history = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for (x,) in dataloader:
        x = x.to(DEVICE)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = mask_mse_loss(x_hat, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(dataset)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Avg Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_state = model.state_dict()

# --- Save best model ---
torch.save(best_model_state, BEST_MODEL_PATH)
print(f"Training complete. Best model saved to {BEST_MODEL_PATH} with loss {best_loss:.6f}")

# --- Show final reconstructions for first 5 images only ---
model.eval()
with torch.no_grad():
    for i in range(min(5, len(dataset))):
        img_tensor = dataset[i][0].unsqueeze(0).to(DEVICE)
        recon = model(img_tensor).squeeze(0).cpu().numpy()
        orig = img_tensor.squeeze(0).cpu().numpy()
        show_comparison(orig[0], recon[0])
