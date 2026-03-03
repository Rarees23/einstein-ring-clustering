import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm

from ConvolutionalAutoencoder import ConvAutoencoder
from config import (
    DATA_DIR,
    SAVED_MODELS_DIR,
    BEST_MODEL_PATH,
    IN_CHANNELS,
    LATENT_DIM,
    BATCH_SIZE,
    NUM_EPOCHS,
    DEVICE,
    set_global_seed,
)
from preprocess import find_fits, load_fits_image, preprocess_image


def main() -> None:
    # Best-effort reproducibility
    set_global_seed(42, deterministic_cudnn=False)

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # ---------------- LOAD DATA ----------------
    fits_files = find_fits(DATA_DIR)
    images = []

    for f in tqdm(fits_files, desc="Loading FITS"):
        try:
            raw = load_fits_image(f)
            images.append(preprocess_image(raw))
        except Exception as e:
            print("Skipping", f, e)

    images = np.array(images, dtype=np.float32)
    dataset = TensorDataset(torch.tensor(images))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ---------------- MODEL ----------------
    model = ConvAutoencoder(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ---------------- LOSS ----------------
    def mask_mse_loss(output, target):
        mask = 0.1 + 0.9 * (target > 0.01).float()
        return ((output - target) ** 2 * mask).mean()

    # ---------------- TRAIN ----------------
    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total = 0

        for (x,) in loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()

            x_hat = model(x)
            loss = mask_mse_loss(x_hat, x)

            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)

        avg = total / len(dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    print("Saved best model to:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
