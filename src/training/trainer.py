from __future__ import annotations

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def mask_mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Higher weight on pixels that are above a small signal threshold.
    mask = 0.1 + 0.9 * (target > 0.01).float()
    return ((output - target) ** 2 * mask).mean()


def train_autoencoder_with_early_stopping(
    model: torch.nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: str,
    num_epochs: int,
    batch_size: int,
    best_model_path: str,
    patience: int = 6,
) -> tuple[float, list[float], list[float]]:
    """
    Train autoencoder with masked MSE loss and early stopping.

    ``train_dataset`` / ``val_dataset`` should yield image tensors shaped (C, H, W).

    Returns:
      best_val_loss, train_loss_history, val_loss_history
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    stall = 0
    history_train: list[float] = []
    history_val: list[float] = []

    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = mask_mse_loss(model(x), x)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)

        train_loss = total / max(1, len(train_dataset))
        history_train.append(float(train_loss))

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                vtotal += mask_mse_loss(model(x), x).item() * x.size(0)
        val_loss = vtotal / max(1, len(val_dataset))
        history_val.append(float(val_loss))

        print(
            f"  epoch {epoch + 1:4d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"{'*' if val_loss < best_val else ''}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            stall = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            stall += 1
            if stall >= patience:
                break

    return best_val, history_train, history_val
