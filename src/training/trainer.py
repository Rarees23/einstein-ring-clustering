from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from src.training.losses import get_loss_fn
from src.training.optimizers import build_optimizer


def train_autoencoder_with_early_stopping(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    num_epochs: int,
    best_model_path: str,
    *,
    learning_rate: float = 1e-3,
    optimizer_kind: str = "adam",
    loss_kind: str = "masked_mse",
    patience: int = 6,
    resume_checkpoint_path: str | None = None,
) -> tuple[float, list[float], list[float]]:
    """AE training with early stopping on validation loss. Batches are NCHW float tensors."""
    if resume_checkpoint_path:
        if not os.path.isfile(resume_checkpoint_path):
            raise FileNotFoundError(f"resume_checkpoint_path not found: {resume_checkpoint_path}")
        state = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"[train] Loaded weights from {resume_checkpoint_path!r}", flush=True)

    loss_fn = get_loss_fn(loss_kind)
    optimizer = build_optimizer(optimizer_kind, model.parameters(), lr=learning_rate)
    print(
        f"[train] optimizer={optimizer_kind} loss={loss_kind} lr={learning_rate} "
        f"early_stopping_patience={patience}",
        flush=True,
    )

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
            loss = loss_fn(model(x), x)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)

        train_loss = total / max(1, len(train_loader.dataset))
        history_train.append(float(train_loss))

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                vtotal += loss_fn(model(x), x).item() * x.size(0)
        val_loss = vtotal / max(1, len(val_loader.dataset))
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
