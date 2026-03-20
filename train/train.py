"""Training loop with early stopping for LSTM and Transformer models."""

import argparse
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.dataset import NILTimeSeriesDataset, split_by_player
from models.lstm_model import NILLSTMModel
from models.transformer_model import NILTransformerModel


def load_config(config_path: str = "train/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for X_seq, y_tier, y_value in dataloader:
        X_seq = X_seq.to(device)
        y_tier = y_tier.to(device)
        y_value = y_value.to(device)

        optimizer.zero_grad()
        tier_logits, value_pred = model(X_seq)
        loss, _ = model.head.compute_loss(tier_logits, value_pred, y_tier, y_value)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    n_batches = 0

    for X_seq, y_tier, y_value in dataloader:
        X_seq = X_seq.to(device)
        y_tier = y_tier.to(device)
        y_value = y_value.to(device)

        tier_logits, value_pred = model(X_seq)
        loss, _ = model.head.compute_loss(tier_logits, value_pred, y_tier, y_value)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    model_type: str = "lstm",
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    config_path: str = "train/config.yaml",
):
    config = load_config(config_path)

    epochs = epochs or config["training"]["epochs"]
    batch_size = batch_size or config["training"]["batch_size"]
    lr = lr or config["training"]["lr"]
    window_size = config["data"]["window_size"]
    patience = config["training"]["patience"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    feature_path = config["data"]["processed_path"]
    if not os.path.exists(feature_path):
        print(f"Feature matrix not found at {feature_path}. Run pipeline first.")
        return

    df = pd.read_csv(feature_path)
    train_df, val_df, test_df = split_by_player(
        df,
        train_frac=config["training"]["train_split"],
        val_frac=config["training"]["val_split"],
    )

    train_ds = NILTimeSeriesDataset(train_df, window_size=window_size)
    val_ds = NILTimeSeriesDataset(val_df, window_size=window_size)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Features: {train_ds.n_features}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Build model
    model_kwargs = dict(
        n_features=train_ds.n_features,
        num_tiers=config["model"]["num_tiers"],
        alpha=config["multitask"]["alpha"],
    )

    if model_type == "lstm":
        model = NILLSTMModel(
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            **model_kwargs,
        )
    elif model_type == "transformer":
        model = NILTransformerModel(
            d_model=config["model"]["hidden_dim"],
            nhead=config["model"]["nhead"],
            num_layers=config["model"]["num_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            **model_kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    print(f"\nModel: {model_type} ({sum(p.numel() for p in model.parameters()):,} parameters)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop with early stopping
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    os.makedirs("models/saved", exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint_path = f"models/saved/{model_type}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "config": config,
                "n_features": train_ds.n_features,
                "feature_cols": train_ds.feature_cols,
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

    # Save final versioned checkpoint
    ts = time.strftime("%Y%m%d_%H%M%S")
    final_path = f"models/saved/{model_type}_epoch{epoch}_val{best_val_loss:.3f}_{ts}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": best_val_loss,
        "config": config,
        "n_features": train_ds.n_features,
        "feature_cols": train_ds.feature_cols,
    }, final_path)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Saved: {checkpoint_path}")
    print(f"Saved: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    train(model_type=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
