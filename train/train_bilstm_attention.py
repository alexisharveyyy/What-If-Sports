"""Train the BiLSTM + Attention multi-task NIL model.

Mirrors ``train/train_multitask_transformer.py`` so the comparison between
backbones is apples-to-apples: same stratified player-level split, same
``MultiTaskNILModel`` head wiring, same AdamW + cosine schedule, same AMP and
early stopping. Only the encoder differs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.lstm_model import BiLSTMWithAttention
from models.multitask_head import MultiTaskLoss, MultiTaskNILModel
from pipeline.preprocess import preprocess
from train.train_multitask_transformer import (
    _epoch,
    build_loaders as _build_transformer_loaders,
    evaluate,
    set_seed,
)


@dataclass
class BiLSTMTrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 7
    max_seq_len: int = 20
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.4
    attention_dim: int = 128
    alpha: float = 1.0
    beta: float = 1.0
    use_uncertainty_weighting: bool = False
    seed: int = 42
    train_frac: float = 0.7
    val_frac: float = 0.15
    use_amp: bool = True
    num_workers: int = 0
    save_dir: Path = _REPO_ROOT / "models" / "saved"
    checkpoint_name: str = "bilstm_attention_best.pt"
    report_name: str = "bilstm_attention_report.json"
    input_csv: Path | None = None


def _to_transformer_cfg_shim(cfg: BiLSTMTrainConfig):
    """Adapt our config into the loader signature used by the transformer
    trainer. The loader only reads max_seq_len/batch/num_workers/seed/splits.
    """
    class _Shim:
        pass
    s = _Shim()
    s.batch_size = cfg.batch_size
    s.num_workers = cfg.num_workers
    s.max_seq_len = cfg.max_seq_len
    s.seed = cfg.seed
    s.train_frac = cfg.train_frac
    s.val_frac = cfg.val_frac
    return s


def build_model(n_features: int, cfg: BiLSTMTrainConfig) -> MultiTaskNILModel:
    encoder = BiLSTMWithAttention(
        n_features=n_features,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        attention_dim=cfg.attention_dim,
    )
    return MultiTaskNILModel(encoder, d_model=cfg.hidden_dim)


def train(cfg: BiLSTMTrainConfig) -> dict:
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"BiLSTM+Attention config: {cfg}")

    df, _, _ = preprocess(input_path=cfg.input_csv, write=False)
    train_loader, val_loader, test_loader, feat_cols = _build_transformer_loaders(
        df, _to_transformer_cfg_shim(cfg)
    )

    n_features = len(feat_cols)
    model = build_model(n_features, cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = MultiTaskLoss(
        alpha=cfg.alpha,
        beta=cfg.beta,
        use_uncertainty_weighting=cfg.use_uncertainty_weighting,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(device.type) if cfg.use_amp and device.type == "cuda" else None

    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.save_dir / cfg.checkpoint_name
    report_path = cfg.save_dir / cfg.report_name

    best_val = math.inf
    epochs_since_improve = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _epoch(model, train_loader, loss_fn, optimizer, device,
                               scaler, cfg.grad_clip)
        val_metrics = _epoch(model, val_loader, loss_fn, None, device, None,
                             cfg.grad_clip)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": scheduler.get_last_lr()[0],
        })
        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if val_metrics["loss"] < best_val - 1e-6:
            best_val = val_metrics["loss"]
            epochs_since_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "n_features": n_features,
                "feature_cols": feat_cols,
                "val_loss": best_val,
            }, best_path)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    print(f"Best val loss: {best_val:.4f}")
    print(f"Saved checkpoint: {best_path}")

    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    print("Evaluating on test split...")
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"Test  acc={test_metrics['accuracy']:.4f}  macroF1={test_metrics['macro_f1']:.4f}  "
        f"MAE=${test_metrics['mae_usd']:,.0f}  RMSE=${test_metrics['rmse_usd']:,.0f}  "
        f"R2={test_metrics['r2']:.4f}"
    )

    report = {
        "config": cfg.__dict__,
        "best_val_loss": best_val,
        "test": {k: v for k, v in test_metrics.items()
                 if k not in ("pred_val", "true_val", "pred_tier", "true_tier")},
        "history": history,
    }
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote report to {report_path}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=BiLSTMTrainConfig.lr)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--input-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = BiLSTMTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        use_amp=not args.no_amp,
        input_csv=Path(args.input_csv) if args.input_csv else None,
    )
    train(cfg)


if __name__ == "__main__":
    main()
