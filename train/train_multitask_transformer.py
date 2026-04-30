"""Train the multi-task NIL transformer.

End-to-end training pipeline:
    1. Load ``data/raw/nil_evaluations_2025.csv``.
    2. Fit/load encoders + scaler via ``pipeline.preprocess``.
    3. Stratified split by ``player_id`` so no player leaks across splits.
    4. Wrap each split in ``NILSequenceDataset`` (pad/truncate to 20 weeks,
       attention mask, log1p target transform).
    5. Train ``MultiTaskNILModel`` with AdamW + cosine schedule, gradient
       clipping, mixed precision, and early stopping on validation loss.
    6. Save the best checkpoint to ``models/saved/multitask_transformer_best.pt``.
    7. Evaluate on the held-out test set; write metrics report and plots to
       ``models/saved/``.
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
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.multitask_head import MultiTaskLoss, MultiTaskNILModel
from models.transformer_model import NILTransformerEncoder
from pipeline.dataset import NILSequenceDataset, stratified_split_by_player
from pipeline.features import (
    SEQUENTIAL_FEATURE_COLS,
    TIER_INT_TO_LABEL,
    feature_columns,
)
from pipeline.preprocess import preprocess


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 7
    max_seq_len: int = 20
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    alpha: float = 1.0
    beta: float = 1.0
    use_uncertainty_weighting: bool = False
    seed: int = 42
    train_frac: float = 0.7
    val_frac: float = 0.15
    use_amp: bool = True
    num_workers: int = 0
    save_dir: Path = _REPO_ROOT / "models" / "saved"
    input_csv: Path | None = None
    checkpoint_name: str = "multitask_transformer_best.pt"
    report_name: str = "training_report.json"
    write_plots: bool = True


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(
    df: pd.DataFrame,
    cfg: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_df, val_df, test_df = stratified_split_by_player(
        df, train_frac=cfg.train_frac, val_frac=cfg.val_frac, seed=cfg.seed
    )

    feat_cols = feature_columns(list(df.columns))
    train_ds = NILSequenceDataset(train_df, max_seq_len=cfg.max_seq_len, feature_cols=feat_cols)
    val_ds = NILSequenceDataset(val_df, max_seq_len=cfg.max_seq_len, feature_cols=feat_cols)
    test_ds = NILSequenceDataset(test_df, max_seq_len=cfg.max_seq_len, feature_cols=feat_cols)

    print(
        f"Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}; "
        f"features={len(feat_cols)}"
    )

    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  pin_memory=torch.cuda.is_available(), collate_fn=_collate)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader, feat_cols


def _collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "length": torch.stack([b["length"] for b in batch]),
        "tier": torch.stack([b["tier"] for b in batch]),
        "valuation": torch.stack([b["valuation"] for b in batch]),
        "player_id": [b["player_id"] for b in batch],
    }


def build_model(n_features: int, cfg: TrainConfig) -> MultiTaskNILModel:
    encoder = NILTransformerEncoder(
        n_features=n_features,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
    )
    return MultiTaskNILModel(encoder, d_model=cfg.d_model)


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: MultiTaskLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: GradScaler | None,
    grad_clip: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"loss": 0.0, "cls": 0.0, "reg": 0.0, "n": 0}
    for batch in loader:
        x = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        tier = batch["tier"].to(device, non_blocking=True)
        valuation = batch["valuation"].to(device, non_blocking=True)
        log_val = torch.log1p(valuation.clamp(min=0.0))

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            with autocast(device_type=device.type, enabled=scaler is not None):
                out = model(x, mask=mask)
                loss, metrics = loss_fn(
                    out["tier_logits"],
                    out["valuation_pred"],
                    tier,
                    log_val,
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        bs = tier.size(0)
        totals["loss"] += metrics["total_loss"] * bs
        totals["cls"] += metrics["cls_loss"] * bs
        totals["reg"] += metrics["reg_loss"] * bs
        totals["n"] += bs

    n = max(totals["n"], 1)
    return {
        "loss": totals["loss"] / n,
        "cls_loss": totals["cls"] / n,
        "reg_loss": totals["reg"] / n,
    }


@torch.no_grad()
def evaluate(
    model: MultiTaskNILModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        precision_recall_fscore_support,
        r2_score,
        root_mean_squared_error,
    )

    model.eval()
    all_logits: list[np.ndarray] = []
    all_tiers: list[np.ndarray] = []
    all_pred_log: list[np.ndarray] = []
    all_true_val: list[np.ndarray] = []

    for batch in loader:
        x = batch["features"].to(device)
        mask = batch["mask"].to(device)
        out = model(x, mask=mask)
        all_logits.append(out["tier_logits"].cpu().numpy())
        all_tiers.append(batch["tier"].cpu().numpy())
        all_pred_log.append(out["valuation_pred"].cpu().numpy())
        all_true_val.append(batch["valuation"].cpu().numpy())

    logits = np.concatenate(all_logits)
    tiers = np.concatenate(all_tiers)
    pred_log = np.concatenate(all_pred_log)
    true_val = np.concatenate(all_true_val)

    pred_tier = logits.argmax(axis=1)
    pred_val = np.expm1(pred_log)

    accuracy = float(accuracy_score(tiers, pred_tier))
    macro_f1 = float(f1_score(tiers, pred_tier, average="macro", zero_division=0))
    precision, recall, _, _ = precision_recall_fscore_support(
        tiers, pred_tier, average=None, labels=list(range(5)), zero_division=0
    )
    cm = confusion_matrix(tiers, pred_tier, labels=list(range(5)))

    mae = float(mean_absolute_error(true_val, pred_val))
    rmse = float(root_mean_squared_error(true_val, pred_val))
    r2 = float(r2_score(true_val, pred_val))

    composite = 0.5 * macro_f1 + 0.5 * max(r2, 0.0)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_tier_precision": {TIER_INT_TO_LABEL[i]: float(p) for i, p in enumerate(precision)},
        "per_tier_recall": {TIER_INT_TO_LABEL[i]: float(r) for i, r in enumerate(recall)},
        "confusion_matrix": cm.tolist(),
        "mae_usd": mae,
        "rmse_usd": rmse,
        "r2": r2,
        "composite": composite,
        "pred_tier": pred_tier.tolist(),
        "true_tier": tiers.tolist(),
        "pred_val": pred_val.tolist(),
        "true_val": true_val.tolist(),
    }


def _save_plots(metrics: dict, residual_groupings: pd.DataFrame, save_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    labels = [TIER_INT_TO_LABEL[i] for i in range(5)]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_title("NIL tier confusion matrix")
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    pred_val = np.asarray(metrics["pred_val"])
    true_val = np.asarray(metrics["true_val"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_val, pred_val, s=6, alpha=0.4)
    lim = max(true_val.max(), pred_val.max())
    ax.plot([0, lim], [0, lim], color="red", lw=1)
    ax.set_xlabel("Actual NIL valuation (USD)")
    ax.set_ylabel("Predicted NIL valuation (USD)")
    ax.set_title("Predicted vs actual NIL valuation")
    fig.tight_layout()
    fig.savefig(save_dir / "pred_vs_actual.png", dpi=150)
    plt.close(fig)

    if not residual_groupings.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, group_col in zip(axes, ("program_tier", "conference")):
            grouped = residual_groupings.groupby(group_col)["residual"]
            ax.boxplot(
                [g.values for _, g in grouped],
                tick_labels=[str(k) for k in grouped.groups.keys()],
                showfliers=False,
            )
            ax.axhline(0, color="red", lw=0.8)
            ax.set_title(f"Residuals by {group_col}")
            ax.set_ylabel("residual (USD)")
            for tick in ax.get_xticklabels():
                tick.set_rotation(40)
                tick.set_ha("right")
        fig.tight_layout()
        fig.savefig(save_dir / "residuals_by_group.png", dpi=150)
        plt.close(fig)


def _attention_visualization(
    model: MultiTaskNILModel,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    n_examples: int = 8,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    seen = 0
    rows: list[np.ndarray] = []
    pids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            mask = batch["mask"].to(device)
            _ = model(x, mask=mask, return_attention=True)
            attn = model.encoder.last_attention.cpu().numpy()
            for i in range(attn.shape[0]):
                if seen >= n_examples:
                    break
                rows.append(attn[i, 1:])
                pids.append(batch["player_id"][i])
                seen += 1
            if seen >= n_examples:
                break

    if not rows:
        return

    arr = np.stack(rows)
    fig, ax = plt.subplots(figsize=(8, max(3, n_examples * 0.4)))
    im = ax.imshow(arr, cmap="viridis", aspect="auto")
    ax.set_xlabel("week index (0 = earliest)")
    ax.set_ylabel("example")
    ax.set_yticks(range(len(pids)))
    ax.set_yticklabels([p[:8] for p in pids])
    ax.set_title("CLS attention weights over weekly snapshots")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(save_dir / "attention_weights.png", dpi=150)
    plt.close(fig)


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {cfg}")

    df, _, _ = preprocess(input_path=cfg.input_csv, write=False)
    train_loader, val_loader, test_loader, feat_cols = build_loaders(df, cfg)

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
        f"R2={test_metrics['r2']:.4f}  composite={test_metrics['composite']:.4f}"
    )

    test_player_meta = (
        df.sort_values("week_number").groupby("player_id", sort=False).tail(1)
        [["player_id", "program_tier", "conference"]]
    )

    pred_val_arr = np.asarray(test_metrics["pred_val"])
    true_val_arr = np.asarray(test_metrics["true_val"])
    test_player_ids = []
    for batch in test_loader:
        test_player_ids.extend(batch["player_id"])
    residual_df = pd.DataFrame({
        "player_id": test_player_ids,
        "residual": pred_val_arr - true_val_arr,
    }).merge(test_player_meta, on="player_id", how="left")

    if cfg.write_plots:
        _save_plots(test_metrics, residual_df, cfg.save_dir)
        _attention_visualization(model, test_loader, device, cfg.save_dir)

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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--uncertainty-weighting", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--input-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        alpha=args.alpha,
        beta=args.beta,
        use_uncertainty_weighting=args.uncertainty_weighting,
        use_amp=not args.no_amp,
        input_csv=Path(args.input_csv) if args.input_csv else None,
    )
    train(cfg)


if __name__ == "__main__":
    main()
