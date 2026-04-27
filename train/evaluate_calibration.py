"""Calibration & uncertainty evaluation for trained NIL models.

Produces:
- ECE before and after temperature scaling
- Reliability diagrams (PNG) before and after
- MC Dropout uncertainty for tier probabilities and valuation
- 95% prediction-interval coverage for the valuation regression head
- A per-sample CSV with predictions, CIs, and uncertainty
- A summary metrics CSV

Usage:
    python eval/evaluate_calibration.py \
        --checkpoint models/saved/lstm_best.pt \
        --model lstm \
        --output_dir reports/calibration
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.dataset import NILTimeSeriesDataset, split_by_player
from models.lstm_model import NILLSTMModel
from models.transformer_model import NILTransformerModel
from models.calibration import (
    CalibratedNILModel,
    expected_calibration_error,
    reliability_diagram,
    mc_dropout_predict,
    regression_interval_coverage,
)


def _build_model(model_type, n_features, config):
    kwargs = dict(
        n_features=n_features,
        num_tiers=config["model"]["num_tiers"],
        alpha=config["multitask"]["alpha"],
    )
    if model_type == "lstm":
        return NILLSTMModel(
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            **kwargs,
        )
    return NILTransformerModel(
        d_model=config["model"]["hidden_dim"],
        nhead=config["model"]["nhead"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        **kwargs,
    )


@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    logits_list, value_list, tier_y, value_y = [], [], [], []
    for X_seq, y_tier, y_value in loader:
        tier_logits, value_pred = model(X_seq.to(device))
        logits_list.append(tier_logits.cpu())
        value_list.append(value_pred.cpu())
        tier_y.append(y_tier)
        value_y.append(y_value)
    return (
        torch.cat(logits_list),
        torch.cat(value_list),
        torch.cat(tier_y).numpy(),
        torch.cat(value_y).numpy(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--feature_matrix", default="data/processed/feature_matrix.csv")
    parser.add_argument("--output_dir", default="reports/calibration")
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ---- Load model and data ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = _build_model(args.model, ckpt["n_features"], config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    df = pd.read_csv(args.feature_matrix)
    _, val_df, test_df = split_by_player(
        df,
        train_frac=config["training"]["train_split"],
        val_frac=config["training"]["val_split"],
    )
    val_ds = NILTimeSeriesDataset(val_df, window_size=config["data"]["window_size"])
    test_ds = NILTimeSeriesDataset(test_df, window_size=config["data"]["window_size"])
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    print(f"Val samples: {len(val_ds)}, Test samples: {len(test_ds)}\n")

    # ---- Calibration: ECE before and after temperature scaling ----
    test_logits, _test_values, tier_y, value_y = _collect(model, test_loader, device)
    probs_uncal = F.softmax(test_logits, dim=-1).numpy()
    ece_before = expected_calibration_error(tier_y, probs_uncal, n_bins=3)
    print("=" * 60)
    print("CALIBRATION (NIL tier classification)")
    print("=" * 60)
    print(f"ECE before temperature scaling: {ece_before:.4f}")
    reliability_diagram(
        tier_y, probs_uncal,
        n_bins=3,
        title=f"Before calibration (ECE={ece_before:.4f})",
        save_path=os.path.join(args.output_dir, f"{args.model}_reliability_before.png"),
    )

    cal = CalibratedNILModel(model).to(device)
    T = cal.fit_temperature(val_loader, device)
    probs_cal = F.softmax(test_logits / T, dim=-1).numpy()
    ece_after = expected_calibration_error(tier_y, probs_cal, n_bins=3)
    print(f"ECE after temperature scaling:  {ece_after:.4f}")
    reliability_diagram(
        tier_y, probs_cal,
        n_bins=3,
        title=f"After calibration (T={T:.3f}, ECE={ece_after:.4f})",
        save_path=os.path.join(args.output_dir, f"{args.model}_reliability_after.png"),
    )

    # ---- Uncertainty: MC Dropout on the test set ----
    print("\n" + "=" * 60)
    print(f"UNCERTAINTY (Monte Carlo Dropout, {args.mc_samples} samples)")
    print("=" * 60)
    tier_means, tier_stds = [], []
    val_means, val_stds, val_lo, val_hi = [], [], [], []
    for X_seq, _y_tier, _y_value in test_loader:
        out = mc_dropout_predict(model, X_seq.to(device), n_samples=args.mc_samples)
        tier_means.append(out["tier_mean"].cpu())
        tier_stds.append(out["tier_std"].cpu())
        val_means.append(out["value_mean"].cpu())
        val_stds.append(out["value_std"].cpu())
        val_lo.append(out["value_ci_low"].cpu())
        val_hi.append(out["value_ci_high"].cpu())

    tier_means = torch.cat(tier_means).numpy()
    tier_stds = torch.cat(tier_stds).numpy()
    val_means = torch.cat(val_means).numpy()
    val_stds = torch.cat(val_stds).numpy()
    val_lo = torch.cat(val_lo).numpy()
    val_hi = torch.cat(val_hi).numpy()

    pred_classes = tier_means.argmax(axis=1)
    pred_uncertainty = tier_stds[np.arange(len(pred_classes)), pred_classes]
    coverage = regression_interval_coverage(value_y, val_lo, val_hi)
    print(f"Mean predictive std on tier prob:    {pred_uncertainty.mean():.4f}")
    print(f"Mean valuation std (USD):            ${val_stds.mean():,.0f}")
    print(f"95% interval coverage on valuation:  {coverage:.3f}  (target ~ 0.95)")
    if coverage < 0.90:
        print("  -> Intervals are too narrow; the model is over-confident.")
    elif coverage > 0.99:
        print("  -> Intervals are too wide; the model is under-confident.")
    else:
        print("  -> Intervals are reasonably calibrated.")

    # ---- Per-sample CSV ----
    out_csv = os.path.join(args.output_dir, f"{args.model}_predictions.csv")
    pd.DataFrame({
        "true_tier": tier_y,
        "pred_tier": pred_classes,
        "pred_tier_prob": tier_means.max(axis=1),
        "pred_tier_prob_std": pred_uncertainty,
        "true_value": value_y,
        "pred_value": val_means,
        "pred_value_std": val_stds,
        "pred_value_ci_low": val_lo,
        "pred_value_ci_high": val_hi,
    }).to_csv(out_csv, index=False)
    print(f"\nPer-sample predictions -> {out_csv}")

    # ---- Summary metrics CSV ----
    summary = {
        "model": args.model,
        "temperature": float(T),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "valuation_interval_coverage_95": coverage,
        "mean_tier_prob_std": float(pred_uncertainty.mean()),
        "mean_valuation_std": float(val_stds.mean()),
    }
    summary_path = os.path.join(args.output_dir, f"{args.model}_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Summary metrics       -> {summary_path}")


if __name__ == "__main__":
    main()