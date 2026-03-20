"""Model evaluation: classification metrics, regression metrics, calibration."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.dataset import NILTimeSeriesDataset, split_by_player
from models.lstm_model import NILLSTMModel
from models.transformer_model import NILTransformerModel


def expected_calibration_error(y_true, y_probs, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)

    return ece / len(y_true)


def evaluate(
    model_path: str,
    model_type: str = "lstm",
    feature_matrix_path: str = "data/processed/feature_matrix.csv",
):
    """Evaluate a saved model on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    n_features = checkpoint["n_features"]

    # Build model
    model_kwargs = dict(
        n_features=n_features,
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
    else:
        model = NILTransformerModel(
            d_model=config["model"]["hidden_dim"],
            nhead=config["model"]["nhead"],
            num_layers=config["model"]["num_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            **model_kwargs,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test data
    df = pd.read_csv(feature_matrix_path)
    _, _, test_df = split_by_player(df)
    test_ds = NILTimeSeriesDataset(test_df, window_size=config["data"]["window_size"])
    test_loader = DataLoader(test_ds, batch_size=64)

    all_tier_logits = []
    all_tier_targets = []
    all_value_preds = []
    all_value_targets = []

    with torch.no_grad():
        for X_seq, y_tier, y_value in test_loader:
            X_seq = X_seq.to(device)
            tier_logits, value_pred = model(X_seq)
            all_tier_logits.append(tier_logits.cpu())
            all_tier_targets.append(y_tier)
            all_value_preds.append(value_pred.cpu())
            all_value_targets.append(y_value)

    tier_logits = torch.cat(all_tier_logits)
    tier_targets = torch.cat(all_tier_targets).numpy()
    tier_probs = torch.softmax(tier_logits, dim=1).numpy()
    tier_preds = tier_probs.argmax(axis=1)

    value_preds = torch.cat(all_value_preds).numpy()
    value_targets = torch.cat(all_value_targets).numpy()

    # Classification metrics
    print("=" * 50)
    print("NIL Tier Classification")
    print("=" * 50)
    print(f"Accuracy:  {accuracy_score(tier_targets, tier_preds):.4f}")
    print(f"F1 Macro:  {f1_score(tier_targets, tier_preds, average='macro'):.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(tier_targets, tier_preds))

    # Calibration
    ece = expected_calibration_error(tier_targets, tier_probs)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")

    # Regression metrics
    print("\n" + "=" * 50)
    print("NIL Valuation Regression")
    print("=" * 50)
    print(f"MAE:   ${mean_absolute_error(value_targets, value_preds):,.0f}")
    print(f"RMSE:  ${root_mean_squared_error(value_targets, value_preds):,.0f}")
    print(f"R²:    {r2_score(value_targets, value_preds):.4f}")

    # Volatility robustness: evaluate on high-variance players
    print("\n" + "=" * 50)
    print("Volatility Robustness (high-variance players)")
    print("=" * 50)
    player_std = df.groupby("player_id")["nil_valuation"].std()
    high_var_threshold = player_std.median()
    high_var_players = player_std[player_std > high_var_threshold].index.tolist()

    hv_test_df = test_df[test_df["player_id"].isin(high_var_players)]
    if len(hv_test_df) > 0:
        hv_ds = NILTimeSeriesDataset(hv_test_df, window_size=config["data"]["window_size"])
        hv_loader = DataLoader(hv_ds, batch_size=64)

        hv_tier_logits, hv_tier_targets, hv_val_preds, hv_val_targets = [], [], [], []
        with torch.no_grad():
            for X_seq, y_tier, y_value in hv_loader:
                tier_logits, value_pred = model(X_seq.to(device))
                hv_tier_logits.append(tier_logits.cpu())
                hv_tier_targets.append(y_tier)
                hv_val_preds.append(value_pred.cpu())
                hv_val_targets.append(y_value)

        hv_tier_preds = torch.cat(hv_tier_logits).argmax(dim=1).numpy()
        hv_tier_targets = torch.cat(hv_tier_targets).numpy()
        hv_val_preds = torch.cat(hv_val_preds).numpy()
        hv_val_targets = torch.cat(hv_val_targets).numpy()

        print(f"Accuracy:  {accuracy_score(hv_tier_targets, hv_tier_preds):.4f}")
        print(f"MAE:       ${mean_absolute_error(hv_val_targets, hv_val_preds):,.0f}")
    else:
        print("No high-variance players in test set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to saved model checkpoint")
    parser.add_argument("--model_type", choices=["lstm", "transformer"], default="lstm")
    args = parser.parse_args()

    evaluate(model_path=args.model_path, model_type=args.model_type)
