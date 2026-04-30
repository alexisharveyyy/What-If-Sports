"""Fit a temperature scalar on the validation set and save a calibrated checkpoint.

Usage:
    python train/calibrate.py --model lstm \
        --checkpoint models/saved/lstm_best.pt
"""

import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path (matches train/train.py convention)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.dataset import NILTimeSeriesDataset, split_by_player
from models.lstm_model import NILLSTMModel
from models.transformer_model import NILTransformerModel
from models.calibration import CalibratedNILModel


def _build_model(model_type: str, n_features: int, config: dict) -> torch.nn.Module:
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
    if model_type == "transformer":
        return NILTransformerModel(
            d_model=config["model"]["hidden_dim"],
            nhead=config["model"]["nhead"],
            num_layers=config["model"]["num_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            **kwargs,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True,
        help="Trained .pt checkpoint, e.g. models/saved/lstm_best.pt",
    )
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--feature_matrix", default="data/processed/feature_matrix.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--out", default=None,
        help="Output path; defaults to <checkpoint>.calibrated.pt",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint (matches train.py's torch.save(...) shape)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    n_features = ckpt["n_features"]

    model = _build_model(args.model, n_features, config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # Validation loader (same split logic as training)
    df = pd.read_csv(args.feature_matrix)
    _, val_df, _ = split_by_player(
        df,
        train_frac=config["training"]["train_split"],
        val_frac=config["training"]["val_split"],
    )
    val_ds = NILTimeSeriesDataset(val_df, window_size=config["data"]["window_size"])
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(f"Validation samples: {len(val_ds)}")

    # Fit T
    cal = CalibratedNILModel(model).to(device)
    T = cal.fit_temperature(val_loader, device)

    # Save calibrated checkpoint (existing fields plus the temperature scalar).
    out_path = args.out or args.checkpoint.replace(".pt", ".calibrated.pt")
    torch.save(
        {
            **ckpt,
            "temperature": T,
            "calibrated_from": args.checkpoint,
        },
        out_path,
    )
    print(f"Saved calibrated checkpoint -> {out_path}")


if __name__ == "__main__":
    main()