"""Core what-if simulation logic."""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import NILLSTMModel
from models.transformer_model import NILTransformerModel
from pipeline.dataset import FEATURE_COLS
from pipeline.features import (
    add_injury_penalty,
    add_lag_features,
    add_momentum_score,
    add_rolling_features,
    add_trend_slope,
)


class WhatIfSimulator:
    """Runs what-if simulations using a trained NIL prediction model."""

    def __init__(self, model_path: str | None = None, model_type: str = "lstm"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = None
        self.feature_cols = None
        self.scaler = None
        self.encoders = None
        self.model_type = model_type

        if model_path:
            self.load_model(model_path)

        # Load scaler/encoders if available
        if os.path.exists("pipeline/scaler.pkl"):
            with open("pipeline/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
        if os.path.exists("pipeline/encoders.pkl"):
            with open("pipeline/encoders.pkl", "rb") as f:
                self.encoders = pickle.load(f)

    def load_model(self, model_path: str):
        """Load a saved PyTorch model checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        n_features = checkpoint["n_features"]
        self.feature_cols = checkpoint.get("feature_cols", FEATURE_COLS[:n_features])

        model_kwargs = dict(
            n_features=n_features,
            num_tiers=self.config["model"]["num_tiers"],
            alpha=self.config["multitask"]["alpha"],
        )

        if self.model_type == "lstm":
            self.model = NILLSTMModel(
                hidden_dim=self.config["model"]["hidden_dim"],
                num_layers=self.config["model"]["num_layers"],
                dropout=0.0,  # No dropout at inference
                **model_kwargs,
            )
        else:
            self.model = NILTransformerModel(
                d_model=self.config["model"]["hidden_dim"],
                nhead=self.config["model"]["nhead"],
                num_layers=self.config["model"]["num_layers"],
                dim_feedforward=self.config["model"]["dim_feedforward"],
                dropout=0.0,
                **model_kwargs,
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def build_snapshot(self, user_inputs: dict) -> pd.DataFrame:
        """Convert user form inputs to a single-row DataFrame."""
        row = {
            "player_id": user_inputs.get("player_id", "USER_001"),
            "snapshot_week": user_inputs.get("snapshot_week", 1),
            "sport": user_inputs.get("sport", "basketball"),
            "school": user_inputs.get("school", "Unknown"),
            "conference": user_inputs.get("conference", "Unknown"),
            "ppg": user_inputs.get("ppg", 0),
            "apg": user_inputs.get("apg", 0),
            "rpg": user_inputs.get("rpg", 0),
            "spg": user_inputs.get("spg", 0),
            "bpg": user_inputs.get("bpg", 0),
            "mpg": user_inputs.get("mpg", 0),
            "fg_pct": user_inputs.get("fg_pct", 0),
            "three_pt_pct": user_inputs.get("three_pt_pct", 0),
            "ft_pct": user_inputs.get("ft_pct", 0),
            "injury_flag": int(user_inputs.get("injury_flag", False)),
            "games_played": user_inputs.get("games_played", 1),
            "program_tier": user_inputs.get("program_tier", 3),
            "nil_valuation": 0,
            "nil_tier": 1,
        }

        # Encode categoricals
        if self.encoders:
            for col, le in self.encoders.items():
                val = row.get(col, "Unknown")
                if val in le.classes_:
                    row[col + "_encoded"] = le.transform([val])[0]
                else:
                    row[col + "_encoded"] = 0
        else:
            row["conference_encoded"] = 0

        return pd.DataFrame([row])

    def _prepare_sequence(self, history_df: pd.DataFrame) -> torch.Tensor:
        """Prepare a feature sequence tensor from player history."""
        # Engineer features
        df = history_df.copy()
        stat_cols = ["ppg", "apg", "rpg", "spg", "bpg", "mpg"]
        df = add_lag_features(df, stat_cols)
        df = add_rolling_features(df, stat_cols)
        df = add_trend_slope(df, stat_cols)
        df = add_injury_penalty(df)
        df = add_momentum_score(df, stat_cols)
        df = df.fillna(0)

        window_size = self.config["data"]["window_size"] if self.config else 8
        available_cols = [c for c in self.feature_cols if c in df.columns]

        features = df[available_cols].values.astype(np.float32)

        # Pad or truncate to window_size
        if len(features) < window_size:
            pad = np.zeros((window_size - len(features), features.shape[1]), dtype=np.float32)
            features = np.vstack([pad, features])
        else:
            features = features[-window_size:]

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, window, features)

    def simulate(self, player_history: list[dict], new_snapshot: dict) -> dict:
        """Run a single what-if simulation.

        Args:
            player_history: List of prior snapshot dicts.
            new_snapshot: The new hypothetical snapshot to evaluate.

        Returns:
            Dict with nil_tier_probs, nil_valuation, and direction.
        """
        # Build full history
        history_dfs = [self.build_snapshot(s) for s in player_history]
        new_df = self.build_snapshot(new_snapshot)
        full_df = pd.concat(history_dfs + [new_df], ignore_index=True)
        full_df["snapshot_week"] = range(1, len(full_df) + 1)

        X_seq = self._prepare_sequence(full_df).to(self.device)

        with torch.no_grad():
            tier_logits, value_pred = self.model(X_seq)
            tier_probs = torch.softmax(tier_logits, dim=1).squeeze().cpu().numpy()
            valuation = value_pred.item()

        # Determine direction
        if len(player_history) > 0:
            prev_val = player_history[-1].get("nil_valuation", valuation)
            pct_change = (valuation - prev_val) / max(prev_val, 1) * 100
            if pct_change > 5:
                direction = "up"
            elif pct_change < -5:
                direction = "down"
            else:
                direction = "stable"
        else:
            direction = "stable"

        return {
            "nil_tier_probs": tier_probs.tolist(),
            "nil_valuation": round(max(0, valuation), 2),
            "direction": direction,
        }

    def simulate_timeline(
        self, base_profile: dict, weeks: int = 4
    ) -> list[dict]:
        """Run multi-week simulation by auto-advancing snapshots."""
        timeline = []
        history = []

        for week in range(1, weeks + 1):
            snapshot = {**base_profile, "snapshot_week": week}

            # Add small random drift to stats for future weeks
            if week > 1:
                for stat in ["ppg", "apg", "rpg", "spg", "bpg", "mpg"]:
                    drift = np.random.normal(0, 0.5)
                    snapshot[stat] = max(0, snapshot.get(stat, 0) + drift)

            result = self.simulate(history, snapshot)
            result["week"] = week
            timeline.append(result)
            history.append(snapshot)

        return timeline
