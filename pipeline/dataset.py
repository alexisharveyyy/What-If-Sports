"""PyTorch Dataset class for NIL time series snapshots."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Feature columns used as model input
FEATURE_COLS = [
    "ppg", "apg", "rpg", "games_played", "program_tier",
    "sport_encoded", "conference_encoded", "injury_flag",
    "ppg_lag1", "ppg_lag2", "ppg_lag3",
    "apg_lag1", "apg_lag2", "apg_lag3",
    "rpg_lag1", "rpg_lag2", "rpg_lag3",
    "ppg_roll3", "apg_roll3", "rpg_roll3",
    "ppg_trend", "apg_trend", "rpg_trend",
    "injury_penalty", "momentum_score",
]


class NILTimeSeriesDataset(Dataset):
    """Dataset that produces fixed-length sequences per player.

    Each sample is a window of `window_size` consecutive snapshots for a single player.
    Targets: nil_tier (int, 0-4) and nil_valuation (float).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 8,
        feature_cols: list[str] | None = None,
    ):
        self.window_size = window_size
        self.feature_cols = feature_cols or FEATURE_COLS

        # Only keep columns that exist in the DataFrame
        available_cols = [c for c in self.feature_cols if c in df.columns]
        self.feature_cols = available_cols

        self.samples = []
        self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame):
        """Extract sliding windows per player."""
        for player_id, group in df.groupby("player_id"):
            group = group.sort_values("snapshot_week").reset_index(drop=True)

            features = group[self.feature_cols].values.astype(np.float32)
            tiers = group["nil_tier"].values.astype(np.int64) - 1  # 0-indexed
            valuations = group["nil_valuation"].values.astype(np.float32)

            n = len(group)
            if n >= self.window_size:
                # Sliding windows
                for i in range(n - self.window_size + 1):
                    end = i + self.window_size
                    self.samples.append((
                        features[i:end],
                        tiers[end - 1],
                        valuations[end - 1],
                    ))
            else:
                # Pad shorter sequences from the left with zeros
                pad_len = self.window_size - n
                padded_features = np.vstack([
                    np.zeros((pad_len, features.shape[1]), dtype=np.float32),
                    features,
                ])
                self.samples.append((
                    padded_features,
                    tiers[-1],
                    valuations[-1],
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, tier, valuation = self.samples[idx]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(tier, dtype=torch.long),
            torch.tensor(valuation, dtype=torch.float32),
        )

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


def split_by_player(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by player_id to prevent data leakage."""
    rng = np.random.RandomState(seed)
    player_ids = df["player_id"].unique()
    rng.shuffle(player_ids)

    n = len(player_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = player_ids[:n_train]
    val_ids = player_ids[n_train:n_train + n_val]
    test_ids = player_ids[n_train + n_val:]

    return (
        df[df["player_id"].isin(train_ids)],
        df[df["player_id"].isin(val_ids)],
        df[df["player_id"].isin(test_ids)],
    )
