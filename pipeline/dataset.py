"""PyTorch Dataset for the multi-task NIL transformer.

Each sample is one player's full season as a fixed-length sequence of weekly
snapshots. Sequences shorter than ``max_seq_len`` are zero-padded on the right,
and an attention-mask vector is returned alongside so the encoder can ignore
padded positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pipeline.features import (
    NUMERIC_FEATURE_COLS,
    SEQUENTIAL_FEATURE_COLS,
    TIER_LABEL_TO_INT,
    feature_columns,
)


# Re-exported so older code paths (baseline.py, simulator) keep working.
FEATURE_COLS: list[str] = NUMERIC_FEATURE_COLS

DEFAULT_MAX_SEQ_LEN: int = 20


class NILSequenceDataset(Dataset):
    """One sample per player. Returns padded sequences plus dual targets.

    The dataset assumes the input frame has already been preprocessed by
    ``pipeline.preprocess`` (encoded categoricals, scaled numerics, integer
    tier labels in ``nil_tier_int``). The regression target is the player's
    most-recent (largest ``week_number``) ``nil_valuation_usd``; the
    classification target is the corresponding ``nil_tier_int``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.feature_cols = feature_cols or feature_columns(list(df.columns))
        if not self.feature_cols:
            raise ValueError(
                "No usable feature columns found. Did you run preprocess()?"
            )

        self._samples: list[dict] = []
        self._build(df)

    def _build(self, df: pd.DataFrame) -> None:
        for player_id, group in df.groupby("player_id", sort=False):
            group = group.sort_values("week_number").reset_index(drop=True)
            seq = group[self.feature_cols].to_numpy(dtype=np.float32)

            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len:]
                last_row = group.iloc[-1]
            else:
                last_row = group.iloc[-1]

            length = seq.shape[0]
            padded = np.zeros(
                (self.max_seq_len, len(self.feature_cols)), dtype=np.float32
            )
            padded[:length] = seq

            mask = np.zeros(self.max_seq_len, dtype=np.bool_)
            mask[:length] = True

            self._samples.append({
                "player_id": str(player_id),
                "features": padded,
                "mask": mask,
                "length": int(length),
                "tier": int(last_row["nil_tier_int"]) if "nil_tier_int" in group.columns
                        else int(TIER_LABEL_TO_INT[str(last_row["nil_tier"])]),
                "valuation": float(last_row["nil_valuation_usd"]),
            })

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]
        return {
            "features": torch.from_numpy(sample["features"]),
            "mask": torch.from_numpy(sample["mask"]),
            "length": torch.tensor(sample["length"], dtype=torch.long),
            "tier": torch.tensor(sample["tier"], dtype=torch.long),
            "valuation": torch.tensor(sample["valuation"], dtype=torch.float32),
            "player_id": sample["player_id"],
        }

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)


# ---------------------------------------------------------------------------
# Backward-compatible classes used by the older LSTM training script.
# ---------------------------------------------------------------------------


class NILTimeSeriesDataset(Dataset):
    """Legacy sliding-window dataset. Retained for the LSTM training path."""

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 8,
        feature_cols: list[str] | None = None,
    ):
        self.window_size = window_size
        self.feature_cols = feature_cols or [
            c for c in NUMERIC_FEATURE_COLS if c in df.columns
        ]
        self.samples: list[tuple[np.ndarray, int, float]] = []
        self._build(df)

    def _build(self, df: pd.DataFrame) -> None:
        sort_col = "week_number" if "week_number" in df.columns else "snapshot_week"
        tier_col = "nil_tier_int" if "nil_tier_int" in df.columns else "nil_tier"
        val_col = "nil_valuation_usd" if "nil_valuation_usd" in df.columns else "nil_valuation"

        for player_id, group in df.groupby("player_id"):
            group = group.sort_values(sort_col).reset_index(drop=True)
            features = group[self.feature_cols].to_numpy(dtype=np.float32)
            if tier_col == "nil_tier":
                tiers = (group[tier_col].astype(int).values - 1).astype(np.int64)
            else:
                tiers = group[tier_col].astype(int).to_numpy()
            valuations = group[val_col].to_numpy(dtype=np.float32)

            n = len(group)
            if n >= self.window_size:
                for i in range(n - self.window_size + 1):
                    end = i + self.window_size
                    self.samples.append(
                        (features[i:end], int(tiers[end - 1]), float(valuations[end - 1]))
                    )
            else:
                pad_len = self.window_size - n
                padded = np.vstack([
                    np.zeros((pad_len, features.shape[1]), dtype=np.float32),
                    features,
                ])
                self.samples.append((padded, int(tiers[-1]), float(valuations[-1])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
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
    """Random player-level split. Use ``stratified_split_by_player`` for
    tier-balanced splits."""
    rng = np.random.RandomState(seed)
    player_ids = df["player_id"].unique()
    rng.shuffle(player_ids)

    n = len(player_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(player_ids[:n_train])
    val_ids = set(player_ids[n_train:n_train + n_val])
    test_ids = set(player_ids[n_train + n_val:])

    return (
        df[df["player_id"].isin(train_ids)].copy(),
        df[df["player_id"].isin(val_ids)].copy(),
        df[df["player_id"].isin(test_ids)].copy(),
    )


def stratified_split_by_player(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    tier_col: str = "nil_tier",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split players by ID stratified on each player's most recent tier label."""
    rng = np.random.RandomState(seed)
    sort_col = "week_number" if "week_number" in df.columns else "snapshot_week"
    last = (
        df.sort_values(sort_col)
        .groupby("player_id", sort=False)
        .tail(1)[["player_id", tier_col]]
    )

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for _, group in last.groupby(tier_col):
        ids = group["player_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    return (
        df[df["player_id"].isin(train_set)].copy(),
        df[df["player_id"].isin(val_set)].copy(),
        df[df["player_id"].isin(test_set)].copy(),
    )
