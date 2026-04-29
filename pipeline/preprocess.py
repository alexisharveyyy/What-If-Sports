"""Preprocess the synthetic NIL evaluation dataset.

Reads ``data/raw/nil_evaluations_2025.csv`` (or the smaller sample), fits label
encoders for the categorical columns, fits a ``StandardScaler`` over the
numeric features, and persists both objects to ``pipeline/encoders.pkl`` and
``pipeline/scaler.pkl``. The processed snapshot DataFrame is also written to
``data/processed/player_snapshots.csv`` for downstream feature engineering.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

_REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORTS))

from pipeline.features import (
    CATEGORICAL_COLS,
    NUMERIC_FEATURE_COLS,
    TIER_LABEL_TO_INT,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = _REPO_ROOT / "data" / "raw" / "nil_evaluations_2025.csv"
SAMPLE_PATH = _REPO_ROOT / "data" / "sample" / "nil_evaluations_sample.csv"
PROCESSED_PATH = _REPO_ROOT / "data" / "processed" / "player_snapshots.csv"
SCALER_PATH = _REPO_ROOT / "pipeline" / "scaler.pkl"
ENCODERS_PATH = _REPO_ROOT / "pipeline" / "encoders.pkl"


def load_nil_dataset(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Load the NIL evaluation CSV. Falls back to the sample if raw is absent."""
    if path is not None:
        return pd.read_csv(path)
    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH)
    if SAMPLE_PATH.exists():
        print(f"Raw dataset missing; using sample at {SAMPLE_PATH}")
        return pd.read_csv(SAMPLE_PATH)
    raise FileNotFoundError(
        "Run pipeline/generate_nil_dataset.py to create the dataset first."
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, fill, and coerce dtypes."""
    df = df.copy()
    df = df.sort_values(["player_id", "week_number"]).reset_index(drop=True)

    for col in NUMERIC_FEATURE_COLS:
        df[col] = df.groupby("player_id")[col].transform(
            lambda s: s.ffill().bfill()
        )
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df["currently_injured"] = df["currently_injured"].astype(bool).astype(int)
    df["games_played"] = df["games_played"].fillna(0).astype(int)
    df["program_tier"] = df["program_tier"].astype(int)
    df["week_number"] = df["week_number"].astype(int)
    df["nil_tier"] = df["nil_tier"].astype(str)
    return df


def fit_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    """Fit label encoders for the categorical UI fields."""
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str).values)
        encoders[col] = le
    return encoders


def apply_encoders(df: pd.DataFrame,
                   encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    df = df.copy()
    for col, le in encoders.items():
        df[f"{col}_encoded"] = le.transform(df[col].astype(str).values)
    df["nil_tier_int"] = df["nil_tier"].map(TIER_LABEL_TO_INT).astype(int)
    return df


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_FEATURE_COLS].values)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    df[NUMERIC_FEATURE_COLS] = scaler.transform(df[NUMERIC_FEATURE_COLS].values)
    return df


def preprocess(input_path: str | os.PathLike | None = None,
               write: bool = True) -> tuple[pd.DataFrame, dict, StandardScaler]:
    df = load_nil_dataset(input_path)
    df = clean(df)
    encoders = fit_encoders(df)
    df = apply_encoders(df, encoders)
    scaler = fit_scaler(df)
    df = apply_scaler(df, scaler)

    if write:
        ENCODERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ENCODERS_PATH.open("wb") as f:
            pickle.dump(encoders, f)
        with SCALER_PATH.open("wb") as f:
            pickle.dump(scaler, f)
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"Processed {len(df):,} rows for {df['player_id'].nunique():,} players")
        print(f"  encoders -> {ENCODERS_PATH}")
        print(f"  scaler   -> {SCALER_PATH}")
        print(f"  snapshot -> {PROCESSED_PATH}")

    return df, encoders, scaler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None,
                        help="Override the input CSV path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess(input_path=args.input)
