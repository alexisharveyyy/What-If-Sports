"""Data cleaning and feature engineering pipeline."""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_raw_data(
    valuations_path: str = "data/raw/on3_valuations.csv",
    stats_path: str = "data/raw/player_stats.csv",
    sample_path: str = "data/sample/sample_players.csv",
) -> pd.DataFrame:
    """Load and merge raw data. Falls back to sample data if raw doesn't exist."""
    if os.path.exists(valuations_path) and os.path.exists(stats_path):
        valuations = pd.read_csv(valuations_path)
        stats = pd.read_csv(stats_path)
        df = valuations.merge(stats, on=["player_id", "snapshot_week"], how="left")
    elif os.path.exists(sample_path):
        print(f"Raw data not found. Using sample data from {sample_path}")
        df = pd.read_csv(sample_path)
    else:
        raise FileNotFoundError("No data found. Run generate_sample.py first.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and basic cleaning."""
    # Sort by player and week for proper forward-fill
    df = df.sort_values(["player_id", "snapshot_week"]).reset_index(drop=True)

    # Forward-fill within each player's timeline
    numeric_cols = ["ppg", "apg", "rpg", "nil_valuation"]
    for col in numeric_cols:
        df[col] = df.groupby("player_id")[col].transform(
            lambda s: s.ffill().fillna(s.median())
        )

    # Fill remaining NaN with column medians
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Ensure injury_flag is int
    df["injury_flag"] = df["injury_flag"].fillna(0).astype(int)
    df["games_played"] = df["games_played"].fillna(1).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns."""
    encoders = {}
    for col in ["sport", "conference"]:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def scale_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Normalize numeric features with StandardScaler."""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def preprocess(sample_path: str = "data/sample/sample_players.csv") -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    df = load_raw_data(sample_path=sample_path)
    df = clean_data(df)
    df, encoders = encode_categoricals(df)

    feature_cols = ["ppg", "apg", "rpg", "games_played", "program_tier"]
    df, scaler = scale_features(df, feature_cols)

    # Save scaler
    os.makedirs("pipeline", exist_ok=True)
    with open("pipeline/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("pipeline/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # Save processed output
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/player_snapshots.csv", index=False)
    print(f"Preprocessed {len(df)} rows → data/processed/player_snapshots.csv")

    return df


if __name__ == "__main__":
    preprocess()
