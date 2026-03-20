"""Lag features, rolling windows, and trend extraction."""

import numpy as np
import pandas as pd
from scipy.stats import linregress


def add_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int] = [1, 2, 3]) -> pd.DataFrame:
    """Add lag features for specified columns within each player."""
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("player_id")[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, cols: list[str], window: int = 3) -> pd.DataFrame:
    """Add rolling mean features within each player."""
    for col in cols:
        df[f"{col}_roll{window}"] = (
            df.groupby("player_id")[col]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
    return df


def add_trend_slope(df: pd.DataFrame, cols: list[str], window: int = 4) -> pd.DataFrame:
    """Compute linear regression slope over the last `window` weeks per feature."""
    def slope(series):
        result = []
        for i in range(len(series)):
            start = max(0, i - window + 1)
            segment = series.iloc[start:i + 1].dropna()
            if len(segment) < 2:
                result.append(0.0)
            else:
                x = np.arange(len(segment))
                try:
                    result.append(linregress(x, segment.values).slope)
                except Exception:
                    result.append(0.0)
        return pd.Series(result, index=series.index)

    for col in cols:
        df[f"{col}_trend"] = df.groupby("player_id")[col].transform(slope)
    return df


def add_injury_penalty(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Cumulative injury weeks in last `window` snapshots."""
    df["injury_penalty"] = (
        df.groupby("player_id")["injury_flag"]
        .transform(lambda s: s.rolling(window, min_periods=1).sum())
    )
    return df


def add_momentum_score(df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    """Composite momentum: sum of deltas vs. prior snapshot."""
    for col in stat_cols:
        df[f"{col}_delta"] = df.groupby("player_id")[col].diff().fillna(0)

    delta_cols = [f"{col}_delta" for col in stat_cols]
    df["momentum_score"] = df[delta_cols].sum(axis=1)
    return df


def engineer_features(
    input_path: str = "data/processed/player_snapshots.csv",
    output_path: str = "data/processed/feature_matrix.csv",
) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = pd.read_csv(input_path)
    stat_cols = ["ppg", "apg", "rpg"]

    df = add_lag_features(df, stat_cols)
    df = add_rolling_features(df, stat_cols, window=3)
    df = add_trend_slope(df, stat_cols, window=4)
    df = add_injury_penalty(df, window=4)
    df = add_momentum_score(df, stat_cols)

    # Fill NaN from lag/rolling at start of sequences
    df = df.fillna(0)

    df.to_csv(output_path, index=False)
    print(f"Feature matrix: {df.shape[0]} rows, {df.shape[1]} columns → {output_path}")

    return df


if __name__ == "__main__":
    engineer_features()
