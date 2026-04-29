"""Feature column definitions plus legacy lag/rolling helpers.

The column lists below are the canonical contract between ``preprocess.py``,
``dataset.py``, and ``train/``. The helper functions at the bottom remain for
backward compatibility with the older LSTM training and baseline scripts that
operated on per-snapshot tabular features rather than raw sequences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress


IDENTITY_COLS: list[str] = [
    "player_id",
    "player_name",
    "school",
    "conference",
    "program_tier",
    "position",
    "class_year",
]


CATEGORICAL_COLS: list[str] = [
    "school",
    "conference",
    "position",
    "class_year",
    "nil_tier",
]


NUMERIC_FEATURE_COLS: list[str] = [
    "ppg",
    "apg",
    "rpg",
    "spg",
    "bpg",
    "mpg",
    "fg_pct",
    "three_pt_pct",
    "ft_pct",
    "games_played",
    "social_media_followers",
    "engagement_rate",
    "market_size_score",
    "performance_score",
]


SEQUENTIAL_FEATURE_COLS: list[str] = NUMERIC_FEATURE_COLS + [
    "currently_injured",
    "program_tier",
    "school_encoded",
    "conference_encoded",
    "position_encoded",
    "class_year_encoded",
    "week_number",
]


TARGET_COLS: list[str] = [
    "nil_tier_int",
    "nil_valuation_usd",
]


TIER_LABELS: list[str] = ["developmental", "low", "mid", "high", "elite"]
TIER_LABEL_TO_INT: dict[str, int] = {label: i for i, label in enumerate(TIER_LABELS)}
TIER_INT_TO_LABEL: dict[int, str] = {i: label for i, label in enumerate(TIER_LABELS)}


def feature_columns(df_columns: list[str]) -> list[str]:
    """Return the subset of ``SEQUENTIAL_FEATURE_COLS`` actually present."""
    return [c for c in SEQUENTIAL_FEATURE_COLS if c in df_columns]


# ---------------------------------------------------------------------------
# Legacy helpers (used by baseline.py and the older LSTM training script).
# ---------------------------------------------------------------------------


def add_lag_features(df: pd.DataFrame, cols: list[str],
                     lags: list[int] = [1, 2, 3]) -> pd.DataFrame:
    """Add lag features for specified columns within each player."""
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("player_id")[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, cols: list[str],
                         window: int = 3) -> pd.DataFrame:
    for col in cols:
        df[f"{col}_roll{window}"] = (
            df.groupby("player_id")[col]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
    return df


def add_trend_slope(df: pd.DataFrame, cols: list[str],
                    window: int = 4) -> pd.DataFrame:
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
    flag_col = "currently_injured" if "currently_injured" in df.columns else "injury_flag"
    df["injury_penalty"] = (
        df.groupby("player_id")[flag_col]
        .transform(lambda s: s.astype(int).rolling(window, min_periods=1).sum())
    )
    return df


def add_momentum_score(df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    for col in stat_cols:
        df[f"{col}_delta"] = df.groupby("player_id")[col].diff().fillna(0)
    delta_cols = [f"{col}_delta" for col in stat_cols]
    df["momentum_score"] = df[delta_cols].sum(axis=1)
    return df
