"""Tests for the data pipeline."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sample.generate_sample import generate_sample_data
from pipeline.preprocess import clean_data, encode_categoricals
from pipeline.features import (
    add_lag_features,
    add_rolling_features,
    add_trend_slope,
    add_injury_penalty,
    add_momentum_score,
)
from pipeline.dataset import NILTimeSeriesDataset


@pytest.fixture
def sample_df():
    return generate_sample_data(n_players=10, n_weeks=10)


class TestPreprocess:
    def test_clean_data_no_nulls(self, sample_df):
        cleaned = clean_data(sample_df)
        assert cleaned[["ppg", "apg", "rpg", "nil_valuation"]].isnull().sum().sum() == 0

    def test_clean_data_preserves_rows(self, sample_df):
        cleaned = clean_data(sample_df)
        assert len(cleaned) == len(sample_df)

    def test_encode_categoricals(self, sample_df):
        df, encoders = encode_categoricals(sample_df.copy())
        assert "conference_encoded" in df.columns
        assert "conference" in encoders

    def test_expected_columns(self, sample_df):
        expected = {
            "player_id", "snapshot_week", "sport", "school", "conference",
            "ppg", "apg", "rpg", "injury_flag", "games_played",
            "program_tier", "nil_valuation", "nil_tier",
        }
        assert expected.issubset(set(sample_df.columns))


class TestFeatures:
    def test_lag_features(self, sample_df):
        df = add_lag_features(sample_df.copy(), ["ppg"], lags=[1, 2])
        assert "ppg_lag1" in df.columns
        assert "ppg_lag2" in df.columns

    def test_rolling_features(self, sample_df):
        df = add_rolling_features(sample_df.copy(), ["ppg"], window=3)
        assert "ppg_roll3" in df.columns
        # Rolling mean should not exceed the range of the original
        df = df.dropna()
        assert df["ppg_roll3"].max() <= df["ppg"].max() + 1e-6

    def test_trend_slope(self, sample_df):
        df = add_trend_slope(sample_df.copy(), ["ppg"], window=4)
        assert "ppg_trend" in df.columns

    def test_injury_penalty(self, sample_df):
        df = add_injury_penalty(sample_df.copy(), window=4)
        assert "injury_penalty" in df.columns
        assert df["injury_penalty"].min() >= 0

    def test_momentum_score(self, sample_df):
        df = add_momentum_score(sample_df.copy(), ["ppg", "apg"])
        assert "momentum_score" in df.columns
        assert "ppg_delta" in df.columns


class TestDataset:
    def test_dataset_shapes(self, sample_df):
        df, _ = encode_categoricals(sample_df.copy())
        df = add_lag_features(df, ["ppg", "apg", "rpg"])
        df = add_rolling_features(df, ["ppg", "apg", "rpg"])
        df = add_trend_slope(df, ["ppg", "apg", "rpg"])
        df = add_injury_penalty(df)
        df = add_momentum_score(df, ["ppg", "apg", "rpg"])
        df = df.fillna(0)

        ds = NILTimeSeriesDataset(df, window_size=8)
        assert len(ds) > 0

        X, y_tier, y_val = ds[0]
        assert X.shape[0] == 8  # window size
        assert X.shape[1] == ds.n_features
        assert y_tier.dim() == 0  # scalar
        assert y_val.dim() == 0

    def test_dataset_tier_range(self, sample_df):
        df, _ = encode_categoricals(sample_df.copy())
        df = df.fillna(0)
        ds = NILTimeSeriesDataset(df, window_size=4)

        for i in range(min(10, len(ds))):
            _, y_tier, _ = ds[i]
            assert 0 <= y_tier.item() <= 4
