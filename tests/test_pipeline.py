"""Tests for the synthetic NIL dataset generator and the preprocessing pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.dataset import DEFAULT_MAX_SEQ_LEN, NILSequenceDataset
from pipeline.features import (
    NUMERIC_FEATURE_COLS,
    SEQUENTIAL_FEATURE_COLS,
    TIER_LABELS,
)
from pipeline.generate_nil_dataset import generate_dataset
from pipeline.preprocess import apply_encoders, apply_scaler, clean, fit_encoders, fit_scaler


REQUIRED_COLUMNS = {
    "player_id", "player_name", "school", "conference", "program_tier",
    "position", "class_year", "week_number", "game_date", "games_played",
    "currently_injured", "ppg", "apg", "rpg", "spg", "bpg", "mpg",
    "fg_pct", "three_pt_pct", "ft_pct", "social_media_followers",
    "engagement_rate", "market_size_score", "performance_score",
    "nil_valuation_usd", "nil_probability_score", "nil_tier",
}


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory) -> pd.DataFrame:
    out_dir = tmp_path_factory.mktemp("nil")
    return generate_dataset(n_players=25, seed=7, output_dir=str(out_dir))


class TestDatasetGeneration:
    def test_schema_is_complete(self, small_dataset):
        assert REQUIRED_COLUMNS.issubset(set(small_dataset.columns))

    def test_player_count(self, small_dataset):
        assert small_dataset["player_id"].nunique() == 25

    def test_row_count_matches_player_weeks(self, small_dataset):
        per_player = small_dataset.groupby("player_id")["week_number"].nunique()
        assert (per_player >= 1).all()
        assert (per_player <= DEFAULT_MAX_SEQ_LEN).all()
        assert len(small_dataset) == int(per_player.sum())

    def test_tier_labels_are_valid(self, small_dataset):
        assert set(small_dataset["nil_tier"].unique()).issubset(set(TIER_LABELS))

    def test_valuation_is_positive(self, small_dataset):
        assert (small_dataset["nil_valuation_usd"] > 0).all()


class TestPreprocess:
    def test_pipeline_round_trip(self, small_dataset):
        df = clean(small_dataset)
        encoders = fit_encoders(df)
        df = apply_encoders(df, encoders)
        scaler = fit_scaler(df)
        scaled = apply_scaler(df, scaler)

        assert "nil_tier_int" in scaled.columns
        for cat in ("school", "conference", "position", "class_year", "nil_tier"):
            assert f"{cat}_encoded" in scaled.columns
        assert scaled[NUMERIC_FEATURE_COLS].std().lt(5).all()


class TestSequenceDataset:
    def test_sequences_pad_to_twenty(self, small_dataset):
        df = clean(small_dataset)
        encoders = fit_encoders(df)
        df = apply_encoders(df, encoders)
        df = apply_scaler(df, fit_scaler(df))

        ds = NILSequenceDataset(df, max_seq_len=20)
        assert len(ds) == 25

        for i in range(len(ds)):
            sample = ds[i]
            assert sample["features"].shape[0] == 20
            assert sample["mask"].shape[0] == 20
            assert sample["features"].shape[1] == ds.n_features
            length = int(sample["length"].item())
            assert int(sample["mask"].sum().item()) == length
            assert sample["features"][length:].abs().sum().item() == 0.0
            assert 0 <= int(sample["tier"].item()) < len(TIER_LABELS)
            assert sample["valuation"].item() > 0

    def test_truncation_for_oversized_sequences(self):
        rows = []
        for week in range(1, 31):
            rows.append({
                "player_id": "P-test",
                "week_number": week,
                "nil_tier_int": 2,
                "nil_valuation_usd": 100_000.0,
                **{c: float(week) for c in NUMERIC_FEATURE_COLS},
                "currently_injured": 0,
                "program_tier": 3,
                "school_encoded": 1,
                "conference_encoded": 1,
                "position_encoded": 1,
                "class_year_encoded": 1,
            })
        df = pd.DataFrame(rows)
        ds = NILSequenceDataset(df, max_seq_len=20)
        sample = ds[0]
        assert sample["features"].shape == (20, ds.n_features)
        assert int(sample["length"].item()) == 20
