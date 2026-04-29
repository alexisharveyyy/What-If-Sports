"""Side-by-side comparison of two simulated players' tier + valuation predictions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.features import NUMERIC_FEATURE_COLS, TIER_INT_TO_LABEL
from simulator.engine import WhatIfSimulator


class PlayerComparator:
    """Compare two simulated players' multi-task model outputs."""

    def __init__(self, simulator: WhatIfSimulator | None = None) -> None:
        self.simulator = simulator or WhatIfSimulator()

    def compare(
        self,
        player_a: dict,
        player_b: dict,
        history_a: list[dict] | None = None,
        history_b: list[dict] | None = None,
    ) -> dict:
        sim_a = self.simulator.simulate(history_a or [], player_a)
        sim_b = self.simulator.simulate(history_b or [], player_b)

        delta_val = sim_a["nil_valuation_usd"] - sim_b["nil_valuation_usd"]
        if sim_a["nil_tier_index"] > sim_b["nil_tier_index"]:
            tier_winner = "a"
        elif sim_b["nil_tier_index"] > sim_a["nil_tier_index"]:
            tier_winner = "b"
        else:
            tier_winner = "tie"

        valuation_winner = (
            "a" if delta_val > 0 else "b" if delta_val < 0 else "tie"
        )

        return {
            "player_a": {
                "label": player_a.get("label", "Player A"),
                "result": sim_a,
            },
            "player_b": {
                "label": player_b.get("label", "Player B"),
                "result": sim_b,
            },
            "delta_valuation_usd": round(delta_val, 2),
            "tier_winner": tier_winner,
            "valuation_winner": valuation_winner,
        }


class CohortComparator:
    """Find similar historical players for a given user-built profile."""

    COMPARISON_FEATURES = [
        "ppg", "apg", "rpg", "spg", "bpg", "mpg",
        "fg_pct", "three_pt_pct", "ft_pct",
        "games_played", "program_tier", "market_size_score",
        "social_media_followers", "engagement_rate",
    ]

    def __init__(self, data_path: str | os.PathLike | None = None) -> None:
        candidates = [
            Path(data_path) if data_path else None,
            _REPO_ROOT / "data" / "raw" / "nil_evaluations_2025.csv",
            _REPO_ROOT / "data" / "sample" / "nil_evaluations_sample.csv",
            _REPO_ROOT / "data" / "processed" / "player_snapshots.csv",
        ]

        self.df = pd.DataFrame()
        for path in candidates:
            if path is not None and Path(path).exists():
                self.df = pd.read_csv(path)
                break

        self.scaler = StandardScaler()
        if self.df.empty:
            self.comparison_cols: list[str] = []
            self.latest = pd.DataFrame()
            return

        sort_col = "week_number" if "week_number" in self.df.columns else "snapshot_week"
        self.latest = (
            self.df.sort_values(sort_col)
            .groupby("player_id")
            .tail(1)
            .reset_index(drop=True)
        )
        self.comparison_cols = [
            c for c in self.COMPARISON_FEATURES if c in self.latest.columns
        ]
        if self.comparison_cols:
            self.scaler.fit(self.latest[self.comparison_cols].fillna(0))

    def find_similar(self, profile: dict, n: int = 10) -> pd.DataFrame:
        if self.latest.empty or not self.comparison_cols:
            return pd.DataFrame()

        user_vec = np.array([[profile.get(c, 0) for c in self.comparison_cols]])
        user_scaled = self.scaler.transform(user_vec)
        hist_scaled = self.scaler.transform(
            self.latest[self.comparison_cols].fillna(0).values
        )
        sims = cosine_similarity(user_scaled, hist_scaled)[0]
        top_idx = np.argsort(sims)[::-1][:n]
        result = self.latest.iloc[top_idx].copy()
        result["similarity"] = sims[top_idx]
        return result

    def compare(self, profile: dict, n: int = 10) -> dict:
        similar = self.find_similar(profile, n=n)
        if similar.empty:
            return {
                "cohort_median_nil": 0,
                "percentile_rank": 50,
                "similar_players": [],
                "residual": 0,
            }

        nil_col = "nil_valuation_usd" if "nil_valuation_usd" in similar.columns else "nil_valuation"
        cohort_nil = similar[nil_col].values
        user_val = profile.get("nil_valuation_usd", profile.get("nil_valuation", 0))

        percentile = (cohort_nil < user_val).mean() * 100 if len(cohort_nil) else 50
        median_nil = float(np.median(cohort_nil))
        residual = user_val - median_nil

        display_cols = ["player_id", "school", nil_col, "ppg", "apg", "rpg", "similarity"]
        cols = [c for c in display_cols if c in similar.columns]
        return {
            "cohort_median_nil": round(median_nil, 2),
            "percentile_rank": round(percentile, 1),
            "similar_players": similar[cols].head(5).to_dict("records"),
            "residual": round(residual, 2),
        }
