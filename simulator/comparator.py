"""Compare user profile vs. historical cohort."""

import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class CohortComparator:
    """Find similar historical players and compare NIL valuations."""

    COMPARISON_FEATURES = [
        "ppg", "apg", "rpg", "games_played", "program_tier", "injury_flag",
    ]

    def __init__(self, data_path: str = "data/processed/feature_matrix.csv"):
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
        elif os.path.exists("data/sample/sample_players.csv"):
            self.df = pd.read_csv("data/sample/sample_players.csv")
        else:
            self.df = pd.DataFrame()

        self.scaler = StandardScaler()
        if len(self.df) > 0:
            available = [c for c in self.COMPARISON_FEATURES if c in self.df.columns]
            self.comparison_cols = available
            # Get latest snapshot per player for comparison
            self.latest = (
                self.df.sort_values("snapshot_week")
                .groupby("player_id")
                .last()
                .reset_index()
            )
            if len(self.latest) > 0 and available:
                self.scaler.fit(self.latest[available].fillna(0))
        else:
            self.comparison_cols = []
            self.latest = pd.DataFrame()

    def find_similar(self, player_profile: dict, n: int = 10) -> pd.DataFrame:
        """Find the n most similar historical players."""
        if len(self.latest) == 0:
            return pd.DataFrame()

        # Build user vector
        user_vec = np.array([[player_profile.get(c, 0) for c in self.comparison_cols]])
        user_scaled = self.scaler.transform(user_vec)

        # Scale historical data
        hist_scaled = self.scaler.transform(
            self.latest[self.comparison_cols].fillna(0).values
        )

        # Compute cosine similarity
        sims = cosine_similarity(user_scaled, hist_scaled)[0]
        top_idx = np.argsort(sims)[::-1][:n]

        result = self.latest.iloc[top_idx].copy()
        result["similarity"] = sims[top_idx]
        return result

    def compare(self, player_profile: dict, n: int = 10) -> dict:
        """Compare a user profile against the historical cohort."""
        similar = self.find_similar(player_profile, n=n)

        if len(similar) == 0:
            return {
                "cohort_median_nil": 0,
                "percentile_rank": 50,
                "similar_players": [],
                "residual": 0,
            }

        cohort_nil = similar["nil_valuation"].values
        user_val = player_profile.get("nil_valuation", 0)

        # Percentile rank within cohort
        percentile = (cohort_nil < user_val).mean() * 100

        # Residual: user vs expected
        median_nil = float(np.median(cohort_nil))
        residual = user_val - median_nil

        # Build player list
        display_cols = ["player_id", "school", "nil_valuation", "ppg", "apg", "rpg", "similarity"]
        available_display = [c for c in display_cols if c in similar.columns]
        player_list = similar[available_display].head(5).to_dict("records")

        return {
            "cohort_median_nil": round(median_nil, 2),
            "percentile_rank": round(percentile, 1),
            "similar_players": player_list,
            "residual": round(residual, 2),
        }
