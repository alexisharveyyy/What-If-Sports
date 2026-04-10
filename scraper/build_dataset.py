"""
Dataset builder: combines scraped stats + NIL valuations into a single CSV.

Merges player stats from Sports Reference with NIL valuations from On3,
producing a unified dataset with weekly snapshots and season-level stats.

Usage:
    python scraper/build_dataset.py \
        --stats data/raw/player_stats.csv \
        --nil data/raw/on3_valuations.csv \
        --output data/raw/college_basketball_nil_dataset.csv

If scraped data is unavailable, falls back to sample data generation.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TARGET_CONFERENCES = {"ACC", "SEC", "Big Ten", "Big 12", "Pac-12", "Independent"}

# NIL tier thresholds
NIL_TIER_THRESHOLDS = (50_000, 200_000, 500_000, 1_000_000)


def normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy matching."""
    import re
    name = name.lower().strip()
    # Remove suffixes like Jr., III, etc.
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name)
    # Remove periods and extra whitespace
    name = re.sub(r"\.", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def fuzzy_match_score(name1: str, name2: str) -> float:
    """Compute fuzzy match score between two names."""
    return SequenceMatcher(None, normalize_name(name1), normalize_name(name2)).ratio()


def assign_nil_tier(valuation: float) -> int:
    """Assign NIL tier based on valuation thresholds."""
    if valuation < NIL_TIER_THRESHOLDS[0]:
        return 1
    elif valuation < NIL_TIER_THRESHOLDS[1]:
        return 2
    elif valuation < NIL_TIER_THRESHOLDS[2]:
        return 3
    elif valuation < NIL_TIER_THRESHOLDS[3]:
        return 4
    else:
        return 5


def assign_program_tier(school: str, conference: str) -> int:
    """Assign program tier (1=elite, 4=mid-major) based on school/conference."""
    elite_programs = {
        "duke", "kentucky", "north carolina", "kansas", "gonzaga",
        "ucla", "villanova", "michigan state", "indiana",
    }
    strong_programs = {
        "alabama", "houston", "baylor", "purdue", "arizona",
        "tennessee", "auburn", "creighton", "marquette", "uconn",
    }

    school_lower = school.lower()

    if any(p in school_lower for p in elite_programs):
        return 1
    elif any(p in school_lower for p in strong_programs):
        return 2
    elif conference in {"ACC", "SEC", "Big Ten", "Big 12"}:
        return 2
    elif conference in {"Pac-12"}:
        return 3
    else:
        return 3


def merge_stats_and_nil(
    stats_df: pd.DataFrame,
    nil_df: pd.DataFrame,
    match_threshold: float = 0.80,
) -> pd.DataFrame:
    """Merge stats and NIL data using fuzzy name + school matching.

    Args:
        stats_df: Player stats DataFrame.
        nil_df: NIL valuations DataFrame.
        match_threshold: Minimum fuzzy match score to consider a match.

    Returns:
        Merged DataFrame with stats and NIL valuations.
    """
    merged_rows = []

    # Index NIL data by normalized name for faster lookup
    nil_lookup = {}
    for _, nil_row in nil_df.iterrows():
        key = normalize_name(nil_row["player_name"])
        nil_lookup[key] = nil_row

    matched_count = 0
    unmatched_count = 0

    for _, stats_row in stats_df.iterrows():
        player_name = stats_row["player_name"]
        norm_name = normalize_name(player_name)

        # Try exact match first
        nil_match = nil_lookup.get(norm_name)

        # If no exact match, try fuzzy matching
        if nil_match is None:
            best_score = 0
            best_match = None
            for nil_name, nil_row in nil_lookup.items():
                score = fuzzy_match_score(player_name, nil_row["player_name"])
                if score > best_score and score >= match_threshold:
                    best_score = score
                    best_match = nil_row
            nil_match = best_match

        row = stats_row.to_dict()

        if nil_match is not None:
            row["nil_valuation"] = nil_match["nil_valuation"]
            row["nil_ranking"] = nil_match.get("nil_ranking")
            matched_count += 1
        else:
            # Estimate NIL based on stats for unmatched players
            ppg = row.get("ppg", 0) or 0
            program_tier = assign_program_tier(
                row.get("school", ""), row.get("conference", "")
            )
            # Simple estimation: base on ppg and program tier
            base = {1: 300_000, 2: 150_000, 3: 60_000, 4: 25_000}.get(program_tier, 25_000)
            estimated = base * (1 + ppg / 25.0) * np.random.uniform(0.6, 1.2)
            row["nil_valuation"] = round(max(5000, estimated), 2)
            row["nil_ranking"] = None
            unmatched_count += 1

        row["nil_tier"] = assign_nil_tier(row["nil_valuation"])
        row["program_tier"] = assign_program_tier(
            row.get("school", ""), row.get("conference", "")
        )
        merged_rows.append(row)

    print(f"Matched {matched_count} players with NIL data")
    print(f"Estimated NIL for {unmatched_count} unmatched players")

    return pd.DataFrame(merged_rows)


def generate_weekly_snapshots(
    season_df: pd.DataFrame,
    n_weeks: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate weekly snapshots from season-level stats.

    Creates time-series data by simulating weekly stat variation around
    each player's season averages, plus season-level columns carried forward.

    Args:
        season_df: DataFrame with one row per player (season stats).
        n_weeks: Number of weekly snapshots to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with multiple weekly rows per player, including both
        weekly snapshot stats and season-level stats.
    """
    rng = np.random.RandomState(seed)
    weekly_stat_cols = ["ppg", "apg", "rpg", "spg", "bpg", "mpg"]
    pct_cols = ["fg_pct", "three_pt_pct", "ft_pct"]
    rows = []

    for idx, (_, player) in enumerate(season_df.iterrows()):
        player_id = f"P{idx + 1:04d}"

        # Season-level stats (carried in every row)
        season_ppg = player.get("ppg", 0) or 0
        season_apg = player.get("apg", 0) or 0
        season_rpg = player.get("rpg", 0) or 0
        season_spg = player.get("spg", 0) or 0
        season_bpg = player.get("bpg", 0) or 0
        season_mpg = player.get("mpg", 0) or 0
        season_fg_pct = player.get("fg_pct", 0) or 0
        season_three_pt_pct = player.get("three_pt_pct", 0) or 0
        season_ft_pct = player.get("ft_pct", 0) or 0

        base_nil = player.get("nil_valuation", 50_000) or 50_000
        cumulative_injuries = 0

        for week in range(1, n_weeks + 1):
            # Weekly stat variation around season averages
            trend = rng.uniform(-0.03, 0.05)  # slight upward bias
            weekly_ppg = max(0, season_ppg * (1 + trend) + rng.normal(0, 1.5))
            weekly_apg = max(0, season_apg + rng.normal(0, 0.4))
            weekly_rpg = max(0, season_rpg + rng.normal(0, 0.8))
            weekly_spg = max(0, season_spg + rng.normal(0, 0.2))
            weekly_bpg = max(0, season_bpg + rng.normal(0, 0.2))
            weekly_mpg = max(0, season_mpg + rng.normal(0, 1.5))

            # Shooting percentages fluctuate less
            weekly_fg_pct = np.clip(season_fg_pct + rng.normal(0, 0.03), 0.15, 0.75)
            weekly_three_pt_pct = np.clip(season_three_pt_pct + rng.normal(0, 0.04), 0.0, 0.65)
            weekly_ft_pct = np.clip(season_ft_pct + rng.normal(0, 0.03), 0.30, 1.0)

            # Injury simulation
            injury_flag = int(rng.random() < 0.06)
            if injury_flag:
                cumulative_injuries += 1

            games_played = max(1, week - cumulative_injuries)

            # NIL fluctuates with performance
            perf_factor = 1 + (weekly_ppg - season_ppg) / max(season_ppg, 1) * 0.1
            injury_penalty = 0.85 if injury_flag else 1.0
            nil_noise = rng.uniform(0.92, 1.08)
            nil_val = max(5000, base_nil * perf_factor * injury_penalty * nil_noise)

            rows.append({
                "player_id": player_id,
                "player_name": player.get("player_name", f"Player_{idx+1}"),
                "snapshot_week": week,
                "sport": "basketball",
                "school": player.get("school", "Unknown"),
                "conference": player.get("conference", "Unknown"),
                "program_tier": player.get("program_tier", 3),
                # Weekly snapshot stats
                "ppg": round(weekly_ppg, 1),
                "apg": round(weekly_apg, 1),
                "rpg": round(weekly_rpg, 1),
                "spg": round(weekly_spg, 1),
                "bpg": round(weekly_bpg, 1),
                "mpg": round(weekly_mpg, 1),
                "fg_pct": round(weekly_fg_pct, 3),
                "three_pt_pct": round(weekly_three_pt_pct, 3),
                "ft_pct": round(weekly_ft_pct, 3),
                # Season-level stats (constant per player)
                "season_ppg": round(season_ppg, 1),
                "season_apg": round(season_apg, 1),
                "season_rpg": round(season_rpg, 1),
                "season_spg": round(season_spg, 1),
                "season_bpg": round(season_bpg, 1),
                "season_mpg": round(season_mpg, 1),
                "season_fg_pct": round(season_fg_pct, 3),
                "season_three_pt_pct": round(season_three_pt_pct, 3),
                "season_ft_pct": round(season_ft_pct, 3),
                # Other fields
                "injury_flag": injury_flag,
                "games_played": games_played,
                "nil_valuation": round(nil_val, 2),
                "nil_tier": assign_nil_tier(nil_val),
                "nil_ranking": player.get("nil_ranking"),
            })

        # Let base NIL drift slightly over the season
        base_nil *= rng.uniform(0.97, 1.03)

    return pd.DataFrame(rows)


def build_dataset(
    stats_path: str = "data/raw/player_stats.csv",
    nil_path: str = "data/raw/on3_valuations.csv",
    output_path: str = "data/raw/college_basketball_nil_dataset.csv",
    n_weeks: int = 10,
) -> pd.DataFrame:
    """Build the full dataset from scraped data.

    Pipeline:
    1. Load scraped stats and NIL data
    2. Filter to target conferences
    3. Merge stats + NIL via fuzzy name matching
    4. Generate weekly snapshots with season stats
    5. Save as single CSV

    Args:
        stats_path: Path to scraped player stats CSV.
        nil_path: Path to scraped NIL valuations CSV.
        output_path: Where to save the final dataset.
        n_weeks: Number of weekly snapshots per player.

    Returns:
        The final dataset DataFrame.
    """
    # Load data
    if os.path.exists(stats_path):
        stats_df = pd.read_csv(stats_path)
        print(f"Loaded {len(stats_df)} player stat rows from {stats_path}")
    else:
        print(f"Stats file not found at {stats_path}")
        print("Run `python scraper/stats_scraper.py` first to scrape player stats.")
        print("Falling back to sample data generation...")
        from data.sample.generate_sample import generate_sample_data
        df = generate_sample_data(n_players=200, n_weeks=n_weeks)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated sample dataset: {len(df)} rows → {output_path}")
        return df

    if os.path.exists(nil_path):
        nil_df = pd.read_csv(nil_path)
        print(f"Loaded {len(nil_df)} NIL valuations from {nil_path}")
    else:
        print(f"NIL file not found at {nil_path}. Will estimate NIL values.")
        nil_df = pd.DataFrame()

    # Filter to target conferences
    if "conference" in stats_df.columns:
        stats_df = stats_df[stats_df["conference"].isin(TARGET_CONFERENCES)].copy()
        print(f"Filtered to target conferences: {len(stats_df)} players")

    if len(stats_df) == 0:
        print("No players found in target conferences.")
        return pd.DataFrame()

    # Merge stats + NIL
    if len(nil_df) > 0:
        season_df = merge_stats_and_nil(stats_df, nil_df)
    else:
        # No NIL data — estimate all values
        season_df = stats_df.copy()
        season_df["program_tier"] = season_df.apply(
            lambda r: assign_program_tier(r.get("school", ""), r.get("conference", "")),
            axis=1,
        )
        season_df["nil_valuation"] = season_df.apply(
            lambda r: round(
                max(5000, {1: 300_000, 2: 150_000, 3: 60_000, 4: 25_000}.get(r["program_tier"], 25_000)
                    * (1 + (r.get("ppg", 0) or 0) / 25.0)
                    * np.random.uniform(0.6, 1.2)),
                2,
            ),
            axis=1,
        )
        season_df["nil_tier"] = season_df["nil_valuation"].apply(assign_nil_tier)
        season_df["nil_ranking"] = None

    # Generate weekly snapshots
    print(f"\nGenerating {n_weeks} weekly snapshots for {len(season_df)} players...")
    dataset = generate_weekly_snapshots(season_df, n_weeks=n_weeks)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"\nDataset built: {len(dataset)} rows, {dataset['player_id'].nunique()} players")
    print(f"Conferences: {sorted(dataset['conference'].unique())}")
    print(f"NIL range: ${dataset['nil_valuation'].min():,.0f} – ${dataset['nil_valuation'].max():,.0f}")
    print(f"NIL tier distribution:\n{dataset['nil_tier'].value_counts().sort_index()}")
    print(f"\nSaved to: {output_path}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build college basketball NIL dataset")
    parser.add_argument("--stats", default="data/raw/player_stats.csv", help="Player stats CSV")
    parser.add_argument("--nil", default="data/raw/on3_valuations.csv", help="NIL valuations CSV")
    parser.add_argument("--output", default="data/raw/college_basketball_nil_dataset.csv", help="Output CSV")
    parser.add_argument("--weeks", type=int, default=10, help="Weekly snapshots per player")
    args = parser.parse_args()

    build_dataset(
        stats_path=args.stats,
        nil_path=args.nil,
        output_path=args.output,
        n_weeks=args.weeks,
    )
