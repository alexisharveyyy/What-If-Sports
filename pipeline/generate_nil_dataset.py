"""Generate a synthetic NIL evaluation dataset for the 2024-2025 season.

Produces weekly player snapshots (up to 20 per player) with per-game averages,
NIL valuation in dollars, and a tier label. The CSV schema mirrors the fields
exposed by the Player Builder UI.

Usage::

    python pipeline/generate_nil_dataset.py
    python pipeline/generate_nil_dataset.py --players 500 --target sample
    python pipeline/generate_nil_dataset.py --seed 1337 --players 10000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from tqdm import tqdm

try:
    from faker import Faker
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'faker'. Install with: pip install faker"
    ) from exc

# Allow running this file directly: `python pipeline/generate_nil_dataset.py`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.config.conferences import CONFERENCES, CONFERENCE_WEIGHTS
from pipeline.config.schools import (
    BLUE_BLOODS,
    SCHOOLS_BY_CONFERENCE,
    market_size_score,
)
from pipeline.config.stat_distributions import (
    CLASS_YEAR_PROBS,
    NUM_WEEKS,
    POSITION_PROBS,
    POSITION_STAT_DISTRIBUTIONS,
    PROGRAM_TIER_PROBS,
    SEASON_START,
    SOCIAL_FOLLOWERS_BY_TIER,
)


def _truncnorm_sample(rng: np.random.Generator, mean: float, std: float,
                      low: float, high: float) -> float:
    a = (low - mean) / std
    b = (high - mean) / std
    return float(truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng))


def _beta_sample(rng: np.random.Generator, low: float, high: float,
                 skew: float = 1.0) -> float:
    """Sample inside [low, high] from a Beta(2, 2) (or skewed) distribution."""
    raw = rng.beta(2.0 * skew, 2.0)
    return low + raw * (high - low)


def _assign_program_tier(rng: np.random.Generator, school: str) -> int:
    if school in BLUE_BLOODS:
        return int(rng.choice([1, 2], p=[0.75, 0.25]))
    return int(rng.choice([1, 2, 3, 4, 5], p=PROGRAM_TIER_PROBS))


def _build_school_pool() -> list[tuple[str, str]]:
    """Build a flat list of (school, conference) entries for sampling."""
    return [
        (school, conf)
        for conf, schools in SCHOOLS_BY_CONFERENCE.items()
        for school in schools
    ]


def _sample_player_skeleton(rng: np.random.Generator, faker: Faker,
                            school_pool: list[tuple[str, str]]) -> dict:
    weights = np.array(
        [CONFERENCE_WEIGHTS[conf] for _, conf in school_pool], dtype=float
    )
    weights /= weights.sum()
    idx = rng.choice(len(school_pool), p=weights)
    school, conference = school_pool[idx]

    program_tier = _assign_program_tier(rng, school)
    position = str(rng.choice(
        list(POSITION_PROBS), p=list(POSITION_PROBS.values())
    ))
    class_year = str(rng.choice(
        list(CLASS_YEAR_PROBS), p=list(CLASS_YEAR_PROBS.values())
    ))

    high_bits = int(rng.integers(0, 2**63 - 1)) & ((1 << 64) - 1)
    low_bits = int(rng.integers(0, 2**63 - 1)) & ((1 << 64) - 1)
    player_uuid = uuid.UUID(int=(high_bits << 64) | low_bits)

    return {
        "player_id": str(player_uuid),
        "player_name": faker.name(),
        "school": school,
        "conference": conference,
        "program_tier": program_tier,
        "position": position,
        "class_year": class_year,
    }


def _baseline_stats(rng: np.random.Generator, position: str) -> dict:
    dist = POSITION_STAT_DISTRIBUTIONS[position]
    counting = {}
    for col in ("ppg", "apg", "rpg", "spg", "bpg", "mpg"):
        spec = dist[col]
        counting[col] = _truncnorm_sample(
            rng, spec["mean"], spec["std"], spec["low"], spec["high"]
        )

    shooting = {}
    for col in ("fg_pct", "three_pt_pct", "ft_pct"):
        spec = dist[col]
        shooting[col] = _beta_sample(rng, spec["low"], spec["high"])

    return {**counting, **shooting}


def _evolve_stats(baseline: dict, week: int, rng: np.random.Generator,
                  hot_streak: int, cold_streak: int) -> dict:
    """Apply week-to-week noise plus optional hot/cold multipliers."""
    multiplier = 1.0
    if hot_streak > 0:
        multiplier *= rng.uniform(1.05, 1.18)
    elif cold_streak > 0:
        multiplier *= rng.uniform(0.82, 0.95)

    out = {}
    counting_caps = {"ppg": 40.0, "apg": 13.0, "rpg": 16.0,
                     "spg": 5.0, "bpg": 6.0, "mpg": 40.0}
    for col in ("ppg", "apg", "rpg", "spg", "bpg", "mpg"):
        noise = rng.normal(0, max(baseline[col] * 0.10, 0.2))
        val = baseline[col] * multiplier + noise
        out[col] = float(np.clip(val, 0.0, counting_caps[col]))

    for col in ("fg_pct", "three_pt_pct", "ft_pct"):
        noise = rng.normal(0, 0.025)
        val = baseline[col] + noise
        out[col] = float(np.clip(val, 0.0, 1.0))

    return out


def _performance_score(stats: dict) -> float:
    """Composite score in roughly [0, 1].

    Weights: ppg 30%, apg 15%, rpg 15%, fg_pct 10%, three_pt_pct 10%, mpg 20%.
    """
    return (
        0.30 * (stats["ppg"] / 28.0)
        + 0.15 * (stats["apg"] / 8.0)
        + 0.15 * (stats["rpg"] / 12.0)
        + 0.10 * (stats["fg_pct"] / 0.65)
        + 0.10 * (stats["three_pt_pct"] / 0.45)
        + 0.20 * (stats["mpg"] / 38.0)
    )


def _social_followers(rng: np.random.Generator, program_tier: int,
                      perf_score: float) -> int:
    spec = SOCIAL_FOLLOWERS_BY_TIER[program_tier]
    log_followers = rng.normal(spec["mean"], spec["std"]) + 0.8 * perf_score
    return int(max(150, math.exp(log_followers)))


def _engagement_rate(rng: np.random.Generator, perf_score: float) -> float:
    base = rng.uniform(0.01, 0.05) + 0.05 * perf_score
    return float(np.clip(base + rng.normal(0, 0.01), 0.001, 0.15))


_TIER_MULTIPLIER = {1: 3.0, 2: 1.6, 3: 0.9, 4: 0.55, 5: 0.4}


def _nil_valuation(rng: np.random.Generator, perf_score: float,
                   program_tier: int, market_score: int,
                   social_followers: int, engagement: float,
                   currently_injured: bool) -> float:
    perf_component = (perf_score ** 1.4) * 250_000
    market_component = market_score * 6_500
    social_component = math.log1p(social_followers) * (1.0 + engagement * 8.0) * 9_500

    raw = (perf_component + market_component + social_component)
    raw *= _TIER_MULTIPLIER[program_tier]

    if currently_injured:
        raw *= rng.uniform(0.65, 0.85)

    raw *= rng.lognormal(mean=0.0, sigma=0.18)
    return float(max(2_500.0, raw))


def _generate_player_rows(rng: np.random.Generator, faker: Faker,
                          school_pool: list[tuple[str, str]],
                          season_start: datetime) -> list[dict]:
    skeleton = _sample_player_skeleton(rng, faker, school_pool)
    baseline = _baseline_stats(rng, skeleton["position"])

    n_weeks = int(rng.integers(8, NUM_WEEKS + 1))
    start_week = int(rng.integers(1, NUM_WEEKS - n_weeks + 2))

    cumulative_games = 0
    streak_state = {"hot": 0, "cold": 0}
    rows: list[dict] = []
    market_score = market_size_score(skeleton["school"], skeleton["program_tier"])

    for offset in range(n_weeks):
        week_number = start_week + offset

        if streak_state["hot"] > 0:
            streak_state["hot"] -= 1
        elif streak_state["cold"] > 0:
            streak_state["cold"] -= 1
        elif rng.random() < 0.10:
            streak_state["hot"] = int(rng.integers(2, 5))
        elif rng.random() < 0.07:
            streak_state["cold"] = int(rng.integers(2, 4))

        injured = rng.random() < 0.05
        if injured:
            week_games = 0
        else:
            week_games = int(rng.integers(1, 4))
        cumulative_games = min(35, cumulative_games + week_games)

        stats = _evolve_stats(
            baseline, week_number, rng,
            streak_state["hot"], streak_state["cold"],
        )

        perf_score = _performance_score(stats)
        social = _social_followers(rng, skeleton["program_tier"], perf_score)
        engagement = _engagement_rate(rng, perf_score)
        valuation = _nil_valuation(
            rng,
            perf_score,
            skeleton["program_tier"],
            market_score,
            social,
            engagement,
            injured,
        )

        game_date = season_start + timedelta(
            weeks=week_number - 1, days=int(rng.integers(0, 7))
        )

        rows.append({
            "player_id": skeleton["player_id"],
            "player_name": skeleton["player_name"],
            "school": skeleton["school"],
            "conference": skeleton["conference"],
            "program_tier": skeleton["program_tier"],
            "position": skeleton["position"],
            "class_year": skeleton["class_year"],
            "week_number": week_number,
            "game_date": game_date.date().isoformat(),
            "games_played": cumulative_games,
            "currently_injured": bool(injured),
            "ppg": round(stats["ppg"], 2),
            "apg": round(stats["apg"], 2),
            "rpg": round(stats["rpg"], 2),
            "spg": round(stats["spg"], 2),
            "bpg": round(stats["bpg"], 2),
            "mpg": round(stats["mpg"], 2),
            "fg_pct": round(stats["fg_pct"], 4),
            "three_pt_pct": round(stats["three_pt_pct"], 4),
            "ft_pct": round(stats["ft_pct"], 4),
            "social_media_followers": social,
            "engagement_rate": round(engagement, 4),
            "market_size_score": market_score,
            "performance_score": round(perf_score, 4),
            "nil_valuation_usd": round(valuation, 2),
        })

    return rows


def _assign_tier_and_probability(df: pd.DataFrame) -> pd.DataFrame:
    """Add nil_tier and nil_probability_score using conference-relative logic."""
    df = df.copy()
    df["nil_tier"] = "mid"
    df["nil_probability_score"] = 0.0

    tier_bins = [
        (0.0, 0.10, "developmental"),
        (0.10, 0.40, "low"),
        (0.40, 0.80, "mid"),
        (0.80, 0.95, "high"),
        (0.95, 1.01, "elite"),
    ]

    out_frames = []
    for conf, conf_df in df.groupby("conference"):
        ranks = conf_df["nil_valuation_usd"].rank(pct=True)
        tiers = pd.Series("mid", index=conf_df.index)
        for low, high, label in tier_bins:
            mask = (ranks >= low) & (ranks < high)
            tiers.loc[mask] = label
        conf_df = conf_df.assign(nil_tier=tiers.values)

        for position, pos_df in conf_df.groupby("position"):
            log_vals = np.log1p(pos_df["nil_valuation_usd"].values)
            shifted = log_vals - log_vals.max()
            probs = np.exp(shifted) / np.exp(shifted).sum()
            conf_df.loc[pos_df.index, "nil_probability_score"] = probs

        out_frames.append(conf_df)

    result = pd.concat(out_frames).sort_index()
    return result


def generate_dataset(
    n_players: int = 10_000,
    seed: int = 42,
    target: str = "raw",
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Generate the dataset and write it to disk.

    Args:
        n_players: Number of unique players to generate.
        seed: Random seed for reproducibility.
        target: One of {"raw", "sample"}; controls the default output path.
        output_dir: Override directory; if set, target is ignored.
    """
    rng = np.random.default_rng(seed)
    faker = Faker()
    Faker.seed(seed)

    school_pool = _build_school_pool()
    season_start = datetime.fromisoformat(SEASON_START)

    all_rows: list[dict] = []
    for _ in tqdm(range(n_players), desc="Generating players", unit="player"):
        all_rows.extend(_generate_player_rows(rng, faker, school_pool, season_start))

    df = pd.DataFrame(all_rows)
    df = _assign_tier_and_probability(df)

    column_order = [
        "player_id", "player_name", "school", "conference", "program_tier",
        "position", "class_year", "week_number", "game_date", "games_played",
        "currently_injured", "ppg", "apg", "rpg", "spg", "bpg", "mpg",
        "fg_pct", "three_pt_pct", "ft_pct", "social_media_followers",
        "engagement_rate", "market_size_score", "performance_score",
        "nil_valuation_usd", "nil_probability_score", "nil_tier",
    ]
    df = df[column_order]

    if output_dir is None:
        if target == "sample":
            out_path = _REPO_ROOT / "data" / "sample" / "nil_evaluations_sample.csv"
        else:
            out_path = _REPO_ROOT / "data" / "raw" / "nil_evaluations_2025.csv"
    else:
        out_path = Path(output_dir) / "nil_evaluations.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _print_summary(df, out_path)
    return df


def _print_summary(df: pd.DataFrame, out_path: Path) -> None:
    print()
    print(f"Wrote {len(df):,} rows for {df['player_id'].nunique():,} players to {out_path}")
    print("\nPlayers per conference:")
    counts = df.groupby("conference")["player_id"].nunique().sort_values(ascending=False)
    for conf, n in counts.items():
        print(f"  {conf:<14} {n:>5}")

    val = df["nil_valuation_usd"]
    print("\nNIL valuation (USD):")
    print(f"  mean   ${val.mean():,.0f}")
    print(f"  median ${val.median():,.0f}")
    print(f"  min    ${val.min():,.0f}")
    print(f"  max    ${val.max():,.0f}")

    print("\nTier distribution:")
    tier_counts = df["nil_tier"].value_counts()
    for tier in ("elite", "high", "mid", "low", "developmental"):
        if tier in tier_counts:
            pct = 100 * tier_counts[tier] / len(df)
            print(f"  {tier:<14} {tier_counts[tier]:>7,}  ({pct:5.1f}%)")

    print("\nStat ranges (sanity check):")
    for col in ("ppg", "apg", "rpg", "fg_pct", "three_pt_pct", "ft_pct", "mpg"):
        s = df[col]
        print(f"  {col:<14} min={s.min():.3f}  median={s.median():.3f}  max={s.max():.3f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, default=10_000,
                        help="Number of unique players (default 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default 42)")
    parser.add_argument("--target", choices=["raw", "sample"], default="raw",
                        help="Write to data/raw or data/sample")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    n = args.players if args.target == "raw" else min(args.players, 500)
    generate_dataset(
        n_players=n,
        seed=args.seed,
        target=args.target,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
