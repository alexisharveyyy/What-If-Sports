"""Position-aware per-game stat distributions for synthetic basketball data.

Distributions are tuned to be plausible season-long averages, not single-game
lines. Each tuple is interpreted by the generator as either a truncated normal
(mean, std, low, high) or a beta-style range (low, high) for shooting splits.
"""

POSITION_PROBS = {
    "PG": 0.20,
    "SG": 0.22,
    "SF": 0.22,
    "PF": 0.20,
    "C": 0.16,
}

CLASS_YEAR_PROBS = {
    "FR": 0.27,
    "SO": 0.26,
    "JR": 0.22,
    "SR": 0.20,
    "GR": 0.05,
}

PROGRAM_TIER_PROBS = [0.10, 0.20, 0.30, 0.25, 0.15]


POSITION_STAT_DISTRIBUTIONS = {
    "PG": {
        "ppg": {"mean": 12.5, "std": 5.0, "low": 1.5, "high": 26.0},
        "apg": {"mean": 5.2, "std": 1.8, "low": 1.5, "high": 10.0},
        "rpg": {"mean": 3.2, "std": 1.2, "low": 0.8, "high": 7.0},
        "spg": {"mean": 1.4, "std": 0.5, "low": 0.2, "high": 3.0},
        "bpg": {"mean": 0.2, "std": 0.15, "low": 0.0, "high": 0.8},
        "mpg": {"mean": 28.0, "std": 6.0, "low": 8.0, "high": 38.0},
        "fg_pct": {"low": 0.36, "high": 0.50},
        "three_pt_pct": {"low": 0.30, "high": 0.42},
        "ft_pct": {"low": 0.70, "high": 0.92},
    },
    "SG": {
        "ppg": {"mean": 13.5, "std": 5.5, "low": 2.0, "high": 28.0},
        "apg": {"mean": 3.0, "std": 1.4, "low": 0.5, "high": 7.0},
        "rpg": {"mean": 3.5, "std": 1.2, "low": 1.0, "high": 7.0},
        "spg": {"mean": 1.2, "std": 0.4, "low": 0.2, "high": 2.6},
        "bpg": {"mean": 0.3, "std": 0.2, "low": 0.0, "high": 1.0},
        "mpg": {"mean": 27.0, "std": 6.0, "low": 8.0, "high": 38.0},
        "fg_pct": {"low": 0.38, "high": 0.52},
        "three_pt_pct": {"low": 0.31, "high": 0.42},
        "ft_pct": {"low": 0.68, "high": 0.90},
    },
    "SF": {
        "ppg": {"mean": 11.5, "std": 4.5, "low": 2.0, "high": 25.0},
        "apg": {"mean": 2.2, "std": 1.0, "low": 0.4, "high": 5.5},
        "rpg": {"mean": 5.5, "std": 1.5, "low": 2.0, "high": 9.0},
        "spg": {"mean": 1.0, "std": 0.4, "low": 0.2, "high": 2.4},
        "bpg": {"mean": 0.6, "std": 0.3, "low": 0.0, "high": 1.8},
        "mpg": {"mean": 26.5, "std": 6.0, "low": 8.0, "high": 37.0},
        "fg_pct": {"low": 0.40, "high": 0.55},
        "three_pt_pct": {"low": 0.28, "high": 0.40},
        "ft_pct": {"low": 0.65, "high": 0.85},
    },
    "PF": {
        "ppg": {"mean": 10.5, "std": 4.0, "low": 2.0, "high": 22.0},
        "apg": {"mean": 1.6, "std": 0.8, "low": 0.2, "high": 4.5},
        "rpg": {"mean": 6.5, "std": 1.8, "low": 2.5, "high": 10.5},
        "spg": {"mean": 0.7, "std": 0.3, "low": 0.1, "high": 1.8},
        "bpg": {"mean": 1.0, "std": 0.5, "low": 0.1, "high": 2.6},
        "mpg": {"mean": 25.5, "std": 6.0, "low": 8.0, "high": 36.0},
        "fg_pct": {"low": 0.42, "high": 0.58},
        "three_pt_pct": {"low": 0.20, "high": 0.36},
        "ft_pct": {"low": 0.60, "high": 0.82},
    },
    "C": {
        "ppg": {"mean": 9.5, "std": 4.0, "low": 1.5, "high": 22.0},
        "apg": {"mean": 1.0, "std": 0.6, "low": 0.1, "high": 3.0},
        "rpg": {"mean": 8.0, "std": 2.2, "low": 3.0, "high": 12.5},
        "spg": {"mean": 0.5, "std": 0.3, "low": 0.0, "high": 1.5},
        "bpg": {"mean": 1.6, "std": 0.7, "low": 0.3, "high": 3.5},
        "mpg": {"mean": 24.0, "std": 6.0, "low": 8.0, "high": 36.0},
        "fg_pct": {"low": 0.48, "high": 0.66},
        "three_pt_pct": {"low": 0.00, "high": 0.30},
        "ft_pct": {"low": 0.50, "high": 0.78},
    },
}


SOCIAL_FOLLOWERS_BY_TIER = {
    1: {"mean": 11.5, "std": 0.8},
    2: {"mean": 10.2, "std": 0.7},
    3: {"mean": 9.0, "std": 0.7},
    4: {"mean": 7.8, "std": 0.7},
    5: {"mean": 6.5, "std": 0.6},
}


SEASON_START = "2024-11-04"
NUM_WEEKS = 20
