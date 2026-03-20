"""Generate synthetic sample player data for development and testing."""

import numpy as np
import pandas as pd


SCHOOLS = [
    ("Alabama", "SEC"), ("Ohio State", "Big Ten"), ("Georgia", "SEC"),
    ("Michigan", "Big Ten"), ("USC", "Big 12"), ("Texas", "SEC"),
    ("Oregon", "Big Ten"), ("LSU", "SEC"), ("Clemson", "ACC"),
    ("Penn State", "Big Ten"), ("Florida State", "ACC"), ("Oklahoma", "SEC"),
    ("Tennessee", "SEC"), ("Notre Dame", "Independent"), ("Miami", "ACC"),
    ("UCLA", "Big Ten"), ("Washington", "Big Ten"), ("Duke", "ACC"),
    ("Kentucky", "SEC"), ("Kansas", "Big 12"),
]

SPORTS = ["basketball", "football"]


def assign_nil_valuation(sport, program_tier, ppg, games_played, injury_flag):
    """Generate a realistic NIL valuation based on player attributes."""
    base = {1: 800_000, 2: 400_000, 3: 150_000, 4: 50_000}[program_tier]
    stat_mult = 1 + (ppg / 30.0)
    games_mult = min(games_played / 10.0, 1.5)
    injury_penalty = 0.7 if injury_flag else 1.0
    sport_mult = 1.2 if sport == "football" else 1.0
    noise = np.random.uniform(0.75, 1.35)
    return max(5000, base * stat_mult * games_mult * injury_penalty * sport_mult * noise)


def nil_tier(valuation, thresholds=(50_000, 200_000, 500_000, 1_000_000)):
    if valuation < thresholds[0]:
        return 1
    elif valuation < thresholds[1]:
        return 2
    elif valuation < thresholds[2]:
        return 3
    elif valuation < thresholds[3]:
        return 4
    else:
        return 5


def generate_sample_data(n_players=50, n_weeks=10, seed=42):
    np.random.seed(seed)
    rows = []

    for pid in range(1, n_players + 1):
        school, conference = SCHOOLS[pid % len(SCHOOLS)]
        sport = SPORTS[pid % len(SPORTS)]
        program_tier = np.random.choice([1, 2, 3, 4], p=[0.15, 0.30, 0.35, 0.20])

        # Base stats for this player
        base_ppg = np.random.uniform(5, 28)
        base_apg = np.random.uniform(0.5, 8) if sport == "basketball" else np.random.uniform(0, 2)
        base_rpg = np.random.uniform(1, 12) if sport == "basketball" else np.random.uniform(0, 3)

        cumulative_injuries = 0

        for week in range(1, n_weeks + 1):
            # Stats drift over time with noise
            trend = np.random.uniform(-0.05, 0.08)  # slight upward bias
            ppg = max(0, base_ppg * (1 + trend * week) + np.random.normal(0, 2))
            apg = max(0, base_apg + np.random.normal(0, 0.5))
            rpg = max(0, base_rpg + np.random.normal(0, 1))

            injury_flag = np.random.random() < 0.08
            if injury_flag:
                cumulative_injuries += 1

            games_played = max(1, week - cumulative_injuries)

            nil_val = assign_nil_valuation(sport, program_tier, ppg, games_played, injury_flag)
            tier = nil_tier(nil_val)

            rows.append({
                "player_id": f"P{pid:03d}",
                "snapshot_week": week,
                "sport": sport,
                "school": school,
                "conference": conference,
                "ppg": round(ppg, 1),
                "apg": round(apg, 1),
                "rpg": round(rpg, 1),
                "injury_flag": int(injury_flag),
                "games_played": games_played,
                "program_tier": int(program_tier),
                "nil_valuation": round(nil_val, 2),
                "nil_tier": tier,
            })

        # Small trend to base stats
        base_ppg += np.random.normal(0.5, 1)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_sample_data()
    df.to_csv("data/sample/sample_players.csv", index=False)
    print(f"Generated {len(df)} rows for {df['player_id'].nunique()} players")
    print(f"NIL tier distribution:\n{df['nil_tier'].value_counts().sort_index()}")
    print(f"Valuation range: ${df['nil_valuation'].min():,.0f} - ${df['nil_valuation'].max():,.0f}")
