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

def assign_nil_valuation(program_tier, ppg, games_played, injury_flag):
    """Generate a realistic NIL valuation based on player attributes."""
    base = {1: 800_000, 2: 400_000, 3: 150_000, 4: 50_000}[program_tier]
    stat_mult = 1 + (ppg / 30.0)
    games_mult = min(games_played / 10.0, 1.5)
    injury_penalty = 0.7 if injury_flag else 1.0
    noise = np.random.uniform(0.75, 1.35)
    return max(5000, base * stat_mult * games_mult * injury_penalty * noise)


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
        program_tier = np.random.choice([1, 2, 3, 4], p=[0.15, 0.30, 0.35, 0.20])

        # Base stats for this player
        base_ppg = np.random.uniform(5, 28)
        base_apg = np.random.uniform(0.5, 8)
        base_rpg = np.random.uniform(1, 12)
        base_spg = np.random.uniform(0.2, 2.5)
        base_bpg = np.random.uniform(0.1, 3.0)
        base_mpg = np.random.uniform(15, 38)
        base_fg_pct = np.random.uniform(0.35, 0.60)
        base_three_pt_pct = np.random.uniform(0.25, 0.45)
        base_ft_pct = np.random.uniform(0.55, 0.90)

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

            spg = max(0, base_spg + np.random.normal(0, 0.3))
            bpg = max(0, base_bpg + np.random.normal(0, 0.3))
            mpg = max(0, base_mpg + np.random.normal(0, 2))
            fg_pct = np.clip(base_fg_pct + np.random.normal(0, 0.03), 0.15, 0.75)
            three_pt_pct = np.clip(base_three_pt_pct + np.random.normal(0, 0.04), 0.10, 0.60)
            ft_pct = np.clip(base_ft_pct + np.random.normal(0, 0.03), 0.30, 1.0)

            nil_val = assign_nil_valuation(program_tier, ppg, games_played, injury_flag)
            tier = nil_tier(nil_val)

            rows.append({
                "player_id": f"P{pid:03d}",
                "snapshot_week": week,
                "sport": "basketball",
                "school": school,
                "conference": conference,
                "ppg": round(ppg, 1),
                "apg": round(apg, 1),
                "rpg": round(rpg, 1),
                "spg": round(spg, 1),
                "bpg": round(bpg, 1),
                "mpg": round(mpg, 1),
                "fg_pct": round(fg_pct, 3),
                "three_pt_pct": round(three_pt_pct, 3),
                "ft_pct": round(ft_pct, 3),
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
