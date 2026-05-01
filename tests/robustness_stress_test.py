"""Scenario robustness stress test for baseline NIL models.

This script perturbs numeric inputs and measures prediction stability:
- tier flip rate
- valuation percent change statistics
- overall stability flag against fixed thresholds
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.preprocess import preprocess


NUMERIC_COLS = [
    "ppg",
    "apg",
    "rpg",
    "spg",
    "bpg",
    "mpg",
    "fg_pct",
    "three_pt_pct",
    "ft_pct",
    "engagement_rate",
    "market_size_score",
    "performance_score",
]

PCT_BOUNDED_COLS = {"fg_pct", "three_pt_pct", "ft_pct", "engagement_rate"}


def run_stress_test(
    sample_size: int,
    perturb_runs: int,
    noise_scale: float,
    seed: int,
) -> None:
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models" / "saved"

    xgb_clf = joblib.load(models_dir / "baseline_xgb_clf.pkl")
    xgb_reg = joblib.load(models_dir / "baseline_xgb_reg.pkl")
    feature_cols = joblib.load(models_dir / "baseline_feature_cols.pkl")

    df, _, _ = preprocess(write=False)
    sample_n = min(sample_size, len(df))
    sample = df.sample(sample_n, random_state=seed).copy()

    x_base = sample[feature_cols].copy()
    base_cls = xgb_clf.predict(x_base)
    base_reg = xgb_reg.predict(x_base)

    rng = np.random.default_rng(seed)
    flip_rates: list[float] = []
    mean_abs_pct_changes: list[float] = []
    max_abs_pct_changes: list[float] = []

    for _ in range(perturb_runs):
        x_perturbed = x_base.copy()
        for col in NUMERIC_COLS:
            if col not in x_perturbed.columns:
                continue

            values = x_perturbed[col].to_numpy(dtype=float)
            noise = rng.normal(
                0.0,
                noise_scale * np.maximum(np.abs(values), 1.0),
                size=values.shape,
            )
            perturbed = values + noise

            if col in PCT_BOUNDED_COLS:
                perturbed = np.clip(perturbed, 0.0, 1.0)
            else:
                perturbed = np.clip(perturbed, 0.0, None)

            x_perturbed[col] = perturbed

        cls_pred = xgb_clf.predict(x_perturbed)
        reg_pred = xgb_reg.predict(x_perturbed)

        flip_rates.append(float(np.mean(cls_pred != base_cls)))
        pct_change = np.abs((reg_pred - base_reg) / np.maximum(np.abs(base_reg), 1.0)) * 100
        mean_abs_pct_changes.append(float(np.mean(pct_change)))
        max_abs_pct_changes.append(float(np.max(pct_change)))

    summary = {
        "samples": len(x_base),
        "perturb_runs": perturb_runs,
        "noise_scale": noise_scale,
        "tier_flip_rate_mean": float(np.mean(flip_rates)),
        "tier_flip_rate_p95": float(np.percentile(flip_rates, 95)),
        "valuation_pct_change_mean": float(np.mean(mean_abs_pct_changes)),
        "valuation_pct_change_p95": float(np.percentile(mean_abs_pct_changes, 95)),
        "valuation_pct_change_max_seen": float(np.max(max_abs_pct_changes)),
    }

    stable = (
        summary["tier_flip_rate_mean"] <= 0.20
        and summary["valuation_pct_change_mean"] <= 15.0
        and summary["valuation_pct_change_max_seen"] <= 45.0
    )

    print("Scenario Robustness Summary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")
    print(f"- stable: {stable}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--perturb-runs", type=int, default=50)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_stress_test(
        sample_size=args.sample_size,
        perturb_runs=args.perturb_runs,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
