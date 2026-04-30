"""Tabular baselines for the multi-task NIL pipeline.

Trains and saves three baselines on the current schema:
    * Logistic regression (classification only)
    * Random forest (classification + regression)
    * XGBoost (classification + regression)

All three use the player-level stratified split from
``pipeline.dataset.stratified_split_by_player`` so they share the same train,
val, and test player IDs as the transformer and BiLSTM models. Targets are
taken from the most-recent week per player so the metrics are directly
comparable with the sequence-model evaluation that produces one prediction
per player.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.dataset import stratified_split_by_player
from pipeline.features import NUMERIC_FEATURE_COLS
from pipeline.preprocess import preprocess


SAVE_DIR = _REPO_ROOT / "models" / "saved"


def _last_week_per_player(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values("week_number")
        .groupby("player_id", sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One row per player using their final-week snapshot. The categorical
    encoders fit by ``preprocess`` are already on the frame as ``*_encoded``
    columns, so we concatenate numerics + the engineered categorical codes.
    """
    numerics = [c for c in NUMERIC_FEATURE_COLS if c in df.columns]
    categoricals = [
        c for c in (
            "school_encoded",
            "conference_encoded",
            "position_encoded",
            "class_year_encoded",
            "program_tier",
            "currently_injured",
        )
        if c in df.columns
    ]
    feature_cols = numerics + categoricals
    return df[feature_cols].to_numpy(dtype=np.float32), feature_cols


def _save(obj: Any, name: str) -> Path:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVE_DIR / name
    with path.open("wb") as f:
        pickle.dump(obj, f)
    return path


def train_baselines(
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    write: bool = True,
) -> dict:
    """Fit all three baselines and report test-set metrics.

    Returns a dict with metrics plus references to the fitted estimators and
    the feature column ordering, which the deck-metrics script needs to
    reload for downstream feature-importance extraction.
    """
    print("Loading + preprocessing dataset...")
    df, _, _ = preprocess(write=False)

    train_df, val_df, test_df = stratified_split_by_player(
        df, train_frac=train_frac, val_frac=val_frac, seed=seed
    )

    train_last = _last_week_per_player(train_df)
    val_last = _last_week_per_player(val_df)
    test_last = _last_week_per_player(test_df)

    X_train, feat_cols = _build_feature_matrix(train_last)
    X_val, _ = _build_feature_matrix(val_last)
    X_test, _ = _build_feature_matrix(test_last)

    y_tier_train = train_last["nil_tier_int"].astype(int).to_numpy()
    y_tier_val = val_last["nil_tier_int"].astype(int).to_numpy()
    y_tier_test = test_last["nil_tier_int"].astype(int).to_numpy()

    y_val_train = train_last["nil_valuation_usd"].astype(float).to_numpy()
    y_val_test = test_last["nil_valuation_usd"].astype(float).to_numpy()

    print(
        f"Players  train={len(train_last):,}  val={len(val_last):,}  "
        f"test={len(test_last):,}  features={len(feat_cols)}"
    )

    results: dict[str, dict] = {}
    artifacts: dict[str, Any] = {"feature_cols": feat_cols}

    # ----- Logistic regression (classification only) -----
    print("\nLogistic regression...")
    logistic = LogisticRegression(
        max_iter=2000, solver="lbfgs", random_state=seed, n_jobs=None
    )
    logistic.fit(X_train, y_tier_train)
    pred_lr = logistic.predict(X_test)
    results["logistic_regression"] = {
        "accuracy": float(accuracy_score(y_tier_test, pred_lr)),
        "macro_f1": float(f1_score(y_tier_test, pred_lr, average="macro", zero_division=0)),
    }
    print(
        f"  acc={results['logistic_regression']['accuracy']:.4f}  "
        f"macroF1={results['logistic_regression']['macro_f1']:.4f}"
    )
    artifacts["logistic"] = logistic

    # ----- Random forest -----
    print("\nRandom forest classifier...")
    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=1, random_state=seed,
        class_weight="balanced_subsample",
    )
    rf_clf.fit(X_train, y_tier_train)
    pred_rf_clf = rf_clf.predict(X_test)

    print("Random forest regressor...")
    rf_reg = RandomForestRegressor(
        n_estimators=400, max_depth=None, n_jobs=1, random_state=seed,
    )
    rf_reg.fit(X_train, y_val_train)
    pred_rf_reg = rf_reg.predict(X_test)

    results["random_forest"] = {
        "accuracy": float(accuracy_score(y_tier_test, pred_rf_clf)),
        "macro_f1": float(f1_score(y_tier_test, pred_rf_clf, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_val_test, pred_rf_reg)),
        "rmse": float(root_mean_squared_error(y_val_test, pred_rf_reg)),
        "r2": float(r2_score(y_val_test, pred_rf_reg)),
    }
    print(
        f"  acc={results['random_forest']['accuracy']:.4f}  "
        f"macroF1={results['random_forest']['macro_f1']:.4f}  "
        f"MAE=${results['random_forest']['mae']:,.0f}  "
        f"R2={results['random_forest']['r2']:.4f}"
    )
    artifacts["rf_clf"] = rf_clf
    artifacts["rf_reg"] = rf_reg

    # ----- XGBoost -----
    print("\nXGBoost classifier...")
    xgb_clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        eval_metric="mlogloss", random_state=seed, n_jobs=1,
        tree_method="hist",
    )
    xgb_clf.fit(X_train, y_tier_train)
    pred_xgb_clf = xgb_clf.predict(X_test)

    print("XGBoost regressor...")
    xgb_reg = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        random_state=seed, n_jobs=1, tree_method="hist",
    )
    xgb_reg.fit(X_train, y_val_train)
    pred_xgb_reg = xgb_reg.predict(X_test)

    results["xgboost"] = {
        "accuracy": float(accuracy_score(y_tier_test, pred_xgb_clf)),
        "macro_f1": float(f1_score(y_tier_test, pred_xgb_clf, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_val_test, pred_xgb_reg)),
        "rmse": float(root_mean_squared_error(y_val_test, pred_xgb_reg)),
        "r2": float(r2_score(y_val_test, pred_xgb_reg)),
    }
    print(
        f"  acc={results['xgboost']['accuracy']:.4f}  "
        f"macroF1={results['xgboost']['macro_f1']:.4f}  "
        f"MAE=${results['xgboost']['mae']:,.0f}  "
        f"R2={results['xgboost']['r2']:.4f}"
    )
    artifacts["xgb_clf"] = xgb_clf
    artifacts["xgb_reg"] = xgb_reg

    if write:
        _save(logistic, "baseline_logistic.pkl")
        _save(rf_clf, "baseline_rf_clf.pkl")
        _save(rf_reg, "baseline_rf_reg.pkl")
        _save(xgb_clf, "baseline_xgb_clf.pkl")
        _save(xgb_reg, "baseline_xgb_reg.pkl")
        _save(feat_cols, "baseline_feature_cols.pkl")
        print(f"\nSaved 5 baseline pickles + feature_cols to {SAVE_DIR}")

    return {"metrics": results, "artifacts": artifacts}


if __name__ == "__main__":
    train_baselines()
