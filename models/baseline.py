"""Baseline models: Logistic Regression + XGBoost."""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

from pipeline.dataset import FEATURE_COLS, split_by_player


def train_baselines(feature_matrix_path: str = "data/processed/feature_matrix.csv"):
    """Train and evaluate baseline models."""
    df = pd.read_csv(feature_matrix_path)

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].fillna(0)
    y_tier = df["nil_tier"].values - 1  # 0-indexed
    y_val = df["nil_valuation"].values

    # Split by player
    train_df, val_df, test_df = split_by_player(df)
    X_train = train_df[available_cols].fillna(0)
    X_test = test_df[available_cols].fillna(0)
    y_tier_train = train_df["nil_tier"].values - 1
    y_tier_test = test_df["nil_tier"].values - 1
    y_val_train = train_df["nil_valuation"].values
    y_val_test = test_df["nil_valuation"].values

    results = {}

    # --- Logistic Regression ---
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, multi_class="multinomial")
    lr.fit(X_train, y_tier_train)
    lr_preds = lr.predict(X_test)
    results["logistic_regression"] = {
        "accuracy": accuracy_score(y_tier_test, lr_preds),
        "f1_macro": f1_score(y_tier_test, lr_preds, average="macro"),
    }
    print(f"  Accuracy: {results['logistic_regression']['accuracy']:.4f}")
    print(f"  F1 Macro: {results['logistic_regression']['f1_macro']:.4f}")

    # --- XGBoost Classifier ---
    print("\nTraining XGBoost Classifier...")
    xgb_cls = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
    )
    xgb_cls.fit(X_train, y_tier_train)
    xgb_cls_preds = xgb_cls.predict(X_test)
    results["xgb_classifier"] = {
        "accuracy": accuracy_score(y_tier_test, xgb_cls_preds),
        "f1_macro": f1_score(y_tier_test, xgb_cls_preds, average="macro"),
    }
    print(f"  Accuracy: {results['xgb_classifier']['accuracy']:.4f}")
    print(f"  F1 Macro: {results['xgb_classifier']['f1_macro']:.4f}")

    # --- XGBoost Regressor ---
    print("\nTraining XGBoost Regressor...")
    xgb_reg = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
    )
    xgb_reg.fit(X_train, y_val_train)
    xgb_reg_preds = xgb_reg.predict(X_test)
    results["xgb_regressor"] = {
        "mae": mean_absolute_error(y_val_test, xgb_reg_preds),
        "rmse": root_mean_squared_error(y_val_test, xgb_reg_preds),
        "r2": r2_score(y_val_test, xgb_reg_preds),
    }
    print(f"  MAE: ${results['xgb_regressor']['mae']:,.0f}")
    print(f"  RMSE: ${results['xgb_regressor']['rmse']:,.0f}")
    print(f"  R²: {results['xgb_regressor']['r2']:.4f}")

    # Calibration
    xgb_probs = xgb_cls.predict_proba(X_test)
    print("\nCalibration (per-tier fraction of positives vs. mean predicted):")
    for tier in range(xgb_probs.shape[1]):
        y_bin = (y_tier_test == tier).astype(int)
        if y_bin.sum() > 0:
            frac_pos, mean_pred = calibration_curve(y_bin, xgb_probs[:, tier], n_bins=5)
            print(f"  Tier {tier + 1}: mean_pred={mean_pred.mean():.3f}, frac_pos={frac_pos.mean():.3f}")

    # Save best model
    os.makedirs("models/saved", exist_ok=True)
    with open("models/saved/baseline_xgb_cls.pkl", "wb") as f:
        pickle.dump(xgb_cls, f)
    with open("models/saved/baseline_xgb_reg.pkl", "wb") as f:
        pickle.dump(xgb_reg, f)
    print("\nSaved baseline models to models/saved/")

    return results


if __name__ == "__main__":
    train_baselines()
