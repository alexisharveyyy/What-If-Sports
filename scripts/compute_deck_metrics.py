"""Compute every deck-ready evaluation number for the What-If Sports models.

Subcommands
-----------
    audit     Print which checkpoints/pickles exist for the current schema.
    train     Train only the missing pieces (baselines, BiLSTM+Attn, ablations).
    evaluate  Score every model on the held-out test split, write
              ``deck/real_values.json`` + assets + audit report.
    all       Run audit, train, evaluate in sequence.

The script reuses ``pipeline.preprocess``, ``pipeline.dataset``, the trainers
under ``train/`` and the model classes under ``models/`` rather than
duplicating logic. Random seeds are pinned at 42 across the board.
"""

from __future__ import annotations

import os

# Cap BLAS/OMP threads BEFORE importing numpy/sklearn/torch to avoid thread
# contention segfaults on macOS arm64 Python 3.14.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.baseline import _build_feature_matrix, _last_week_per_player, train_baselines
from models.lstm_model import BiLSTMWithAttention
from models.multitask_head import MultiTaskNILModel
from models.transformer_model import NILTransformerEncoder
from pipeline.dataset import NILSequenceDataset, stratified_split_by_player
from pipeline.features import (
    NUMERIC_FEATURE_COLS,
    TIER_INT_TO_LABEL,
    TIER_LABELS,
    feature_columns,
)
from pipeline.preprocess import preprocess
from train.train_bilstm_attention import BiLSTMTrainConfig
from train.train_bilstm_attention import train as train_bilstm
from train.train_multitask_transformer import (
    TrainConfig as TransformerTrainConfig,
    _collate,
    train as train_transformer,
)


SEED = 42
SAVE_DIR = _REPO_ROOT / "models" / "saved"
DECK_DIR = _REPO_ROOT / "deck"
ASSETS_DIR = DECK_DIR / "assets"

BASELINE_FILES = {
    "logistic": SAVE_DIR / "baseline_logistic.pkl",
    "rf_clf": SAVE_DIR / "baseline_rf_clf.pkl",
    "rf_reg": SAVE_DIR / "baseline_rf_reg.pkl",
    "xgb_clf": SAVE_DIR / "baseline_xgb_clf.pkl",
    "xgb_reg": SAVE_DIR / "baseline_xgb_reg.pkl",
    "feature_cols": SAVE_DIR / "baseline_feature_cols.pkl",
}
TRANSFORMER_FULL = SAVE_DIR / "multitask_transformer_best.pt"
TRANSFORMER_REG_ONLY = SAVE_DIR / "transformer_reg_only.pt"
TRANSFORMER_CLS_ONLY = SAVE_DIR / "transformer_cls_only.pt"
BILSTM_PATH = SAVE_DIR / "bilstm_attention_best.pt"


def _set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _check_baseline_schema() -> str:
    """Return 'current', 'stale', or 'missing' for the existing XGB pickles."""
    if not BASELINE_FILES["xgb_clf"].exists():
        return "missing"
    if not BASELINE_FILES["feature_cols"].exists():
        return "stale"
    try:
        with BASELINE_FILES["feature_cols"].open("rb") as f:
            cols = pickle.load(f)
        if any(c.endswith("_encoded") or c == "performance_score" for c in cols):
            return "current"
        return "stale"
    except Exception:
        return "stale"


def _check_lstm_attention() -> str:
    """Return whether models/lstm_model.py exposes BiLSTMWithAttention."""
    try:
        import importlib

        mod = importlib.import_module("models.lstm_model")
        return "present" if hasattr(mod, "BiLSTMWithAttention") else "missing"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def audit() -> dict[str, Any]:
    """Print and return what's already on disk vs. missing."""
    print("=" * 60)
    print("Audit of trained artifacts")
    print("=" * 60)

    rows = [
        ("multi-task transformer (full)", TRANSFORMER_FULL.exists(), TRANSFORMER_FULL),
        ("multi-task transformer (reg-only)", TRANSFORMER_REG_ONLY.exists(), TRANSFORMER_REG_ONLY),
        ("multi-task transformer (cls-only)", TRANSFORMER_CLS_ONLY.exists(), TRANSFORMER_CLS_ONLY),
        ("BiLSTM + attention", BILSTM_PATH.exists(), BILSTM_PATH),
    ]
    for name, present, path in rows:
        flag = "[OK]    " if present else "[MISS]  "
        print(f"  {flag}{name:<38} {path.relative_to(_REPO_ROOT)}")

    print()
    print("  Baselines:")
    for key in ("logistic", "rf_clf", "rf_reg", "xgb_clf", "xgb_reg", "feature_cols"):
        p = BASELINE_FILES[key]
        flag = "[OK]    " if p.exists() else "[MISS]  "
        print(f"    {flag}{key:<22} {p.relative_to(_REPO_ROOT)}")
    schema = _check_baseline_schema()
    print(f"    Schema check on existing baseline pickles: {schema}")

    print()
    attn = _check_lstm_attention()
    print(f"  BiLSTMWithAttention class exposed by models.lstm_model: {attn}")
    print()

    return {
        "transformer_full": TRANSFORMER_FULL.exists(),
        "transformer_reg_only": TRANSFORMER_REG_ONLY.exists(),
        "transformer_cls_only": TRANSFORMER_CLS_ONLY.exists(),
        "bilstm_attention": BILSTM_PATH.exists(),
        "baseline_schema": schema,
        "lstm_attention_class": attn,
        "baselines_present": all(BASELINE_FILES[k].exists() for k in BASELINE_FILES),
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train_missing(force: bool = False) -> dict[str, str]:
    """Train every missing artifact. Returns a dict describing what ran."""
    actions: dict[str, str] = {}
    schema = _check_baseline_schema()

    if force or schema != "current":
        print("\n>>> Training baselines on current multi-task schema")
        train_baselines(seed=SEED, write=True)
        actions["baselines"] = "trained"
    else:
        actions["baselines"] = "skipped"

    if force or not BILSTM_PATH.exists():
        print("\n>>> Training BiLSTM + Attention")
        train_bilstm(BiLSTMTrainConfig(seed=SEED))
        actions["bilstm_attention"] = "trained"
    else:
        actions["bilstm_attention"] = "skipped"

    if force or not TRANSFORMER_REG_ONLY.exists():
        print("\n>>> Training transformer reg-only ablation")
        cfg = TransformerTrainConfig(
            alpha=0.0, beta=1.0, seed=SEED,
            checkpoint_name="transformer_reg_only.pt",
            report_name="transformer_reg_only_report.json",
            write_plots=False,
        )
        train_transformer(cfg)
        actions["transformer_reg_only"] = "trained"
    else:
        actions["transformer_reg_only"] = "skipped"

    if force or not TRANSFORMER_CLS_ONLY.exists():
        print("\n>>> Training transformer cls-only ablation")
        cfg = TransformerTrainConfig(
            alpha=1.0, beta=0.0, seed=SEED,
            checkpoint_name="transformer_cls_only.pt",
            report_name="transformer_cls_only_report.json",
            write_plots=False,
        )
        train_transformer(cfg)
        actions["transformer_cls_only"] = "trained"
    else:
        actions["transformer_cls_only"] = "skipped"

    return actions


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _data_setup() -> dict[str, Any]:
    """Run preprocess + stratified split. Reused by everything below."""
    df, _, _ = preprocess(write=False)
    train_df, val_df, test_df = stratified_split_by_player(
        df, train_frac=0.7, val_frac=0.15, seed=SEED
    )
    feat_cols = feature_columns(list(df.columns))
    return {
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "feat_cols": feat_cols,
    }


def _build_loader(df: pd.DataFrame, feat_cols: list[str], batch_size: int = 64,
                  max_seq_len: int = 20) -> DataLoader:
    ds = NILSequenceDataset(df, max_seq_len=max_seq_len, feature_cols=feat_cols)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)


@torch.no_grad()
def _model_outputs(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    logits, tiers, log_pred, true_val, pids = [], [], [], [], []
    for batch in loader:
        x = batch["features"].to(device)
        mask = batch["mask"].to(device)
        out = model(x, mask=mask)
        logits.append(out["tier_logits"].cpu().numpy())
        tiers.append(batch["tier"].cpu().numpy())
        log_pred.append(out["valuation_pred"].cpu().numpy())
        true_val.append(batch["valuation"].cpu().numpy())
        pids.extend(batch["player_id"])
    return {
        "logits": np.concatenate(logits),
        "tier_target": np.concatenate(tiers),
        "log_pred": np.concatenate(log_pred),
        "valuation_target": np.concatenate(true_val),
        "player_ids": pids,
    }


def _metrics_from_outputs(outputs: dict[str, np.ndarray]) -> dict[str, float]:
    pred_tier = outputs["logits"].argmax(axis=1)
    pred_val = np.expm1(np.clip(outputs["log_pred"], 0.0, 18.0))
    return {
        "accuracy": float(accuracy_score(outputs["tier_target"], pred_tier)),
        "macro_f1": float(f1_score(outputs["tier_target"], pred_tier,
                                   average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(outputs["valuation_target"], pred_val)),
        "rmse": float(root_mean_squared_error(outputs["valuation_target"], pred_val)),
        "r2": float(r2_score(outputs["valuation_target"], pred_val)),
        "pred_tier": pred_tier,
        "pred_val": pred_val,
    }


def _expected_calibration_error(probs: np.ndarray, targets: np.ndarray,
                                n_bins: int = 15) -> float:
    """Standard binned ECE on top-1 confidences."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(targets)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _fit_temperature(logits_val: np.ndarray, targets_val: np.ndarray,
                     max_iter: int = 300) -> float:
    """Optimize a single temperature scalar on validation logits via LBFGS."""
    logits = torch.tensor(logits_val, dtype=torch.float32)
    targets = torch.tensor(targets_val, dtype=torch.long)
    log_T = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def _closure():
        optimizer.zero_grad()
        T = log_T.exp()
        loss = F.cross_entropy(logits / T, targets)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(log_T.exp().item())


def _calibrate(model: nn.Module, val_loader: DataLoader, test_loader: DataLoader,
               device: torch.device) -> dict[str, float]:
    """Fit temperature on val, apply to test, return ECE before/after."""
    val_out = _model_outputs(model, val_loader, device)
    test_out = _model_outputs(model, test_loader, device)

    test_probs_before = _softmax(test_out["logits"])
    ece_before = _expected_calibration_error(test_probs_before, test_out["tier_target"])

    T = _fit_temperature(val_out["logits"], val_out["tier_target"])
    scaled_logits = test_out["logits"] / max(T, 1e-3)
    test_probs_after = _softmax(scaled_logits)
    ece_after = _expected_calibration_error(test_probs_after, test_out["tier_target"])

    return {
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "temperature": float(T),
        "reduction_pct": float(
            (ece_before - ece_after) / ece_before * 100.0 if ece_before > 0 else 0.0
        ),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def _load_transformer(path: Path, device: torch.device) -> tuple[MultiTaskNILModel, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    encoder = NILTransformerEncoder(
        n_features=ckpt["n_features"],
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 8),
        num_layers=cfg.get("num_layers", 4),
        dim_feedforward=cfg.get("dim_feedforward", 512),
        dropout=0.0,
        max_seq_len=cfg.get("max_seq_len", 20),
    )
    model = MultiTaskNILModel(encoder, d_model=cfg.get("d_model", 128))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def _load_bilstm(path: Path, device: torch.device) -> tuple[MultiTaskNILModel, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    encoder = BiLSTMWithAttention(
        n_features=ckpt["n_features"],
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=0.0,
        attention_dim=cfg.get("attention_dim", 128),
    )
    model = MultiTaskNILModel(encoder, d_model=cfg.get("hidden_dim", 128))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def _load_baselines() -> dict[str, Any]:
    out = {}
    for key, path in BASELINE_FILES.items():
        with path.open("rb") as f:
            out[key] = pickle.load(f)
    return out


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


def _eval_baselines(setup: dict, baselines: dict) -> dict:
    test_last = _last_week_per_player(setup["test_df"])
    X_test, _ = _build_feature_matrix(test_last)
    y_tier = test_last["nil_tier_int"].astype(int).to_numpy()
    y_val = test_last["nil_valuation_usd"].astype(float).to_numpy()

    out: dict[str, dict[str, float]] = {}

    pred_lr = baselines["logistic"].predict(X_test)
    out["logistic_regression"] = {
        "accuracy": float(accuracy_score(y_tier, pred_lr)),
        "macro_f1": float(f1_score(y_tier, pred_lr, average="macro", zero_division=0)),
    }

    pred_rf_clf = baselines["rf_clf"].predict(X_test)
    pred_rf_reg = baselines["rf_reg"].predict(X_test)
    out["random_forest"] = {
        "accuracy": float(accuracy_score(y_tier, pred_rf_clf)),
        "macro_f1": float(f1_score(y_tier, pred_rf_clf, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_val, pred_rf_reg)),
        "rmse": float(root_mean_squared_error(y_val, pred_rf_reg)),
        "r2": float(r2_score(y_val, pred_rf_reg)),
    }

    pred_xgb_clf = baselines["xgb_clf"].predict(X_test)
    pred_xgb_reg = baselines["xgb_reg"].predict(X_test)
    out["xgboost"] = {
        "accuracy": float(accuracy_score(y_tier, pred_xgb_clf)),
        "macro_f1": float(f1_score(y_tier, pred_xgb_clf, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_val, pred_xgb_reg)),
        "rmse": float(root_mean_squared_error(y_val, pred_xgb_reg)),
        "r2": float(r2_score(y_val, pred_xgb_reg)),
    }
    out["_xgb_test_prediction"] = {
        "player_ids": test_last["player_id"].tolist(),
        "pred_val": pred_xgb_reg.tolist(),
        "true_val": y_val.tolist(),
    }
    return out


def _volatile_cohort_metrics(test_df: pd.DataFrame, xgb_pred_map: dict,
                             bilstm_outputs: dict) -> dict:
    """MAE on the top-CV quartile of test players, for XGB vs. BiLSTM+Attn."""
    cv = (
        test_df.groupby("player_id")["nil_valuation_usd"]
        .agg(lambda s: s.std(ddof=0) / max(s.mean(), 1.0))
    )
    threshold = cv.quantile(0.75)
    volatile_ids = set(cv[cv >= threshold].index.tolist())

    xgb_pid = xgb_pred_map["player_ids"]
    xgb_pred = np.array(xgb_pred_map["pred_val"])
    xgb_true = np.array(xgb_pred_map["true_val"])
    xgb_mask = np.array([pid in volatile_ids for pid in xgb_pid])
    xgb_mae = float(mean_absolute_error(xgb_true[xgb_mask], xgb_pred[xgb_mask])) if xgb_mask.any() else float("nan")

    bilstm_pid = bilstm_outputs["player_ids"]
    bilstm_pred = np.expm1(np.clip(bilstm_outputs["log_pred"], 0, None))
    bilstm_true = bilstm_outputs["valuation_target"]
    bilstm_mask = np.array([pid in volatile_ids for pid in bilstm_pid])
    bilstm_mae = float(mean_absolute_error(bilstm_true[bilstm_mask], bilstm_pred[bilstm_mask])) if bilstm_mask.any() else float("nan")

    return {
        "cv_threshold": float(threshold),
        "volatile_player_count": int(len(volatile_ids)),
        "xgb_mae": xgb_mae,
        "bilstm_mae": bilstm_mae,
        "delta_pct": float((bilstm_mae - xgb_mae) / xgb_mae * 100.0) if xgb_mae > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Asset generation
# ---------------------------------------------------------------------------


def _attention_assets(setup: dict, bilstm_model: nn.Module,
                      device: torch.device) -> dict:
    """Pull attention weights for three archetypes from the test set."""
    test_df = setup["test_df"]

    grouped = test_df.groupby("player_id")
    cv = grouped["nil_valuation_usd"].agg(lambda s: s.std(ddof=0) / max(s.mean(), 1.0))
    mean_val = grouped["nil_valuation_usd"].mean()
    injury_run = grouped["currently_injured"].agg(lambda s: int(s.astype(int).sum()))

    consistent_ids = mean_val[mean_val.rank(pct=True) >= 0.85].index
    consistent_pool = cv.loc[consistent_ids].sort_values()
    consistent_pid = consistent_pool.index[0] if len(consistent_pool) else None

    volatile_pid = cv.idxmax() if len(cv) else None
    injury_pid = injury_run.idxmax() if injury_run.max() > 0 else None

    archetypes = {
        "consistent_high_tier": consistent_pid,
        "volatile": volatile_pid,
        "injury_affected": injury_pid,
    }

    feat_cols = setup["feat_cols"]
    test_loader = _build_loader(test_df, feat_cols)
    bilstm_model.eval()

    pid_to_attention: dict[str, np.ndarray] = {}
    pid_to_length: dict[str, int] = {}

    needed = {pid for pid in archetypes.values() if pid is not None}
    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            mask = batch["mask"].to(device)
            _ = bilstm_model(x, mask=mask, return_attention=True)
            attn = bilstm_model.encoder.last_attention.cpu().numpy()
            for i, pid in enumerate(batch["player_id"]):
                if pid in needed:
                    pid_to_attention[pid] = attn[i]
                    pid_to_length[pid] = int(batch["length"][i].item())

    out = {}
    for label, pid in archetypes.items():
        if pid is None or pid not in pid_to_attention:
            continue
        weeks = test_df[test_df["player_id"] == pid].sort_values("week_number")
        out[label] = {
            "player_id": pid,
            "school": str(weeks["school"].iloc[-1]),
            "position": str(weeks["position"].iloc[-1]),
            "valid_weeks": pid_to_length[pid],
            "attention": [float(a) for a in pid_to_attention[pid]],
            "weekly_stats": [
                {
                    "week": int(w),
                    "ppg": float(p),
                    "rpg": float(r),
                    "apg": float(a),
                    "currently_injured": bool(inj),
                }
                for w, p, r, a, inj in zip(
                    weeks["week_number"], weeks["ppg"], weeks["rpg"],
                    weeks["apg"], weeks["currently_injured"],
                )
            ],
        }
    return out


def _feature_importance(baselines: dict) -> list[dict]:
    """Top-6 gain-based importances from the XGBoost regressor."""
    reg = baselines["xgb_reg"]
    cols = baselines["feature_cols"]
    booster = reg.get_booster()
    gain = booster.get_score(importance_type="gain")
    indexed = []
    for k, v in gain.items():
        if k.startswith("f"):
            try:
                idx = int(k[1:])
                if idx < len(cols):
                    indexed.append((cols[idx], float(v)))
                continue
            except ValueError:
                pass
        indexed.append((k, float(v)))

    indexed.sort(key=lambda kv: kv[1], reverse=True)
    top = indexed[:6]
    if not top:
        return []
    top_score = top[0][1]
    return [
        {"feature": name, "gain": value, "normalized": round(100.0 * value / top_score, 2)}
        for name, value in top
    ]


# ---------------------------------------------------------------------------
# Evaluate orchestrator
# ---------------------------------------------------------------------------


def evaluate(write: bool = True) -> dict:
    _set_global_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluation device: {device}")

    setup = _data_setup()
    feat_cols = setup["feat_cols"]
    train_df, val_df, test_df = setup["train_df"], setup["val_df"], setup["test_df"]

    train_loader = _build_loader(train_df, feat_cols)
    val_loader = _build_loader(val_df, feat_cols)
    test_loader = _build_loader(test_df, feat_cols)

    baselines = _load_baselines()
    print("Baselines loaded.")

    baseline_results = _eval_baselines(setup, baselines)
    xgb_test_pred = baseline_results.pop("_xgb_test_prediction")

    # ---- Sequence models ----
    print("Loading transformer (full)...")
    full_model, _ = _load_transformer(TRANSFORMER_FULL, device)
    full_outputs = _model_outputs(full_model, test_loader, device)
    full_metrics = _metrics_from_outputs(full_outputs)
    full_calib = _calibrate(full_model, val_loader, test_loader, device)

    print("Loading transformer (reg-only)...")
    reg_model, _ = _load_transformer(TRANSFORMER_REG_ONLY, device)
    reg_outputs = _model_outputs(reg_model, test_loader, device)
    reg_metrics = _metrics_from_outputs(reg_outputs)

    print("Loading transformer (cls-only)...")
    cls_model, _ = _load_transformer(TRANSFORMER_CLS_ONLY, device)
    cls_outputs = _model_outputs(cls_model, test_loader, device)
    cls_metrics = _metrics_from_outputs(cls_outputs)
    cls_probs_before = _softmax(cls_outputs["logits"])
    cls_ece_before = _expected_calibration_error(cls_probs_before, cls_outputs["tier_target"])

    print("Loading BiLSTM + Attention...")
    bilstm_model, _ = _load_bilstm(BILSTM_PATH, device)
    bilstm_outputs = _model_outputs(bilstm_model, test_loader, device)
    bilstm_metrics = _metrics_from_outputs(bilstm_outputs)
    bilstm_calib = _calibrate(bilstm_model, val_loader, test_loader, device)

    # ---- Volatile cohort ----
    volatile = _volatile_cohort_metrics(test_df, xgb_test_pred, bilstm_outputs)

    # ---- Cross-model deltas ----
    xgb = baseline_results["xgboost"]
    delta_vs_xgb = {
        "mae_pct": float((bilstm_metrics["mae"] - xgb["mae"]) / xgb["mae"] * 100.0),
        "rmse_pct": float((bilstm_metrics["rmse"] - xgb["rmse"]) / xgb["rmse"] * 100.0),
        "r2_abs": float(bilstm_metrics["r2"] - xgb["r2"]),
        "f1_abs": float(bilstm_metrics["macro_f1"] - xgb["macro_f1"]),
    }

    # ---- Compose JSON output ----
    real_values: dict[str, Any] = {
        "seed": SEED,
        "dataset": {
            "total_player_weeks": int(len(setup["df"])),
            "unique_players": int(setup["df"]["player_id"].nunique()),
            "split_sizes": {
                "train": int(train_df["player_id"].nunique()),
                "val": int(val_df["player_id"].nunique()),
                "test": int(test_df["player_id"].nunique()),
            },
            "tier_distribution": {
                label: int((setup["df"]["nil_tier"] == label).sum())
                for label in TIER_LABELS
            },
        },
        "baselines": baseline_results,
        "bilstm_attention": {
            "mae": bilstm_metrics["mae"],
            "rmse": bilstm_metrics["rmse"],
            "r2": bilstm_metrics["r2"],
            "accuracy": bilstm_metrics["accuracy"],
            "macro_f1": bilstm_metrics["macro_f1"],
            "ece_before": bilstm_calib["ece_before"],
            "ece_after": bilstm_calib["ece_after"],
            "volatile_cohort_mae": volatile["bilstm_mae"],
            "delta_vs_xgboost": delta_vs_xgb,
        },
        "transformer_reg_only": {
            "mae": reg_metrics["mae"],
            "rmse": reg_metrics["rmse"],
            "r2": reg_metrics["r2"],
        },
        "transformer_cls_only": {
            "accuracy": cls_metrics["accuracy"],
            "macro_f1": cls_metrics["macro_f1"],
            "ece_before": float(cls_ece_before),
        },
        "transformer_multitask": {
            "mae": full_metrics["mae"],
            "rmse": full_metrics["rmse"],
            "r2": full_metrics["r2"],
            "accuracy": full_metrics["accuracy"],
            "macro_f1": full_metrics["macro_f1"],
            "ece_before": full_calib["ece_before"],
            "ece_after": full_calib["ece_after"],
        },
        "calibration": {
            "bilstm_attention": {
                "ece_before": bilstm_calib["ece_before"],
                "ece_after": bilstm_calib["ece_after"],
                "reduction_pct": bilstm_calib["reduction_pct"],
                "temperature": bilstm_calib["temperature"],
            },
            "transformer_multitask": {
                "ece_before": full_calib["ece_before"],
                "ece_after": full_calib["ece_after"],
                "reduction_pct": full_calib["reduction_pct"],
                "temperature": full_calib["temperature"],
            },
        },
        "volatile_cohort": {
            "definition": "test players with weekly nil_valuation_usd CV >= 75th percentile",
            **volatile,
        },
    }

    # ---- Asset files ----
    attention_payload = _attention_assets(setup, bilstm_model, device)
    importance_payload = _feature_importance(baselines)

    if write:
        DECK_DIR.mkdir(parents=True, exist_ok=True)
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        (DECK_DIR / "real_values.json").write_text(
            json.dumps(real_values, indent=2, default=str)
        )
        (ASSETS_DIR / "attention_heatmaps.json").write_text(
            json.dumps(attention_payload, indent=2, default=str)
        )
        (ASSETS_DIR / "feature_importance.json").write_text(
            json.dumps(importance_payload, indent=2, default=str)
        )
        report_path = DECK_DIR / "replacement_report.md"
        report_path.write_text(_render_report(real_values, importance_payload))

        print(f"\nWrote {DECK_DIR / 'real_values.json'}")
        print(f"Wrote {ASSETS_DIR / 'attention_heatmaps.json'}")
        print(f"Wrote {ASSETS_DIR / 'feature_importance.json'}")
        print(f"Wrote {report_path}")

    return {
        "values": real_values,
        "attention": attention_payload,
        "importance": importance_payload,
    }


# ---------------------------------------------------------------------------
# Audit report rendering
# ---------------------------------------------------------------------------


def _render_report(values: dict, importance: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Deck metrics replacement report")
    lines.append("")
    lines.append(f"Seed: **{values['seed']}**  ")
    ds = values["dataset"]
    lines.append(
        f"Dataset: **{ds['total_player_weeks']:,}** player-weeks across "
        f"**{ds['unique_players']:,}** unique players "
        f"(train {ds['split_sizes']['train']:,} / "
        f"val {ds['split_sizes']['val']:,} / "
        f"test {ds['split_sizes']['test']:,}, stratified by NIL tier).")
    lines.append("")

    lines.append("## Baselines (test split, last-week-per-player)")
    lines.append("")
    lines.append("| Model | Accuracy | Macro F1 | MAE | RMSE | R² |")
    lines.append("|---|---|---|---|---|---|")
    b = values["baselines"]
    lines.append(
        f"| Logistic regression | {b['logistic_regression']['accuracy']:.4f} | "
        f"{b['logistic_regression']['macro_f1']:.4f} | — | — | — |"
    )
    rf = b["random_forest"]
    lines.append(
        f"| Random forest | {rf['accuracy']:.4f} | {rf['macro_f1']:.4f} | "
        f"${rf['mae']:,.0f} | ${rf['rmse']:,.0f} | {rf['r2']:.4f} |"
    )
    xgb = b["xgboost"]
    lines.append(
        f"| XGBoost | {xgb['accuracy']:.4f} | {xgb['macro_f1']:.4f} | "
        f"${xgb['mae']:,.0f} | ${xgb['rmse']:,.0f} | {xgb['r2']:.4f} |"
    )
    lines.append("")

    lines.append("## Sequence models (test split, full-season inputs)")
    lines.append("")
    lines.append("| Model | Accuracy | Macro F1 | MAE | RMSE | R² | ECE before | ECE after |")
    lines.append("|---|---|---|---|---|---|---|---|")

    bilstm = values["bilstm_attention"]
    lines.append(
        f"| BiLSTM + Attention | {bilstm['accuracy']:.4f} | {bilstm['macro_f1']:.4f} | "
        f"${bilstm['mae']:,.0f} | ${bilstm['rmse']:,.0f} | {bilstm['r2']:.4f} | "
        f"{bilstm['ece_before']:.3f} | {bilstm['ece_after']:.3f} |"
    )

    treg = values["transformer_reg_only"]
    lines.append(
        f"| Transformer (reg-only) | — | — | "
        f"${treg['mae']:,.0f} | ${treg['rmse']:,.0f} | {treg['r2']:.4f} | — | — |"
    )
    tcls = values["transformer_cls_only"]
    lines.append(
        f"| Transformer (cls-only) | {tcls['accuracy']:.4f} | {tcls['macro_f1']:.4f} | "
        f"— | — | — | {tcls['ece_before']:.3f} | — |"
    )
    tfull = values["transformer_multitask"]
    lines.append(
        f"| Transformer (multi-task) | {tfull['accuracy']:.4f} | {tfull['macro_f1']:.4f} | "
        f"${tfull['mae']:,.0f} | ${tfull['rmse']:,.0f} | {tfull['r2']:.4f} | "
        f"{tfull['ece_before']:.3f} | {tfull['ece_after']:.3f} |"
    )
    lines.append("")

    vol = values["volatile_cohort"]
    lines.append("## Volatile cohort (top-CV quartile)")
    lines.append("")
    lines.append(f"- CV threshold: {vol['cv_threshold']:.4f}  ")
    lines.append(f"- Volatile players in test split: **{vol['volatile_player_count']}**  ")
    lines.append(f"- XGBoost MAE: **${vol['xgb_mae']:,.0f}**  ")
    lines.append(f"- BiLSTM + Attention MAE: **${vol['bilstm_mae']:,.0f}**  ")
    lines.append(f"- Δ vs. XGBoost: **{vol['delta_pct']:+.2f}%**")
    lines.append("")

    cal = values["calibration"]
    lines.append("## Temperature scaling (15 bins)")
    lines.append("")
    lines.append("| Model | T | ECE before | ECE after | Reduction |")
    lines.append("|---|---|---|---|---|")
    for name, label in (
        ("transformer_multitask", "Transformer (multi-task)"),
        ("bilstm_attention", "BiLSTM + Attention"),
    ):
        c = cal[name]
        lines.append(
            f"| {label} | {c['temperature']:.3f} | {c['ece_before']:.3f} | "
            f"{c['ece_after']:.3f} | {c['reduction_pct']:.2f}% |"
        )
    lines.append("")

    lines.append("## Feature importance (XGBoost regressor, top-6 by gain)")
    lines.append("")
    lines.append("| Rank | Feature | Gain | Normalized |")
    lines.append("|---|---|---|---|")
    for rank, item in enumerate(importance, 1):
        lines.append(
            f"| {rank} | `{item['feature']}` | {item['gain']:.2f} | {item['normalized']}/100 |"
        )
    lines.append("")

    lines.append("## Sanity check")
    lines.append("")
    issues: list[str] = []
    if tfull["mae"] > xgb["mae"]:
        issues.append(
            f"Multi-task transformer MAE (${tfull['mae']:,.0f}) is **higher** than XGBoost "
            f"(${xgb['mae']:,.0f}). The transformer should usually win — investigate before presenting."
        )
    if tfull["ece_after"] > tfull["ece_before"]:
        issues.append(
            f"Transformer ECE rose after temperature scaling "
            f"({tfull['ece_before']:.3f} -> {tfull['ece_after']:.3f}); validation set may be too small."
        )
    if bilstm["ece_after"] > bilstm["ece_before"]:
        issues.append(
            f"BiLSTM ECE rose after temperature scaling "
            f"({bilstm['ece_before']:.3f} -> {bilstm['ece_after']:.3f})."
        )
    if vol["delta_pct"] > 0:
        issues.append(
            f"BiLSTM **lost** to XGBoost on volatile players ({vol['delta_pct']:+.2f}% vs. expectation of negative)."
        )
    counts = ds["tier_distribution"]
    total_tier = sum(counts.values()) or 1
    rare_tiers = [t for t, n in counts.items() if n / total_tier < 0.07]
    if rare_tiers:
        issues.append(
            f"Tiers with <7% of player-weeks may inflate macro F1 noise: {', '.join(rare_tiers)}"
        )
    if not issues:
        lines.append("No flags — every metric pattern matches expectations.")
    else:
        for issue in issues:
            lines.append(f"- {issue}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", help="Print which artifacts exist for the current schema")
    train_p = sub.add_parser("train", help="Train missing pieces only")
    train_p.add_argument("--force", action="store_true",
                         help="Retrain even when artifacts exist")
    sub.add_parser("evaluate", help="Score every model and write deck/real_values.json")
    all_p = sub.add_parser("all", help="audit -> train -> evaluate")
    all_p.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "audit":
        audit()
    elif args.cmd == "train":
        audit()
        train_missing(force=args.force)
    elif args.cmd == "evaluate":
        evaluate()
    elif args.cmd == "all":
        audit()
        train_missing(force=args.force)
        evaluate()


if __name__ == "__main__":
    main()
