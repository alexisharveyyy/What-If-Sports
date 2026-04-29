"""What-if simulation engine backed by the multi-task NIL transformer."""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.multitask_head import MultiTaskNILModel
from models.transformer_model import NILTransformerEncoder
from pipeline.config.schools import market_size_score
from pipeline.features import (
    NUMERIC_FEATURE_COLS,
    SEQUENTIAL_FEATURE_COLS,
    TIER_INT_TO_LABEL,
)


DEFAULT_MODEL_PATH = _REPO_ROOT / "models" / "saved" / "multitask_transformer_best.pt"
DEFAULT_SCALER_PATH = _REPO_ROOT / "pipeline" / "scaler.pkl"
DEFAULT_ENCODERS_PATH = _REPO_ROOT / "pipeline" / "encoders.pkl"


@dataclass
class SimulationResult:
    nil_tier: str
    nil_tier_index: int
    tier_probs: dict[str, float]
    nil_valuation_usd: float
    direction: str

    def to_dict(self) -> dict:
        valuation = round(self.nil_valuation_usd, 2)
        ordered_probs = [
            self.tier_probs[TIER_INT_TO_LABEL[i]]
            for i in range(len(TIER_INT_TO_LABEL))
        ]
        return {
            "nil_tier": self.nil_tier,
            "nil_tier_index": self.nil_tier_index,
            "tier_probs": self.tier_probs,
            "nil_tier_probs": ordered_probs,
            "nil_valuation_usd": valuation,
            "nil_valuation": valuation,
            "direction": self.direction,
        }


class WhatIfSimulator:
    """Wraps the trained ``MultiTaskNILModel`` for player-builder requests."""

    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
        scaler_path: str | os.PathLike | None = None,
        encoders_path: str | os.PathLike | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: MultiTaskNILModel | None = None
        self.config: dict | None = None
        self.feature_cols: list[str] | None = None
        self.scaler = None
        self.encoders: dict | None = None
        self.max_seq_len: int = 20

        scaler_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH
        encoders_path = Path(encoders_path) if encoders_path else DEFAULT_ENCODERS_PATH
        if scaler_path.exists():
            with scaler_path.open("rb") as f:
                self.scaler = pickle.load(f)
        if encoders_path.exists():
            with encoders_path.open("rb") as f:
                self.encoders = pickle.load(f)

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if path.exists():
            self.load_model(path)

    def load_model(self, path: str | os.PathLike) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.config = ckpt.get("config", {})
        self.feature_cols = ckpt.get("feature_cols") or SEQUENTIAL_FEATURE_COLS
        self.max_seq_len = self.config.get("max_seq_len", 20)

        encoder = NILTransformerEncoder(
            n_features=ckpt["n_features"],
            d_model=self.config.get("d_model", 128),
            nhead=self.config.get("nhead", 8),
            num_layers=self.config.get("num_layers", 4),
            dim_feedforward=self.config.get("dim_feedforward", 512),
            dropout=0.0,
            max_seq_len=self.max_seq_len,
        )
        self.model = MultiTaskNILModel(
            encoder, d_model=self.config.get("d_model", 128)
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def build_snapshot(self, inputs: dict, week_number: int = 1) -> dict:
        """Convert player-builder UI fields to a single-week row."""
        school = inputs.get("school", "Unknown")
        program_tier = int(inputs.get("program_tier", 3))
        market_score = inputs.get("market_size_score") or market_size_score(
            school, program_tier
        )

        ppg = float(inputs.get("ppg", 0))
        apg = float(inputs.get("apg", 0))
        rpg = float(inputs.get("rpg", 0))
        fg_pct = float(inputs.get("fg_pct", 0.4))
        three_pt_pct = float(inputs.get("three_pt_pct", 0.3))
        mpg = float(inputs.get("mpg", 20))
        perf = inputs.get("performance_score") or _performance_score(
            ppg, apg, rpg, fg_pct, three_pt_pct, mpg
        )

        return {
            "school": school,
            "conference": inputs.get("conference", "Independent"),
            "position": inputs.get("position", "SF"),
            "class_year": inputs.get("class_year", "FR"),
            "program_tier": program_tier,
            "week_number": week_number,
            "ppg": ppg,
            "apg": apg,
            "rpg": rpg,
            "spg": float(inputs.get("spg", 0)),
            "bpg": float(inputs.get("bpg", 0)),
            "mpg": mpg,
            "fg_pct": fg_pct,
            "three_pt_pct": three_pt_pct,
            "ft_pct": float(inputs.get("ft_pct", 0.7)),
            "games_played": int(inputs.get("games_played", 1)),
            "currently_injured": bool(
                inputs.get("currently_injured", inputs.get("injury_flag", False))
            ),
            "social_media_followers": int(inputs.get("social_media_followers", 5000)),
            "engagement_rate": float(inputs.get("engagement_rate", 0.03)),
            "market_size_score": float(market_score),
            "performance_score": float(perf),
        }

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.encoders is not None:
            for col, encoder in self.encoders.items():
                if col not in df.columns:
                    continue
                values = df[col].astype(str).values
                known = set(encoder.classes_)
                fallback = encoder.classes_[0]
                values = np.array([v if v in known else fallback for v in values])
                df[f"{col}_encoded"] = encoder.transform(values)
        df["currently_injured"] = df["currently_injured"].astype(int)
        return df

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.scaler is not None:
            df[NUMERIC_FEATURE_COLS] = self.scaler.transform(
                df[NUMERIC_FEATURE_COLS].values
            )
        return df

    def _to_tensor(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        cols = [c for c in (self.feature_cols or SEQUENTIAL_FEATURE_COLS)
                if c in df.columns]
        seq = df[cols].to_numpy(dtype=np.float32)
        length = min(len(seq), self.max_seq_len)
        seq = seq[-self.max_seq_len:] if len(seq) > self.max_seq_len else seq

        padded = np.zeros((self.max_seq_len, len(cols)), dtype=np.float32)
        padded[:length] = seq
        mask = np.zeros(self.max_seq_len, dtype=np.bool_)
        mask[:length] = True

        return (
            torch.from_numpy(padded).unsqueeze(0).to(self.device),
            torch.from_numpy(mask).unsqueeze(0).to(self.device),
        )

    def simulate(
        self,
        player_history: list[dict] | None,
        new_snapshot: dict,
    ) -> dict:
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Train the multitask transformer first."
            )

        history = player_history or []
        rows = [self.build_snapshot(s, week_number=i + 1)
                for i, s in enumerate(history)]
        rows.append(self.build_snapshot(new_snapshot, week_number=len(rows) + 1))

        df = pd.DataFrame(rows)
        df = self._encode(df)
        df = self._scale(df)

        x, mask = self._to_tensor(df)
        with torch.no_grad():
            out = self.model(x, mask=mask)
            probs = torch.softmax(out["tier_logits"], dim=1).squeeze(0).cpu().numpy()
            tier_idx = int(np.argmax(probs))
            log_pred = float(out["valuation_pred"].squeeze(0).cpu())
            valuation = float(np.expm1(max(log_pred, 0.0)))

        prev_val = history[-1].get("nil_valuation_usd") if history else None
        if prev_val:
            pct = (valuation - prev_val) / max(prev_val, 1.0) * 100
            direction = "up" if pct > 5 else "down" if pct < -5 else "stable"
        else:
            direction = "stable"

        result = SimulationResult(
            nil_tier=TIER_INT_TO_LABEL[tier_idx],
            nil_tier_index=tier_idx,
            tier_probs={TIER_INT_TO_LABEL[i]: float(p) for i, p in enumerate(probs)},
            nil_valuation_usd=valuation,
            direction=direction,
        )
        return result.to_dict()

    def simulate_timeline(self, base_profile: dict, weeks: int = 4) -> list[dict]:
        timeline: list[dict] = []
        history: list[dict] = []
        for week in range(1, weeks + 1):
            snapshot = {**base_profile}
            if week > 1:
                for stat in ("ppg", "apg", "rpg", "spg", "bpg", "mpg"):
                    snapshot[stat] = max(
                        0.0,
                        snapshot.get(stat, 0.0) + np.random.normal(0, 0.5),
                    )
            result = self.simulate(history, snapshot)
            result["week"] = week
            timeline.append(result)
            history.append({**snapshot, "nil_valuation_usd": result["nil_valuation_usd"]})
        return timeline


def _performance_score(ppg: float, apg: float, rpg: float, fg_pct: float,
                       three_pt_pct: float, mpg: float) -> float:
    return (
        0.30 * (ppg / 28.0)
        + 0.15 * (apg / 8.0)
        + 0.15 * (rpg / 12.0)
        + 0.10 * (fg_pct / 0.65)
        + 0.10 * (three_pt_pct / 0.45)
        + 0.20 * (mpg / 38.0)
    )
