"""Tests for calibration and MC Dropout utilities."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import NILLSTMModel
from models.calibration import (
    CalibratedNILModel,
    expected_calibration_error,
    mc_dropout_predict,
    regression_interval_coverage,
)


BATCH = 8
SEQ_LEN = 6
N_FEATURES = 10
NUM_TIERS = 5


def _fake_loader(n_batches=3):
    """Generator yielding (X_seq, y_tier, y_value) triples in the project's format."""
    for _ in range(n_batches):
        yield (
            torch.randn(BATCH, SEQ_LEN, N_FEATURES),
            torch.randint(0, NUM_TIERS, (BATCH,)),
            torch.randn(BATCH) * 1e4,
        )


class TestECE:
    def test_perfect_calibration_is_zero(self):
        # 1-hot probabilities matching true labels -> ECE = 0
        n, c = 100, NUM_TIERS
        rng = np.random.default_rng(0)
        y = rng.integers(0, c, size=n)
        probs = np.eye(c)[y]
        assert expected_calibration_error(y, probs) == pytest.approx(0.0, abs=1e-6)

    def test_fully_wrong_max_ece(self):
        # Always predict class 0 with prob 1.0, but true labels are class 1
        n = 100
        y = np.ones(n, dtype=int)
        probs = np.zeros((n, NUM_TIERS))
        probs[:, 0] = 1.0
        assert expected_calibration_error(y, probs) == pytest.approx(1.0, abs=1e-6)


class TestTemperatureScaling:
    def test_init_temperature_is_one(self):
        model = NILLSTMModel(n_features=N_FEATURES, num_tiers=NUM_TIERS)
        cal = CalibratedNILModel(model)
        assert cal.temperature.item() == pytest.approx(1.0)

    def test_forward_shapes(self):
        model = NILLSTMModel(n_features=N_FEATURES, num_tiers=NUM_TIERS)
        cal = CalibratedNILModel(model)
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        tier_logits, value_pred = cal(x)
        assert tier_logits.shape == (BATCH, NUM_TIERS)
        assert value_pred.shape == (BATCH,)

    def test_fit_runs_and_keeps_temperature_positive(self):
        torch.manual_seed(0)
        model = NILLSTMModel(n_features=N_FEATURES, num_tiers=NUM_TIERS)
        cal = CalibratedNILModel(model)
        cal.fit_temperature(_fake_loader(), device=torch.device("cpu"))
        assert cal.temperature.item() > 0

    def test_dividing_by_temperature_changes_logits(self):
        model = NILLSTMModel(n_features=N_FEATURES, num_tiers=NUM_TIERS)
        model.eval()
        cal = CalibratedNILModel(model)
        with torch.no_grad():
            cal.temperature.fill_(2.0)
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        raw_logits, _ = model(x)
        cal_logits, _ = cal(x)
        assert torch.allclose(cal_logits, raw_logits / 2.0, atol=1e-5)


class TestMCDropout:
    def test_returns_all_required_keys(self):
        model = NILLSTMModel(n_features=N_FEATURES, num_tiers=NUM_TIERS, dropout=0.5)
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        out = mc_dropout_predict(model, x, n_samples=5)
        for k in [
            "tier_mean", "tier_std", "tier_ci_low", "tier_ci_high",
            "value_mean", "value_std", "value_ci_low", "value_ci_high",
        ]:
            assert k in out
        assert out["tier_mean"].shape == (BATCH, NUM_TIERS)
        assert out["value_mean"].shape == (BATCH,)

    def test_dropout_produces_variance(self):
        torch.manual_seed(0)
        model = NILLSTMModel(n_features=N_FEATURES, num_tiers=NUM_TIERS, dropout=0.5)
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        out = mc_dropout_predict(model, x, n_samples=20)
        assert out["tier_std"].sum().item() > 0
        assert out["value_std"].sum().item() > 0

    def test_ci_bounds_are_ordered(self):
        model = NILLSTMModel(n_features=N_FEATURES, num_tiers=NUM_TIERS, dropout=0.5)
        x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        out = mc_dropout_predict(model, x, n_samples=10)
        assert torch.all(out["value_ci_low"] <= out["value_ci_high"])
        assert torch.all(out["tier_ci_low"] <= out["tier_ci_high"])


class TestRegressionCoverage:
    def test_full_coverage_when_intervals_contain_truth(self):
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.0, 1.0, 2.0])
        hi = np.array([2.0, 3.0, 4.0])
        assert regression_interval_coverage(y, lo, hi) == 1.0

    def test_zero_coverage_when_intervals_miss(self):
        y = np.array([10.0, 20.0])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        assert regression_interval_coverage(y, lo, hi) == 0.0