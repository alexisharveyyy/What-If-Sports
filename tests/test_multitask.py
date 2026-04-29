"""Tests for the multi-task NIL transformer model and loss."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multitask_head import (
    MultiTaskLoss,
    MultiTaskNILModel,
    NILTierClassificationHead,
    NILValuationRegressionHead,
)
from models.transformer_model import NILTransformerEncoder


BATCH = 6
SEQ_LEN = 20
N_FEATURES = 21
D_MODEL = 64
NUM_TIERS = 5


@pytest.fixture
def encoder() -> NILTransformerEncoder:
    return NILTransformerEncoder(
        n_features=N_FEATURES,
        d_model=D_MODEL,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        max_seq_len=SEQ_LEN,
    )


@pytest.fixture
def model(encoder) -> MultiTaskNILModel:
    return MultiTaskNILModel(encoder, d_model=D_MODEL, num_tiers=NUM_TIERS)


@pytest.fixture
def batch():
    x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
    mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.bool)
    mask[1, 14:] = False
    mask[3, 8:] = False
    return x, mask


class TestModelForward:
    def test_returns_expected_shapes(self, model, batch):
        x, mask = batch
        out = model(x, mask=mask)
        assert isinstance(out, dict)
        assert out["tier_logits"].shape == (BATCH, NUM_TIERS)
        assert out["valuation_pred"].shape == (BATCH,)
        assert out["shared"].shape == (BATCH, D_MODEL)

    def test_valuation_is_non_negative(self, model, batch):
        x, mask = batch
        out = model(x, mask=mask)
        assert (out["valuation_pred"] >= 0).all().item()

    def test_runs_without_mask(self, model):
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert out["tier_logits"].shape == (2, NUM_TIERS)
        assert out["valuation_pred"].shape == (2,)

    def test_attention_capture(self, model, batch):
        x, mask = batch
        model(x, mask=mask, return_attention=True)
        attn = model.encoder.last_attention
        assert attn is not None
        assert attn.shape == (BATCH, SEQ_LEN + 1)


class TestHeadsStandalone:
    def test_classification_head_logits(self):
        head = NILTierClassificationHead(input_dim=D_MODEL, num_tiers=NUM_TIERS)
        out = head(torch.randn(BATCH, D_MODEL))
        assert out.shape == (BATCH, NUM_TIERS)

    def test_regression_head_positive(self):
        head = NILValuationRegressionHead(input_dim=D_MODEL)
        out = head(torch.randn(BATCH, D_MODEL))
        assert out.shape == (BATCH,)
        assert (out >= 0).all().item()


class TestMultiTaskLoss:
    def _make_inputs(self):
        logits = torch.randn(BATCH, NUM_TIERS)
        valuation = torch.rand(BATCH) * 5
        tier_target = torch.randint(0, NUM_TIERS, (BATCH,))
        val_target = torch.rand(BATCH) * 5
        return logits, valuation, tier_target, val_target

    def test_returns_scalar(self):
        loss_fn = MultiTaskLoss(alpha=1.0, beta=1.0)
        logits, valuation, tier, val = self._make_inputs()
        total, metrics = loss_fn(logits, valuation, tier, val)
        assert total.dim() == 0
        assert "cls_loss" in metrics
        assert "reg_loss" in metrics
        assert "total_loss" in metrics

    def test_weighting_applied_correctly(self):
        loss_fn = MultiTaskLoss(alpha=2.0, beta=3.0)
        logits, valuation, tier, val = self._make_inputs()
        total, metrics = loss_fn(logits, valuation, tier, val)
        expected = 2.0 * metrics["cls_loss"] + 3.0 * metrics["reg_loss"]
        assert total.item() == pytest.approx(expected, rel=1e-5)

    def test_uncertainty_weighting_runs(self):
        loss_fn = MultiTaskLoss(use_uncertainty_weighting=True)
        logits, valuation, tier, val = self._make_inputs()
        total, metrics = loss_fn(logits, valuation, tier, val)
        assert total.dim() == 0
        assert "log_var_cls" in metrics
        assert "log_var_reg" in metrics
        assert {p.requires_grad for p in loss_fn.parameters()} == {True}
