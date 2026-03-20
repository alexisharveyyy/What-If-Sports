"""Tests for model forward passes and output shapes."""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import NILLSTMModel
from models.transformer_model import NILTransformerModel
from models.multitask_head import MultiTaskHead


BATCH = 4
SEQ_LEN = 8
N_FEATURES = 20
NUM_TIERS = 5


@pytest.fixture
def random_input():
    return torch.randn(BATCH, SEQ_LEN, N_FEATURES)


class TestMultiTaskHead:
    def test_forward_shapes(self):
        head = MultiTaskHead(input_dim=128, num_tiers=NUM_TIERS)
        shared = torch.randn(BATCH, 128)
        tier_logits, value_pred = head(shared)
        assert tier_logits.shape == (BATCH, NUM_TIERS)
        assert value_pred.shape == (BATCH,)

    def test_compute_loss(self):
        head = MultiTaskHead(input_dim=128, num_tiers=NUM_TIERS)
        shared = torch.randn(BATCH, 128)
        tier_logits, value_pred = head(shared)
        tier_target = torch.randint(0, NUM_TIERS, (BATCH,))
        value_target = torch.randn(BATCH)

        loss, details = head.compute_loss(tier_logits, value_pred, tier_target, value_target)
        assert loss.dim() == 0  # scalar
        assert "cls_loss" in details
        assert "reg_loss" in details
        assert "total_loss" in details


class TestLSTMModel:
    def test_forward_shapes(self, random_input):
        model = NILLSTMModel(n_features=N_FEATURES, hidden_dim=128, num_tiers=NUM_TIERS)
        tier_logits, value_pred = model(random_input)
        assert tier_logits.shape == (BATCH, NUM_TIERS)
        assert value_pred.shape == (BATCH,)

    def test_different_seq_lengths(self):
        model = NILLSTMModel(n_features=N_FEATURES)
        for seq_len in [4, 8, 16]:
            x = torch.randn(2, seq_len, N_FEATURES)
            tier_logits, value_pred = model(x)
            assert tier_logits.shape == (2, 5)
            assert value_pred.shape == (2,)


class TestTransformerModel:
    def test_forward_shapes(self, random_input):
        model = NILTransformerModel(n_features=N_FEATURES, d_model=128, nhead=4, num_tiers=NUM_TIERS)
        tier_logits, value_pred = model(random_input)
        assert tier_logits.shape == (BATCH, NUM_TIERS)
        assert value_pred.shape == (BATCH,)

    def test_different_seq_lengths(self):
        model = NILTransformerModel(n_features=N_FEATURES, d_model=128, nhead=4)
        for seq_len in [4, 8, 16]:
            x = torch.randn(2, seq_len, N_FEATURES)
            tier_logits, value_pred = model(x)
            assert tier_logits.shape == (2, 5)
            assert value_pred.shape == (2,)
