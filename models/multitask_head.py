"""Multi-task heads + loss for the NIL transformer.

Provides:
    - ``NILTierClassificationHead``: 5-way classifier over NIL tiers.
    - ``NILValuationRegressionHead``: positive-valued dollar regressor.
    - ``MultiTaskNILModel``: shared encoder + both heads, returns a dict.
    - ``MultiTaskLoss``: cross-entropy + Huber with optional Kendall-Gal
      learnable uncertainty weighting.
    - ``MultiTaskHead``: legacy combined head retained for the older LSTM/
      Transformer training paths so existing checkpoints keep loading.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NILTierClassificationHead(nn.Module):
    """Two-layer MLP producing logits over the five NIL tiers."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64,
                 num_tiers: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tiers),
        )

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        return self.net(shared_repr)


class NILValuationRegressionHead(nn.Module):
    """Two-layer MLP returning a strictly positive dollar prediction.

    A softplus activation on the final output guarantees non-negativity, which
    matches the natural lower bound of NIL valuations.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        raw = self.net(shared_repr).squeeze(-1)
        return F.softplus(raw)


class MultiTaskNILModel(nn.Module):
    """Composes a shared encoder with both heads.

    ``forward`` returns a dict so callers can pick the outputs they need
    without unpacking a fixed tuple order.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_model: int | None = None,
        num_tiers: int = 5,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model is None:
            d_model = getattr(encoder, "d_model", None)
        if d_model is None:
            raise ValueError(
                "Could not infer d_model from encoder; pass it explicitly."
            )

        self.encoder = encoder
        self.tier_head = NILTierClassificationHead(
            input_dim=d_model,
            hidden_dim=head_hidden_dim,
            num_tiers=num_tiers,
            dropout=head_dropout,
        )
        self.valuation_head = NILValuationRegressionHead(
            input_dim=d_model,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        if mask is not None or return_attention:
            shared = self.encoder(x, mask=mask, return_attention=return_attention)
        else:
            shared = self.encoder(x)
        return {
            "tier_logits": self.tier_head(shared),
            "valuation_pred": self.valuation_head(shared),
            "shared": shared,
        }


class MultiTaskLoss(nn.Module):
    """Weighted classification + regression loss.

    Standard mode:
        ``total = alpha * CE + beta * Huber``

    Uncertainty mode (Kendall and Gal, "Multi-Task Learning Using Uncertainty
    to Weigh Losses"):
        ``total = sum_t (1 / (2 * sigma_t**2)) * L_t + log(sigma_t)``

    where each ``log(sigma_t)`` is a learnable scalar parameter.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        huber_delta: float = 1.0,
        use_uncertainty_weighting: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.SmoothL1Loss(beta=huber_delta)

        if use_uncertainty_weighting:
            self.log_var_cls = nn.Parameter(torch.zeros(1))
            self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        tier_logits: torch.Tensor,
        valuation_pred: torch.Tensor,
        tier_target: torch.Tensor,
        valuation_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cls_loss = self.classification_loss(tier_logits, tier_target)
        reg_loss = self.regression_loss(valuation_pred, valuation_target)

        if self.use_uncertainty_weighting:
            precision_cls = torch.exp(-self.log_var_cls)
            precision_reg = torch.exp(-self.log_var_reg)
            total = (
                precision_cls * cls_loss
                + precision_reg * reg_loss
                + self.log_var_cls
                + self.log_var_reg
            ).squeeze()
            metrics = {
                "cls_loss": float(cls_loss.detach().cpu()),
                "reg_loss": float(reg_loss.detach().cpu()),
                "log_var_cls": float(self.log_var_cls.detach().cpu()),
                "log_var_reg": float(self.log_var_reg.detach().cpu()),
                "total_loss": float(total.detach().cpu()),
            }
        else:
            total = self.alpha * cls_loss + self.beta * reg_loss
            metrics = {
                "cls_loss": float(cls_loss.detach().cpu()),
                "reg_loss": float(reg_loss.detach().cpu()),
                "total_loss": float(total.detach().cpu()),
            }

        return total, metrics


class MultiTaskHead(nn.Module):
    """Legacy combined head retained for backward compatibility.

    The training scripts under ``train/train.py`` and the LSTM model expect a
    single object that exposes both heads plus a ``compute_loss`` helper.
    Newer code should use ``MultiTaskNILModel`` + ``MultiTaskLoss`` instead.

    Added improvements:
        - Adds configurable hidden size and dropout.
        - Uses a slightly deeper shared MLP before splitting into the tier and
          valuation heads.
        - Keeps the same return format so older training code still works.
        - Uses SmoothL1Loss for valuation because it is less sensitive to NIL
          outliers than plain MSE.
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_tiers: int = 5,
        alpha: float = 0.5,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.alpha = alpha

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.tier_head = NILTierClassificationHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_tiers=num_tiers,
            dropout=dropout,
        )

        self.value_head = NILValuationRegressionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.SmoothL1Loss()

    def forward(self, shared_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_repr = self.shared(shared_repr)
        return self.tier_head(shared_repr), self.value_head(shared_repr)

    def compute_loss(
        self,
        tier_logits: torch.Tensor,
        value_pred: torch.Tensor,
        tier_target: torch.Tensor,
        value_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        cls = self.cls_loss_fn(tier_logits, tier_target)
        reg = self.reg_loss_fn(value_pred, value_target)

        total = self.alpha * cls + (1 - self.alpha) * reg

        return total, {
            "cls_loss": float(cls.detach().cpu()),
            "reg_loss": float(reg.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }
