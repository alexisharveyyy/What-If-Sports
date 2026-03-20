"""Multi-task prediction heads for NIL tier classification + valuation regression."""

import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    """Combined classification + regression head.

    Args:
        input_dim: Size of the shared representation vector.
        num_tiers: Number of NIL tier classes.
        alpha: Weight for classification loss (1-alpha for regression).
    """

    def __init__(self, input_dim: int = 128, num_tiers: int = 5, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

        self.tier_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_tiers),
        )

        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.HuberLoss()

    def forward(self, shared_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (tier_logits, value_pred)."""
        tier_logits = self.tier_head(shared_repr)
        value_pred = self.value_head(shared_repr).squeeze(-1)
        return tier_logits, value_pred

    def compute_loss(
        self,
        tier_logits: torch.Tensor,
        value_pred: torch.Tensor,
        tier_target: torch.Tensor,
        value_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined multi-task loss."""
        cls_loss = self.cls_loss_fn(tier_logits, tier_target)
        reg_loss = self.reg_loss_fn(value_pred, value_target)
        total_loss = self.alpha * cls_loss + (1 - self.alpha) * reg_loss

        return total_loss, {
            "cls_loss": cls_loss.item(),
            "reg_loss": reg_loss.item(),
            "total_loss": total_loss.item(),
        }
