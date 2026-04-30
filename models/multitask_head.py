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