"""PyTorch LSTM/GRU recurrent model for NIL prediction."""

import torch
import torch.nn as nn

from models.multitask_head import MultiTaskHead


class NILLSTMModel(nn.Module):
    """Bidirectional LSTM model for NIL time series prediction.

    Args:
        n_features: Number of input features per timestep.
        hidden_dim: LSTM hidden size.
        num_layers: Number of LSTM layers.
        dropout: Dropout rate.
        num_tiers: Number of NIL tier classes.
        alpha: Multi-task loss weight.

    Added improvements:
        - Supports either LSTM or GRU through the ``rnn_type`` argument.
        - Adds LayerNorm after the recurrent encoder to stabilize training.
        - Uses mean + max pooling over all timesteps instead of only the last
          timestep, so the model can use information from the full athlete
          history.
        - Adds a deeper projection layer for a stronger shared representation.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_tiers: int = 5,
        alpha: float = 0.5,
        rnn_type: str = "lstm",
    ):
        super().__init__()

        rnn_type = rnn_type.lower()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be either 'lstm' or 'gru'.")

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        # Mean pooling + max pooling doubles the bidirectional output size.
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.head = MultiTaskHead(
            input_dim=hidden_dim,
            num_tiers=num_tiers,
            alpha=alpha,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            (tier_logits, value_pred) tuple.

        Added behavior:
            Instead of using only the last timestep, this forward pass combines
            mean pooling and max pooling across the full sequence. This helps
            the model capture both long-term trends and standout moments in an
            athlete's performance or popularity history.
        """
        rnn_out, _ = self.rnn(x)
        rnn_out = self.norm(rnn_out)

        mean_pool = rnn_out.mean(dim=1)
        max_pool = rnn_out.max(dim=1).values

        pooled = torch.cat([mean_pool, max_pool], dim=1)
        pooled = self.dropout(pooled)

        shared_repr = self.projection(pooled)

        return self.head(shared_repr)