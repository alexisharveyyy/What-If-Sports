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
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_tiers: int = 5,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Bidirectional doubles the hidden dim
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

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
        """
        # x: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)

        # Use last timestep output
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim*2)
        last_hidden = self.dropout(last_hidden)

        shared_repr = self.projection(last_hidden)  # (batch, hidden_dim)
        shared_repr = torch.relu(shared_repr)

        return self.head(shared_repr)
