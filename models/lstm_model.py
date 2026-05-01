"""PyTorch BiLSTM with Temporal Attention model for NIL prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multitask_head import MultiTaskHead


class TemporalAttention(nn.Module):
    """Learned attention over BiLSTM hidden states across all timesteps.

    Instead of discarding all but the last hidden state, this computes a
    weighted sum over every timestep so the model can learn which past
    weeks (e.g. a breakout performance 3 weeks ago) matter most for the
    current NIL forecast.

    Args:
        hidden_dim: Size of the BiLSTM output at each timestep
                    (hidden_size * 2 because bidirectional).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Single-layer MLP that scores each timestep
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted context vector.

        Args:
            lstm_out: BiLSTM outputs of shape (batch, seq_len, hidden_dim).

        Returns:
            context: Weighted sum over timesteps, shape (batch, hidden_dim).
            weights: Attention weights for interpretability, shape (batch, seq_len).
        """
        # Score each timestep: (batch, seq_len, 1)
        scores = self.attention(lstm_out)

        # Normalise across the time dimension
        weights = F.softmax(scores, dim=1)          # (batch, seq_len, 1)

        # Weighted sum: (batch, hidden_dim)
        context = (weights * lstm_out).sum(dim=1)

        return context, weights.squeeze(-1)          # weights: (batch, seq_len)


class NILLSTMModel(nn.Module):
    """Bidirectional LSTM + Temporal Attention model for NIL time series prediction.

    Architecture
    ------------
    Input (batch, seq_len, n_features)
        │
        ▼
    BiLSTM  (hidden_dim per direction → hidden_dim*2 total)
        │
        ▼
    TemporalAttention  (learns which past weeks matter most)
        │
        ▼
    Linear projection  (hidden_dim*2 → hidden_dim)  + ReLU
        │
        ▼
    MultiTaskHead
        ├── tier_head  → tier_logits  (batch, num_tiers)
        └── value_head → value_pred   (batch,)

    Args:
        n_features:  Number of input features per timestep.
        hidden_dim:  LSTM hidden size per direction.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout rate (applied after attention and inside LSTM).
        num_tiers:   Number of NIL tier classes.
        alpha:       Multi-task loss weighting (classification share).
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

        # Temporal attention over all BiLSTM timestep outputs
        self.attention = TemporalAttention(hidden_dim=hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)

        # Project from hidden_dim*2 (bidirectional) → hidden_dim
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
            tier_logits: Shape (batch, num_tiers).
            value_pred:  Shape (batch,).
        """
        # BiLSTM over full sequence
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_dim*2)

        # Attention-weighted context instead of last-step only
        context, _ = self.attention(lstm_out)   # (batch, hidden_dim*2)
        context = self.dropout(context)

        # Project down to hidden_dim for the task heads
        shared_repr = F.relu(self.projection(context))  # (batch, hidden_dim)

        return self.head(shared_repr)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-timestep attention weights for a given input.

        Useful for visualising which snapshot weeks drove the prediction.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            weights: Shape (batch, seq_len).
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            _, weights = self.attention(lstm_out)
        return weights
