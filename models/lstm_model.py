"""PyTorch LSTM/GRU recurrent models for NIL prediction.

Two architectures live in this module:
    - ``NILLSTMModel``: original bidirectional LSTM with last-step pooling and
      the legacy ``MultiTaskHead``. Kept for backward compatibility with old
      checkpoints.
    - ``BiLSTMWithAttention``: bidirectional LSTM + Bahdanau additive attention
      that returns a pooled context vector. Designed to plug into the new
      ``MultiTaskNILModel`` wrapper alongside the transformer encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class BiLSTMWithAttention(nn.Module):
    """Bidirectional LSTM encoder with additive (Bahdanau-style) attention.

    Forward signature mirrors ``NILTransformerEncoder`` so the same
    ``MultiTaskNILModel`` wrapper can host either backbone. Each timestep's
    bidirectional hidden state is scored by a small MLP, masked against the
    padding mask, and softmax-normalized into a probability distribution over
    weeks. The pooled context vector is returned along with optional attention
    capture for interpretability.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.d_model = hidden_dim

        self.input_norm = nn.LayerNorm(n_features)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        bi_hidden = hidden_dim * 2
        self.lstm_norm = nn.LayerNorm(bi_hidden)
        self.attention_proj = nn.Linear(bi_hidden, attention_dim)
        self.attention_score = nn.Linear(attention_dim, 1, bias=False)
        self.context_proj = nn.Linear(bi_hidden, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Encode ``(batch, seq_len, n_features)`` into ``(batch, hidden_dim)``.

        ``mask`` is a boolean tensor where ``True`` marks valid positions; it
        prevents attention from attending to padded weeks.
        """
        x = self.input_norm(x)
        hidden_states, _ = self.lstm(x)  # (B, S, 2H)
        hidden_states = self.lstm_norm(hidden_states)

        scored = self.attention_score(
            torch.tanh(self.attention_proj(hidden_states))
        ).squeeze(-1)  # (B, S)

        if mask is not None:
            scored = scored.masked_fill(~mask.bool(), float("-inf"))

        attn_weights = F.softmax(scored, dim=-1)  # (B, S)
        if return_attention:
            self.last_attention = attn_weights.detach()

        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        pooled = self.context_proj(context)
        return self.dropout(pooled)

