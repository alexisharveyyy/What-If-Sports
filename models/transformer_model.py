"""Compact Transformer encoder model for NIL prediction."""

import math

import torch
import torch.nn as nn

from models.multitask_head import MultiTaskHead


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class NILTransformerModel(nn.Module):
    """Compact Transformer encoder for NIL time series prediction.

    Uses a learnable CLS token for pooling.

    Args:
        n_features: Number of input features per timestep.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward network dimension.
        dropout: Dropout rate.
        num_tiers: Number of NIL tier classes.
        alpha: Multi-task loss weight.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        num_tiers: int = 5,
        alpha: float = 0.5,
    ):
        super().__init__()

        # Project input features to model dimension
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)

        self.head = MultiTaskHead(
            input_dim=d_model,
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
        batch_size = x.size(0)

        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+seq_len, d_model)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, 1+seq_len, d_model)

        # CLS token pooling
        cls_output = x[:, 0, :]  # (batch, d_model)
        cls_output = self.dropout(cls_output)

        return self.head(cls_output)
