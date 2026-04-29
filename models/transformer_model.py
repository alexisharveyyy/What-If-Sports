"""Shared Transformer encoder for the multi-task NIL model.

The encoder consumes ``(batch, seq_len, n_features)`` tensors of weekly player
snapshots and returns a single pooled representation per player. Pooling is
done via a learnable ``[CLS]`` token whose final hidden state aggregates
information from every visible week. Padded positions are skipped via a
``key_padding_mask`` so the attention layers only attend to real games.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.multitask_head import MultiTaskHead


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding from "Attention Is All You Need"."""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class NILTransformerEncoder(nn.Module):
    """Shared encoder over weekly player sequences.

    Args:
        n_features: Width of the per-week feature vector.
        d_model: Internal model dimension.
        nhead: Number of attention heads.
        num_layers: Number of stacked encoder layers.
        dim_feedforward: Hidden size of the feedforward sub-layer.
        dropout: Dropout probability across the stack.
        max_seq_len: Maximum supported sequence length (excluding the [CLS] slot).
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 20,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Linear(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model, max_len=max_seq_len + 1
        )
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.last_attention: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Encode a batch of weekly snapshots.

        Args:
            x: ``(batch, seq_len, n_features)`` input.
            mask: Optional ``(batch, seq_len)`` boolean mask where ``True``
                marks **valid** (non-padded) positions.
            return_attention: When ``True``, store CLS-row attention weights
                from the final layer in ``self.last_attention`` for
                interpretability.

        Returns:
            ``(batch, d_model)`` pooled representation taken from the [CLS] slot.
        """
        batch_size = x.size(0)
        x = self.input_projection(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.positional_encoding(x)
        x = self.input_dropout(x)

        if mask is not None:
            cls_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            attn_mask = torch.cat([cls_valid, mask.bool()], dim=1)
            key_padding_mask = ~attn_mask
        else:
            key_padding_mask = None

        if return_attention:
            attn_weights = self._collect_attention(x, key_padding_mask)
            self.last_attention = attn_weights

        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        return encoded[:, 0]

    def _collect_attention(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Re-run the final encoder layer with attention weights returned.

        Returns the CLS row of the final layer's attention, averaged over
        heads, shape ``(batch, 1 + seq_len)``.
        """
        layers = list(self.encoder.layers)
        h = x
        for layer in layers[:-1]:
            h = layer(h, src_key_padding_mask=key_padding_mask)
        last = layers[-1]
        normed = last.norm1(h) if last.norm_first else h
        _, attn = last.self_attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        return attn[:, 0, :].detach()


class NILTransformerModel(nn.Module):
    """Backward-compatible wrapper that pairs the shared encoder with the
    legacy ``MultiTaskHead`` used by ``train/train.py``.
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
        max_seq_len: int = 20,
    ) -> None:
        super().__init__()
        self.encoder = NILTransformerEncoder(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.head = MultiTaskHead(
            input_dim=d_model,
            num_tiers=num_tiers,
            alpha=alpha,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        repr_ = self.encoder(x)
        return self.head(repr_)
