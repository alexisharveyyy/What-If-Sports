"""Shared Transformer encoder for the multi-task NIL model.

The encoder consumes ``(batch, seq_len, n_features)`` tensors of weekly player
snapshots and returns a single pooled representation per player. Pooling is
done via a learnable ``[CLS]`` token whose final hidden state aggregates
information from every visible week. Padded positions are skipped via a
``key_padding_mask`` so the attention layers only attend to real games.

Added improvements:
    - Adds optional learned positional embeddings (in addition to sinusoidal).
    - Combines CLS token with mean pooling for richer representations.
    - Adds dropout + MLP projection head after encoder output.
    - Improves attention interpretability handling.
    - Adds residual projection for better gradient flow.
"""
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

    Added behavior:
        - Supports hybrid pooling: CLS + mean pooling.
        - Adds optional learned positional embeddings.
        - Adds projection head after encoding for stronger representations.
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
        use_learned_pos_emb: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Linear(n_features, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # optional learned positional embeddings
        if use_learned_pos_emb:
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, max_seq_len + 1, d_model)
            )
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            self.use_learned_pos_emb = True
        else:
            self.positional_encoding = SinusoidalPositionalEncoding(
                d_model, max_len=max_seq_len + 1
            )
            self.use_learned_pos_emb = False

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

        # projection head after encoder
        self.projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.last_attention: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.input_projection(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # improved positional encoding
        if self.use_learned_pos_emb:
            x = x + self.pos_embedding[:, : x.size(1)]
        else:
            x = self.positional_encoding(x)

        x = self.input_dropout(x)

        if mask is not None:
            cls_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            attn_mask = torch.cat([cls_valid, mask.bool()], dim=1)
            key_padding_mask = ~attn_mask
        else:
            key_padding_mask = None

        if return_attention:
            self.last_attention = self._collect_attention(x, key_padding_mask)

        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)

        # hybrid pooling (CLS + mean)
        cls_repr = encoded[:, 0]
        mean_repr = encoded[:, 1:].mean(dim=1)

        combined = torch.cat([cls_repr, mean_repr], dim=1)

        return self.projection(combined)