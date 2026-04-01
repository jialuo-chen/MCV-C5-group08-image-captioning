"""Q-Former bridge: learned query tokens cross-attend to ViT patch features.

Compresses variable-length ViT patch tokens (e.g. 197) into a fixed number
of query tokens (e.g. 32) in the decoder embedding space. Inspired by
BLIP-2's Querying Transformer architecture.

Input:  (batch, num_patches, encoder_dim)  — raw ViT features
Output: (batch, num_queries, decoder_dim)  — visual prefix for the LM decoder
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QFormerBridge(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        num_queries: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.decoder_dim = decoder_dim

        self.query_embeds = nn.Parameter(torch.empty(num_queries, decoder_dim))
        nn.init.trunc_normal_(self.query_embeds, std=0.02)

        self.input_proj = nn.Linear(encoder_dim, decoder_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(decoder_dim)

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_features: (batch, num_patches, encoder_dim)
        Returns:
            (batch, num_queries, decoder_dim)
        """
        batch = encoder_features.size(0)
        memory = self.input_proj(encoder_features)
        queries = self.query_embeds.unsqueeze(0).expand(batch, -1, -1)
        out = self.layers(queries, memory)
        return self.out_norm(out)
