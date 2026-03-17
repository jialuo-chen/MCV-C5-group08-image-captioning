"""Attention mechanisms for image captioning.

- BahdanauAttention (additive)
- LuongAttention (multiplicative)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention.

    Given encoder spatial features and a decoder hidden state,
    computes a context vector as a weighted sum of encoder features.

    Parameters
    ----------
    encoder_dim : int
        Dimension of encoder feature vectors.
    decoder_dim : int
        Dimension of decoder hidden state.
    attention_dim : int
        Internal projection dimension.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int = 256) -> None:
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(
        self,
        encoder_features: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        encoder_features : Tensor
            ``(batch, num_positions, encoder_dim)``
        decoder_hidden : Tensor
            ``(batch, decoder_dim)``

        Returns
        -------
        context : Tensor
            ``(batch, encoder_dim)``
        weights : Tensor
            ``(batch, num_positions)``
        """
        # Project
        enc_proj = self.encoder_proj(encoder_features)   # (batch, pos, attn_dim)
        dec_proj = self.decoder_proj(decoder_hidden)      # (batch, attn_dim)
        dec_proj = dec_proj.unsqueeze(1)                  # (batch, 1, attn_dim)

        # Score
        energy = torch.tanh(enc_proj + dec_proj)          # (batch, pos, attn_dim)
        scores = self.score(energy).squeeze(-1)           # (batch, pos)
        weights = F.softmax(scores, dim=-1)               # (batch, pos)

        # Context
        context = (encoder_features * weights.unsqueeze(-1)).sum(dim=1)  # (batch, encoder_dim)
        return context, weights


class LuongAttention(nn.Module):
    """Multiplicative (Luong) attention with 'general' scoring.

    Parameters
    ----------
    encoder_dim : int
        Dimension of encoder feature vectors.
    decoder_dim : int
        Dimension of decoder hidden state.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(decoder_dim, encoder_dim, bias=False)

    def forward(
        self,
        encoder_features: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        encoder_features : Tensor
            ``(batch, num_positions, encoder_dim)``
        decoder_hidden : Tensor
            ``(batch, decoder_dim)``

        Returns
        -------
        context : Tensor
            ``(batch, encoder_dim)``
        weights : Tensor
            ``(batch, num_positions)``
        """
        proj = self.proj(decoder_hidden)                  # (batch, encoder_dim)
        scores = torch.bmm(encoder_features, proj.unsqueeze(-1)).squeeze(-1)  # (batch, pos)
        weights = F.softmax(scores, dim=-1)
        context = (encoder_features * weights.unsqueeze(-1)).sum(dim=1)
        return context, weights


def build_attention(cfg) -> nn.Module | None:
    """Build attention module from config, or *None* if disabled."""
    if not cfg.attention.enabled:
        return None
    attn_type = cfg.attention.type
    encoder_dim = cfg.encoder.feature_dim
    decoder_dim = cfg.decoder.hidden_size
    attn_dim = cfg.attention.get("attention_dim", 256)

    if attn_type == "bahdanau":
        return BahdanauAttention(encoder_dim, decoder_dim, attn_dim)
    elif attn_type == "luong":
        return LuongAttention(encoder_dim, decoder_dim)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")
