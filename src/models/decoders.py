"""Decoder modules for image captioning.

Two decoder families:

1. **RNNDecoder** — GRU / LSTM trained from scratch with our tokenizer.
   Supports optional attention over encoder spatial features.
2. **HFLMDecoder** — Pretrained HuggingFace causal LM (e.g. xLSTM) fine-tuned
   for captioning by prepending projected image features as prefix tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.attention import BahdanauAttention, LuongAttention


# ===================================================================
# Custom RNN Decoder (GRU / LSTM)
# ===================================================================

class RNNDecoder(nn.Module):
    """Autoregressive decoder using GRU or LSTM cells.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size of the tokenizer.
    embed_size : int
        Token embedding dimension.
    hidden_size : int
        RNN hidden dimension.
    num_layers : int
        Number of recurrent layers.
    rnn_type : str
        ``"gru"`` or ``"lstm"``.
    dropout : float
        Dropout applied between RNN layers (only if num_layers > 1).
    attention : nn.Module | None
        Optional attention module (Bahdanau / Luong).
    encoder_dim : int | None
        Encoder feature dim (needed when attention is used).
    pad_id : int
        Padding token id.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 1,
        rnn_type: str = "gru",
        dropout: float = 0.0,
        attention: nn.Module | None = None,
        encoder_dim: int | None = None,
        pad_id: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.pad_id = pad_id
        self.attention = attention
        self.use_attention = attention is not None

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)

        # Input to RNN: embed_size (+ encoder_dim if attention)
        rnn_input_size = embed_size
        if self.use_attention:
            rnn_input_size += (encoder_dim or hidden_size)

        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            rnn_input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Project encoder pooled feature → initial hidden state
        self.init_h = nn.Linear(encoder_dim or hidden_size, hidden_size * num_layers)
        if self.rnn_type == "lstm":
            self.init_c = nn.Linear(encoder_dim or hidden_size, hidden_size * num_layers)

    # ----- helpers ---------------------------------------------------------

    def _init_hidden(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """Initialize hidden state from encoder features ``(batch, feature_dim)``."""
        batch = encoder_features.size(0)
        h = self.init_h(encoder_features)  # (batch, hidden_size * num_layers)
        h = h.view(batch, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        if self.rnn_type == "lstm":
            c = self.init_c(encoder_features)
            c = c.view(batch, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
            return (h, c)
        return h

    # ----- forward (teacher-forced training) --------------------------------

    def forward(
        self,
        encoder_pooled: torch.Tensor,
        captions: torch.Tensor,
        encoder_spatial: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Parameters
        ----------
        encoder_pooled : Tensor
            ``(batch, feature_dim)`` — pooled encoder output.
        captions : Tensor
            ``(batch, seq_len)`` — ground-truth caption token ids.
        encoder_spatial : Tensor | None
            ``(batch, num_positions, encoder_dim)`` — spatial features for attention.

        Returns
        -------
        logits : Tensor
            ``(batch, vocab_size, seq_len)`` — ready for CrossEntropyLoss.
        """
        # Exclude last token (we don't predict after <EOS>)
        captions_input = captions[:, :-1]  # input tokens
        embeddings = self.dropout(self.embedding(captions_input))  # (batch, seq-1, embed)

        hidden = self._init_hidden(encoder_pooled)

        if self.use_attention and encoder_spatial is not None:
            # Step-by-step with attention
            outputs = []
            seq_len = embeddings.size(1)
            for t in range(seq_len):
                emb_t = embeddings[:, t, :]  # (batch, embed)
                h_for_attn = hidden[0][-1] if self.rnn_type == "lstm" else hidden[-1]
                context, _ = self.attention(encoder_spatial, h_for_attn)
                rnn_input = torch.cat([emb_t, context], dim=-1).unsqueeze(1)
                out, hidden = self.rnn(rnn_input, hidden)
                outputs.append(out.squeeze(1))
            outputs = torch.stack(outputs, dim=1)  # (batch, seq-1, hidden)
        else:
            # Full sequence (no attention)
            outputs, hidden = self.rnn(embeddings, hidden)

        logits = self.fc_out(outputs)  # (batch, seq-1, vocab_size)
        return logits.permute(0, 2, 1)  # (batch, vocab_size, seq-1)

    # ----- generate (inference) ---------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        encoder_pooled: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_length: int = 201,
        encoder_spatial: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Greedy decoding.

        Returns
        -------
        list[list[int]]
            Generated token id sequences (one per batch element).
        """
        batch_size = encoder_pooled.size(0)
        device = encoder_pooled.device
        hidden = self._init_hidden(encoder_pooled)

        current_token = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        sequences: list[list[int]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_length):
            emb = self.embedding(current_token)  # (batch, 1, embed)

            if self.use_attention and encoder_spatial is not None:
                h_for_attn = hidden[0][-1] if self.rnn_type == "lstm" else hidden[-1]
                context, _ = self.attention(encoder_spatial, h_for_attn)
                rnn_input = torch.cat([emb.squeeze(1), context], dim=-1).unsqueeze(1)
            else:
                rnn_input = emb

            out, hidden = self.rnn(rnn_input, hidden)
            logits = self.fc_out(out.squeeze(1))  # (batch, vocab_size)
            next_token = logits.argmax(dim=-1)     # (batch,)

            for i in range(batch_size):
                if not finished[i]:
                    tok = next_token[i].item()
                    if tok == eos_id:
                        finished[i] = True
                    else:
                        sequences[i].append(tok)

            if all(finished):
                break

            current_token = next_token.unsqueeze(1)

        return sequences


# ===================================================================
# HuggingFace Causal LM Decoder (e.g. xLSTM)
# ===================================================================

class HFLMDecoder(nn.Module):
    """Decoder that wraps a pretrained HuggingFace causal LM.

    Image features are projected into the LM's embedding space and prepended
    as prefix tokens before the caption.

    Parameters
    ----------
    pretrained : str
        HuggingFace model id (e.g. ``"NX-AI/xLSTM-7b"``).
    encoder_dim : int
        Image encoder feature dimension.
    num_prefix_tokens : int
        Number of prefix tokens to generate from image features.
    freeze_lm : bool
        If *True*, freeze the LM backbone and only train the projection.
    """

    def __init__(
        self,
        pretrained: str,
        encoder_dim: int,
        num_prefix_tokens: int = 1,
        freeze_lm: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.lm = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        lm_embed_dim = self.lm.get_input_embeddings().weight.shape[1]
        self.num_prefix_tokens = num_prefix_tokens
        self.prefix_proj = nn.Linear(encoder_dim, lm_embed_dim * num_prefix_tokens)

        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False

    def _get_prefix_embeds(self, encoder_pooled: torch.Tensor) -> torch.Tensor:
        """Project encoder features to prefix embeddings."""
        batch = encoder_pooled.size(0)
        lm_embed_dim = self.lm.get_input_embeddings().weight.shape[1]
        prefix = self.prefix_proj(encoder_pooled)  # (batch, lm_embed_dim * n)
        return prefix.view(batch, self.num_prefix_tokens, lm_embed_dim)

    def forward(
        self,
        encoder_pooled: torch.Tensor,
        captions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Parameters
        ----------
        encoder_pooled : Tensor
            ``(batch, encoder_dim)``
        captions : Tensor
            ``(batch, seq_len)`` — tokenized caption ids (using the LM tokenizer).

        Returns
        -------
        loss : Tensor
            Scalar causal LM loss.
        """
        prefix_embeds = self._get_prefix_embeds(encoder_pooled)
        caption_embeds = self.lm.get_input_embeddings()(captions)
        inputs_embeds = torch.cat([prefix_embeds, caption_embeds], dim=1)

        # Build labels: -100 for prefix positions (ignored in loss), then caption ids
        prefix_labels = torch.full(
            (captions.size(0), self.num_prefix_tokens),
            -100,
            dtype=torch.long,
            device=captions.device,
        )
        labels = torch.cat([prefix_labels, captions], dim=1)
        # Shift labels left by 1 for causal LM (model does this internally)

        outputs = self.lm(inputs_embeds=inputs_embeds, labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        encoder_pooled: torch.Tensor,
        max_length: int = 100,
        **generate_kwargs,
    ) -> list[str]:
        """Generate captions using the LM's generate method.

        Returns
        -------
        list[str]
            Decoded caption strings.
        """
        prefix_embeds = self._get_prefix_embeds(encoder_pooled)
        batch_size = encoder_pooled.size(0)
        device = encoder_pooled.device

        # Use a BOS token as the start
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        start_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        start_embeds = self.lm.get_input_embeddings()(start_ids)
        inputs_embeds = torch.cat([prefix_embeds, start_embeds], dim=1)

        # Create attention mask
        attention_mask = torch.ones(
            (batch_size, self.num_prefix_tokens + 1),
            dtype=torch.long,
            device=device,
        )

        outputs = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions


# ===================================================================
# Factory
# ===================================================================

def build_decoder(cfg, vocab_size: int | None = None, pad_id: int = 0, attention: nn.Module | None = None):
    """Build a decoder from config.

    Parameters
    ----------
    cfg : Config
        Full experiment config.
    vocab_size : int | None
        Vocabulary size (required for RNN decoders).
    pad_id : int
        Padding token id.
    attention : nn.Module | None
        Attention module (for RNN decoders).
    """
    dec_type = cfg.decoder.type

    if dec_type == "rnn":
        return RNNDecoder(
            vocab_size=vocab_size,
            embed_size=cfg.decoder.embed_size,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            rnn_type=cfg.decoder.name,
            dropout=cfg.decoder.get("dropout", 0.0),
            attention=attention,
            encoder_dim=cfg.encoder.feature_dim,
            pad_id=pad_id,
        )
    elif dec_type == "hf_lm":
        return HFLMDecoder(
            pretrained=cfg.decoder.pretrained,
            encoder_dim=cfg.encoder.feature_dim,
            num_prefix_tokens=cfg.decoder.get("num_prefix_tokens", 1),
            freeze_lm=cfg.decoder.get("freeze_lm", False),
        )
    else:
        raise ValueError(f"Unknown decoder type: {dec_type}")
