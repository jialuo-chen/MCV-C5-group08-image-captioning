"""Captioning model: composes encoder + decoder (+ optional attention).

Provides a unified interface for training (``forward``) and inference (``generate``),
plus checkpoint save/load with full config and tokenizer state.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from src.models.encoders import ImageEncoder
from src.models.decoders import RNNDecoder, HFLMDecoder, build_decoder
from src.models.attention import build_attention
from src.data.tokenizer import BaseTokenizer


class CaptioningModel(nn.Module):
    """End-to-end image captioning model.

    Attributes
    ----------
    encoder : ImageEncoder
    decoder : RNNDecoder | HFLMDecoder
    is_hf_lm : bool
        Whether decoder is an HF causal LM.
    use_attention : bool
        Whether attention is active (only for RNN decoders).
    """

    def __init__(
        self,
        encoder: ImageEncoder,
        decoder: RNNDecoder | HFLMDecoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_hf_lm = isinstance(decoder, HFLMDecoder)
        self.use_attention = (
            not self.is_hf_lm and hasattr(decoder, "use_attention") and decoder.use_attention
        )

    def forward(self, images: torch.Tensor, captions: torch.Tensor):
        """Training forward pass.

        Parameters
        ----------
        images : Tensor
            ``(batch, 3, H, W)``
        captions : Tensor
            ``(batch, seq_len)`` — ground-truth caption token ids.

        Returns
        -------
        For RNN decoders: logits ``(batch, vocab_size, seq_len-1)``
        For HF LM decoders: scalar loss
        """
        if self.is_hf_lm:
            encoder_pooled = self.encoder(images, spatial=False)
            return self.decoder(encoder_pooled, captions)
        else:
            encoder_pooled = self.encoder(images, spatial=False)
            encoder_spatial = self.encoder(images, spatial=True) if self.use_attention else None
            return self.decoder(encoder_pooled, captions, encoder_spatial)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        tokenizer: BaseTokenizer | None = None,
        max_length: int = 201,
        **kwargs,
    ) -> list[str]:
        """Generate captions for a batch of images.

        Parameters
        ----------
        images : Tensor
            ``(batch, 3, H, W)``
        tokenizer : BaseTokenizer | None
            Required for RNN decoders to decode token ids to strings.
        max_length : int
            Maximum generation length.

        Returns
        -------
        list[str]
            Generated captions.
        """
        self.eval()
        encoder_pooled = self.encoder(images, spatial=False)

        if self.is_hf_lm:
            return self.decoder.generate(encoder_pooled, max_length=max_length, **kwargs)
        else:
            encoder_spatial = self.encoder(images, spatial=True) if self.use_attention else None
            sequences = self.decoder.generate(
                encoder_pooled,
                sos_id=tokenizer.sos_id,
                eos_id=tokenizer.eos_id,
                max_length=max_length,
                encoder_spatial=encoder_spatial,
            )
            return [tokenizer.decode(seq) for seq in sequences]

    # ----- checkpoint save / load -------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        config: dict,
        tokenizer: BaseTokenizer | None = None,
        epoch: int = 0,
        metrics: dict | None = None,
    ) -> None:
        """Save model checkpoint, config, and tokenizer state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": dict(config),
            "epoch": epoch,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
        }
        torch.save(checkpoint, path)

        # Save tokenizer alongside checkpoint
        if tokenizer is not None:
            tok_path = path.parent / "tokenizer.json"
            tokenizer.save(tok_path)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> tuple["CaptioningModel", dict]:
        """Load model from checkpoint.

        Returns the model and the checkpoint dict (containing config, epoch, metrics).
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]

        # Rebuild model from config
        from src.utils.config import Config
        from src.data.tokenizer import CharTokenizer, WordTokenizer, SubwordTokenizer
        cfg = Config(config)

        # Resolve vocab_size / pad_id from saved tokenizer (required for RNN decoders)
        vocab_size: int | None = None
        pad_id: int = 0
        if cfg.decoder.type == "rnn":
            tok_path = path.parent / "tokenizer.json"
            tok_type = cfg.tokenizer.type
            if tok_type == "char":
                tok = CharTokenizer.load(tok_path) if tok_path.exists() else CharTokenizer()
            elif tok_type == "word":
                tok = WordTokenizer.load(tok_path)
            elif tok_type == "subword":
                tok = SubwordTokenizer.load(tok_path)
            else:
                tok = CharTokenizer()
            vocab_size = tok.vocab_size
            pad_id = tok.pad_id

        model = build_captioning_model(cfg, vocab_size=vocab_size, pad_id=pad_id)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model, checkpoint

    # ----- factory ----------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "CaptioningModel":
        """Build model from config (delegates to ``build_captioning_model``)."""
        return build_captioning_model(cfg)


def build_captioning_model(cfg, vocab_size: int | None = None, pad_id: int = 0) -> CaptioningModel:
    """Build a complete captioning model from config.

    Parameters
    ----------
    cfg : Config
        Full experiment config.
    vocab_size : int | None
        Vocabulary size (required for RNN decoders; ignored for HF LM).
    pad_id : int
        Padding token id.
    """
    encoder = ImageEncoder.from_config(cfg)
    attention = build_attention(cfg) if cfg.decoder.type == "rnn" else None
    decoder = build_decoder(cfg, vocab_size=vocab_size, pad_id=pad_id, attention=attention)
    return CaptioningModel(encoder, decoder)
