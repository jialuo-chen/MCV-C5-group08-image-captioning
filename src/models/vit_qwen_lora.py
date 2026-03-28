"""ViT encoder + Qwen3.5 decoder with LoRA adapters.

Uses a frozen ViT (pretrained or fine-tuned in Task 2) as an image feature
extractor and fine-tunes a Qwen3.5 causal-LM decoder with PEFT LoRA.
A learnable projection maps ViT patch tokens into the Qwen embedding space.

Qwen3.5 has a hybrid architecture with GatedDeltaNet (linear attention) +
Gated Attention layers + MLP (FFN) blocks.  LoRA target presets:

* ``"all"``       – every nn.Linear (attention + linear + lm_head)
* ``"linear"``    – MLP/FFN layers only (gate_proj, up_proj, down_proj)
* ``"attention"`` – all attention layers (standard + GatedDeltaNet)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

LORA_TARGETS: dict[str, list[str]] = {
    # "all" is resolved dynamically from decoder nn.Linear module names.
    "all": [],
    "linear": [
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "attention": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "out_proj",
    ],
}


def _resolve_lora_target_modules(
    decoder: nn.Module,
    lora_target: str,
) -> list[str]:
    """Resolve LoRA target_modules for PEFT.

    For Qwen3.5 models, PEFT requires explicit ``target_modules``. For the
    ``"all"`` preset, build this list from all ``nn.Linear`` module suffixes
    present in the loaded decoder (e.g., q_proj, gate_proj, in_proj_qkv,
    lm_head). This keeps behavior model-family aware across 0.8B/4B/etc.
    """
    if lora_target == "all":
        linear_suffixes = {
            name.rsplit(".", 1)[-1]
            for name, module in decoder.named_modules()
            if isinstance(module, nn.Linear)
        }
        if not linear_suffixes:
            raise ValueError("No nn.Linear modules found to target for LoRA")
        return sorted(linear_suffixes)

    if lora_target in LORA_TARGETS:
        return LORA_TARGETS[lora_target]

    valid = ", ".join(sorted(LORA_TARGETS.keys()))
    raise ValueError(f"Unknown lora target '{lora_target}'. Valid options: {valid}")


class ViTQwenLoRA(nn.Module):
    """Frozen ViT encoder + Qwen3.5 decoder with LoRA + learnable projection.

    Parameters
    ----------
    encoder_id : str
        HuggingFace vision model id (e.g. ``google/vit-base-patch16-224``).
    decoder_id : str
        HuggingFace causal-LM id (e.g. ``Qwen/Qwen3.5-0.8B``).
    lora_r : int
        LoRA rank.
    lora_alpha : int
        LoRA scaling factor.
    lora_dropout : float
        LoRA dropout.
    lora_target : str
        One of ``"all"``, ``"linear"``, ``"attention"``.
    encoder_checkpoint : str | None
        Path to a fine-tuned ViT checkpoint directory (``save_pretrained``
        format). If *None*, uses the pretrained ``encoder_id`` weights.
    num_prefix_tokens : int
        Number of ViT patch tokens forwarded as visual prefix (0 = use all).
    """

    def __init__(
        self,
        encoder_id: str,
        decoder_id: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target: str = "all",
        encoder_checkpoint: str | None = None,
        num_prefix_tokens: int = 0,  # 0 means use all patch tokens
    ) -> None:
        super().__init__()

        enc_path = encoder_checkpoint or encoder_id
        self.encoder = AutoModel.from_pretrained(enc_path, trust_remote_code=True)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        enc_cfg = self.encoder.config
        self.encoder_dim: int = getattr(enc_cfg, "hidden_size", 768)
        self.num_prefix_tokens = num_prefix_tokens

        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            decoder_id,
            trust_remote_code=True,
            padding_side="left",  # decoder-only → left-pad for batched generation
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        target_modules = _resolve_lora_target_modules(self.decoder, lora_target)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        self.decoder = get_peft_model(self.decoder, lora_cfg)
        self.decoder.print_trainable_parameters()

        dec_embed_dim: int = self.decoder.get_input_embeddings().weight.shape[1]
        self.projection = nn.Linear(self.encoder_dim, dec_embed_dim)

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract ViT patch tokens.  Returns ``(batch, num_patches, encoder_dim)``."""
        with torch.no_grad():
            outputs = self.encoder(pixel_values=pixel_values)
        features = outputs.last_hidden_state
        if self.num_prefix_tokens > 0:
            features = features[:, 1 : 1 + self.num_prefix_tokens, :]
        return features

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training forward pass.  Returns scalar loss.

        Parameters
        ----------
        pixel_values : ``(batch, 3, H, W)``
        input_ids : ``(batch, seq_len)`` — tokenized caption ids.
        attention_mask : ``(batch, seq_len)`` or *None*.
        labels : ``(batch, seq_len)`` — same as input_ids with ``-100`` for padding.
        """
        vit_features = self._encode_images(pixel_values)  # (B, P, enc_dim)
        vit_projected = self.projection(
            vit_features.to(self.projection.weight.dtype)
        )  # (B, P, dec_dim)

        caption_embeds = self.decoder.get_input_embeddings()(input_ids)  # (B, S, dec_dim)
        inputs_embeds = torch.cat(
            [vit_projected, caption_embeds], dim=1
        )  # (B, P+S, dec_dim)

        batch, prefix_len = pixel_values.size(0), vit_projected.size(1)

        prefix_mask = torch.ones(
            batch, prefix_len, dtype=torch.long, device=pixel_values.device
        )
        if attention_mask is not None:
            full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            full_mask = torch.cat([prefix_mask, torch.ones_like(input_ids)], dim=1)

        if labels is not None:
            prefix_labels = torch.full(
                (batch, prefix_len), -100, dtype=torch.long, device=labels.device
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            full_labels = None

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 128,
        **generate_kwargs,
    ) -> list[str]:
        """Generate captions for a batch of images."""
        vit_features = self._encode_images(pixel_values)
        vit_projected = self.projection(vit_features.to(self.projection.weight.dtype))

        batch = pixel_values.size(0)
        prefix_len = vit_projected.size(1)

        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        start_ids = torch.full(
            (batch, 1), bos_id, dtype=torch.long, device=pixel_values.device
        )
        start_embeds = self.decoder.get_input_embeddings()(start_ids)
        inputs_embeds = torch.cat([vit_projected, start_embeds], dim=1)

        attention_mask = torch.ones(
            (batch, prefix_len + 1), dtype=torch.long, device=pixel_values.device
        )

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def save_checkpoint(self, save_dir: str) -> None:
        """Save LoRA adapter + projection weights."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.decoder.save_pretrained(save_path / "lora_adapter")
        self.tokenizer.save_pretrained(save_path / "lora_adapter")
        torch.save(self.projection.state_dict(), save_path / "projection.pt")

    @classmethod
    def load_checkpoint(
        cls,
        save_dir: str,
        encoder_id: str,
        decoder_id: str,
        device: str = "cuda",
        **kwargs,
    ) -> "ViTQwenLoRA":
        """Load a saved LoRA checkpoint."""
        save_path = Path(save_dir)
        model = cls.__new__(cls)
        nn.Module.__init__(model)

        model.encoder = AutoModel.from_pretrained(encoder_id, trust_remote_code=True)
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.encoder.eval()
        model.encoder_dim = getattr(model.encoder.config, "hidden_size", 768)
        model.num_prefix_tokens = kwargs.get("num_prefix_tokens", 0)

        model.tokenizer = AutoTokenizer.from_pretrained(
            save_path / "lora_adapter",
            trust_remote_code=True,
            padding_side="left",  # decoder-only → left-pad for batched generation
        )
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token

        base_decoder = AutoModelForCausalLM.from_pretrained(
            decoder_id, dtype=torch.bfloat16, trust_remote_code=True
        )
        model.decoder = PeftModel.from_pretrained(
            base_decoder, save_path / "lora_adapter"
        )

        dec_embed_dim = model.decoder.get_input_embeddings().weight.shape[1]
        model.projection = nn.Linear(model.encoder_dim, dec_embed_dim)
        model.projection.load_state_dict(
            torch.load(
                save_path / "projection.pt", map_location=device, weights_only=True
            )
        )
        model.to(device)
        return model
