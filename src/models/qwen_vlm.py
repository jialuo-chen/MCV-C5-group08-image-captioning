"""Wrapper for Qwen3.5 multimodal models for image captioning.

Qwen3.5 is a natively multimodal hybrid reasoning LLM that accepts both
vision and text inputs.  It must be loaded with
``Qwen3_5ForConditionalGeneration`` (NOT ``AutoModelForCausalLM``, which
maps to the text-only ``Qwen3_5ForCausalLM``).  The processor's
``apply_chat_template`` with ``tokenize=True, return_dict=True`` returns
all required tensors (input_ids, pixel_values, image_grid_thw, etc.)
ready for ``generate()``.  Thinking is disabled via
``enable_thinking=False``.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def _resolve_torch_dtype(value: Any):
    """Resolve config dtype value into a transformers-compatible dtype arg."""
    if value is None:
        return torch.bfloat16
    if value == "auto":
        return "auto"
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if key in mapping:
            return mapping[key]
    raise ValueError(
        "Invalid torch_dtype for multimodal model loading. "
        "Use one of: auto, bfloat16/bf16, float16/fp16, float32/fp32."
    )


class QwenVLMCaptioner:
    """Qwen3.5 multimodal captioner for image captioning."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        prompt: str = "Describe this image briefly.",
        max_new_tokens: int = 128,
        torch_dtype=None,
        loader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._dtype = _resolve_torch_dtype(torch_dtype)
        self._loader_kwargs = dict(loader_kwargs or {})

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",  # decoder-only → left-pad for batched generation
        )

        # For large MoE checkpoints (e.g. Qwen3.5-35B-A3B-FP8), callers can
        # override device_map/max_memory/offload settings via loader_kwargs.
        model_load_kwargs = {
            "dtype": self._dtype,
            "device_map": self._loader_kwargs.pop("device_map", device),
            "trust_remote_code": True,
        }
        model_load_kwargs.update(self._loader_kwargs)

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            **model_load_kwargs,
        )
        self.model.eval()

        self._input_device = self._infer_input_device()

        # Silence repeated "Setting pad_token_id to eos_token_id" warnings
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self._pad_token_id = self.processor.tokenizer.pad_token_id

    def _infer_input_device(self) -> torch.device:
        """Pick a safe input device, including for sharded device_map='auto'."""
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for mapped in hf_device_map.values():
                if isinstance(mapped, int):
                    return torch.device(f"cuda:{mapped}")
                if isinstance(mapped, str) and mapped.startswith("cuda"):
                    return torch.device(mapped)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def generate_captions(self, images: list[Image.Image]) -> list[str]:
        """Generate captions for a batch of PIL images.

        Builds one conversation per image and processes the full batch in
        a single ``apply_chat_template`` + ``generate()`` call so that
        the GPU stays busy instead of idling between per-image CPU steps.
        """
        if not images:
            return []

        batch_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            for img in images
        ]

        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        ).to(self._input_device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._pad_token_id,
        )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        captions = self.processor.batch_decode(generated, skip_special_tokens=True)
        return captions
