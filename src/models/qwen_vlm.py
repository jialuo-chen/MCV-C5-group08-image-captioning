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

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


class QwenVLMCaptioner:
    """Qwen3.5 multimodal captioner for image captioning."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        prompt: str = "Describe this image briefly.",
        max_new_tokens: int = 128,
        torch_dtype=None,
    ) -> None:
        self.model_name = model_name
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._dtype = torch_dtype or torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",  # decoder-only → left-pad for batched generation
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=self._dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Silence repeated "Setting pad_token_id to eos_token_id" warnings
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self._pad_token_id = self.processor.tokenizer.pad_token_id

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
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._pad_token_id,
        )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        captions = self.processor.batch_decode(generated, skip_special_tokens=True)
        return captions
