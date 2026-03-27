"""Wrappers for pretrained HuggingFace image-captioning models.

Supported models:
- ``nlpconnect/vit-gpt2-image-captioning``  (VisionEncoderDecoderModel)
- ``Salesforce/blip-image-captioning-base``  (BLIP)
- ``microsoft/git-base-coco``               (GIT)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)


class PretrainedCaptioner(ABC):
    """Base class for pretrained captioning models."""

    @abstractmethod
    def generate_captions(self, images: list[Image.Image]) -> list[str]:
        """Generate a caption for each image in the batch."""
        ...


class VitGPT2Captioner(PretrainedCaptioner):
    """``nlpconnect/vit-gpt2-image-captioning`` wrapper."""

    def __init__(
        self, model_name: str, device: str = "cpu", max_new_tokens: int = 128
    ) -> None:
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_captions(self, images: list[Image.Image]) -> list[str]:
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(
            self.device
        )
        output_ids = self.model.generate(
            pixel_values,
            max_new_tokens=self.max_new_tokens,
            max_length=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


class BLIPCaptioner(PretrainedCaptioner):
    """``Salesforce/blip-image-captioning-base`` wrapper."""

    def __init__(
        self, model_name: str, device: str = "cpu", max_new_tokens: int = 128
    ) -> None:
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    @torch.no_grad()
    def generate_captions(self, images: list[Image.Image]) -> list[str]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)


class GITCaptioner(PretrainedCaptioner):
    """``microsoft/git-base-coco`` wrapper."""

    def __init__(
        self, model_name: str, device: str = "cpu", max_new_tokens: int = 128
    ) -> None:
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_captions(self, images: list[Image.Image]) -> list[str]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)


_CAPTIONER_MAP: dict[str, type[PretrainedCaptioner]] = {
    "nlpconnect/vit-gpt2-image-captioning": VitGPT2Captioner,
    "Salesforce/blip-image-captioning-base": BLIPCaptioner,
    "Salesforce/blip-image-captioning-large": BLIPCaptioner,
    "microsoft/git-base-coco": GITCaptioner,
    "microsoft/git-large-coco": GITCaptioner,
}

_FAMILY_MAP: dict[str, type[PretrainedCaptioner]] = {
    "blip": BLIPCaptioner,
    "git": GITCaptioner,
    "vit-gpt2": VitGPT2Captioner,
}


def build_pretrained_captioner(
    model_name: str,
    device: str = "cpu",
    max_new_tokens: int = 128,
) -> PretrainedCaptioner:
    """Instantiate a pretrained captioner by HuggingFace model id."""
    cls = _CAPTIONER_MAP.get(model_name)
    if cls is None:
        name_lower = model_name.lower()
        for key, fallback_cls in _FAMILY_MAP.items():
            if key in name_lower:
                cls = fallback_cls
                break
    if cls is None:
        raise ValueError(
            f"Unknown pretrained captioner: {model_name}. "
            f"Supported: {list(_CAPTIONER_MAP.keys())}"
        )
    return cls(model_name, device=device, max_new_tokens=max_new_tokens)
