"""Inference script.

Load a trained checkpoint and generate captions for a single image,
a directory of images, or a list of image paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from src.data.dataset import get_image_transform
from src.data.tokenizer import BaseTokenizer, CharTokenizer, WordTokenizer, SubwordTokenizer
from src.models.captioner import CaptioningModel
from src.models.decoders import HFLMDecoder
from src.utils.config import Config


def _load_tokenizer(ckpt_dir: Path, cfg: Config) -> BaseTokenizer:
    """Load tokenizer from checkpoint directory."""
    tok_path = ckpt_dir / "tokenizer.json"
    tok_type = cfg.tokenizer.type
    if tok_type == "char":
        return CharTokenizer.load(tok_path) if tok_path.exists() else CharTokenizer()
    elif tok_type == "word":
        return WordTokenizer.load(tok_path)
    elif tok_type == "subword":
        return SubwordTokenizer.load(tok_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tok_type}")


def infer(cfg: Config, checkpoint_path: str, image_paths: list[str], output_file: str | None = None) -> list[dict]:
    """Run inference on a list of images.

    Parameters
    ----------
    cfg : Config
        Experiment configuration.
    checkpoint_path : str
        Path to the trained model checkpoint.
    image_paths : list[str]
        Paths to input images.
    output_file : str | None
        If given, save results as JSON.

    Returns
    -------
    list[dict]
        List of ``{"image": str, "caption": str}`` dicts.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint_path)

    # -- Load model ---------------------------------------------------------
    model, checkpoint = CaptioningModel.from_checkpoint(ckpt_path, device=str(device))
    model.eval()
    is_hf_lm = isinstance(model.decoder, HFLMDecoder)

    saved_cfg = Config(checkpoint["config"])

    # -- Load tokenizer -----------------------------------------------------
    if is_hf_lm:
        tokenizer = CharTokenizer()
    else:
        tokenizer = _load_tokenizer(ckpt_path.parent, saved_cfg)

    # -- Image transform ----------------------------------------------------
    transform = get_image_transform("val")

    # -- Run inference ------------------------------------------------------
    results = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        captions = model.generate(
            img_tensor,
            tokenizer=tokenizer,
            max_length=cfg.inference.get("max_length", cfg.tokenizer.max_length),
        )
        result = {"image": str(img_path), "caption": captions[0]}
        results.append(result)
        print(f"{Path(img_path).name}: {captions[0]}")

    # -- Save results -------------------------------------------------------
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {output_file}")

    return results


def collect_image_paths(path: str) -> list[str]:
    """Collect image paths from a file path or directory."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in exts)
    else:
        raise FileNotFoundError(f"Path not found: {path}")
