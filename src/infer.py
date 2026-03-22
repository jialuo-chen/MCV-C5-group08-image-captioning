from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image

from src.data.dataset import get_image_transform
from src.data.tokenizer import (
    BaseTokenizer,
    CharTokenizer,
    SubwordTokenizer,
    WordTokenizer,
)
from src.models.captioner import CaptioningModel
from src.models.decoders import HFLMDecoder
from src.utils.config import Config
from src.utils.logger import ExperimentLogger


def _load_tokenizer(ckpt_dir: Path, cfg: Config) -> BaseTokenizer:
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


def infer(
    cfg: Config,
    checkpoint_path: str,
    image_paths: list[str],
    output_file: str | None = None,
) -> list[dict]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint_path)

    model, checkpoint = CaptioningModel.from_checkpoint(ckpt_path, device=str(device))
    model.eval()
    is_hf_lm = isinstance(model.decoder, HFLMDecoder)

    saved_cfg = Config(checkpoint["config"])

    if is_hf_lm:
        tokenizer = CharTokenizer()
    else:
        tokenizer = _load_tokenizer(ckpt_path.parent, saved_cfg)

    transform = get_image_transform("val")

    results = []
    t_start = time.time()
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

    total_time = time.time() - t_start

    log_dir = (
        Path(output_file).parent if output_file else ckpt_path.parent.parent / "results"
    )
    exp_logger = ExperimentLogger(log_dir, dict(saved_cfg))
    exp_logger.log_model_info(model, device=str(device))
    exp_logger.log_inference(
        {
            "checkpoint": str(ckpt_path),
            "num_images": len(image_paths),
            "total_time_s": round(total_time, 3),
            "avg_latency_ms": round((total_time / max(len(image_paths), 1)) * 1000, 2),
        }
    )
    exp_logger.save()

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {output_file}")

    return results


def collect_image_paths(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in exts)
    else:
        raise FileNotFoundError(f"Path not found: {path}")
