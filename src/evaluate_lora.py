"""Evaluate a LoRA-finetuned ViT+Qwen model on the VizWiz test set."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

from src.data.dataset import build_vision_datasets, vision_collate_fn
from src.evaluation.metrics import compute_metrics, format_metrics
from src.models.vit_qwen_lora import ViTQwenLoRA
from src.utils.config import Config


def evaluate_lora(
    cfg: Config,
    checkpoint_path: str,
    output_dir: str | None = None,
) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    max_gen_length = cfg.inference.get("max_length", 128)
    batch_size = cfg.training.get("batch_size", 4)

    encoder_id = cfg.encoder.pretrained
    decoder_id = cfg.decoder.pretrained

    print(f"Loading LoRA checkpoint from: {checkpoint_path}")
    model = ViTQwenLoRA.load_checkpoint(
        checkpoint_path,
        encoder_id=encoder_id,
        decoder_id=decoder_id,
        device=device,
        encoder_checkpoint=cfg.encoder.get("checkpoint"),
        num_prefix_tokens=cfg.encoder.get("num_prefix_tokens", 0),
    )
    model.eval()

    image_processor = AutoImageProcessor.from_pretrained(encoder_id)
    tokenizer = model.tokenizer

    _, _, test_ds = build_vision_datasets(cfg, image_processor, tokenizer)
    print(f"Test set: {len(test_ds)} images")

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.training.get("num_workers", 4),
        collate_fn=vision_collate_fn,
        pin_memory=True,
    )

    all_predictions: list[str] = []
    all_references: list[list[str]] = []
    total_time = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            pixel_values = batch["pixel_values"].to(device)

            t0 = time.perf_counter()
            captions = model.generate(pixel_values, max_new_tokens=max_gen_length)
            total_time += time.perf_counter() - t0

            all_predictions.extend(captions)

            batch_start = batch_idx * test_loader.batch_size
            for i in range(len(captions)):
                idx = batch_start + i
                if idx < len(test_ds):
                    all_references.append(test_ds.get_all_captions(idx))

    metrics = compute_metrics(all_predictions, all_references)
    num_images = len(all_predictions)
    avg_ms = (total_time / num_images * 1000.0) if num_images else 0.0

    print(f"\nTest Results: {format_metrics(metrics)}")
    print(f"Average inference time per image: {avg_ms:.3f} ms")

    ckpt_label = Path(checkpoint_path).parent.parent.name
    out_dir = (
        Path(output_dir)
        if output_dir
        else Path(cfg.output_dir) / f"eval_lora_{ckpt_label}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": f"LoRA({cfg.encoder.name}+{cfg.decoder.name})",
        "checkpoint": checkpoint_path,
        "metrics": metrics,
        "num_images": num_images,
        "average_inference_time_per_image_ms": round(avg_ms, 6),
        "total_inference_time_s": round(total_time, 6),
    }
    results_file = out_dir / "eval_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {results_file}")
    return metrics
