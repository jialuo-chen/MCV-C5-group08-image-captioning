from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import VizWizCaptionDataset, caption_collate_fn
from src.data.tokenizer import (
    BaseTokenizer,
    CharTokenizer,
    SubwordTokenizer,
    WordTokenizer,
)
from src.evaluation.metrics import compute_metrics, format_metrics
from src.models.captioner import CaptioningModel
from src.models.decoders import HFLMDecoder
from src.utils.config import Config


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


def evaluate(cfg: Config, checkpoint_path: str) -> dict[str, float]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint_path)

    model, checkpoint = CaptioningModel.from_checkpoint(ckpt_path, device=str(device))
    model.eval()
    is_hf_lm = isinstance(model.decoder, HFLMDecoder)

    saved_cfg = Config(checkpoint["config"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    if is_hf_lm:
        tokenizer = CharTokenizer()  # placeholder
    else:
        tokenizer = _load_tokenizer(ckpt_path.parent, saved_cfg)
    print(f"Tokenizer: {saved_cfg.tokenizer.type}, vocab_size={tokenizer.vocab_size}")

    root = Path(cfg.dataset.root)
    test_ds = VizWizCaptionDataset(
        annotation_file=str(root / cfg.dataset.val_ann),
        image_dir=str(root / cfg.dataset.val_img_dir),
        tokenizer=tokenizer,
        max_length=cfg.tokenizer.max_length,
        split="val",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=caption_collate_fn,
        pin_memory=True,
    )
    print(f"Test set: {len(test_ds)} images")

    all_predictions: list[str] = []
    all_references: list[list[str]] = []
    all_image_paths: list[str] = []
    total_inference_time_s = 0.0

    for images, cap_tensors, cap_texts, img_paths in tqdm(
        test_loader, desc="Evaluating"
    ):
        images = images.to(device)
        batch_start_time = time.perf_counter()
        generated = model.generate(
            images,
            tokenizer=tokenizer,
            max_length=cfg.inference.get("max_length", cfg.tokenizer.max_length),
        )
        total_inference_time_s += time.perf_counter() - batch_start_time
        all_predictions.extend(generated)
        all_image_paths.extend(img_paths)

        batch_start = len(all_references)
        for i in range(len(generated)):
            idx = batch_start + i
            if idx < len(test_ds):
                refs = test_ds.get_all_captions(idx)
                all_references.append(refs)

    metrics = compute_metrics(all_predictions, all_references)
    num_images = len(all_predictions)
    avg_inference_ms = (
        (total_inference_time_s / num_images * 1000.0) if num_images > 0 else 0.0
    )
    print(f"\nTest Results: {format_metrics(metrics)}")
    print(f"Average inference time per image: {avg_inference_ms:.3f} ms")

    results_dir = ckpt_path.parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "metrics": metrics,
        "num_images": num_images,
        "average_inference_time_per_image (ms)": round(avg_inference_ms, 6),
        "total_inference_time_s": round(total_inference_time_s, 6),
        "checkpoint": str(ckpt_path),
        "samples": [
            {"image": img, "prediction": pred, "references": refs}
            for img, pred, refs in zip(
                all_image_paths[:20],
                all_predictions[:20],
                all_references[:20],
            )
        ],
    }
    results_file = results_dir / "eval_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {results_file}")

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=f"eval-{ckpt_path.stem}",
            config=dict(cfg),
        )
        wandb.log({f"test_{k}": v for k, v in metrics.items()})
        wandb.finish()

    return metrics
