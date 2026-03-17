"""Evaluation script.

Loads a trained checkpoint and evaluates on the test set (VizWiz val split),
computing BLEU-1, BLEU-2, ROUGE-L, and METEOR.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import VizWizCaptionDataset, caption_collate_fn
from src.data.tokenizer import BaseTokenizer, CharTokenizer, WordTokenizer, SubwordTokenizer
from src.models.captioner import CaptioningModel
from src.models.decoders import HFLMDecoder
from src.evaluation.metrics import compute_metrics, format_metrics
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


def evaluate(cfg: Config, checkpoint_path: str) -> dict[str, float]:
    """Evaluate a model checkpoint on the test set.

    Parameters
    ----------
    cfg : Config
        Experiment configuration.
    checkpoint_path : str
        Path to the model checkpoint.

    Returns
    -------
    dict[str, float]
        Metric scores.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint_path)

    # -- Load model ---------------------------------------------------------
    model, checkpoint = CaptioningModel.from_checkpoint(ckpt_path, device=str(device))
    model.eval()
    is_hf_lm = isinstance(model.decoder, HFLMDecoder)

    saved_cfg = Config(checkpoint["config"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # -- Load tokenizer -----------------------------------------------------
    if is_hf_lm:
        tokenizer = CharTokenizer()  # placeholder
    else:
        tokenizer = _load_tokenizer(ckpt_path.parent, saved_cfg)
    print(f"Tokenizer: {saved_cfg.tokenizer.type}, vocab_size={tokenizer.vocab_size}")

    # -- Build test dataset -------------------------------------------------
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

    # -- Generate & evaluate ------------------------------------------------
    all_predictions: list[str] = []
    all_references: list[list[str]] = []
    all_image_paths: list[str] = []

    for images, cap_tensors, cap_texts, img_paths in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        generated = model.generate(
            images,
            tokenizer=tokenizer,
            max_length=cfg.inference.get("max_length", cfg.tokenizer.max_length),
        )
        all_predictions.extend(generated)
        all_image_paths.extend(img_paths)

        batch_start = len(all_references)
        for i in range(len(generated)):
            idx = batch_start + i
            if idx < len(test_ds):
                refs = test_ds.get_all_captions(idx)
                all_references.append(refs)

    metrics = compute_metrics(all_predictions, all_references)
    print(f"\nTest Results: {format_metrics(metrics)}")

    # -- Save results -------------------------------------------------------
    results_dir = ckpt_path.parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "metrics": metrics,
        "num_images": len(all_predictions),
        "checkpoint": str(ckpt_path),
        "samples": [
            {"image": img, "prediction": pred, "references": refs}
            for img, pred, refs in zip(
                all_image_paths[:20], all_predictions[:20], all_references[:20],
            )
        ],
    }
    results_file = results_dir / "eval_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {results_file}")

    # -- WandB logging (if enabled) -----------------------------------------
    if cfg.wandb.enabled:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=f"eval-{ckpt_path.stem}",
            config=dict(cfg),
        )
        wandb.log({f"test_{k}": v for k, v in metrics.items()})
        wandb.finish()

    return metrics
