"""Evaluate a multimodal VLM (Qwen3.5) on VizWiz image captioning.

Usage:
    c5-caption evaluate-multimodal --config configs/eval_qwen_multimodal.yaml \
        --model Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import VizWizEvalDataset
from src.evaluation.metrics import compute_metrics, format_metrics
from src.models.qwen_vlm import QwenVLMCaptioner
from src.utils.config import Config


def _eval_collate_fn(batch):
    """Collate PIL images + refs + paths into separate lists (no stacking)."""
    images, refs, paths = zip(*batch)
    return list(images), list(refs), list(paths)


def evaluate_multimodal(
    cfg: Config,
    model_name: str,
    output_dir: str | None = None,
    prompt: str | None = None,
) -> dict[str, float]:
    """Evaluate a multimodal VLM on the VizWiz test set.

    Parameters
    ----------
    cfg : Config
        Experiment configuration (dataset paths, device, inference settings).
    model_name : str
        HuggingFace model id (e.g. ``Qwen/Qwen3.5-9B``).
    output_dir : str | None
        Where to save results. Auto-generated if *None*.
    prompt : str | None
        Text prompt for the model. Falls back to ``cfg.multimodal.prompt``.
    """
    device = cfg.device  # explicit CUDA — never fall back to CPU
    prompt = prompt or cfg.get("multimodal", {}).get(
        "prompt", "Describe this image briefly."
    )
    max_new_tokens = cfg.get("multimodal", {}).get(
        "max_new_tokens", cfg.inference.get("max_length", 128)
    )
    batch_size = cfg.training.get("batch_size", 4)
    num_workers = cfg.training.get("num_workers", 4)

    root = Path(cfg.dataset.root)
    test_ds = VizWizEvalDataset(
        annotation_file=str(root / cfg.dataset.val_ann),
        image_dir=str(root / cfg.dataset.val_img_dir),
    )
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_eval_collate_fn,
        pin_memory=False,  # PIL images, not tensors
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    print(f"Test set: {len(test_ds)} images  |  batch_size={batch_size}")
    print(f'Model: {model_name}  |  prompt: "{prompt}"')

    captioner = QwenVLMCaptioner(
        model_name=model_name,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    all_predictions: list[str] = []
    all_references: list[list[str]] = []
    all_image_paths: list[str] = []
    total_time = 0.0

    for batch_images, batch_refs, batch_paths in tqdm(loader, desc="Evaluating"):
        t0 = time.perf_counter()
        captions = captioner.generate_captions(batch_images)
        total_time += time.perf_counter() - t0

        all_predictions.extend(captions)
        all_references.extend(batch_refs)
        all_image_paths.extend(batch_paths)

    metrics = compute_metrics(all_predictions, all_references)
    num_images = len(all_predictions)
    avg_ms = (total_time / num_images * 1000.0) if num_images else 0.0

    print(f"\nTest Results: {format_metrics(metrics)}")
    print(f"Average inference time per image: {avg_ms:.3f} ms")

    out_dir = (
        Path(output_dir)
        if output_dir
        else Path(cfg.output_dir) / f"eval_{model_name.replace('/', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": model_name,
        "prompt": prompt,
        "metrics": metrics,
        "num_images": num_images,
        "average_inference_time_per_image_ms": round(avg_ms, 6),
        "total_inference_time_s": round(total_time, 6),
        "samples": [
            {"image": img, "prediction": pred, "references": refs}
            for img, pred, refs in zip(
                all_image_paths[:20], all_predictions[:20], all_references[:20]
            )
        ],
    }
    results_file = out_dir / "eval_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {results_file}")
    return metrics
