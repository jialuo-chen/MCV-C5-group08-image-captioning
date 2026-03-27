"""Evaluate pretrained HuggingFace captioning models on VizWiz.

Supports:
- Direct pretrained model evaluation via ``--model`` (Task 1)
- Fine-tuned VisionEncoderDecoderModel evaluation via ``--checkpoint`` (Task 2)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

from src.data.dataset import VizWizEvalDataset
from src.evaluation.metrics import compute_metrics, format_metrics
from src.models.pretrained_captioner import build_pretrained_captioner
from src.utils.config import Config


def evaluate_pretrained(
    cfg: Config,
    model_name: str | None = None,
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
) -> dict[str, float]:
    """Evaluate a pretrained or fine-tuned HF captioning model on VizWiz.

    Either *model_name* (HF hub id) or *checkpoint_path* (local dir saved
    with ``save_pretrained``) must be provided.
    """
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    max_new_tokens = cfg.inference.get(
        "max_new_tokens", cfg.inference.get("max_length", 128)
    )
    batch_size = cfg.training.get("batch_size", 16)

    root = Path(cfg.dataset.root)
    test_ds = VizWizEvalDataset(
        annotation_file=str(root / cfg.dataset.val_ann),
        image_dir=str(root / cfg.dataset.val_img_dir),
    )
    print(f"Test set: {len(test_ds)} images")

    if checkpoint_path is not None:
        captioner = _load_finetuned_ved(checkpoint_path, device, max_new_tokens)
        label = Path(checkpoint_path).name
    elif model_name is not None:
        captioner = build_pretrained_captioner(
            model_name, device=device, max_new_tokens=max_new_tokens
        )
        label = model_name
    else:
        raise ValueError("Provide either --model (HF id) or --checkpoint (local path)")

    print(f"Model: {label}  |  device: {device}")

    all_predictions: list[str] = []
    all_references: list[list[str]] = []
    all_image_paths: list[str] = []
    total_time = 0.0

    for start in tqdm(range(0, len(test_ds), batch_size), desc="Evaluating"):
        end = min(start + batch_size, len(test_ds))
        batch_images, batch_refs, batch_paths = [], [], []
        for i in range(start, end):
            img, refs, path = test_ds[i]
            batch_images.append(img)
            batch_refs.append(refs)
            batch_paths.append(path)

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
        else Path(cfg.output_dir) / f"eval_{label.replace('/', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": label,
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


class _FinetunedVEDCaptioner:
    """Thin wrapper around a locally-saved VisionEncoderDecoderModel."""

    def __init__(self, model_dir: str, device: str, max_new_tokens: int) -> None:
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Try to load processor from same dir, fallback to encoder config
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_dir)
        except Exception:
            enc_name = self.model.config.encoder.name_or_path
            self.processor = AutoImageProcessor.from_pretrained(enc_name)

    @torch.no_grad()
    def generate_captions(self, images) -> list[str]:
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


def _load_finetuned_ved(checkpoint_path: str, device: str, max_new_tokens: int):
    return _FinetunedVEDCaptioner(checkpoint_path, device, max_new_tokens)
