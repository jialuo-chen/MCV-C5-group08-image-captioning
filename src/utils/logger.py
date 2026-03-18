"""Experiment logger.

Logs training/inference stats to a structured JSON file inside the run's output
directory.  Captures:

- Model info: encoder/decoder parameter counts, total parameters, FLOPs estimate
- Hyperparameters: full config snapshot
- Per-epoch: train_loss, val_loss, metrics (bleu1, bleu2, rougeL, meteor), lr, epoch time
- Summary: best epoch, best metrics, total training time
- Inference: checkpoint used, number of images, average latency
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable and total parameters, broken down by encoder/decoder/other."""
    encoder_total = 0
    encoder_trainable = 0
    decoder_total = 0
    decoder_trainable = 0
    other_total = 0
    other_trainable = 0

    for name, param in model.named_parameters():
        n = param.numel()
        if name.startswith("encoder"):
            encoder_total += n
            if param.requires_grad:
                encoder_trainable += n
        elif name.startswith("decoder"):
            decoder_total += n
            if param.requires_grad:
                decoder_trainable += n
        else:
            other_total += n
            if param.requires_grad:
                other_trainable += n

    total = encoder_total + decoder_total + other_total
    trainable = encoder_trainable + decoder_trainable + other_trainable

    return {
        "total_params": total,
        "trainable_params": trainable,
        "encoder_total_params": encoder_total,
        "encoder_trainable_params": encoder_trainable,
        "decoder_total_params": decoder_total,
        "decoder_trainable_params": decoder_trainable,
        "other_total_params": other_total,
        "other_trainable_params": other_trainable,
    }


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def estimate_flops(model: nn.Module, input_size: tuple = (1, 3, 224, 224), device: str = "cpu") -> int | None:
    """Estimate FLOPs for a single forward pass using torch.utils.flop_counter."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        dummy_img = torch.randn(*input_size, device=device)
        # Caption length doesn't matter much for FLOPs estimate; use short sequence
        dummy_cap = torch.zeros(input_size[0], 20, dtype=torch.long, device=device)

        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            model(dummy_img, dummy_cap)
        return flop_counter.get_total_flops()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Experiment Logger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """Logs experiment stats to a JSON file.

    Usage::

        logger = ExperimentLogger(output_dir, config)
        logger.log_model_info(model, device)
        logger.start_training()
        for epoch in ...:
            logger.log_epoch(epoch_data)
        logger.end_training()
        logger.save()
    """

    def __init__(self, output_dir: str | Path, config: dict) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "experiment_log.json"

        self.data: dict[str, Any] = {
            "hyperparameters": self._extract_hyperparams(config),
            "model_info": {},
            "training": {
                "epochs": [],
            },
            "summary": {},
        }
        self._train_start: float | None = None
        self._epoch_start: float | None = None

    @staticmethod
    def _extract_hyperparams(config: dict) -> dict:
        """Extract a clean hyperparameter dict from the full config."""
        return {
            "encoder": dict(config.get("encoder", {})),
            "decoder": dict(config.get("decoder", {})),
            "attention": dict(config.get("attention", {})),
            "tokenizer": dict(config.get("tokenizer", {})),
            "training": dict(config.get("training", {})),
            "seed": config.get("seed"),
            "device": config.get("device"),
        }

    def log_model_info(self, model: nn.Module, device: str = "cpu") -> dict:
        """Log parameter counts and FLOPs. Returns the info dict."""
        param_info = count_parameters(model)
        flops = estimate_flops(model, device=device)
        info = {
            **param_info,
            "flops": flops,
            "flops_readable": _format_number(flops) if flops else "N/A",
        }
        self.data["model_info"] = info
        return info

    def start_training(self) -> None:
        self._train_start = time.time()

    def start_epoch(self) -> None:
        self._epoch_start = time.time()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: dict[str, float],
        lr: float,
        is_best: bool = False,
    ) -> None:
        """Log one epoch's results."""
        epoch_time = time.time() - self._epoch_start if self._epoch_start else 0.0
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "bleu1": round(metrics.get("bleu1", 0.0), 6),
            "bleu2": round(metrics.get("bleu2", 0.0), 6),
            "rougeL": round(metrics.get("rougeL", 0.0), 6),
            "meteor": round(metrics.get("meteor", 0.0), 6),
            "lr": lr,
            "epoch_time_s": round(epoch_time, 2),
            "is_best": is_best,
        }
        self.data["training"]["epochs"].append(entry)

    def end_training(self, best_epoch: int, best_metrics: dict[str, float]) -> None:
        """Log training summary."""
        total_time = time.time() - self._train_start if self._train_start else 0.0
        epochs_data = self.data["training"]["epochs"]
        train_losses = [e["train_loss"] for e in epochs_data]
        val_losses = [e["val_loss"] for e in epochs_data]

        self.data["summary"] = {
            "total_epochs_run": len(epochs_data),
            "total_training_time_s": round(total_time, 2),
            "total_training_time_readable": _format_time(total_time),
            "best_epoch": best_epoch,
            "best_metrics": {k: round(v, 6) for k, v in best_metrics.items()},
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "min_train_loss": min(train_losses) if train_losses else None,
            "min_val_loss": min(val_losses) if val_losses else None,
            "avg_epoch_time_s": round(
                sum(e["epoch_time_s"] for e in epochs_data) / max(len(epochs_data), 1), 2
            ),
        }

    def log_inference(self, info: dict) -> None:
        """Log inference session info."""
        self.data["inference"] = info

    def save(self) -> Path:
        """Write the log to disk and return the path."""
        self.log_path.write_text(json.dumps(self.data, indent=2))
        return self.log_path


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_number(n: int | float) -> str:
    """Human-readable number (e.g. 1.2M, 3.4G)."""
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(int(n))


def _format_time(seconds: float) -> str:
    """Human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60
    return f"{hours:.1f}h"


def print_model_summary(info: dict) -> None:
    """Pretty-print model parameter and FLOPs info."""
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"  Total parameters:      {info['total_params']:>12,}")
    print(f"  Trainable parameters:  {info['trainable_params']:>12,}")
    print(f"  ─ Encoder total:       {info['encoder_total_params']:>12,}")
    print(f"  ─ Encoder trainable:   {info['encoder_trainable_params']:>12,}")
    print(f"  ─ Decoder total:       {info['decoder_total_params']:>12,}")
    print(f"  ─ Decoder trainable:   {info['decoder_trainable_params']:>12,}")
    if info.get("other_total_params", 0) > 0:
        print(f"  ─ Other total:         {info['other_total_params']:>12,}")
        print(f"  ─ Other trainable:     {info['other_trainable_params']:>12,}")
    flops = info.get("flops_readable", "N/A")
    print(f"  FLOPs (single fwd):    {flops:>12}")
    print("=" * 60)
