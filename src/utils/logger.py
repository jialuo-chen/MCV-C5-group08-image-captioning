from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(model: nn.Module) -> dict[str, int]:
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
        "readable_total_params": _format_number(total),
        "trainable_params": trainable,
        "encoder_total_params": encoder_total,
        "encoder_trainable_params": encoder_trainable,
        "decoder_total_params": decoder_total,
        "decoder_trainable_params": decoder_trainable,
        "other_total_params": other_total,
        "other_trainable_params": other_trainable,
    }


def estimate_flops(
    model: nn.Module, input_size: tuple = (1, 3, 224, 224), device: str = "cpu"
) -> dict[str, int | None]:
    """Estimate FLOPs for the model, returning per-component and total counts.

    For ViTQwenLoRA models, computes encoder, projection, and decoder FLOPs
    separately to avoid issues with PEFT wrappers and mixed dtypes.

    Returns a dict with keys: ``encoder_flops``, ``projection_flops``,
    ``decoder_flops``, ``total_flops``.
    """
    from src.models.vit_qwen_lora import ViTQwenLoRA

    result: dict[str, int | None] = {
        "encoder_flops": None,
        "projection_flops": None,
        "decoder_flops": None,
        "total_flops": None,
    }

    if isinstance(model, ViTQwenLoRA):
        result = _estimate_flops_vit_qwen(model, input_size, device)
    else:
        try:
            dummy_img = torch.randn(*input_size, device=device)
            dummy_cap = torch.zeros(input_size[0], 20, dtype=torch.long, device=device)
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                model(dummy_img, dummy_cap)
            total = flop_counter.get_total_flops()
            result["total_flops"] = total
        except Exception:
            pass

    return result


@torch.no_grad()
def _estimate_flops_vit_qwen(
    model, input_size: tuple, device: str
) -> dict[str, int | None]:
    """Component-wise FLOPs for ViTQwenLoRA (encoder + projection + decoder)."""
    batch = input_size[0]
    seq_len = 20  # short caption for estimation
    encoder_flops = None
    projection_flops = None
    decoder_flops = None

    dummy_pixels = torch.randn(*input_size, device=device)

    # --- Encoder FLOPs ---
    try:
        counter = FlopCounterMode(display=False)
        with counter:
            enc_out = model.encoder(pixel_values=dummy_pixels)
        encoder_flops = counter.get_total_flops()
        vit_features = enc_out.last_hidden_state
    except Exception:
        # Fall back: run encoder without counting to get features for next steps
        try:
            enc_out = model.encoder(pixel_values=dummy_pixels)
            vit_features = enc_out.last_hidden_state
        except Exception:
            return {
                "encoder_flops": None,
                "projection_flops": None,
                "decoder_flops": None,
                "total_flops": None,
            }

    if model.num_prefix_tokens > 0:
        vit_features = vit_features[:, 1 : 1 + model.num_prefix_tokens, :]

    # --- Projection FLOPs ---
    try:
        proj_input = vit_features.to(model.projection.weight.dtype)
        counter = FlopCounterMode(display=False)
        with counter:
            vit_projected = model.projection(proj_input)
        projection_flops = counter.get_total_flops()
    except Exception:
        try:
            vit_projected = model.projection(
                vit_features.to(model.projection.weight.dtype)
            )
        except Exception:
            vit_projected = None

    # --- Decoder FLOPs ---
    if vit_projected is not None:
        try:
            dummy_ids = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
            caption_embeds = model.decoder.get_input_embeddings()(dummy_ids)
            inputs_embeds = torch.cat([vit_projected, caption_embeds], dim=1).to(
                caption_embeds.dtype
            )
            prefix_len = vit_projected.size(1)
            attn_mask = torch.ones(
                batch, prefix_len + seq_len, dtype=torch.long, device=device
            )
            counter = FlopCounterMode(display=False)
            with counter:
                model.decoder(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
            decoder_flops = counter.get_total_flops()
        except Exception:
            pass

    parts = [encoder_flops, projection_flops, decoder_flops]
    total_flops = sum(f for f in parts if f is not None) if any(parts) else None

    return {
        "encoder_flops": encoder_flops,
        "projection_flops": projection_flops,
        "decoder_flops": decoder_flops,
        "total_flops": total_flops,
    }


class ExperimentLogger:
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
        param_info = count_parameters(model)
        flops_info = estimate_flops(model, device=device)
        info = {
            **param_info,
            "encoder_flops": flops_info.get("encoder_flops"),
            "encoder_flops_readable": (
                _format_number(flops_info["encoder_flops"])
                if flops_info.get("encoder_flops")
                else "N/A"
            ),
            "projection_flops": flops_info.get("projection_flops"),
            "projection_flops_readable": (
                _format_number(flops_info["projection_flops"])
                if flops_info.get("projection_flops")
                else "N/A"
            ),
            "decoder_flops": flops_info.get("decoder_flops"),
            "decoder_flops_readable": (
                _format_number(flops_info["decoder_flops"])
                if flops_info.get("decoder_flops")
                else "N/A"
            ),
            "total_flops": flops_info.get("total_flops"),
            "total_flops_readable": (
                _format_number(flops_info["total_flops"])
                if flops_info.get("total_flops")
                else "N/A"
            ),
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
        epoch_time = time.time() - self._epoch_start if self._epoch_start else 0.0
        entry = {
            "epoch": int(epoch),
            "train_loss": float(round(float(train_loss), 6)),
            "val_loss": float(round(float(val_loss), 6)),
            "bleu1": float(round(float(metrics.get("bleu1", 0.0)), 6)),
            "bleu2": float(round(float(metrics.get("bleu2", 0.0)), 6)),
            "rougeL": float(round(float(metrics.get("rougeL", 0.0)), 6)),
            "meteor": float(round(float(metrics.get("meteor", 0.0)), 6)),
            "lr": float(lr),
            "epoch_time_s": float(round(float(epoch_time), 2)),
            "is_best": bool(is_best),
        }
        self.data["training"]["epochs"].append(entry)

    def end_training(self, best_epoch: int, best_metrics: dict[str, float]) -> None:
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
                sum(e["epoch_time_s"] for e in epochs_data) / max(len(epochs_data), 1),
                2,
            ),
        }

    def log_inference(self, info: dict) -> None:
        self.data["inference"] = info

    def log_test_eval(self, metrics: dict[str, float]) -> None:
        self.data["test_eval"] = {k: round(float(v), 6) for k, v in metrics.items()}

    def save(self) -> Path:
        self.log_path.write_text(json.dumps(self.data, indent=2, default=_json_default))
        return self.log_path


def _format_number(n: int | float) -> str:
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
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60
    return f"{hours:.1f}h"


def _json_default(obj):
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def print_model_summary(info: dict) -> None:
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"  Total parameters:      {info['readable_total_params']:>12}")
    print(f"  Trainable parameters:  {info['trainable_params']:>12}")
    print(f"  ─ Encoder total:       {info['encoder_total_params']:>12}")
    print(f"  ─ Encoder trainable:   {info['encoder_trainable_params']:>12}")
    print(f"  ─ Decoder total:       {info['decoder_total_params']:>12}")
    print(f"  ─ Decoder trainable:   {info['decoder_trainable_params']:>12}")
    if info.get("other_total_params", 0) > 0:
        print(f"  ─ Other total:         {info['other_total_params']:>12}")
        print(f"  ─ Other trainable:     {info['other_trainable_params']:>12}")
    # Per-component FLOPs (new format)
    if info.get("encoder_flops_readable") and info["encoder_flops_readable"] != "N/A":
        print(f"  FLOPs (encoder):       {info['encoder_flops_readable']:>12}")
    if (
        info.get("projection_flops_readable")
        and info["projection_flops_readable"] != "N/A"
    ):
        print(f"  FLOPs (projection):    {info['projection_flops_readable']:>12}")
    if info.get("decoder_flops_readable") and info["decoder_flops_readable"] != "N/A":
        print(f"  FLOPs (decoder):       {info['decoder_flops_readable']:>12}")
    total_flops = info.get("total_flops_readable") or info.get("flops_readable", "N/A")
    print(f"  FLOPs (total fwd):     {total_flops:>12}")
    print("=" * 60)
