"""Fine-tune a Qwen3.5 decoder with LoRA using a frozen ViT encoder.

Usage:
    c5-caption finetune-lora --config configs/lora_qwen_0.8b.yaml
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

from src.data.dataset import build_vision_datasets, vision_collate_fn
from src.evaluation.metrics import compute_metrics, format_metrics
from src.models.vit_qwen_lora import ViTQwenLoRA
from src.utils.config import Config
from src.utils.logger import ExperimentLogger, print_model_summary


def _build_optimizer(cfg: Config, params) -> torch.optim.Optimizer:
    name = cfg.training.optimizer.lower()
    lr = cfg.training.lr
    wd = cfg.training.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def _build_scheduler(cfg: Config, optimizer):
    sched = cfg.training.get("scheduler")
    if sched is None:
        return None
    params = cfg.training.get("scheduler_params", {})
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, **params
        )
    elif sched == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    raise ValueError(f"Unknown scheduler: {sched}")


def train_lora(cfg: Config) -> float:
    """Fine-tune a Qwen3.5 decoder with LoRA on VizWiz.

    Returns the best METEOR score achieved during training.
    """
    device = torch.device(cfg.device)
    print(f"Device: {device}")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    encoder_id = cfg.encoder.pretrained
    decoder_id = cfg.decoder.pretrained
    lora_cfg = cfg.get("lora", {})
    print(f"Encoder: {encoder_id} (frozen)")
    print(f"Decoder: {decoder_id} + LoRA (r={lora_cfg.get('r', 16)})")

    model = ViTQwenLoRA(
        encoder_id=encoder_id,
        decoder_id=decoder_id,
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        lora_target=lora_cfg.get("target", "all"),
        encoder_checkpoint=cfg.encoder.get("checkpoint"),
        num_prefix_tokens=cfg.encoder.get("num_prefix_tokens", 0),
    )
    model = model.to(device)

    image_processor = AutoImageProcessor.from_pretrained(encoder_id)
    tokenizer = model.tokenizer

    run_name = (
        cfg.get("run_name")
        or f"lora-{cfg.encoder.name}-{cfg.decoder.name}-{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(cfg.output_dir) / run_name
    ckpt_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    wandb_run = None
    if cfg.wandb.enabled:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=run_name,
            config=dict(cfg),
            tags=cfg.wandb.get("tags", []),
        )

    train_ds, val_ds, test_ds = build_vision_datasets(cfg, image_processor, tokenizer)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=vision_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=vision_collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=vision_collate_fn,
        pin_memory=True,
    )

    exp_logger = ExperimentLogger(output_dir, dict(cfg))
    model_info = exp_logger.log_model_info(model, device=str(device))
    print_model_summary(model_info)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found.")
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.2f}%)"
    )

    optimizer = _build_optimizer(cfg, trainable_params)
    scheduler = _build_scheduler(cfg, optimizer)

    use_amp = cfg.training.get("mixed_precision", False)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    grad_accum_steps = cfg.training.get("gradient_accumulation_steps", 1)

    best_metric = -1.0
    best_epoch = 0
    best_metrics_dict: dict[str, float] = {}
    patience_counter = 0
    patience = cfg.training.get("early_stopping_patience")
    max_gen_length = cfg.inference.get("max_length", 128)
    exp_logger.start_training()

    # --- Pre-fine-tuning evaluation on the VizWiz test set ---
    print("\n" + "=" * 60)
    print("PRE-FINETUNE EVALUATION (VizWiz test set)")
    print("=" * 60)
    pre_metrics = _run_evaluation(model, test_loader, device, max_gen_length)
    exp_logger.log_pre_finetune_eval(pre_metrics)
    print(f"  {format_metrics(pre_metrics)}")
    print("=" * 60 + "\n")

    for epoch in range(cfg.training.epochs):
        exp_logger.start_epoch()
        model.train()
        model.encoder.eval()  # encoder always frozen
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()

                if (step + 1) % grad_accum_steps == 0:
                    if cfg.training.grad_clip:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.training.grad_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = loss / grad_accum_steps
                loss.backward()

                if (step + 1) % grad_accum_steps == 0:
                    if cfg.training.grad_clip:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.training.grad_clip
                        )
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum_steps
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)

        val_metrics = _run_validation(model, val_loader, device, max_gen_length)
        val_loss = val_metrics.pop("val_loss", 0.0)

        print(
            f"Epoch {epoch + 1}: train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
            f"{format_metrics(val_metrics)}"
        )

        is_best = val_metrics.get("meteor", 0.0) > best_metric
        exp_logger.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_loss=val_loss,
            metrics=val_metrics,
            lr=optimizer.param_groups[0]["lr"],
            is_best=is_best,
        )

        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )

        if scheduler:
            scheduler.step()

        model.save_checkpoint(str(ckpt_dir / "last"))

        current_metric = val_metrics.get("meteor", 0.0)
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_metrics_dict = dict(val_metrics)
            model.save_checkpoint(str(ckpt_dir / "best"))
            print(f"  -> New best model (meteor={best_metric * 100:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience and patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

    exp_logger.end_training(best_epoch=best_epoch, best_metrics=best_metrics_dict)

    # --- Post-fine-tuning evaluation: load best checkpoint and evaluate on test set ---
    print("\n" + "=" * 60)
    print("POST-FINETUNE EVALUATION (VizWiz test set, best checkpoint)")
    print("=" * 60)
    best_ckpt_path = str(ckpt_dir / "best")
    best_model = ViTQwenLoRA.load_checkpoint(
        best_ckpt_path,
        encoder_id=encoder_id,
        decoder_id=decoder_id,
        device=str(device),
        encoder_checkpoint=cfg.encoder.get("checkpoint"),
        num_prefix_tokens=cfg.encoder.get("num_prefix_tokens", 0),
    )
    best_model.eval()
    post_metrics = _run_evaluation(best_model, test_loader, device, max_gen_length)
    exp_logger.log_post_finetune_eval(post_metrics)
    print(f"  {format_metrics(post_metrics)}")
    print("=" * 60)

    # --- Compute and log deltas ---
    exp_logger.log_finetune_deltas(pre_metrics, post_metrics)
    print("\nFINE-TUNING DELTAS (post - pre):")
    for key in pre_metrics:
        if key in post_metrics:
            delta = post_metrics[key] - pre_metrics[key]
            sign = "+" if delta >= 0 else ""
            print(f"  {key}: {sign}{delta * 100:.2f}%")

    # Clean up the reloaded model to free memory
    del best_model
    torch.cuda.empty_cache()

    log_path = exp_logger.save()
    print(f"\nTraining complete. Best METEOR: {best_metric * 100:.2f}%")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Experiment log: {log_path}")

    if wandb_run:
        wandb_run.finish()
    return best_metric


@torch.no_grad()
def _run_evaluation(
    model: ViTQwenLoRA,
    loader: DataLoader,
    device: torch.device,
    max_gen_length: int,
) -> dict[str, float]:
    """Run generation-only evaluation on a dataset (no loss computation)."""
    model.eval()
    all_predictions: list[str] = []
    all_references: list[list[str]] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating on test set")):
        pixel_values = batch["pixel_values"].to(device)
        captions = model.generate(pixel_values, max_new_tokens=max_gen_length)
        all_predictions.extend(captions)

        dataset = loader.dataset
        batch_start = batch_idx * loader.batch_size
        for i in range(len(captions)):
            idx = batch_start + i
            if idx < len(dataset):
                refs = dataset.get_all_captions(idx)
                all_references.append(refs)

    return compute_metrics(all_predictions, all_references)


@torch.no_grad()
def _run_validation(
    model: ViTQwenLoRA,
    val_loader: DataLoader,
    device: torch.device,
    max_gen_length: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions: list[str] = []
    all_references: list[list[str]] = []

    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        loss = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += loss.item()
        num_batches += 1

        captions = model.generate(pixel_values, max_new_tokens=max_gen_length)
        all_predictions.extend(captions)

        dataset = val_loader.dataset
        batch_start = (num_batches - 1) * val_loader.batch_size
        for i in range(len(captions)):
            idx = batch_start + i
            if idx < len(dataset):
                refs = dataset.get_all_captions(idx)
                all_references.append(refs)

    avg_loss = total_loss / max(num_batches, 1)
    metrics = compute_metrics(all_predictions, all_references)
    metrics["val_loss"] = avg_loss
    return metrics
