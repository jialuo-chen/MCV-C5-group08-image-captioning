"""Fine-tune a VisionEncoderDecoderModel (ViT/CLIP encoder + GPT2/T5/SmolLM decoder).

Usage:
    c5-caption finetune --config configs/vit_gpt2.yaml
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
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
)

from src.data.dataset import build_vision_datasets, vision_collate_fn
from src.evaluation.metrics import compute_metrics, format_metrics
from src.utils.config import Config
from src.utils.logger import ExperimentLogger, print_model_summary


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


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


def _setup_tokenizer(tokenizer, model):
    """Configure special tokens for the decoder tokenizer & model."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    return tokenizer


def train_vit_decoder(cfg: Config) -> float:
    """Fine-tune a VisionEncoderDecoderModel.

    Returns the best METEOR score achieved during training.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    encoder_id = cfg.encoder.pretrained
    decoder_id = cfg.decoder.pretrained
    print(f"Encoder: {encoder_id}  |  Decoder: {decoder_id}")

    # Load decoder config first and ensure it's set up for cross-attention
    # (required by VisionEncoderDecoderModel for models like GPT2 that
    # weren't originally trained as encoder-decoder decoders).
    decoder_config = AutoConfig.from_pretrained(decoder_id)
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_id, decoder_id, decoder_config=decoder_config
    )

    image_processor = AutoImageProcessor.from_pretrained(encoder_id)
    tokenizer = AutoTokenizer.from_pretrained(decoder_id)
    tokenizer = _setup_tokenizer(tokenizer, model)

    run_name = (
        cfg.get("run_name")
        or f"{cfg.encoder.name}-{cfg.decoder.name}-{time.strftime('%Y%m%d_%H%M%S')}"
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

    model = model.to(device)
    freeze_encoder = bool(cfg.training.get("freeze_encoder", False))
    freeze_decoder = bool(cfg.training.get("freeze_decoder", False))
    if freeze_encoder:
        _set_module_trainable(model.encoder, trainable=False)
        print("Encoder frozen.")
    if freeze_decoder:
        _set_module_trainable(model.decoder, trainable=False)
        print("Decoder frozen.")

    exp_logger = ExperimentLogger(output_dir, dict(cfg))
    model_info = exp_logger.log_model_info(model, device=str(device))
    print_model_summary(model_info)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters — unfreeze encoder or decoder.")

    optimizer = _build_optimizer(cfg, trainable_params)
    scheduler = _build_scheduler(cfg, optimizer)

    use_amp = cfg.training.get("mixed_precision", False)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_metric = -1.0
    best_epoch = 0
    best_metrics_dict: dict[str, float] = {}
    patience_counter = 0
    patience = cfg.training.get("early_stopping_patience")
    exp_logger.start_training()
    max_gen_length = cfg.inference.get("max_length", 128)

    for epoch in range(cfg.training.epochs):
        exp_logger.start_epoch()
        model.train()
        if freeze_encoder:
            model.encoder.eval()
        if freeze_decoder:
            model.decoder.eval()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                if cfg.training.grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                if cfg.training.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)

        val_metrics = _run_validation(
            model, val_loader, tokenizer, device, max_gen_length
        )
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

        last_dir = ckpt_dir / "last"
        model.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)
        image_processor.save_pretrained(last_dir)

        current_metric = val_metrics.get("meteor", 0.0)
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_metrics_dict = dict(val_metrics)
            best_dir = ckpt_dir / "best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            image_processor.save_pretrained(best_dir)
            print(f"  -> New best model (meteor={best_metric * 100:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience and patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

    exp_logger.end_training(best_epoch=best_epoch, best_metrics=best_metrics_dict)
    log_path = exp_logger.save()
    print(f"\nTraining complete. Best METEOR: {best_metric * 100:.2f}%")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Experiment log: {log_path}")

    if wandb_run:
        wandb_run.finish()
    return best_metric


@torch.no_grad()
def _run_validation(
    model: VisionEncoderDecoderModel,
    val_loader: DataLoader,
    tokenizer,
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
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()
        num_batches += 1

        gen_ids = model.generate(
            pixel_values,
            max_new_tokens=max_gen_length,
            max_length=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        captions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
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
