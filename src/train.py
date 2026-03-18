"""Training loop for image captioning models.

Supports both RNN decoders (CrossEntropyLoss) and HF LM decoders (causal LM loss).
Includes WandB logging, checkpointing, gradient clipping, and optional mixed precision.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import VizWizCaptionDataset, caption_collate_fn, build_datasets
from src.data.tokenizer import BaseTokenizer, build_tokenizer
from src.models.captioner import CaptioningModel, build_captioning_model
from src.models.decoders import HFLMDecoder
from src.evaluation.metrics import compute_metrics, format_metrics
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
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _build_scheduler(cfg: Config, optimizer):
    sched = cfg.training.get("scheduler")
    if sched is None:
        return None
    params = cfg.training.get("scheduler_params", {})
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, **params,
        )
    elif sched == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    else:
        raise ValueError(f"Unknown scheduler: {sched}")


def _collect_train_captions(annotation_file: str) -> list[str]:
    """Extract all caption strings from a VizWiz annotation file."""
    from src.data.vizwiz import VizWiz
    vw = VizWiz(annotation_file, ignore_rejected=True, ignore_precanned=True)
    captions = []
    for ann in vw.dataset.get("annotations", []):
        cap = ann.get("caption", "")
        if cap:
            captions.append(cap)
    return captions


# ===================================================================
# Training
# ===================================================================

def train(cfg: Config) -> None:
    """Run the full training loop.

    Steps:
    1. Build tokenizer (train vocab if needed).
    2. Build datasets and dataloaders.
    3. Build model.
    4. Train for N epochs with validation after each.
    5. Save best and last checkpoints.
    6. Log to WandB if enabled.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Seed ---------------------------------------------------------------
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # -- Run name -----------------------------------------------------------
    run_name = cfg.get("run_name")
    if not run_name:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.encoder.name}-{cfg.decoder.name}-{cfg.tokenizer.type}-{ts}"
    output_dir = Path(cfg.output_dir) / run_name
    ckpt_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # -- WandB --------------------------------------------------------------
    wandb_run = None
    if cfg.wandb.enabled:
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=run_name,
            config=dict(cfg),
            tags=cfg.wandb.get("tags", []),
        )

    # -- Tokenizer ----------------------------------------------------------
    root = Path(cfg.dataset.root)
    train_ann_path = str(root / cfg.dataset.train_ann)

    is_hf_lm = cfg.decoder.type == "hf_lm"
    tokenizer: BaseTokenizer | None = None

    if not is_hf_lm:
        captions = _collect_train_captions(train_ann_path)
        print(f"Collected {len(captions)} training captions for tokenizer.")
        tokenizer = build_tokenizer(cfg.tokenizer, captions=captions)
        print(f"Tokenizer: {cfg.tokenizer.type}, vocab_size={tokenizer.vocab_size}")

    # -- Datasets -----------------------------------------------------------
    if is_hf_lm:
        # HF LM uses its own tokenizer — we still need a BaseTokenizer wrapper
        # For simplicity, use CharTokenizer as a pass-through for dataset loading.
        # The actual encoding will be handled in the training loop.
        from src.data.tokenizer import CharTokenizer
        tokenizer = CharTokenizer()  # placeholder for dataset loading

    train_ds, val_ds, test_ds = build_datasets(cfg, tokenizer)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=caption_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=caption_collate_fn,
        pin_memory=True,
    )

    # -- Model --------------------------------------------------------------
    if is_hf_lm:
        model = build_captioning_model(cfg)
    else:
        model = build_captioning_model(cfg, vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    model = model.to(device)

    # -- Logger -------------------------------------------------------------
    exp_logger = ExperimentLogger(output_dir, dict(cfg))
    model_info = exp_logger.log_model_info(model, device=str(device))
    print_model_summary(model_info)

    # -- Optimizer, scheduler, criterion ------------------------------------
    optimizer = _build_optimizer(cfg, model.parameters())
    scheduler = _build_scheduler(cfg, optimizer)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id) if not is_hf_lm else None

    # -- Mixed precision ----------------------------------------------------
    use_amp = cfg.training.get("mixed_precision", False)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # -- Training loop ------------------------------------------------------
    best_metric = -1.0
    best_epoch = 0
    best_metrics_dict: dict[str, float] = {}
    patience_counter = 0
    patience = cfg.training.get("early_stopping_patience")
    exp_logger.start_training()

    for epoch in range(cfg.training.epochs):
        exp_logger.start_epoch()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}")
        for images, cap_tensors, cap_texts, img_paths in pbar:
            images = images.to(device)
            cap_tensors = cap_tensors.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    if is_hf_lm:
                        loss = model(images, cap_tensors)
                    else:
                        logits = model(images, cap_tensors)
                        targets = cap_tensors[:, 1:]  # shift: predict from SOS onward
                        loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                if cfg.training.grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if is_hf_lm:
                    loss = model(images, cap_tensors)
                else:
                    logits = model(images, cap_tensors)
                    targets = cap_tensors[:, 1:]
                    loss = criterion(logits, targets)
                loss.backward()
                if cfg.training.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)

        # -- Validation -----------------------------------------------------
        val_metrics = _run_validation(model, val_loader, tokenizer, device, cfg, criterion)
        val_loss = val_metrics.pop("val_loss", 0.0)

        print(
            f"Epoch {epoch + 1}: train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
            f"{format_metrics(val_metrics)}"
        )

        # -- Logging --------------------------------------------------------
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        if wandb_run:
            wandb_run.log(log_dict)

        # -- Logger epoch ---------------------------------------------------
        is_best = val_metrics.get("meteor", 0.0) > best_metric
        exp_logger.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_loss=val_loss,
            metrics=val_metrics,
            lr=optimizer.param_groups[0]["lr"],
            is_best=is_best,
        )

        # -- Scheduler step -------------------------------------------------
        if scheduler:
            scheduler.step()

        # -- Checkpointing --------------------------------------------------
        # Save last
        model.save_checkpoint(
            ckpt_dir / "last.pt",
            config=dict(cfg),
            tokenizer=tokenizer if not is_hf_lm else None,
            epoch=epoch + 1,
            metrics=val_metrics,
        )

        # Save best (by METEOR)
        current_metric = val_metrics.get("meteor", 0.0)
        if current_metric > best_metric:
            best_metric = current_metric
            model.save_checkpoint(
                ckpt_dir / "best.pt",
                config=dict(cfg),
                tokenizer=tokenizer if not is_hf_lm else None,
                epoch=epoch + 1,
                metrics=val_metrics,
            )
            best_epoch = epoch + 1
            best_metrics_dict = dict(val_metrics)
            print(f"  -> New best model (meteor={best_metric * 100:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1

        # -- Early stopping -------------------------------------------------
        if patience and patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

    exp_logger.end_training(best_epoch=best_epoch, best_metrics=best_metrics_dict)
    log_path = exp_logger.save()
    print(f"\nTraining complete. Best METEOR: {best_metric * 100:.2f}%")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Experiment log saved to: {log_path}")

    if wandb_run:
        wandb_run.finish()


# ===================================================================
# Validation helper
# ===================================================================

@torch.no_grad()
def _run_validation(
    model: CaptioningModel,
    val_loader: DataLoader,
    tokenizer: BaseTokenizer,
    device: torch.device,
    cfg: Config,
    criterion: nn.Module | None,
) -> dict[str, float]:
    """Run validation: compute loss + captioning metrics."""
    model.eval()
    is_hf_lm = isinstance(model.decoder, HFLMDecoder)

    total_loss = 0.0
    num_batches = 0
    all_predictions: list[str] = []
    all_references: list[list[str]] = []

    for images, cap_tensors, cap_texts, img_paths in val_loader:
        images = images.to(device)
        cap_tensors = cap_tensors.to(device)

        # Loss
        if is_hf_lm:
            loss = model(images, cap_tensors)
        else:
            logits = model(images, cap_tensors)
            targets = cap_tensors[:, 1:]
            loss = criterion(logits, targets)
        total_loss += loss.item()
        num_batches += 1

        # Generate captions for metrics
        generated = model.generate(
            images,
            tokenizer=tokenizer,
            max_length=cfg.inference.get("max_length", cfg.tokenizer.max_length),
        )
        all_predictions.extend(generated)

        # Collect references (all captions for each image)
        dataset = val_loader.dataset
        batch_start = (num_batches - 1) * val_loader.batch_size
        for i in range(len(generated)):
            idx = batch_start + i
            if idx < len(dataset):
                refs = dataset.get_all_captions(idx)
                all_references.append(refs)

    avg_loss = total_loss / max(num_batches, 1)

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_references)
    metrics["val_loss"] = avg_loss
    return metrics
