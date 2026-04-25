"""Fine-tune a Qwen3.5 decoder with LoRA using a frozen ViT encoder.

Usage:
    c5-caption finetune-lora --config configs/lora_qwen_0.8b.yaml
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoImageProcessor

from src.data.dataset import build_vision_datasets, vision_collate_fn
from src.evaluation.metrics import compute_metrics, format_metrics
from src.models.vit_qwen_lora import ViTQwenLoRA
from src.utils.config import Config, load_config
from src.utils.logger import ExperimentLogger, print_model_summary


def _build_optimizer(cfg: Config, params) -> torch.optim.Optimizer:
    name = cfg.training.optimizer.lower()
    lr = cfg.training.lr
    wd = cfg.training.get("weight_decay", 0.0)
    mo = cfg.training.get("momentum", 0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=mo)
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
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.training.get("step_size", 2), **params
        )
    elif sched == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, **params
        )
    elif sched == "none":
        return None
    raise ValueError(f"Unknown scheduler: {sched}")


def _bool_like(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on", "auto"}
    return bool(value)


def _distributed_cfg(cfg: Config) -> Config:
    return cfg.training.get("distributed", Config())


def _distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_distributed_env() -> bool:
    return _distributed_world_size() > 1


def _should_launch_distributed(cfg: Config) -> bool:
    if _is_distributed_env():
        return False
    if str(cfg.device) != "cuda" or not torch.cuda.is_available():
        return False
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        return False

    dist_cfg = _distributed_cfg(cfg)
    enabled = dist_cfg.get("enabled", "auto")
    if isinstance(enabled, str):
        enabled = enabled.lower()
        if enabled == "auto":
            requested_processes = _resolve_nproc_per_node(cfg)
            return requested_processes > 1
        return enabled in {"1", "true", "yes", "on"}
    return bool(enabled)


def _resolve_nproc_per_node(cfg: Config) -> int:
    if not torch.cuda.is_available():
        return 1
    dist_cfg = _distributed_cfg(cfg)
    requested = dist_cfg.get("nproc_per_node", "auto")
    gpu_count = torch.cuda.device_count()
    if requested in (None, "auto"):
        return gpu_count
    return max(1, min(int(requested), gpu_count))


def maybe_launch_distributed_training(
    config_path: str,
    overrides: list[str] | None = None,
) -> bool:
    overrides = overrides or []
    cfg = load_config(config_path, overrides=overrides)
    if not _should_launch_distributed(cfg):
        return False

    nproc_per_node = _resolve_nproc_per_node(cfg)
    if nproc_per_node <= 1:
        return False

    main_script = Path(__file__).resolve().parents[1] / "main.py"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        str(main_script),
        "finetune-lora",
        "--config",
        str(config_path),
    ]
    if overrides:
        cmd.extend(["--override", *overrides])

    print(f"Launching distributed LoRA training on {nproc_per_node} GPUs...")
    subprocess.run(cmd, check=True)
    return True


def _setup_runtime(cfg: Config) -> dict[str, object]:
    dist_cfg = _distributed_cfg(cfg)
    world_size = _distributed_world_size()
    is_distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if str(cfg.device) == "cuda" and torch.cuda.is_available():
        if is_distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")

        use_tf32 = _bool_like(cfg.training.get("tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cudnn.benchmark = _bool_like(
            cfg.training.get("cudnn_benchmark", True)
        )
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device(cfg.device)

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(
            backend=dist_cfg.get("backend", "nccl"),
            timeout=timedelta(minutes=int(dist_cfg.get("timeout_minutes", 60))),
        )

    return {
        "device": device,
        "is_distributed": is_distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main_process": rank == 0,
        "eval_rank0_only": _bool_like(dist_cfg.get("eval_rank0_only", True)),
    }


def _cleanup_runtime(runtime: dict[str, object]) -> None:
    if runtime["is_distributed"] and dist.is_initialized():
        dist.destroy_process_group()


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _step_scheduler(scheduler, metric: float | None = None) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()


def _build_loader(
    cfg: Config,
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    sampler,
    collate_fn,
    generator: torch.Generator | None,
    drop_last: bool = False,
) -> DataLoader:
    num_workers = int(cfg.training.get("num_workers", 0))
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": _bool_like(cfg.training.get("pin_memory", True))
        and torch.cuda.is_available(),
        "drop_last": drop_last,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = _bool_like(
            cfg.training.get("persistent_workers", True)
        )
        loader_kwargs["prefetch_factor"] = int(cfg.training.get("prefetch_factor", 2))
    return DataLoader(**loader_kwargs)


def _resolve_amp(
    cfg: Config, device: torch.device
) -> tuple[bool, torch.dtype | None, torch.amp.GradScaler | None, str]:
    amp_cfg = cfg.training.get("mixed_precision", False)
    if amp_cfg in (False, None):
        return False, None, None, "fp32"
    if device.type != "cuda":
        raise ValueError("mixed_precision requires CUDA.")

    if amp_cfg is True:
        mode = "fp16"
    else:
        mode = str(amp_cfg).lower()

    if mode in {"bf16", "bfloat16"}:
        return True, torch.bfloat16, None, "bf16"
    if mode in {"fp16", "float16", "16", "16-mixed"}:
        return True, torch.float16, torch.amp.GradScaler("cuda"), "fp16"
    raise ValueError(
        f"Unsupported mixed_precision setting '{amp_cfg}'. Use false, true/fp16, or bf16."
    )


def _broadcast_metrics(metrics: dict[str, float] | None) -> dict[str, float]:
    shared = [metrics]
    dist.broadcast_object_list(shared, src=0)
    return shared[0]


def train_lora(cfg: Config, epoch_callback=None) -> float:
    """Fine-tune a Qwen3.5 decoder with LoRA on VizWiz.

    Returns the best METEOR score achieved during training.

    Parameters
    ----------
    epoch_callback : callable | None
        Called after each epoch as ``epoch_callback(metric_value, epoch)``.
        If it returns ``True`` the training loop stops (used by Optuna pruning).
    """
    runtime = _setup_runtime(cfg)
    device = runtime["device"]
    is_distributed = runtime["is_distributed"]
    is_main_process = runtime["is_main_process"]
    rank = runtime["rank"]
    local_rank = runtime["local_rank"]

    torch.manual_seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed + rank)

    if is_main_process:
        print(f"Device: {device}")
        print(
            f"Distributed: {is_distributed} | world_size={runtime['world_size']} | local_rank={local_rank}"
        )

    encoder_id = cfg.encoder.pretrained
    decoder_id = cfg.decoder.pretrained
    lora_cfg = cfg.get("lora", {})
    proj_cfg = cfg.get("projection", {})
    if is_main_process:
        print(f"Encoder: {encoder_id} (frozen)")
        print(f"Decoder: {decoder_id} + LoRA (r={lora_cfg.get('r', 16)})")
        print(f"Projection: {proj_cfg.get('type', 'linear')}")
        if cfg.training.get("init_checkpoint"):
            print(f"Warm start checkpoint: {cfg.training.init_checkpoint}")

    proj_type = proj_cfg.get("type", "linear")
    proj_kwargs = {k: v for k, v in proj_cfg.items() if k != "type"}

    init_checkpoint = cfg.training.get("init_checkpoint")
    if init_checkpoint:
        model = ViTQwenLoRA.load_checkpoint(
            init_checkpoint,
            encoder_id=encoder_id,
            decoder_id=decoder_id,
            device=str(device),
            encoder_checkpoint=cfg.encoder.get("checkpoint"),
            num_prefix_tokens=cfg.encoder.get("num_prefix_tokens", 0),
            is_trainable=True,
        )
        model = model.to(device)
    else:
        model = ViTQwenLoRA(
            encoder_id=encoder_id,
            decoder_id=decoder_id,
            lora_r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("r", 16) * lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            lora_target=lora_cfg.get("target", "all"),
            encoder_checkpoint=cfg.encoder.get("checkpoint"),
            num_prefix_tokens=cfg.encoder.get("num_prefix_tokens", 0),
            projection_type=proj_type,
            projection_kwargs=proj_kwargs,
        )
        model = model.to(device)
    if is_distributed:
        dist_cfg = _distributed_cfg(cfg)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=_bool_like(
                dist_cfg.get("find_unused_parameters", False)
            ),
            gradient_as_bucket_view=_bool_like(
                dist_cfg.get("gradient_as_bucket_view", True)
            ),
            broadcast_buffers=_bool_like(dist_cfg.get("broadcast_buffers", False)),
            static_graph=_bool_like(dist_cfg.get("static_graph", False)),
        )
    base_model = _unwrap_model(model)

    image_processor = AutoImageProcessor.from_pretrained(encoder_id)
    tokenizer = base_model.tokenizer

    run_name = (
        cfg.get("run_name")
        or f"lora-{cfg.encoder.name}-{cfg.decoder.name}-{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(cfg.output_dir) / run_name
    ckpt_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    if is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")
    if is_distributed:
        dist.barrier()

    wandb_run = None
    if cfg.wandb.enabled and is_main_process:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=run_name,
            config=dict(cfg),
            tags=cfg.wandb.get("tags", []),
        )

    train_ds, val_ds, test_ds = build_vision_datasets(cfg, image_processor, tokenizer)
    if is_main_process:
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.seed + rank)

    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=runtime["world_size"],
            rank=rank,
            shuffle=True,
            seed=cfg.seed,
            drop_last=_bool_like(cfg.training.get("drop_last", False)),
        )
        if is_distributed
        else None
    )

    train_loader = _build_loader(
        cfg,
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        sampler=train_sampler,
        collate_fn=vision_collate_fn,
        generator=loader_generator,
        drop_last=_bool_like(cfg.training.get("drop_last", False)),
    )
    val_loader = None
    test_loader = None
    if (not is_distributed) or runtime["eval_rank0_only"] is False or is_main_process:
        val_loader = _build_loader(
            cfg,
            val_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            sampler=None,
            collate_fn=vision_collate_fn,
            generator=None,
        )
        test_loader = _build_loader(
            cfg,
            test_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            sampler=None,
            collate_fn=vision_collate_fn,
            generator=None,
        )

    exp_logger = ExperimentLogger(output_dir, dict(cfg)) if is_main_process else None
    if exp_logger is not None:
        model_info = exp_logger.log_model_info(base_model, device=str(device))
        print_model_summary(model_info)

    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found.")
    if is_main_process:
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in base_model.parameters())
        print(
            f"Trainable: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.2f}%)"
        )

    optimizer = _build_optimizer(cfg, trainable_params)
    scheduler = _build_scheduler(cfg, optimizer)

    use_amp, amp_dtype, scaler, amp_mode = _resolve_amp(cfg, device)
    grad_accum_steps = cfg.training.get("gradient_accumulation_steps", 1)
    if is_main_process:
        print(f"Mixed precision: {amp_mode}")

    best_metric = -1.0
    best_epoch = 0
    best_metrics_dict: dict[str, float] = {}
    patience_counter = 0
    patience = cfg.training.get("early_stopping_patience")
    max_gen_length = cfg.inference.get("max_length", 128)
    if exp_logger is not None:
        exp_logger.start_training()

    try:
        for epoch in range(cfg.training.epochs):
            if exp_logger is not None:
                exp_logger.start_epoch()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            base_model.encoder.eval()  # encoder always frozen
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{cfg.training.epochs}",
                disable=not is_main_process,
            )
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(pbar):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                should_step = (step + 1) % grad_accum_steps == 0 or (step + 1) == len(
                    train_loader
                )
                sync_context = (
                    model.no_sync()
                    if is_distributed and grad_accum_steps > 1 and not should_step
                    else nullcontext()
                )

                with sync_context:
                    autocast_context = (
                        torch.amp.autocast("cuda", dtype=amp_dtype)
                        if use_amp and device.type == "cuda"
                        else nullcontext()
                    )
                    with autocast_context:
                        loss = model(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        scaled_loss = loss / grad_accum_steps

                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                if should_step:
                    if scaler is not None:
                        if cfg.training.grad_clip:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(
                                base_model.parameters(), cfg.training.grad_clip
                            )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if cfg.training.grad_clip:
                            nn.utils.clip_grad_norm_(
                                base_model.parameters(), cfg.training.grad_clip
                            )
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += float(loss.item())
                num_batches += 1
                if is_main_process:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            loss_stats = torch.tensor(
                [epoch_loss, float(num_batches)], device=device, dtype=torch.float64
            )
            if is_distributed:
                dist.all_reduce(loss_stats)
            avg_loss = loss_stats[0].item() / max(loss_stats[1].item(), 1.0)

            if is_distributed:
                dist.barrier()

            if is_main_process:
                val_metrics = _run_validation(
                    base_model,
                    val_loader,
                    device,
                    max_gen_length,
                    disable_tqdm=False,
                )
            else:
                val_metrics = None

            if is_distributed:
                val_metrics = _broadcast_metrics(val_metrics)

            val_loss = val_metrics.pop("val_loss", 0.0)
            current_metric = val_metrics.get("meteor", 0.0)
            _step_scheduler(scheduler, current_metric)

            is_best = current_metric > best_metric

            if is_main_process:
                print(
                    f"Epoch {epoch + 1}: train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"{format_metrics(val_metrics)}"
                )
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

                base_model.save_checkpoint(str(ckpt_dir / "last"))

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch + 1
                best_metrics_dict = dict(val_metrics)
                if is_main_process:
                    base_model.save_checkpoint(str(ckpt_dir / "best"))
                    print(f"  -> New best model (meteor={best_metric * 100:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch_callback is not None:
                should_stop = epoch_callback(current_metric, epoch + 1)
                if should_stop:
                    if is_main_process:
                        print(f"Trial pruned at epoch {epoch + 1}.")
                    break

            if patience and patience_counter >= patience:
                if is_main_process:
                    print(f"Early stopping after {patience} epochs without improvement.")
                break

        if exp_logger is not None:
            exp_logger.end_training(best_epoch=best_epoch, best_metrics=best_metrics_dict)

        if is_distributed:
            dist.barrier()

        if is_main_process:
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
            post_metrics, post_samples = _run_evaluation(
                best_model,
                test_loader,
                device,
                max_gen_length,
                disable_tqdm=False,
            )
            exp_logger.log_test_eval(post_metrics, samples=post_samples)
            print(f"  {format_metrics(post_metrics)}")
            print("=" * 60)

            del best_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            log_path = exp_logger.save()
            print(f"\nTraining complete. Best METEOR: {best_metric * 100:.2f}%")
            print(f"Checkpoints saved to: {ckpt_dir}")
            print(f"Experiment log: {log_path}")

        if is_distributed:
            dist.barrier()
    finally:
        if wandb_run:
            wandb_run.finish()
        _cleanup_runtime(runtime)
    return best_metric


@torch.no_grad()
def _run_evaluation(
    model: ViTQwenLoRA,
    loader: DataLoader,
    device: torch.device,
    max_gen_length: int,
    num_samples: int = 15,
    disable_tqdm: bool = False,
) -> tuple[dict[str, float], list[dict]]:
    """Run generation-only evaluation. Returns (metrics, sample_predictions)."""
    model.eval()
    all_predictions: list[str] = []
    all_references: list[list[str]] = []
    all_image_paths: list[str] = []

    for batch_idx, batch in enumerate(
        tqdm(loader, desc="Evaluating on test set", disable=disable_tqdm)
    ):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        captions = model.generate(pixel_values, max_new_tokens=max_gen_length)
        all_predictions.extend(captions)
        all_image_paths.extend(batch["image_paths"])

        dataset = loader.dataset
        batch_start = batch_idx * loader.batch_size
        for i in range(len(captions)):
            idx = batch_start + i
            if idx < len(dataset):
                refs = dataset.get_all_captions(idx)
                all_references.append(refs)

    metrics = compute_metrics(all_predictions, all_references)

    rng = random.Random(42)
    indices = list(range(len(all_predictions)))
    rng.shuffle(indices)
    samples = [
        {
            "image": Path(all_image_paths[i]).name,
            "prediction": all_predictions[i],
            "references": all_references[i],
        }
        for i in indices[:num_samples]
    ]
    return metrics, samples


@torch.no_grad()
def _run_validation(
    model: ViTQwenLoRA,
    val_loader: DataLoader,
    device: torch.device,
    max_gen_length: int,
    disable_tqdm: bool = False,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions: list[str] = []
    all_references: list[list[str]] = []

    for batch in tqdm(val_loader, desc="Validating", disable=disable_tqdm):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

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
