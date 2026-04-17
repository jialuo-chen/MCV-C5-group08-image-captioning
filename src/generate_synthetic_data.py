"""Generate synthetic training data using Stable Diffusion + VizWiz-format output.

The prompt used to generate each image becomes the ground-truth caption,
producing a ready-to-use training set compatible with VizWizVisionDataset.

Supports single-GPU and multi-GPU generation via config flags:
  - num_gpus: "auto" (all available), or an integer
  - cpu_offload: true/false (for low-VRAM GPUs like RTX 3090 24GB)

Usage:
    uv run python main.py generate-synthetic --config configs/sd_synthetic_data.yaml
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from diffusers import AutoPipelineForText2Image

from src.generate_sd import DTYPE_MAP, SCHEDULER_MAP
from src.utils.config import Config


def _load_prompts(cfg: Config) -> list[dict]:
    """Load prompts as list of {prompt, caption, category}.

    Supports:
    - ``prompts_file``: path to a JSON with list of prompt dicts
    - ``prompts``: inline list in config (strings or dicts)
    """
    prompts_file = cfg.get("prompts_file")
    if prompts_file and Path(prompts_file).is_file():
        data = json.loads(Path(prompts_file).read_text())
        entries = []
        for item in data:
            if isinstance(item, str):
                entries.append({"prompt": item, "caption": item, "category": "general"})
            else:
                entries.append(
                    {
                        "prompt": item.get("prompt", ""),
                        "caption": item.get("caption", item.get("prompt", "")),
                        "category": item.get("category", "general"),
                    }
                )
        return entries

    prompts_list = cfg.get("prompts", [])
    entries = []
    for item in prompts_list:
        if isinstance(item, str):
            entries.append({"prompt": item, "caption": item, "category": "general"})
        elif isinstance(item, dict):
            if "pattern" in item and "variables" in item:
                entries.append(item)
            else:
                entries.append(
                    {
                        "prompt": item.get("prompt", ""),
                        "caption": item.get("caption", item.get("prompt", "")),
                        "category": item.get("category", "general"),
                    }
                )
    return entries


def _expand_all_work_items(
    prompt_entries: list[dict],
    imgs_template: int,
    imgs_static: int,
    rng: random.Random,
    available_sizes: list[tuple[int, int]],
    vizwiz_cfg: dict,
) -> list[dict]:
    """Expand all prompt entries into a flat list of work items (deterministic)."""
    vizwiz_enabled = vizwiz_cfg.get("enabled", False)
    vizwiz_prefixes = vizwiz_cfg.get("prefixes", [])
    clean_fraction = vizwiz_cfg.get("clean_fraction", 0.2)

    work_items = []
    for entry in prompt_entries:
        is_template = "pattern" in entry and "variables" in entry
        category = entry.get("category", "general")
        num_images = imgs_template if is_template else imgs_static
        caption_synonyms = entry.get("caption_synonyms", [])

        for _ in range(num_images):
            if is_template:
                fills = {k: rng.choice(v) for k, v in entry["variables"].items()}
                base_prompt = entry["pattern"].format(**fills)
                base_caption = entry.get(
                    "caption_pattern", entry["pattern"] + "."
                ).format(**fills)
            else:
                base_prompt = entry["prompt"]
                base_caption = entry["caption"]

            if caption_synonyms and rng.random() < 0.5:
                caption = rng.choice(caption_synonyms)
                if is_template:
                    caption = caption.format(**fills)
            else:
                caption = base_caption

            if vizwiz_enabled and vizwiz_prefixes and rng.random() > clean_fraction:
                prefix = rng.choice(vizwiz_prefixes)
                first_char = base_prompt[0].lower() if base_prompt else ""
                prompt = f"{prefix} {first_char}{base_prompt[1:]}"
            else:
                prompt = base_prompt

            w, h = rng.choice(available_sizes)
            work_items.append(
                {
                    "prompt": prompt,
                    "caption": caption,
                    "category": category,
                    "width": w,
                    "height": h,
                }
            )
    return work_items


def _build_vizwiz_annotations(
    generated: list[dict],
) -> dict:
    """Build a VizWiz-compatible annotation dict from generated image metadata."""
    images = []
    annotations = []
    for i, item in enumerate(generated):
        img_id = i + 1
        images.append(
            {
                "id": img_id,
                "file_name": item["file_name"],
                "width": item.get("width", 1024),
                "height": item.get("height", 1024),
            }
        )
        annotations.append(
            {
                "id": img_id,
                "image_id": img_id,
                "caption": item["caption"],
                "is_rejected": False,
                "is_precanned": False,
            }
        )
    return {"images": images, "annotations": annotations}


def _generate_on_device(
    device_id: int,
    work_items: list[dict],
    global_indices: list[int],
    model_id: str,
    dtype: torch.dtype,
    scheduler_name: str,
    guidance_scale: float,
    num_steps: int,
    negative_prompt: str | None,
    batch_size: int,
    cpu_offload: bool,
    seed: int,
    img_dir: Path,
    result_file: Path,
) -> None:
    """Load pipeline on *device_id* and generate images for *work_items*.

    Results are written to *result_file* as JSON so the caller can merge them.
    Safe to run inside an ``mp.Process``.
    """
    device = f"cuda:{device_id}"

    print(f"[GPU {device_id}] Loading pipeline: {model_id}")
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)

    if cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=device_id)
    else:
        pipe = pipe.to(device)

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    if not cpu_offload and hasattr(pipe, "transformer"):
        try:
            pipe.transformer = torch.compile(
                pipe.transformer, mode="reduce-overhead", fullgraph=True
            )
        except Exception:
            pass

    if scheduler_name != "default" and scheduler_name in SCHEDULER_MAP:
        sched_cls = SCHEDULER_MAP[scheduler_name]
        pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)

    gen_device = "cpu" if cpu_offload else device
    generator = torch.Generator(gen_device).manual_seed(seed + device_id)

    size_groups: dict[tuple[int, int], list[int]] = {}
    for local_idx, item in enumerate(work_items):
        sz = (item["width"], item["height"])
        size_groups.setdefault(sz, []).append(local_idx)

    generated: list[dict] = []
    done = 0
    t_start = time.time()

    for (w, h), local_indices in size_groups.items():
        for batch_start in range(0, len(local_indices), batch_size):
            batch_local = local_indices[batch_start : batch_start + batch_size]
            batch_prompts = [work_items[i]["prompt"] for i in batch_local]
            bs = len(batch_prompts)
            neg = [negative_prompt] * bs if negative_prompt else None

            result = pipe(
                prompt=batch_prompts,
                negative_prompt=neg,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                num_images_per_prompt=1,
                width=w,
                height=h,
                generator=generator,
            )

            for j, image in enumerate(result.images):
                li = batch_local[j]
                gi = global_indices[li]
                item = work_items[li]
                fname = f"synthetic_{gi:05d}.png"
                image.save(img_dir / fname)
                generated.append(
                    {
                        "file_name": fname,
                        "prompt": item["prompt"],
                        "caption": item["caption"],
                        "category": item["category"],
                        "width": w,
                        "height": h,
                    }
                )
                done += 1

            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(work_items) - done) / rate if rate > 0 else 0
            print(
                f"  [GPU {device_id}] {done}/{len(work_items)} | "
                f"{rate:.1f} img/s | ETA {eta / 60:.0f}m"
            )

    result_file.write_text(json.dumps(generated))
    print(f"[GPU {device_id}] Done: {done} images")


def generate_synthetic_data(cfg: Config) -> Path:
    """Generate synthetic images and save them with VizWiz-format annotations."""
    sd_cfg = cfg.get("stable_diffusion", {})
    model_id = sd_cfg.get("model_id", "stabilityai/stable-diffusion-3.5-large-turbo")
    scheduler_name = sd_cfg.get("scheduler", "default")
    guidance_scale = sd_cfg.get("guidance_scale", 0.0)
    num_steps = sd_cfg.get("num_inference_steps", 4)
    imgs_template = sd_cfg.get(
        "images_per_template_prompt", sd_cfg.get("images_per_prompt", 50)
    )
    imgs_static = sd_cfg.get(
        "images_per_static_prompt", sd_cfg.get("images_per_prompt", 50)
    )
    seed = sd_cfg.get("seed", cfg.get("seed", 42))
    dtype_str = sd_cfg.get("dtype", "bfloat16")
    negative_prompt = sd_cfg.get("negative_prompt", None)
    batch_size = sd_cfg.get("batch_size", 4)
    cpu_offload = sd_cfg.get("cpu_offload", False)

    num_gpus_cfg = sd_cfg.get("num_gpus", "auto")
    available_gpus = torch.cuda.device_count()
    if num_gpus_cfg == "auto":
        num_gpus = max(1, available_gpus)
    else:
        num_gpus = max(1, min(int(num_gpus_cfg), available_gpus))

    size_cfg = sd_cfg.get("image_sizes", None)
    if size_cfg:
        available_sizes = [
            (s, s) if isinstance(s, int) else (s[0], s[1]) for s in size_cfg
        ]
    else:
        w = sd_cfg.get("width", 1024)
        h = sd_cfg.get("height", 1024)
        available_sizes = [(w, h)]

    output_root = Path(cfg.get("output_dir", "outputs")) / cfg.get(
        "run_name", "synthetic_data"
    )
    img_dir = output_root / "images"
    ann_dir = output_root / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    prompt_entries = _load_prompts(cfg)
    if not prompt_entries:
        raise ValueError(
            "No prompts found. Set 'prompts' list or 'prompts_file' in config."
        )

    dtype = DTYPE_MAP.get(dtype_str, torch.bfloat16)
    vizwiz_cfg_dict = cfg.get("vizwiz_style", {})
    rng = random.Random(seed)

    work_items = _expand_all_work_items(
        prompt_entries, imgs_template, imgs_static, rng, available_sizes, vizwiz_cfg_dict
    )
    total_imgs = len(work_items)
    global_indices = list(range(1, total_imgs + 1))

    print(f"Total: {total_imgs} images | GPUs: {num_gpus} | cpu_offload: {cpu_offload}")
    print(f"  Image sizes: {available_sizes} | batch_size={batch_size}")

    t_start = time.time()

    if num_gpus == 1:
        result_file = output_root / "_results_gpu0.json"
        _generate_on_device(
            device_id=0,
            work_items=work_items,
            global_indices=global_indices,
            model_id=model_id,
            dtype=dtype,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            cpu_offload=cpu_offload,
            seed=seed,
            img_dir=img_dir,
            result_file=result_file,
        )
        generated = json.loads(result_file.read_text())
        result_file.unlink()
    else:
        shards: list[list[dict]] = [[] for _ in range(num_gpus)]
        shard_indices: list[list[int]] = [[] for _ in range(num_gpus)]
        for i, item in enumerate(work_items):
            gpu = i % num_gpus
            shards[gpu].append(item)
            shard_indices[gpu].append(global_indices[i])

        result_files = [output_root / f"_results_gpu{r}.json" for r in range(num_gpus)]

        ctx = mp.get_context("spawn")
        processes = []
        for rank in range(num_gpus):
            p = ctx.Process(
                target=_generate_on_device,
                args=(
                    rank,
                    shards[rank],
                    shard_indices[rank],
                    model_id,
                    dtype,
                    scheduler_name,
                    guidance_scale,
                    num_steps,
                    negative_prompt,
                    batch_size,
                    cpu_offload,
                    seed,
                    img_dir,
                    result_files[rank],
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                print(f"WARNING: worker exited with code {p.exitcode}")

        generated = []
        for rf in result_files:
            if rf.exists():
                generated.extend(json.loads(rf.read_text()))
                rf.unlink()
        generated.sort(key=lambda x: x["file_name"])

    elapsed = time.time() - t_start
    rate = len(generated) / elapsed if elapsed > 0 else 0
    print(
        f"\nGenerated {len(generated)} images in {elapsed / 60:.1f}m ({rate:.1f} img/s)"
    )

    vizwiz_ann = _build_vizwiz_annotations(generated)
    ann_path = ann_dir / "synthetic.json"
    ann_path.write_text(json.dumps(vizwiz_ann, indent=2))

    meta_path = output_root / "generation_metadata.json"
    meta = {
        "model_id": model_id,
        "num_gpus": num_gpus,
        "cpu_offload": cpu_offload,
        "scheduler": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_steps,
        "images_per_template_prompt": imgs_template,
        "images_per_static_prompt": imgs_static,
        "image_sizes": available_sizes,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "total_images": len(generated),
        "items": generated,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Images → {img_dir}")
    print(f"VizWiz annotations → {ann_path}")
    print(f"Metadata → {meta_path}")
    return output_root
