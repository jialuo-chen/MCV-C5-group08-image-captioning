"""Generate synthetic training data using Stable Diffusion + VizWiz-format output.

The prompt used to generate each image becomes the ground-truth caption,
producing a ready-to-use training set compatible with VizWizVisionDataset.

Usage:
    uv run python main.py generate-synthetic --config configs/sd_synthetic_data.yaml
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
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


def generate_synthetic_data(cfg: Config) -> Path:
    """Generate synthetic images and save them with VizWiz-format annotations."""
    sd_cfg = cfg.get("stable_diffusion", {})
    model_id = sd_cfg.get("model_id", "stabilityai/stable-diffusion-3.5-large-turbo")
    scheduler_name = sd_cfg.get("scheduler", "default")
    guidance_scale = sd_cfg.get("guidance_scale", 0.0)
    num_steps = sd_cfg.get("num_inference_steps", 4)
    images_per_prompt = sd_cfg.get("images_per_prompt", 1)
    width = sd_cfg.get("width", 1024)
    height = sd_cfg.get("height", 1024)
    seed = sd_cfg.get("seed", cfg.get("seed", 42))
    dtype_str = sd_cfg.get("dtype", "bfloat16")
    negative_prompt = sd_cfg.get("negative_prompt", None)

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

    print(f"Loading pipeline: {model_id}")
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    if hasattr(pipe, "transformer"):
        pipe.transformer = torch.compile(
            pipe.transformer, mode="reduce-overhead", fullgraph=True
        )

    if scheduler_name != "default" and scheduler_name in SCHEDULER_MAP:
        sched_cls = SCHEDULER_MAP[scheduler_name]
        pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)
        print(f"Scheduler: {scheduler_name}")

    vizwiz_cfg = cfg.get("vizwiz_style", {})
    vizwiz_enabled = vizwiz_cfg.get("enabled", False)
    vizwiz_prefixes = vizwiz_cfg.get("prefixes", [])
    clean_fraction = vizwiz_cfg.get("clean_fraction", 0.2)
    batch_size = sd_cfg.get("batch_size", 4)

    generator = torch.Generator("cuda").manual_seed(seed)
    rng = random.Random(seed)
    generated = []
    img_counter = 0

    total = len(prompt_entries) * images_per_prompt
    print(
        f"Generating {total} images ({len(prompt_entries)} prompts x {images_per_prompt} each, batch_size={batch_size})"
    )
    if vizwiz_enabled:
        print(
            f"VizWiz-style augmentation ON ({len(vizwiz_prefixes)} prefixes, {clean_fraction:.0%} clean)"
        )

    t_start = time.time()
    for p_idx, entry in enumerate(prompt_entries):
        is_template = "pattern" in entry and "variables" in entry
        category = entry.get("category", "general")
        display = entry["pattern"] if is_template else entry["prompt"]
        print(f"[{p_idx + 1}/{len(prompt_entries)}] ({category}) {display[:80]}...")

        all_prompts = []
        all_captions = []
        for _ in range(images_per_prompt):
            if is_template:
                fills = {k: rng.choice(v) for k, v in entry["variables"].items()}
                base_prompt = entry["pattern"].format(**fills)
                caption = entry.get("caption_pattern", entry["pattern"] + ".").format(
                    **fills
                )
            else:
                base_prompt = entry["prompt"]
                caption = entry["caption"]

            if vizwiz_enabled and vizwiz_prefixes and rng.random() > clean_fraction:
                prefix = rng.choice(vizwiz_prefixes)
                first_char = base_prompt[0].lower() if base_prompt else ""
                all_prompts.append(f"{prefix} {first_char}{base_prompt[1:]}")
            else:
                all_prompts.append(base_prompt)
            all_captions.append(caption)

        for batch_start in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[batch_start : batch_start + batch_size]
            batch_captions = all_captions[batch_start : batch_start + batch_size]
            bs = len(batch_prompts)
            neg = [negative_prompt] * bs if negative_prompt else None

            result = pipe(
                prompt=batch_prompts,
                negative_prompt=neg,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                num_images_per_prompt=1,
                width=width,
                height=height,
                generator=generator,
            )

            for i, image in enumerate(result.images):
                img_counter += 1
                fname = f"synthetic_{img_counter:05d}.png"
                image.save(img_dir / fname)

                generated.append(
                    {
                        "file_name": fname,
                        "prompt": batch_prompts[i],
                        "caption": batch_captions[i],
                        "category": category,
                        "width": width,
                        "height": height,
                    }
                )

        elapsed_total = time.time() - t_start
        imgs_done = (p_idx + 1) * images_per_prompt
        rate = imgs_done / elapsed_total
        eta = (total - imgs_done) / rate if rate > 0 else 0
        print(
            f"  {images_per_prompt} images done | {rate:.1f} img/s | ETA {eta / 60:.0f}m"
        )

    vizwiz_ann = _build_vizwiz_annotations(generated)
    ann_path = ann_dir / "synthetic.json"
    ann_path.write_text(json.dumps(vizwiz_ann, indent=2))

    meta_path = output_root / "generation_metadata.json"
    meta = {
        "model_id": model_id,
        "scheduler": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_steps,
        "images_per_prompt": images_per_prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "total_images": img_counter,
        "items": generated,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nGenerated {img_counter} images → {img_dir}")
    print(f"VizWiz annotations → {ann_path}")
    print(f"Metadata → {meta_path}")
    return output_root
