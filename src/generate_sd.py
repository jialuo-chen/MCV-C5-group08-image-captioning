"""Generate images using Stable Diffusion models via HuggingFace diffusers.

Usage:
    uv run python main.py generate-sd --config configs/sd_inference.yaml
    uv run python main.py generate-sd --config configs/sd_inference.yaml --prompt "a cat on a table"
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from diffusers import (
    AutoPipelineForText2Image,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
)

from src.utils.config import Config

SCHEDULER_MAP = {
    "flow_match_euler": FlowMatchEulerDiscreteScheduler,
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "dpm++": DPMSolverMultistepScheduler,
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _load_prompts(prompt_source: str) -> list[str]:
    """Load prompts from a string, text file, or JSON file."""
    path = Path(prompt_source)
    if path.is_file():
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return [p if isinstance(p, str) else p["prompt"] for p in data]
            return [data["prompt"]]
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [prompt_source]


def generate_sd(cfg: Config, prompt_override: str | None = None) -> Path:
    """Generate images from text prompts using a Stable Diffusion pipeline.

    Returns the output directory path.
    """
    sd_cfg = cfg.get("stable_diffusion", cfg)
    model_id = sd_cfg.get("model_id", "stabilityai/stable-diffusion-3.5-large-turbo")
    scheduler_name = sd_cfg.get("scheduler", "default")
    guidance_scale = sd_cfg.get("guidance_scale", 0.0)
    num_steps = sd_cfg.get("num_inference_steps", 4)
    num_images = sd_cfg.get("num_images_per_prompt", 1)
    width = sd_cfg.get("width", 1024)
    height = sd_cfg.get("height", 1024)
    seed = sd_cfg.get("seed", cfg.get("seed", 42))
    dtype_str = sd_cfg.get("dtype", "bfloat16")
    negative_prompt = sd_cfg.get("negative_prompt", None)
    output_dir = (
        Path(sd_cfg.get("output_dir", cfg.get("output_dir", "outputs")))
        / "sd_generations"
    )

    prompt_source = prompt_override or sd_cfg.get(
        "prompt", "a photograph of an object on a table"
    )
    prompts = _load_prompts(prompt_source)

    dtype = DTYPE_MAP.get(dtype_str, torch.bfloat16)

    print(f"Loading pipeline: {model_id} ({dtype_str})")
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    if scheduler_name != "default" and scheduler_name in SCHEDULER_MAP:
        sched_cls = SCHEDULER_MAP[scheduler_name]
        pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)
        print(f"Scheduler overridden to: {scheduler_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator("cpu").manual_seed(seed)

    all_metadata = []
    img_counter = 0

    print(f"Generating {len(prompts)} prompt(s) x {num_images} image(s) each")
    print(f"  Steps: {num_steps} | CFG: {guidance_scale} | Size: {width}x{height}")

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[{prompt_idx + 1}/{len(prompts)}] {prompt[:80]}...")
        t0 = time.time()

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            generator=generator,
        )

        elapsed = time.time() - t0

        for i, image in enumerate(result.images):
            img_counter += 1
            fname = f"sd_{img_counter:05d}.png"
            image.save(output_dir / fname)

            meta = {
                "file_name": fname,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_id": model_id,
                "scheduler": scheduler_name,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_steps,
                "width": width,
                "height": height,
                "seed": seed,
                "generation_time_s": round(elapsed / num_images, 2),
            }
            all_metadata.append(meta)

    meta_path = output_dir / "generation_metadata.json"
    meta_path.write_text(json.dumps(all_metadata, indent=2))

    print(f"\nGenerated {img_counter} images → {output_dir}")
    print(f"Metadata saved → {meta_path}")
    return output_dir
