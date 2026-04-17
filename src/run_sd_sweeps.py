"""Run the Stable Diffusion experiment sweeps described in docs/task_ab_experiment.md.

Loads each model pipeline only once (expensive) and iterates the axes it is
assigned to. For every sweep it saves the individual images plus a composed
grid (one PNG per sweep) ready to drop into slides.

Usage:
    uv run python main.py run-sd-sweeps --config configs/sd_sweeps.yaml
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image, ImageDraw, ImageFont

from src.generate_sd import DTYPE_MAP, SCHEDULER_MAP
from src.utils.config import Config


def _resolve_variant(sweep: dict, variant: dict, reference: dict) -> dict:
    """Merge reference defaults + sweep-level + variant-level into one settings dict."""
    merged = dict(reference)
    for key in ("model_id", "scheduler", "guidance_scale", "num_inference_steps", "negative_prompt"):
        if key in sweep:
            merged[key] = sweep[key]
        if key in variant:
            merged[key] = variant[key]
    merged["label"] = variant["label"]
    return merged


def _collect_tasks(cfg: Config) -> list[dict]:
    """Flatten all (sweep, variant, prompt) triples into a single task list."""
    reference = cfg.get("reference", {})
    prompts_map = cfg.get("prompts", {})
    tasks = []
    for sweep in cfg.get("sweeps", []):
        prompt_keys = sweep.get("prompts", [])
        for variant in sweep.get("variants", []):
            settings = _resolve_variant(sweep, variant, reference)
            for pk in prompt_keys:
                p = prompts_map[pk]
                tasks.append(
                    {
                        "sweep": sweep["name"],
                        "axis": sweep.get("axis", ""),
                        "variant_label": settings["label"],
                        "prompt_key": pk,
                        "prompt_label": p["label"],
                        "prompt_text": p["text"],
                        "model_id": settings["model_id"],
                        "scheduler": settings.get("scheduler", "dpm++"),
                        "guidance_scale": float(settings.get("guidance_scale", 7.5)),
                        "num_inference_steps": int(settings.get("num_inference_steps", 25)),
                        "negative_prompt": settings.get("negative_prompt", "") or None,
                    }
                )
    return tasks


def _load_pipeline(model_id: str, dtype: torch.dtype, cpu_offload: bool) -> "AutoPipelineForText2Image":
    print(f"\n=== Loading pipeline: {model_id} ===")
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    return pipe


def _apply_scheduler(pipe, scheduler_name: str) -> None:
    if scheduler_name and scheduler_name in SCHEDULER_MAP:
        sched_cls = SCHEDULER_MAP[scheduler_name]
        pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)


def _build_grid(
    images_by_cell: dict[tuple[int, int], Image.Image],
    col_labels: list[str],
    row_labels: list[str],
    title: str,
    cell_size: int = 512,
) -> Image.Image:
    """Compose a labeled grid. images_by_cell is keyed by (row, col)."""
    n_rows = len(row_labels)
    n_cols = len(col_labels)
    margin_top = 80
    margin_left = 140
    label_h = 40
    W = margin_left + n_cols * cell_size
    H = margin_top + n_rows * (cell_size + label_h)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font_title = ImageFont.load_default()
        font = ImageFont.load_default()

    draw.text((20, 20), title, fill="black", font=font_title)

    for c, cl in enumerate(col_labels):
        x = margin_left + c * cell_size + cell_size // 2
        draw.text((x, margin_top - 30), cl, fill="black", font=font, anchor="mm")

    for r, rl in enumerate(row_labels):
        y = margin_top + r * (cell_size + label_h) + cell_size // 2
        draw.text((margin_left // 2, y), rl, fill="black", font=font, anchor="mm")

    for (r, c), img in images_by_cell.items():
        thumb = img.resize((cell_size, cell_size), Image.LANCZOS)
        x = margin_left + c * cell_size
        y = margin_top + r * (cell_size + label_h)
        canvas.paste(thumb, (x, y))

    return canvas


def run_sd_sweeps(cfg: Config) -> Path:
    """Entry point: execute every sweep and write outputs."""
    seed = int(cfg.get("seed", 42))
    dtype = DTYPE_MAP.get(cfg.get("dtype", "bfloat16"), torch.bfloat16)
    cpu_offload = bool(cfg.get("cpu_offload", False))
    width = int(cfg.get("width", 1024))
    height = int(cfg.get("height", 1024))

    out_root = Path(cfg.get("output_dir", "outputs")) / cfg.get("run_name", "sd_sweeps")
    img_dir = out_root / "images"
    grid_dir = out_root / "grids"
    img_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    tasks = _collect_tasks(cfg)
    print(f"Total generations: {len(tasks)}")

    # Group by model_id so we load each pipeline just once.
    by_model: dict[str, list[dict]] = {}
    for t in tasks:
        by_model.setdefault(t["model_id"], []).append(t)

    results: list[dict] = []
    t_start = time.time()

    for model_id, model_tasks in by_model.items():
        pipe = _load_pipeline(model_id, dtype, cpu_offload)
        last_scheduler = None

        for t in model_tasks:
            if t["scheduler"] != last_scheduler:
                _apply_scheduler(pipe, t["scheduler"])
                last_scheduler = t["scheduler"]

            generator = torch.Generator("cpu").manual_seed(seed)
            print(
                f"[{t['sweep']}] {t['variant_label']} | {t['prompt_label']} "
                f"| sched={t['scheduler']} cfg={t['guidance_scale']} "
                f"steps={t['num_inference_steps']}"
            )
            t0 = time.time()
            result = pipe(
                prompt=t["prompt_text"],
                negative_prompt=t["negative_prompt"],
                guidance_scale=t["guidance_scale"],
                num_inference_steps=t["num_inference_steps"],
                num_images_per_prompt=1,
                width=width,
                height=height,
                generator=generator,
            )
            elapsed = time.time() - t0
            image = result.images[0]
            fname = (
                f"{t['sweep']}__{_slug(t['variant_label'])}__{t['prompt_key']}.png"
            )
            image.save(img_dir / fname)
            results.append({**t, "file_name": fname, "seconds": round(elapsed, 2)})

        del pipe
        torch.cuda.empty_cache()

    # Metadata
    (out_root / "sweep_results.json").write_text(json.dumps(results, indent=2))

    # Build one grid per sweep
    sweeps = cfg.get("sweeps", [])
    prompts_map = cfg.get("prompts", {})
    for sweep in sweeps:
        _build_sweep_grid(sweep, prompts_map, results, img_dir, grid_dir)

    total = time.time() - t_start
    print(f"\nDone in {total / 60:.1f} min. Outputs → {out_root}")
    return out_root


def _build_sweep_grid(
    sweep: dict,
    prompts_map: dict,
    results: list[dict],
    img_dir: Path,
    grid_dir: Path,
) -> None:
    sname = sweep["name"]
    variants = sweep["variants"]
    prompt_keys = sweep["prompts"]
    col_labels = [v["label"] for v in variants]
    row_labels = [prompts_map[pk]["label"] for pk in prompt_keys]

    cells: dict[tuple[int, int], Image.Image] = {}
    for r, pk in enumerate(prompt_keys):
        for c, v in enumerate(variants):
            match = next(
                (
                    x for x in results
                    if x["sweep"] == sname
                    and x["variant_label"] == v["label"]
                    and x["prompt_key"] == pk
                ),
                None,
            )
            if match is None:
                continue
            cells[(r, c)] = Image.open(img_dir / match["file_name"]).convert("RGB")

    title = f"{sname}  (axis: {sweep.get('axis', '')})"
    grid = _build_grid(cells, col_labels, row_labels, title)
    grid.save(grid_dir / f"{sname}.png")
    print(f"Grid → {grid_dir / f'{sname}.png'}")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")
