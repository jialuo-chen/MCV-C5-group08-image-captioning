# Tasks A & B — Experiment Plan

This document describes how we plan the Stable Diffusion exploration experiments for Task A (model zoo) and Task B (inference parameters).

## Goal

Run a small, focused set of image generations that let us draw clear conclusions and fit on a few slides. We avoid full grid-search because it produces too many images to review and compare.

## Guiding principles

1. **One axis at a time.** When we study a parameter, we fix all other parameters to a sensible default. This makes every difference between images attributable to the axis we are varying.
2. **Same prompts across sweeps.** We use the same 2 reference prompts in every sweep, so slides are directly comparable side by side.
3. **Same seed inside a sweep.** Different seeds would hide the effect of the parameter under noise.
4. **Small, readable outputs.** Each sweep produces a labeled image grid that can go directly into a slide.

## Reference configuration

All sweeps (except the one that varies it) use this fixed point:

| Parameter | Default |
|---|---|
| Model | `stabilityai/stable-diffusion-xl-base-1.0` |
| Scheduler | DPM++ (`DPMSolverMultistep`) |
| Guidance scale (CFG) | 7.5 |
| Inference steps | 25 |
| Negative prompt | empty |
| Seed | fixed (same value for the whole sweep) |
| Image size | 1024×1024 |

## Reference prompts

Two prompts used in every sweep:

- **P1 (simple object):** `"a red apple on a wooden table, studio lighting"`
- **P2 (hard / compositional):** `"a person holding a sign that reads 'HELLO', realistic photo"`

P1 is easy and lets us see baseline quality. P2 stresses text rendering and composition, where parameter differences are more visible.

## Task A — Model sweep

We compare three models that cover the design space, not the full zoo:

| Model | Role |
|---|---|
| `sd2-community/stable-diffusion-2-1` | classic baseline (community mirror of SD 2.1) |
| `stabilityai/stable-diffusion-xl-base-1.0` | larger, higher quality |
| `stabilityai/sdxl-turbo` | distilled / fast |

Note: the original `stabilityai/stable-diffusion-2-1` repository is no longer publicly available on HuggingFace, so we use the community mirror which hosts the same weights.

For each model we generate P1 and P2 with the model's recommended inference settings:

- SD 2.1 and SDXL base → scheduler DPM++, CFG 7.5, 25 steps.
- SDXL Turbo → CFG 0.0, 4 steps (the turbo model is distilled to these values; using CFG or more steps is not meaningful).

Output: 3 models × 2 prompts = **6 images**, arranged in a 3×2 grid for one slide.

## Task B — Parameter sweeps

All Task B sweeps run on **SDXL base** only. Running them on a turbo model is not useful because turbo models ignore CFG and are fixed to ~4 steps.

### B1 — Scheduler

Values: `DDPM`, `DDIM`, `DPM++`.
Everything else at the reference configuration.
Output: 3 × 2 prompts = **6 images**.

### B2 — Guidance scale (CFG)

Values: `1.0`, `7.5`, `12.0`.
Chosen to show underguidance (1.0), balanced (7.5), and overguidance artifacts (12.0).
Output: 3 × 2 prompts = **6 images**.

### B3 — Inference steps

Values: `4`, `25`, `50`.
Chosen to show the underconverged regime (4), the converged regime (25), and the point of diminishing returns (50).
Output: 3 × 2 prompts = **6 images**.

### B4 — Negative prompt

Two runs on a single prompt that tends to fail without a negative prompt (for example P2, which often renders broken text and malformed hands):

- Without negative prompt (empty).
- With negative prompt: `"blurry, low quality, extra fingers, deformed hands, unreadable text, watermark"`.

Output: 2 × 1 prompt = **2 images**, shown side by side.

## Totals

| Sweep | Images |
|---|---|
| A — models | 6 |
| B1 — scheduler | 6 |
| B2 — CFG | 6 |
| B3 — steps | 6 |
| B4 — negative prompt | 2 |
| **Total** | **26** |

Every B sweep is now 3 values × 2 prompts, so all slides share the same 3×2 grid layout.

At a few seconds per image on a modern GPU, the full experiment completes in roughly 10–15 minutes of generation time, plus model loading.

## Deliverable per sweep

For each sweep we produce:

1. The raw images saved to disk, named with their parameter value (for example `b2_cfg_7.5_p1.png`).
2. A composed grid image (with parameter labels) ready to drop into a slide.
3. A one-line qualitative note of what we observe.

## What this plan does *not* cover

- Full combinatorial grids (too many images, unreadable).
- Seed variability studies (orthogonal to A/B and not required for grade C).
- Turbo models on parameter sweeps (CFG/steps are effectively fixed for turbo).
- Quantitative metrics such as CLIP score (not required for grade C; qualitative comparison is enough).

## Implementation

Three files implement the plan:

- [configs/sd_sweeps.yaml](../configs/sd_sweeps.yaml) — declarative spec of every sweep (prompts, reference point, variants per axis).
- [src/run_sd_sweeps.py](../src/run_sd_sweeps.py) — runner that groups generations by `model_id`, loads each pipeline once, swaps scheduler in place, generates with a fixed seed, saves individual images and a labeled grid per sweep.
- `main.py` — exposes `run-sd-sweeps` as a CLI subcommand.

### How to run

```bash
uv run python main.py run-sd-sweeps --config configs/sd_sweeps.yaml
```

### Outputs

```
outputs/sd_sweeps/
├── images/          # raw images, named <sweep>__<variant>__<prompt>.png
├── grids/           # one composed PNG per sweep (3×2 with labels) — drop into slides
└── sweep_results.json
```

### Key implementation choices

- **Pipelines loaded once per model.** All variants that share a `model_id` run back-to-back so we pay the load cost only once per model.
- **Scheduler swap is in-place.** No VRAM cost when changing scheduler between variants.
- **Same seed per sweep.** `torch.Generator("cpu").manual_seed(seed)` is reset before every generation so the seed is identical across variants of a sweep — differences are attributable to the axis.
- **Reference defaults.** The `reference` block in YAML fills any parameter not specified by a variant. Adding a new sweep only requires listing the values that actually change.
- **Grids are the deliverable.** Each sweep's 3×2 grid PNG goes directly into the slide for that sweep.

### Overriding from CLI

Any YAML field can be overridden at the command line, e.g. to do a dry run with reduced resolution:

```bash
uv run python main.py run-sd-sweeps --config configs/sd_sweeps.yaml \
    --override width=512 height=512
```
