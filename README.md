# C5 Generative Vision Workbench

This repository has two major tracks:

1. Diffusion generation and synthetic data creation.
2. Image captioning training and evaluation on VizWiz.

## Setup

Requirements:
- Python 3.12+
- uv
- CUDA GPU recommended

Install dependencies:

uv sync

Quick sanity check:

uv run python main.py --help

## Reviewer Quick Start

Start with diffusion/generation first, then captioning.

### A) Diffusion Models and Generation (start here)

Code entry path:
1. `main.py` (commands: `generate-sd`, `generate-synthetic`, `run-sd-sweeps`)
2. `src/generate_sd.py`
3. `src/generate_synthetic_data.py`
4. `src/run_sd_sweeps.py`
5. `configs/` files for SD workflows (for example `sd_*.yaml`)

Fast test path:
1. `uv run python main.py generate-sd --help`
2. `uv run python main.py generate-synthetic --help`
3. `uv run python main.py run-sd-sweeps --help`

Example commands:
- `uv run python main.py generate-sd --config configs/sd_inference.yaml`
- `uv run python main.py generate-synthetic --config configs/sd_synthetic_data.yaml`
- `uv run python main.py run-sd-sweeps --config configs/sd_sweeps.yaml`

### B) Image Captioning

Code entry path:
1. `main.py` (captioning commands)
2. `src/utils/config.py`
3. Training/eval modules by pipeline:
   - classic: `src/train.py`, `src/evaluate.py`
   - VisionEncoderDecoder: `src/train_vit_decoder.py`, `src/evaluate_pretrained.py`
   - LoRA: `src/train_lora.py`, `src/evaluate_lora.py`
   - multimodal eval: `src/evaluate_multimodal.py`, `src/models/qwen_vlm.py`

Fast test path:
1. `uv run python main.py evaluate-multimodal --help`
2. `uv run python main.py evaluate-lora --help`
3. `uv run python main.py train --help`

Example commands:
- `uv run python main.py train --config <config.yaml>`
- `uv run python main.py evaluate --config <config.yaml> --checkpoint <path>`
- `uv run python main.py infer --config <config.yaml> --checkpoint <path> --image <img_or_dir>`
- `uv run python main.py finetune --config configs/vit_gpt2.yaml`
- `uv run python main.py evaluate-pretrained --config configs/eval_pretrained.yaml --model nlpconnect/vit-gpt2-image-captioning`
- `uv run python main.py finetune-lora --config configs/lora_qwen_0.8b.yaml`
- `uv run python main.py evaluate-lora --config configs/lora_qwen_0.8b.yaml --checkpoint outputs/<run>/checkpoints/best`

## Minimal Smoke Checks

There is no dedicated unit-test suite in this repository.

1. Syntax check:

uv run python -m compileall main.py src

2. CLI wiring:

uv run python main.py --help

3. Diffusion CLI checks:

uv run python main.py generate-sd --help
uv run python main.py run-sd-sweeps --help

4. Captioning CLI checks:

uv run python main.py evaluate-multimodal --help
uv run python main.py evaluate-lora --help

## Project Layout

- `main.py`: single CLI dispatcher
- `configs/`: all config files (diffusion, training, eval, sweeps)
- `src/generate_sd.py`, `src/generate_synthetic_data.py`, `src/run_sd_sweeps.py`: diffusion/synthetic pipeline
- `src/data/`: VizWiz access, dataset classes, tokenizers
- `src/models/`: captioning and multimodal model modules
- `src/train.py`, `src/train_vit_decoder.py`, `src/train_lora.py`: training entrypoints
- `src/evaluate.py`, `src/evaluate_pretrained.py`, `src/evaluate_multimodal.py`, `src/evaluate_lora.py`: evaluation entrypoints
- `src/optuna_sweep.py`, `src/optuna_visualize.py`: hyperparameter search and plots
- `src/generate_*plots.py`, `src/generate_task2_presentation.py`, `src/generate_task_de_plots.py`: reporting plots
- `outputs/`: generated images, checkpoints, metrics, and artifacts

## Notes

- Commands supporting configs accept `--override key=value`.
- Most runs write to `outputs/` unless `output_dir` is overridden.
- GPU is recommended for both diffusion and captioning workflows.
