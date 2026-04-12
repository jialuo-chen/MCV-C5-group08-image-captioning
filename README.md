# C5 Image Captioning (VizWiz)

A modular image-captioning project with multiple workflows:
- classic encoder-decoder training
- VisionEncoderDecoder fine-tuning
- multimodal VLM evaluation (Qwen)
- ViT + Qwen LoRA fine-tuning and evaluation
- Optuna sweep and plot generation

## Reviewer Quick Start

If you only have a few minutes, start here:

1. Read `main.py` to understand all supported commands and entrypoints.
2. Read `src/utils/config.py` to see how YAML configs are loaded and overridden.
3. Read the specific pipeline module:
   - `src/train.py` and `src/evaluate.py` for classic captioning pipeline.
   - `src/train_vit_decoder.py` and `src/evaluate_pretrained.py` for VisionEncoderDecoder pipeline.
   - `src/train_lora.py` and `src/evaluate_lora.py` for LoRA pipeline.
   - `src/evaluate_multimodal.py` and `src/models/qwen_vlm.py` for direct multimodal evaluation.
4. Inspect example configs in `configs/` matching the pipeline you are reviewing.

## Setup

Requirements:
- Python 3.12+
- uv
- CUDA GPU recommended

Install dependencies:

uv sync

Run commands through uv so the local environment is used:

uv run python main.py --help

## Main Commands

Training and evaluation:
- `uv run python main.py train --config <config.yaml>`
- `uv run python main.py evaluate --config <config.yaml> --checkpoint <path>`
- `uv run python main.py infer --config <config.yaml> --checkpoint <path> --image <img_or_dir>`
- `uv run python main.py visualize --config <config.yaml> --checkpoint <path>`

Pretrained and multimodal:
- `uv run python main.py finetune --config configs/vit_gpt2.yaml`
- `uv run python main.py evaluate-pretrained --config configs/eval_pretrained.yaml --model nlpconnect/vit-gpt2-image-captioning`
- `uv run python main.py evaluate-multimodal --config configs/eval_qwen_multimodal.yaml --model Qwen/Qwen3.5-0.8B`

LoRA:
- `uv run python main.py finetune-lora --config configs/lora_qwen_0.8b.yaml`
- `uv run python main.py evaluate-lora --config configs/lora_qwen_0.8b.yaml --checkpoint outputs/<run>/checkpoints/best`

Optuna and analysis:
- `uv run python main.py optuna-sweep --config configs/optuna_lora_2b.yaml`
- `uv run python main.py optuna-viz --study-dir outputs/optuna_lora_qwen_2b`
- `uv run python main.py quantitative-plots --outputs-dir outputs --out-dir outputs/presentation_plots`

## Minimal Smoke Checks

There is no dedicated unit-test suite in this repository. For quick verification:

1. Syntax/compile check:

uv run python -m compileall main.py src

2. CLI wiring check:

uv run python main.py --help

3. Pipeline command check:

uv run python main.py evaluate-multimodal --help
uv run python main.py evaluate-lora --help

## Project Layout

- `main.py`: single CLI dispatcher
- `configs/`: experiment/eval/sweep config files
- `src/data/`: VizWiz access, datasets, tokenizers
- `src/models/`: captioning models, VLM wrappers, LoRA bridge modules
- `src/train.py`: classic encoder-decoder training
- `src/train_vit_decoder.py`: VisionEncoderDecoder fine-tuning
- `src/train_lora.py`: LoRA fine-tuning for ViT+Qwen
- `src/evaluate.py`, `src/evaluate_pretrained.py`, `src/evaluate_multimodal.py`, `src/evaluate_lora.py`: evaluation entrypoints
- `src/optuna_sweep.py`, `src/optuna_visualize.py`: hyperparameter search and visualization
- `src/generate_*plots.py`, `src/generate_task2_presentation.py`: report/presentation figures
- `outputs/`: run artifacts and evaluation JSON files

## Notes For Reviewers

- Config overrides are supported via `--override key=value` on commands that accept configs.
- Most workflows write outputs under `outputs/` unless an explicit output directory is passed.
- Some workflows assume GPU availability for practical runtime.
