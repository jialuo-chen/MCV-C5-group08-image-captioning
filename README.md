# C5 — Image Captioning on VizWiz

Image captioning framework built for the VizWiz dataset. Train, evaluate, and compare different encoder–decoder architectures for generating image descriptions, all driven by YAML config files and a single CLI.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package & project manager)
- CUDA-capable GPU (recommended)

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd c5_image_caption

# Install all dependencies
uv sync
```

That's it — `uv sync` reads `pyproject.toml` and installs everything into a local `.venv`.

## Project Structure

```
c5_image_caption/
├── main.py                  # CLI entry point (train / evaluate / infer / sweep / visualize)
├── pyproject.toml           # Dependencies and project metadata
├── configs/                 # YAML configuration files for each experiment
│   ├── baseline.yaml            # Baseline: ResNet-18 + GRU + character-level
│   ├── baseline_lstm.yaml       # Decoder swap: LSTM instead of GRU
│   ├── baseline_subword.yaml    # Text swap: BPE subword tokenizer
│   ├── baseline_word.yaml       # Text swap: word-level tokenizer
│   ├── baseline_attention.yaml  # Adds Bahdanau attention to baseline
│   ├── resnet50_gru.yaml        # Encoder swap: ResNet-50
│   ├── resnet50_xlstm.yaml      # ResNet-50 + xLSTM (HuggingFace pretrained)
│   └── sweep.yaml               # WandB hyperparameter sweep definition
├── src/
│   ├── data/
│   │   ├── vizwiz.py            # VizWiz API for loading annotations
│   │   ├── dataset.py           # PyTorch Dataset & data splitting logic
│   │   └── tokenizer.py         # Character / word / subword tokenizers
│   ├── models/
│   │   ├── encoders.py          # Image encoders (ResNet-18/34/50, VGG-16/19)
│   │   ├── decoders.py          # RNN decoders (GRU/LSTM) & HuggingFace LM decoder
│   │   ├── attention.py         # Bahdanau & Luong attention modules
│   │   └── captioner.py         # Puts encoder + decoder together into one model
│   ├── evaluation/
│   │   └── metrics.py           # BLEU-1, BLEU-2, ROUGE-L, METEOR
│   ├── utils/
│   │   ├── config.py            # YAML config loader with defaults & CLI overrides
│   │   └── logger.py            # Experiment logger (params, FLOPs, epoch stats)
│   ├── train.py                 # Training loop (loss, checkpoints, WandB logging)
│   ├── evaluate.py              # Run evaluation on the test set
│   ├── infer.py                 # Generate captions for individual images
│   └── visualize.py             # Generate captioning visualization plots
├── notebooks/                   # Reference notebooks (baseline model, demo eval)
├── outputs/                     # Training outputs: checkpoints & results (gitignored)
└── .gitignore
```

## How It Works

### Configs

Every experiment is defined by a YAML file in `configs/`. A config specifies:

- **Encoder** — which image backbone to use (e.g. ResNet-18, ResNet-50)
- **Decoder** — which text generator to use (GRU, LSTM, or a pretrained HuggingFace LM like xLSTM)
- **Tokenizer** — how to represent text (character-level, subword BPE, or word-level)
- **Attention** — optionally enable Bahdanau or Luong attention
- **Training** — learning rate, batch size, epochs, optimizer, grad clipping, etc.
- **WandB** — enable logging to Weights & Biases

To try a different model combination, just pick a different config or create a new one.

### Data Splits

- **Training data** — 90% of the VizWiz training set
- **Validation data** — 10% of the VizWiz training set (for monitoring during training)
- **Test data** — the full VizWiz validation set (used for final evaluation / inference)

### Metrics

All models are evaluated using: **BLEU-1**, **BLEU-2**, **ROUGE-L**, and **METEOR**.

## CLI Usage

### Train a model

```bash
uv run python main.py train --config configs/baseline.yaml
```

Override any config value from the command line:

```bash
uv run python main.py train --config configs/baseline.yaml \
    --override training.lr=0.0001 training.epochs=10 training.batch_size=32
```

Checkpoints are saved to `outputs/<run_name>/checkpoints/` (both `best.pt` and `last.pt`).

### Evaluate a trained model

```bash
uv run python main.py evaluate \
    --config configs/baseline.yaml \
    --checkpoint outputs/<run_name>/checkpoints/best.pt
```

Prints BLEU-1, BLEU-2, ROUGE-L, METEOR scores and saves a results JSON.

### Generate captions (inference)

For a single image:

```bash
uv run python main.py infer \
    --config configs/baseline.yaml \
    --checkpoint outputs/<run_name>/checkpoints/best.pt \
    --image /path/to/image.jpg
```

For a whole folder of images:

```bash
uv run python main.py infer \
    --config configs/baseline.yaml \
    --checkpoint outputs/<run_name>/checkpoints/best.pt \
    --image /path/to/images/ \
    --output results.json
```

### Run a WandB hyperparameter sweep

```bash
uv run python main.py sweep --config configs/sweep.yaml
```

The sweep config (`configs/sweep.yaml`) defines which parameters to search and references a base experiment config.

### Visualize model predictions

Generate side-by-side plots of ground-truth vs predicted captions on test images:

```bash
uv run python main.py visualize \
    --config configs/baseline.yaml \
    --checkpoint outputs/<run_name>/checkpoints/best.pt
```

Customize the number of samples and add a model label:

```bash
uv run python main.py visualize \
    --config configs/baseline.yaml \
    --checkpoint outputs/<run_name>/checkpoints/best.pt \
    --num-images 10 \
    --model-type "Baseline" \
    --output outputs/plots/
```

Saves a grid image plus individual per-image plots.

## Experiment Logger

Every training run automatically produces an `experiment_log.json` inside the output directory with:

- **Model info** — total / trainable parameters (broken down by encoder and decoder), estimated FLOPs
- **Hyperparameters** — full config snapshot (encoder, decoder, attention, tokenizer, training)
- **Per-epoch stats** — train loss, val loss, BLEU-1, BLEU-2, ROUGE-L, METEOR, learning rate, epoch wall-clock time
- **Training summary** — best epoch, best metrics, total training time, min losses, average epoch time

Inference runs also log checkpoint path, number of images, total time, and average latency.

## Available Configs

| Config | Encoder | Decoder | Tokenizer | Attention | Purpose |
|--------|---------|---------|-----------|-----------|---------|
| `baseline.yaml` | ResNet-18 | GRU | char | — | Baseline model |
| `baseline_lstm.yaml` | ResNet-18 | LSTM | char | — | Decoder change |
| `baseline_subword.yaml` | ResNet-18 | GRU | subword (BPE) | — | Text representation change |
| `baseline_word.yaml` | ResNet-18 | GRU | word | — | Text representation change |
| `baseline_attention.yaml` | ResNet-18 | GRU | char | Bahdanau | + Attention mechanism |
| `resnet50_gru.yaml` | ResNet-50 | GRU | char | — | Encoder change |
| `resnet50_xlstm.yaml` | ResNet-50 | xLSTM (HF) | model's own | — | Pretrained LM decoder |

## WandB Integration

To enable Weights & Biases logging for any training run, either:

1. Set `wandb.enabled: true` in your config YAML, or
2. Override from CLI: `--override wandb.enabled=true`

Make sure you're logged in (`wandb login`) before running.
