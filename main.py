"""CLI entry point for the image captioning framework.

Subcommands
-----------
train     Train a model from a YAML config.
evaluate  Evaluate a checkpoint on the test set.
infer     Generate captions for images using a trained model.
sweep     Run a WandB hyperparameter sweep.

Usage
-----
    uv run python main.py train --config configs/baseline.yaml
    uv run python main.py evaluate --config configs/baseline.yaml --checkpoint outputs/.../best.pt
    uv run python main.py infer --config configs/baseline.yaml --checkpoint outputs/.../best.pt --image path/to/image.jpg
    uv run python main.py sweep --config configs/sweep.yaml
"""

from __future__ import annotations

import argparse
import sys


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides as key=value pairs (e.g., training.lr=0.001).",
    )


def cmd_train(args: argparse.Namespace) -> None:
    from src.utils.config import load_config
    from src.train import train

    cfg = load_config(args.config, overrides=args.override)
    train(cfg)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.utils.config import load_config
    from src.evaluate import evaluate

    cfg = load_config(args.config, overrides=args.override)
    evaluate(cfg, checkpoint_path=args.checkpoint)


def cmd_infer(args: argparse.Namespace) -> None:
    from src.utils.config import load_config
    from src.infer import infer, collect_image_paths

    cfg = load_config(args.config, overrides=args.override)
    image_paths = collect_image_paths(args.image)
    infer(cfg, checkpoint_path=args.checkpoint, image_paths=image_paths, output_file=args.output)


def cmd_sweep(args: argparse.Namespace) -> None:
    """Launch a WandB sweep from a sweep YAML config."""
    import yaml
    import wandb
    from src.utils.config import load_config
    from src.train import train

    with open(args.config) as f:
        sweep_cfg = yaml.safe_load(f)

    # The sweep config should have a "base_config" field pointing to the experiment config
    base_config_path = sweep_cfg.pop("base_config")
    sweep_id = wandb.sweep(sweep_cfg, project=sweep_cfg.get("project", "c5-image-caption"))

    def sweep_agent():
        wandb.init()
        # Merge wandb sweep params as overrides
        overrides = [f"{k}={v}" for k, v in wandb.config.items()]
        cfg = load_config(base_config_path, overrides=overrides)
        cfg.wandb.enabled = True
        train(cfg)

    wandb.agent(sweep_id, function=sweep_agent, count=sweep_cfg.get("count", 10))


def cmd_visualize(args: argparse.Namespace) -> None:
    from src.utils.config import load_config
    from src.visualize import visualize

    cfg = load_config(args.config, overrides=args.override)
    visualize(
        cfg,
        checkpoint_path=args.checkpoint,
        num_images=args.num_images,
        output_dir=args.output,
        model_type=args.model_type,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="c5-caption",
        description="Image Captioning Framework — VizWiz",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train a model.")
    _add_common_args(p_train)

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a checkpoint.")
    _add_common_args(p_eval)
    p_eval.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")

    # --- infer ---
    p_infer = subparsers.add_parser("infer", help="Run inference on images.")
    _add_common_args(p_infer)
    p_infer.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    p_infer.add_argument("--image", type=str, required=True, help="Image path or directory.")
    p_infer.add_argument("--output", type=str, default=None, help="Output JSON file.")

    # --- sweep ---
    p_sweep = subparsers.add_parser("sweep", help="Run a WandB hyperparameter sweep.")
    p_sweep.add_argument("--config", type=str, required=True, help="Path to sweep YAML config.")

    # --- visualize ---
    p_vis = subparsers.add_parser("visualize", help="Generate caption visualizations.")
    _add_common_args(p_vis)
    p_vis.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    p_vis.add_argument("--num-images", type=int, default=5, help="Number of images to visualize (default: 5).")
    p_vis.add_argument("--output", type=str, default=None, help="Output directory for plots.")
    p_vis.add_argument("--model-type", type=str, default=None, help="Model label for plot title.")

    args = parser.parse_args()

    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "infer": cmd_infer,
        "sweep": cmd_sweep,
        "visualize": cmd_visualize,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
