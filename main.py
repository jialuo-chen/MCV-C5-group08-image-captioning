from __future__ import annotations

import argparse

import wandb
import yaml

from src.evaluate import evaluate
from src.evaluate_multimodal import evaluate_multimodal
from src.evaluate_pretrained import evaluate_pretrained
from src.generate_presentation_plots import generate_all_plots
from src.infer import collect_image_paths, infer
from src.optuna_sweep import run_optuna_sweep
from src.optuna_visualize import load_and_visualize
from src.train import train
from src.train_lora import train_lora
from src.train_vit_decoder import train_vit_decoder
from src.utils.config import load_config
from src.visualize import visualize


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file."
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides as key=value pairs (e.g., training.lr=0.001).",
    )


def cmd_train(args: argparse.Namespace) -> None:

    cfg = load_config(args.config, overrides=args.override)
    train(cfg)


def cmd_evaluate(args: argparse.Namespace) -> None:

    cfg = load_config(args.config, overrides=args.override)
    evaluate(cfg, checkpoint_path=args.checkpoint)


def cmd_infer(args: argparse.Namespace) -> None:

    cfg = load_config(args.config, overrides=args.override)
    image_paths = collect_image_paths(args.image)
    infer(
        cfg,
        checkpoint_path=args.checkpoint,
        image_paths=image_paths,
        output_file=args.output,
    )


def cmd_sweep(args: argparse.Namespace) -> None:
    """Launch a WandB sweep from a sweep YAML config."""

    with open(args.config) as f:
        sweep_cfg = yaml.safe_load(f)

    base_config_path = sweep_cfg.pop("base_config")
    sweep_id = wandb.sweep(
        sweep_cfg, project=sweep_cfg.get("project", "c5-image-caption")
    )

    def sweep_agent():
        wandb.init()
        overrides = [f"{k}={v}" for k, v in wandb.config.items()]
        cfg = load_config(base_config_path, overrides=overrides)
        cfg.wandb.enabled = True
        train(cfg)

    wandb.agent(sweep_id, function=sweep_agent, count=sweep_cfg.get("count", 10))


def cmd_optuna_sweep(args: argparse.Namespace) -> None:
    run_optuna_sweep(args.config)


def cmd_optuna_viz(args: argparse.Namespace) -> None:
    load_and_visualize(args.study_dir)


def cmd_quantitative_plots(args: argparse.Namespace) -> None:
    generate_all_plots(args.outputs_dir, args.out_dir)


def cmd_visualize(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, overrides=args.override)
    visualize(
        cfg,
        checkpoint_path=args.checkpoint,
        num_images=args.num_images,
        output_dir=args.output,
        model_type=args.model_type,
    )


def cmd_evaluate_pretrained(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, overrides=args.override)
    evaluate_pretrained(
        cfg,
        model_name=args.model,
        checkpoint_path=getattr(args, "checkpoint", None),
        output_dir=args.output_dir,
    )


def cmd_finetune(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, overrides=args.override)
    train_vit_decoder(cfg)


def cmd_evaluate_multimodal(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, overrides=args.override)
    evaluate_multimodal(
        cfg,
        model_name=args.model,
        output_dir=args.output_dir,
        prompt=getattr(args, "prompt", None),
    )


def cmd_finetune_lora(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, overrides=args.override)
    train_lora(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="c5-caption",
        description="Image Captioning Framework — VizWiz",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train a model.")
    _add_common_args(p_train)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate a checkpoint.")
    _add_common_args(p_eval)
    p_eval.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint."
    )

    p_infer = subparsers.add_parser("infer", help="Run inference on images.")
    _add_common_args(p_infer)
    p_infer.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint."
    )
    p_infer.add_argument(
        "--image", type=str, required=True, help="Image path or directory."
    )
    p_infer.add_argument("--output", type=str, default=None, help="Output JSON file.")

    p_sweep = subparsers.add_parser("sweep", help="Run a WandB hyperparameter sweep.")
    p_sweep.add_argument(
        "--config", type=str, required=True, help="Path to sweep YAML config."
    )

    p_optuna = subparsers.add_parser(
        "optuna-sweep", help="Run an Optuna hyperparameter sweep."
    )
    p_optuna.add_argument(
        "--config", type=str, required=True, help="Path to Optuna sweep YAML config."
    )

    p_optuna_viz = subparsers.add_parser(
        "optuna-viz", help="Generate visualizations from a completed Optuna sweep."
    )
    p_optuna_viz.add_argument(
        "--study-dir",
        type=str,
        required=True,
        help="Path to Optuna output directory containing study.pkl.",
    )

    p_vis = subparsers.add_parser("visualize", help="Generate caption visualizations.")
    _add_common_args(p_vis)
    p_vis.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint."
    )
    p_vis.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of images to visualize (default: 5).",
    )
    p_vis.add_argument(
        "--output", type=str, default=None, help="Output directory for plots."
    )
    p_vis.add_argument(
        "--model-type", type=str, default=None, help="Model label for plot title."
    )

    quant_plots = subparsers.add_parser(
        "quantitative-plots",
        help="Generate quantitative result plots for the presentation.",
    )
    quant_plots.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Directory containing experiment outputs (default: outputs).",
    )
    quant_plots.add_argument(
        "--out-dir",
        type=str,
        default="outputs/presentation_plots",
        help="Output directory for plots (default: outputs/presentation_plots).",
    )

    p_eval_pt = subparsers.add_parser(
        "evaluate-pretrained",
        help="Evaluate a pretrained HF captioning model on VizWiz.",
    )
    _add_common_args(p_eval_pt)
    p_eval_pt.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model id (e.g. nlpconnect/vit-gpt2-image-captioning).",
    )
    p_eval_pt.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a fine-tuned VisionEncoderDecoderModel checkpoint directory.",
    )
    p_eval_pt.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results."
    )

    p_finetune = subparsers.add_parser(
        "finetune",
        help="Fine-tune a VisionEncoderDecoderModel (ViT/CLIP encoder + GPT2/T5/SmolLM decoder).",
    )
    _add_common_args(p_finetune)

    p_eval_mm = subparsers.add_parser(
        "evaluate-multimodal",
        help="Evaluate a multimodal VLM (e.g. Qwen) on VizWiz.",
    )
    _add_common_args(p_eval_mm)
    p_eval_mm.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace multimodal model id (e.g. Qwen/Qwen3.5-9B).",
    )
    p_eval_mm.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results."
    )
    p_eval_mm.add_argument(
        "--prompt",
        type=str,
        default=None,
        help='Text prompt for the model (default: "Describe this image briefly.").',
    )

    p_lora = subparsers.add_parser(
        "finetune-lora",
        help="Fine-tune a Qwen decoder with LoRA using a frozen ViT encoder.",
    )
    _add_common_args(p_lora)

    args = parser.parse_args()

    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "infer": cmd_infer,
        "sweep": cmd_sweep,
        "optuna-sweep": cmd_optuna_sweep,
        "optuna-viz": cmd_optuna_viz,
        "visualize": cmd_visualize,
        "quantitative-plots": cmd_quantitative_plots,
        "evaluate-pretrained": cmd_evaluate_pretrained,
        "finetune": cmd_finetune,
        "evaluate-multimodal": cmd_evaluate_multimodal,
        "finetune-lora": cmd_finetune_lora,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
