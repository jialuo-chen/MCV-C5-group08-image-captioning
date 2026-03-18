"""Visualization generator.

Loads a trained checkpoint, samples N images from the test set, generates
captions, and produces a grid of plots showing the image together with
ground-truth and predicted captions.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import v2

from src.data.dataset import VizWizCaptionDataset, get_image_transform
from src.data.tokenizer import BaseTokenizer, CharTokenizer, WordTokenizer, SubwordTokenizer
from src.models.captioner import CaptioningModel
from src.models.decoders import HFLMDecoder
from src.utils.config import Config


def _load_tokenizer(ckpt_dir: Path, cfg: Config) -> BaseTokenizer:
    tok_path = ckpt_dir / "tokenizer.json"
    tok_type = cfg.tokenizer.type
    if tok_type == "char":
        return CharTokenizer.load(tok_path) if tok_path.exists() else CharTokenizer()
    elif tok_type == "word":
        return WordTokenizer.load(tok_path)
    elif tok_type == "subword":
        return SubwordTokenizer.load(tok_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tok_type}")


def _display_transform(image_size: int = 224):
    """Transform for display: resize only, no normalization."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((image_size, image_size), antialias=True),
    ])


def _wrap_text(text: str, max_chars: int = 60) -> str:
    """Wrap text so it fits in a plot title."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        if current and len(current) + len(w) + 1 > max_chars:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        lines.append(current)
    return "\n".join(lines)


def visualize(
    cfg: Config,
    checkpoint_path: str,
    num_images: int = 5,
    output_dir: str | None = None,
    model_type: str | None = None,
    seed: int = 42,
) -> Path:
    """Generate visualization plots.

    Parameters
    ----------
    cfg : Config
        Experiment configuration.
    checkpoint_path : str
        Path to trained model checkpoint.
    num_images : int
        Number of test images to visualize.
    output_dir : str | None
        Directory to save plots. Defaults to ``outputs/<run>/visualizations/``.
    model_type : str | None
        Label shown on the plot subtitle (e.g. "Baseline", "ResNet-50 + GRU").
        Defaults to ``<encoder_name> + <decoder_name>``.
    seed : int
        Random seed for image selection.

    Returns
    -------
    Path
        Directory where plots were saved.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint_path)

    # -- Load model ---------------------------------------------------------
    model, checkpoint = CaptioningModel.from_checkpoint(ckpt_path, device=str(device))
    model.eval()
    is_hf_lm = isinstance(model.decoder, HFLMDecoder)
    saved_cfg = Config(checkpoint["config"])

    # -- Tokenizer ----------------------------------------------------------
    if is_hf_lm:
        tokenizer = CharTokenizer()
    else:
        tokenizer = _load_tokenizer(ckpt_path.parent, saved_cfg)

    # -- Test dataset -------------------------------------------------------
    root = Path(cfg.dataset.root)
    test_ds = VizWizCaptionDataset(
        annotation_file=str(root / cfg.dataset.val_ann),
        image_dir=str(root / cfg.dataset.val_img_dir),
        tokenizer=tokenizer,
        max_length=cfg.tokenizer.max_length,
        split="val",
    )
    print(f"Test set: {len(test_ds)} images")

    # -- Sample images ------------------------------------------------------
    rng = random.Random(seed)
    indices = rng.sample(range(len(test_ds)), min(num_images, len(test_ds)))

    # -- Output directory ---------------------------------------------------
    if output_dir:
        vis_dir = Path(output_dir)
    else:
        vis_dir = ckpt_path.parent.parent / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # -- Model type label ---------------------------------------------------
    if not model_type:
        enc = saved_cfg.encoder.name
        dec = saved_cfg.decoder.name
        model_type = f"{enc} + {dec}"

    # -- Transforms ---------------------------------------------------------
    model_transform = get_image_transform("val")
    display_transform = _display_transform()

    # -- Generate and plot --------------------------------------------------
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 6))
    if num_images == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        item = test_ds.items[idx]
        img_path = Path(test_ds.image_dir) / item["file_name"]
        gt_captions = item["captions"]

        # Load image for model
        raw_img = Image.open(img_path).convert("RGB")
        img_tensor = model_transform(raw_img).unsqueeze(0).to(device)

        # Generate caption
        pred = model.generate(
            img_tensor,
            tokenizer=tokenizer,
            max_length=cfg.inference.get("max_length", cfg.tokenizer.max_length),
        )[0]

        # Display image (un-normalized)
        display_img = display_transform(raw_img)
        ax.imshow(display_img.permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_axis_off()

        # Title: GT + Predicted
        gt_text = _wrap_text(gt_captions[0])
        pred_text = _wrap_text(pred)
        ax.set_title(
            f"GT: {gt_text}\n\nPred: {pred_text}",
            fontsize=9,
            wrap=True,
        )

    fig.suptitle(f"Model: {model_type}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = vis_dir / "captions_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to: {out_path}")

    # Also save individual images
    for i, idx in enumerate(indices):
        item = test_ds.items[idx]
        img_path = Path(test_ds.image_dir) / item["file_name"]
        gt_captions = item["captions"]

        raw_img = Image.open(img_path).convert("RGB")
        img_tensor = model_transform(raw_img).unsqueeze(0).to(device)
        pred = model.generate(
            img_tensor,
            tokenizer=tokenizer,
            max_length=cfg.inference.get("max_length", cfg.tokenizer.max_length),
        )[0]

        fig_single, ax_single = plt.subplots(figsize=(6, 7))
        display_img = display_transform(raw_img)
        ax_single.imshow(display_img.permute(1, 2, 0).clamp(0, 1).numpy())
        ax_single.set_axis_off()

        gt_text = _wrap_text(gt_captions[0])
        pred_text = _wrap_text(pred)
        ax_single.set_title(
            f"GT: {gt_text}\n\nPred: {pred_text}",
            fontsize=10,
            wrap=True,
        )
        fig_single.suptitle(f"Model: {model_type}", fontsize=12, fontweight="bold")
        fig_single.tight_layout()

        img_name = Path(item["file_name"]).stem
        fig_single.savefig(vis_dir / f"{img_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_single)

    print(f"Individual plots saved to: {vis_dir}")
    return vis_dir
