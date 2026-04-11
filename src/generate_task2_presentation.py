"""Generate all plots and tables for the Task 2 presentation report.

Usage:
    python src/generate_task2_presentation.py [--output-dir outputs/task2_presentation]

Produces:
    01_multimodal_results_bar.png        — T2.1 grouped bar chart (4 models × 4 metrics)
    02_prompt_comparison_9b.png          — Prompt engineering comparison on 9B
    03_architecture_diagram.png          — ViT + Projection + LoRA decoder pipeline
    04_lora_targets_diagram.png          — LoRA target module options
    05_hpo_trial_scatter_0.8b.png        — Optuna trial scatter for 0.8B
    06_hpo_trial_scatter_2b.png          — Optuna trial scatter for 2B
    07_hpo_best_params_comparison.png    — Side-by-side best params table (0.8B vs 2B)
    08_lora_results_bar.png              — LoRA final eval bar chart
    09_full_comparison_table.png         — T2.3 required table (all methods)
    10_method_tradeoff_scatter.png       — METEOR vs inference time scatter
    11_metric_profile_radar.png          — Radar chart of top methods
    12_qualitative_examples.png          — Sample predictions comparison
    13_hpo_optimizer_impact.png          — SGD vs AdamW vs Adam boxplot
    14_projection_comparison.png         — Linear vs Q-Former scatter
    15_challenges_solutions_table.png    — Challenges & solutions visual table
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

METRICS = ["bleu1", "bleu2", "rougeL", "meteor"]
METRIC_LABELS = {
    "bleu1": "BLEU-1",
    "bleu2": "BLEU-2",
    "rougeL": "ROUGE-L",
    "meteor": "METEOR",
}

# Color palettes
COLORS_MULTIMODAL = {
    "0.8B": "#42A5F5",
    "2B": "#66BB6A",
    "4B": "#FFA726",
    "9B": "#EF5350",
}
COLORS_LORA = {
    "LoRA 0.8B": "#1565C0",
    "LoRA 2B": "#2E7D32",
}
COLORS_METHOD = {
    "Multimodal 0.8B": "#42A5F5",
    "Multimodal 2B": "#66BB6A",
    "Multimodal 4B": "#FFA726",
    "Multimodal 9B": "#EF5350",
    "LoRA 0.8B": "#1565C0",
    "LoRA 2B": "#2E7D32",
    "Task 1 Best": "#9E9E9E",
}

# ─── Hardcoded results (from eval_results.json files) ───

MULTIMODAL_RESULTS = {
    "Qwen3.5-0.8B": {
        "bleu1": 0.5371,
        "bleu2": 0.3352,
        "rougeL": 0.4169,
        "meteor": 0.4580,
        "ms_per_img": 481.0,
        "prompt": "Describe this image in a short sentence.",
    },
    "Qwen3.5-2B": {
        "bleu1": 0.5594,
        "bleu2": 0.3506,
        "rougeL": 0.4243,
        "meteor": 0.4755,
        "ms_per_img": 771.2,
        "prompt": "Describe this image in a short sentence.",
    },
    "Qwen3.5-4B": {
        "bleu1": 0.5192,
        "bleu2": 0.3191,
        "rougeL": 0.4029,
        "meteor": 0.4752,
        "ms_per_img": 1235.8,
        "prompt": "Describe this image in a short sentence.",
    },
    "Qwen3.5-9B": {
        "bleu1": 0.6115,
        "bleu2": 0.4117,
        "rougeL": 0.4744,
        "meteor": 0.4911,
        "ms_per_img": 1002.3,
        "prompt": "Describe this image in a short sentence.",
    },
}

# 9B prompt comparison
PROMPT_RESULTS_9B = {
    "Describe this image.": {
        "bleu1": 0.1401,
        "bleu2": 0.0699,
        "rougeL": 0.1507,
        "meteor": 0.2720,
        "ms_per_img": 1632.6,
    },
    "Describe this image briefly.": {
        "bleu1": 0.1710,
        "bleu2": 0.0902,
        "rougeL": 0.1655,
        "meteor": 0.3028,
        "ms_per_img": 1646.6,
    },
    "Describe this image in a short sentence.": {
        "bleu1": 0.6115,
        "bleu2": 0.4117,
        "rougeL": 0.4744,
        "meteor": 0.4911,
        "ms_per_img": 1002.3,
    },
}

LORA_RESULTS = {
    "LoRA(ViT+Qwen3.5-0.8B)": {
        "bleu1": 0.6696,
        "bleu2": 0.4196,
        "rougeL": 0.4379,
        "meteor": 0.4095,
        "ms_per_img": 20.3,
    },
    "LoRA(ViT+Qwen3.5-2B)": {
        "bleu1": 0.6792,
        "bleu2": 0.4309,
        "rougeL": 0.4431,
        "meteor": 0.4197,
        "ms_per_img": 11.1,
    },
}

TASK1_BEST = {
    "Task1-Best(R50+LSTM+Sub+Attn)": {
        "bleu1": 0.6382,
        "bleu2": 0.4078,
        "rougeL": 0.4287,
        "meteor": 0.3826,
        "ms_per_img": 1.95,
    },
}

# ─── HPO best parameters ───

HPO_BEST_0_8B = {
    "study": "lora-optuna-qwen-0.8b-qformer",
    "best_trial": 26,
    "best_meteor": 0.4145,
    "n_trials": 30,
    "n_complete": 30,
    "n_failed": 2,
    "params": {
        "Learning Rate": "7.84e-4",
        "Optimizer": "AdamW",
        "Scheduler": "Cosine",
        "Weight Decay": "1e-4",
        "LoRA Rank (r)": "8",
        "LoRA Alpha": "2",
        "LoRA Target": "linear_and_head",
        "Projection": "Linear",
        "Max Token Length": "128",
    },
}

HPO_BEST_2B = {
    "study": "lora-optuna-qwen-2b",
    "best_trial": 27,
    "best_meteor": 0.4215,
    "n_trials": 30,
    "n_complete": 30,
    "n_failed": 3,
    "params": {
        "Learning Rate": "4.02e-5",
        "Optimizer": "AdamW",
        "Scheduler": "Step",
        "Weight Decay": "0.1",
        "LoRA Rank (r)": "4",
        "LoRA Alpha": "4",
        "LoRA Target": "attention",
        "Projection": "Q-Former (32q, 1L, 2048d)",
        "Max Token Length": "128",
    },
}

# ─── Qualitative samples (from eval JSONs) ───

QUALITATIVE_SAMPLES = [
    {
        "image_name": "VizWiz_val_00000001.jpg",
        "references": [
            "A person is holding a bottle that has medicine for the night time.",
            "a person holding a small black bottle of NIGHT TIME",
        ],
        "multimodal_9b": "A hand holding a bottle of Night Time medicine.",
        "multimodal_0_8b": "A hand holds a bottle of Night Time medicine.",
        "lora_2b": "A man is holding a bottle of medicine.",
        "lora_0_8b": "A man is holding a bottle of medicine in his hand.",
    },
    {
        "image_name": "VizWiz_val_00000002.jpg",
        "references": [
            "A library book with pictures of two dogs on the cover on a wooden table.",
            "The book cover shows two dogs in the snow",
        ],
        "multimodal_9b": "A book with a sticker on it.",
        "multimodal_0_8b": "A photograph of a book titled 'A Memoir'.",
        "lora_2b": "A book is on a table with a picture on it.",
        "lora_0_8b": "A book is on a wooden table.",
    },
    {
        "image_name": "VizWiz_val_00000000.jpg",
        "references": [
            "A computer screen shows a repair prompt on the screen.",
            "Part of a computer monitor showing a computer repair message.",
        ],
        "multimodal_9b": 'A computer screen with a message that says "cannot repair this computer automatically".',
        "multimodal_0_8b": "A computer screen displays a message that it cannot repair the computer automatically.",
        "lora_2b": "A computer monitor is on a desk with the screen turned off.",
        "lora_0_8b": "A computer monitor with a screen and keyboard.",
    },
]


def pct(v: float) -> float:
    return v * 100.0


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "legend.framealpha": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.2,
        }
    )


# ─────────────────────────────────────────────────────────────────────
# PLOT 1: Multimodal results grouped bar chart
# ─────────────────────────────────────────────────────────────────────


def plot_multimodal_results_bar(out_dir: Path) -> None:
    models = list(MULTIMODAL_RESULTS.keys())
    n_metrics = len(METRICS)
    n_models = len(models)
    x = np.arange(n_metrics)
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        vals = [pct(MULTIMODAL_RESULTS[model][m]) for m in METRICS]
        size_label = model.replace("Qwen3.5-", "")
        bars = ax.bar(
            x + (i - (n_models - 1) / 2) * width,
            vals,
            width * 0.88,
            label=model,
            color=COLORS_MULTIMODAL[size_label],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Score (%)")
    ax.set_title("T2.1 — Direct Multimodal Evaluation (zero-shot, no thinking)")
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(0, 72)

    fig.tight_layout()
    fig.savefig(out_dir / "01_multimodal_results_bar.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  01_multimodal_results_bar.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 2: Prompt comparison on 9B
# ─────────────────────────────────────────────────────────────────────


def plot_prompt_comparison_9b(out_dir: Path) -> None:
    prompts = list(PROMPT_RESULTS_9B.keys())
    short_labels = [
        '"Describe this image."',
        '"Describe this image briefly."',
        '"...in a short sentence."',
    ]
    colors = ["#EF9A9A", "#FFCC80", "#81C784"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: metric bars
    ax = axes[0]
    n_prompts = len(prompts)
    x = np.arange(len(METRICS))
    width = 0.25

    for i, (prompt, label) in enumerate(zip(prompts, short_labels)):
        vals = [pct(PROMPT_RESULTS_9B[prompt][m]) for m in METRICS]
        bars = ax.bar(
            x + (i - (n_prompts - 1) / 2) * width,
            vals,
            width * 0.88,
            label=label,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Score (%)")
    ax.set_title("Qwen3.5-9B: Prompt Impact on Metrics")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(5, 65)

    # Right: inference time + verbosity note
    ax2 = axes[1]
    ms_vals = [PROMPT_RESULTS_9B[p]["ms_per_img"] for p in prompts]
    bars2 = ax2.barh(
        range(n_prompts), ms_vals, color=colors, edgecolor="white", height=0.6
    )
    ax2.set_yticks(range(n_prompts))
    ax2.set_yticklabels(short_labels, fontsize=10)
    ax2.set_xlabel("Inference Time (ms/image)")
    ax2.set_title("Inference Time per Prompt")
    ax2.invert_yaxis()

    for bar, ms in zip(bars2, ms_vals):
        ax2.text(
            bar.get_width() + 20,
            bar.get_y() + bar.get_height() / 2,
            f"{ms:.0f} ms",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Add annotation about verbose output
    ax2.annotate(
        "Verbose prompts → longer\noutputs → lower metrics\n(mismatch with references)",
        xy=(1600, 0.5),
        fontsize=9,
        style="italic",
        color="#666",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", alpha=0.8),
    )

    fig.suptitle(
        "Prompt Engineering — Qwen3.5-9B", fontsize=18, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_dir / "02_prompt_comparison_9b.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  02_prompt_comparison_9b.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 3: Architecture diagram (ViT + Projection + LoRA Decoder)
# ─────────────────────────────────────────────────────────────────────


def plot_architecture_diagram(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title(
        "T2.2 — ViT + Projection + LoRA Decoder Architecture",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    def draw_box(
        x,
        y,
        w,
        h,
        text,
        color,
        text_color="white",
        fontsize=12,
        subtext=None,
        border_color=None,
    ):
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor=border_color or color,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2 + (0.15 if subtext else 0),
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=text_color,
        )
        if subtext:
            ax.text(
                x + w / 2,
                y + h / 2 - 0.25,
                subtext,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                style="italic",
            )

    def draw_arrow(x1, y1, x2, y2, text=None):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#333", lw=2),
        )
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.25
            ax.text(
                mx,
                my,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="#555",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="#DDD",
                    alpha=0.9,
                ),
            )

    # Input image
    draw_box(0.3, 3.0, 2.0, 2.0, "Input\nImage", "#78909C", subtext="224×224×3")

    # Arrow → ViT
    draw_arrow(2.3, 4.0, 3.5, 4.0)

    # ViT Encoder (frozen)
    draw_box(3.5, 2.5, 2.5, 3.0, "ViT-Base\nEncoder", "#2196F3", subtext="FROZEN")
    # Frozen indicator
    ax.text(
        5.6, 5.3, "[FROZEN]", fontsize=8, ha="center", color="#1565C0", fontweight="bold"
    )

    # Arrow → Projection
    draw_arrow(6.0, 4.0, 7.2, 4.0, "197 × 768")

    # Projection module
    draw_box(7.2, 2.5, 2.8, 3.0, "Projection\nModule", "#FF9800", subtext="TRAINABLE")

    # Two sub-options inside projection
    draw_box(
        7.2,
        1.0,
        0.8,
        1.0,
        "Linear",
        "#FFB74D",
        text_color="#333",
        fontsize=9,
        subtext="768→D",
    )
    draw_box(
        9.2,
        1.0,
        0.8,
        1.0,
        "Q-Former",
        "#FFB74D",
        text_color="#333",
        fontsize=9,
        subtext="BLIP-2",
    )
    ax.text(
        8.55,
        1.5,
        "or",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#E65100",
    )

    # Arrow → Decoder
    draw_arrow(10.0, 4.0, 11.2, 4.0, "N × D")

    # Qwen Decoder (LoRA)
    draw_box(11.2, 2.0, 3.0, 4.0, "Qwen3.5\nDecoder", "#4CAF50", subtext="LoRA-adapted")
    # Trainable indicator
    ax.text(
        13.8, 5.8, "[TRAIN]", fontsize=8, ha="center", color="#2E7D32", fontweight="bold"
    )

    # LoRA detail boxes
    draw_box(
        11.2, 0.5, 0.8, 1.0, "Frozen\nWeights", "#C8E6C9", text_color="#333", fontsize=8
    )
    draw_box(
        13.4,
        0.5,
        0.8,
        1.0,
        "LoRA\nAdapters",
        "#A5D6A7",
        text_color="#333",
        fontsize=8,
        border_color="#2E7D32",
    )
    ax.text(
        12.6,
        1.0,
        "+",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#333",
    )

    # Arrow → Output
    draw_arrow(14.2, 4.0, 15.4, 4.0)

    # Output caption
    draw_box(15.4, 3.0, 2.2, 2.0, "Generated\nCaption", "#7E57C2", subtext="text output")

    # Legend at bottom
    legend_y = 7.2
    for i, (label, color) in enumerate(
        [
            ("Frozen (no gradients)", "#2196F3"),
            ("Trainable (full)", "#FF9800"),
            ("LoRA-adapted (1-5% params)", "#4CAF50"),
        ]
    ):
        rect = mpatches.FancyBboxPatch(
            (3.5 + i * 4.5, legend_y),
            0.4,
            0.4,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor=color,
        )
        ax.add_patch(rect)
        ax.text(4.1 + i * 4.5, legend_y + 0.2, label, va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "03_architecture_diagram.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  03_architecture_diagram.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 4: LoRA target modules diagram
# ─────────────────────────────────────────────────────────────────────


def plot_lora_targets_diagram(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(
        "LoRA Target Module Options — What Gets Fine-Tuned",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    targets = [
        ("attention", "q/k/v/o_proj", "#42A5F5", "Standard attention\nprojections only"),
        (
            "full_attention",
            "attention +\nGatedDeltaNet",
            "#1E88E5",
            "All attention incl.\nlinear attention layers",
        ),
        (
            "linear",
            "gate/up/down\nMLP layers",
            "#66BB6A",
            "MLP feed-forward\nlayers only",
        ),
        (
            "linear_and_head",
            "linear +\nlm_head",
            "#43A047",
            "MLP + output\nprojection head",
        ),
        ("all", "attention +\nlinear", "#FFA726", "All transformer\nlayers combined"),
        (
            "all_and_head",
            "all +\nlm_head",
            "#EF5350",
            "Everything including\noutput head",
        ),
    ]

    for i, (name, modules, color, desc) in enumerate(targets):
        col = i % 3
        row = i // 3
        x = 1.0 + col * 4.3
        y = 4.5 - row * 3.0

        # Box
        rect = mpatches.FancyBboxPatch(
            (x, y),
            3.5,
            2.2,
            boxstyle="round,pad=0.15",
            facecolor=color,
            edgecolor=color,
            alpha=0.15,
            linewidth=2,
        )
        ax.add_patch(rect)
        rect_border = mpatches.FancyBboxPatch(
            (x, y),
            3.5,
            2.2,
            boxstyle="round,pad=0.15",
            facecolor="none",
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect_border)

        # Target name
        ax.text(
            x + 1.75,
            y + 1.75,
            f'"{name}"',
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=color,
            fontfamily="monospace",
        )
        # What modules
        ax.text(
            x + 1.75,
            y + 1.15,
            modules,
            ha="center",
            va="center",
            fontsize=10,
            color="#333",
        )
        # Description
        ax.text(
            x + 1.75,
            y + 0.4,
            desc,
            ha="center",
            va="center",
            fontsize=8,
            color="#666",
            style="italic",
        )

    # Winner annotations
    # ax.annotate(
    #     "✓ Best for 0.8B",
    #     xy=(5.7, 2.0),
    #     fontsize=11,
    #     fontweight="bold",
    #     color="#2E7D32",
    #     ha="center",
    # )
    # ax.annotate(
    #     "✓ Best for 2B",
    #     xy=(1.0 + 1.75, 5.0),
    #     fontsize=11,
    #     fontweight="bold",
    #     color="#1565C0",
    #     ha="center",
    # )

    fig.tight_layout()
    fig.savefig(out_dir / "04_lora_targets_diagram.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  04_lora_targets_diagram.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 5 & 6: HPO trial scatter plots
# ─────────────────────────────────────────────────────────────────────


def _load_all_trials(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def plot_hpo_trial_scatter(
    trials_path: Path, model_name: str, best_trial_num: int, out_dir: Path, filename: str
) -> None:
    trials = _load_all_trials(trials_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: METEOR by trial number, colored by optimizer
    ax = axes[0]
    optimizer_colors = {"adamw": "#4CAF50", "adam": "#FF9800", "sgd": "#F44336"}
    for trial in trials:
        if trial["state"] != "COMPLETE":
            continue
        opt = trial["params"]["training.optimizer"]
        color = optimizer_colors.get(opt, "#999")
        size = 200 if trial["number"] == best_trial_num else 60
        ax.scatter(
            trial["number"],
            pct(trial["value"]),
            c=color,
            s=size,
            marker="*" if trial["number"] == best_trial_num else "o",
            edgecolors="black" if trial["number"] == best_trial_num else "white",
            linewidths=2 if trial["number"] == best_trial_num else 0.5,
            zorder=5 if trial["number"] == best_trial_num else 3,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markersize=10,
            label=n.upper(),
        )
        for n, c in optimizer_colors.items()
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markersize=15,
            markeredgecolor="black",
            label="Best Trial",
        )
    )
    ax.legend(handles=handles, loc="lower right")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("METEOR (%)")
    ax.set_title(f"{model_name} — METEOR by Trial (colored by optimizer)")

    # Right: METEOR by learning rate (log scale)
    ax2 = axes[1]
    for trial in trials:
        if trial["state"] != "COMPLETE":
            continue
        lr = trial["params"]["training.lr"]
        opt = trial["params"]["training.optimizer"]
        color = optimizer_colors.get(opt, "#999")
        size = 200 if trial["number"] == best_trial_num else 60
        ax2.scatter(
            lr,
            pct(trial["value"]),
            c=color,
            s=size,
            marker="*" if trial["number"] == best_trial_num else "o",
            edgecolors="black" if trial["number"] == best_trial_num else "white",
            linewidths=2 if trial["number"] == best_trial_num else 0.5,
            zorder=5 if trial["number"] == best_trial_num else 3,
        )

    ax2.set_xscale("log")
    ax2.set_xlabel("Learning Rate (log scale)")
    ax2.set_ylabel("METEOR (%)")
    ax2.set_title(f"{model_name} — METEOR vs Learning Rate")
    ax2.legend(handles=handles, loc="lower right")

    fig.suptitle(f"Optuna HPO — {model_name}", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {filename}")


# ─────────────────────────────────────────────────────────────────────
# PLOT 7: Best params comparison table (0.8B vs 2B)
# ─────────────────────────────────────────────────────────────────────


def plot_hpo_best_params_comparison(out_dir: Path) -> None:
    params_order = [
        "Learning Rate",
        "Optimizer",
        "Scheduler",
        "Weight Decay",
        "LoRA Rank (r)",
        "LoRA Alpha",
        "LoRA Target",
        "Projection",
        "Max Token Length",
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    ax.set_title(
        "Best Hyperparameters — 0.8B vs 2B (Key Differences Highlighted)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    headers = [
        "Hyperparameter",
        "Qwen3.5-0.8B (Trial #26)",
        "Qwen3.5-2B (Trial #27)",
        "Different?",
    ]
    cell_text = []
    cell_colors = []

    for param in params_order:
        v1 = HPO_BEST_0_8B["params"][param]
        v2 = HPO_BEST_2B["params"][param]
        diff = "✓ Yes" if v1 != v2 else "—"
        cell_text.append([param, v1, v2, diff])
        if v1 != v2:
            cell_colors.append(["#FFF9C4", "#E3F2FD", "#E8F5E9", "#FFECB3"])
        else:
            cell_colors.append(["white", "white", "white", "white"])

    # Add summary rows
    cell_text.append(["", "", "", ""])
    cell_colors.append(["#F5F5F5"] * 4)
    cell_text.append(
        [
            "Best METEOR",
            f"{HPO_BEST_0_8B['best_meteor']:.4f}",
            f"{HPO_BEST_2B['best_meteor']:.4f}",
            "",
        ]
    )
    cell_colors.append(["#E8EAF6", "#E8EAF6", "#E8EAF6", "#E8EAF6"])

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.25, 0.30, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Color header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color cells
    for i, row_colors in enumerate(cell_colors):
        for j, color in enumerate(row_colors):
            table[i + 1, j].set_facecolor(color)

    fig.tight_layout()
    fig.savefig(out_dir / "07_hpo_best_params_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  07_hpo_best_params_comparison.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 8: LoRA final eval bar chart
# ─────────────────────────────────────────────────────────────────────


def plot_lora_results_bar(out_dir: Path) -> None:
    models = list(LORA_RESULTS.keys())
    short_labels = ["ViT+Qwen3.5-0.8B", "ViT+Qwen3.5-2B"]
    colors = [COLORS_LORA["LoRA 0.8B"], COLORS_LORA["LoRA 2B"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: metrics
    ax = axes[0]
    x = np.arange(len(METRICS))
    width = 0.35
    for i, (model, label) in enumerate(zip(models, short_labels)):
        vals = [pct(LORA_RESULTS[model][m]) for m in METRICS]
        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width * 0.9,
            label=label,
            color=colors[i],
            edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Score (%)")
    ax.set_title("LoRA Fine-Tuned Models — Metrics")
    ax.legend()
    ax.set_ylim(0, 80)

    # Right: inference time comparison
    ax2 = axes[1]
    all_models_ms = {
        "Multimodal 0.8B": MULTIMODAL_RESULTS["Qwen3.5-0.8B"]["ms_per_img"],
        "Multimodal 2B": MULTIMODAL_RESULTS["Qwen3.5-2B"]["ms_per_img"],
        "Multimodal 9B": MULTIMODAL_RESULTS["Qwen3.5-9B"]["ms_per_img"],
        "LoRA 0.8B": LORA_RESULTS["LoRA(ViT+Qwen3.5-0.8B)"]["ms_per_img"],
        "LoRA 2B": LORA_RESULTS["LoRA(ViT+Qwen3.5-2B)"]["ms_per_img"],
    }
    bar_colors = ["#42A5F5", "#66BB6A", "#EF5350", "#1565C0", "#2E7D32"]
    names = list(all_models_ms.keys())
    ms_vals = list(all_models_ms.values())

    bars = ax2.barh(
        range(len(names)), ms_vals, color=bar_colors, edgecolor="white", height=0.6
    )
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=10)
    ax2.set_xlabel("Inference Time (ms/image)")
    ax2.set_title("Inference Speed Comparison")
    ax2.invert_yaxis()
    ax2.set_xscale("log")

    for bar, ms in zip(bars, ms_vals):
        ax2.text(
            bar.get_width() * 1.1,
            bar.get_y() + bar.get_height() / 2,
            f"{ms:.1f} ms",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Add speedup annotation
    speedup_08 = (
        MULTIMODAL_RESULTS["Qwen3.5-0.8B"]["ms_per_img"]
        / LORA_RESULTS["LoRA(ViT+Qwen3.5-0.8B)"]["ms_per_img"]
    )
    speedup_2b = (
        MULTIMODAL_RESULTS["Qwen3.5-2B"]["ms_per_img"]
        / LORA_RESULTS["LoRA(ViT+Qwen3.5-2B)"]["ms_per_img"]
    )
    ax2.text(
        0.95,
        0.05,
        f"LoRA speedup: {speedup_08:.0f}×–{speedup_2b:.0f}× faster",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#2E7D32",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", alpha=0.9),
    )

    fig.suptitle(
        "T2.2 — LoRA Fine-Tuned Model Evaluation", fontsize=16, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_dir / "08_lora_results_bar.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  08_lora_results_bar.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 9: Full comparison table (T2.3)
# ─────────────────────────────────────────────────────────────────────


def plot_full_comparison_table(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    ax.set_title(
        "T2.3 — Full Results Comparison Table", fontsize=18, fontweight="bold", pad=20
    )

    headers = ["Method", "Model", "BLEU-1", "BLEU-2", "ROUGE-L", "METEOR", "ms/img"]

    rows = []
    row_colors = []

    # Multimodal rows
    for name, data in MULTIMODAL_RESULTS.items():
        rows.append(
            [
                "Multimodal\n(zero-shot)",
                name,
                f"{data['bleu1']:.3f}",
                f"{data['bleu2']:.3f}",
                f"{data['rougeL']:.3f}",
                f"{data['meteor']:.3f}",
                f"{data['ms_per_img']:.0f}",
            ]
        )
        row_colors.append("#E3F2FD")

    # LoRA rows
    for name, data in LORA_RESULTS.items():
        short_name = name.replace("LoRA(", "").replace(")", "")
        rows.append(
            [
                "LoRA\n(fine-tuned)",
                short_name,
                f"{data['bleu1']:.3f}",
                f"{data['bleu2']:.3f}",
                f"{data['rougeL']:.3f}",
                f"{data['meteor']:.3f}",
                f"{data['ms_per_img']:.1f}",
            ]
        )
        row_colors.append("#E8F5E9")

    # Previous week reference
    for name, data in TASK1_BEST.items():
        rows.append(
            [
                "Previous Week Best\n(reference)",
                "R50+LSTM\n+Sub+Attn",
                f"{data['bleu1']:.3f}",
                f"{data['bleu2']:.3f}",
                f"{data['rougeL']:.3f}",
                f"{data['meteor']:.3f}",
                f"{data['ms_per_img']:.1f}",
            ]
        )
        row_colors.append("#F5F5F5")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.16, 0.10, 0.10, 0.10, 0.10, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    # Header styling
    for j in range(len(headers)):
        table[0, j].set_facecolor("#263238")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Row colors
    for i, color in enumerate(row_colors):
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    # Bold the best values
    # Find best in each metric column
    metric_cols = {2: "bleu1", 3: "bleu2", 4: "rougeL", 5: "meteor"}
    for col_idx, metric_key in metric_cols.items():
        all_vals = []
        for row in rows:
            try:
                all_vals.append(float(row[col_idx]))
            except (ValueError, IndexError):
                all_vals.append(-1)
        best_idx = int(np.argmax(all_vals))
        cell = table[best_idx + 1, col_idx]
        cell.set_text_props(fontweight="bold", color="#1B5E20")
        cell.set_facecolor("#C8E6C9")

    # Best (lowest) inference time
    ms_vals = []
    for row in rows:
        try:
            ms_vals.append(float(row[6]))
        except (ValueError, IndexError):
            ms_vals.append(float("inf"))
    best_ms_idx = int(np.argmin(ms_vals))
    table[best_ms_idx + 1, 6].set_text_props(fontweight="bold", color="#1B5E20")
    table[best_ms_idx + 1, 6].set_facecolor("#C8E6C9")

    fig.tight_layout()
    fig.savefig(out_dir / "09_full_comparison_table.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  09_full_comparison_table.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 10: METEOR vs Inference Time scatter (trade-off)
# ─────────────────────────────────────────────────────────────────────


def plot_method_tradeoff_scatter(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    all_methods = {}
    for name, data in MULTIMODAL_RESULTS.items():
        label = f"modal {name.replace('Qwen3.5-', '')}"
        all_methods[label] = data
    for name, data in LORA_RESULTS.items():
        label = name.replace("LoRA(ViT+Qwen3.5-", "LoRA ").replace(")", "")
        all_methods[label] = data
    for name, data in TASK1_BEST.items():
        all_methods["Previous Week Best"] = data

    for label, data in all_methods.items():
        color = COLORS_METHOD.get(label, "#999")
        size = 200 if "9B" in label or "2B" in label else 120
        marker = "s" if "LoRA" in label else ("D" if "Task" in label else "o")
        ax.scatter(
            data["ms_per_img"],
            pct(data["meteor"]),
            c=color,
            s=size,
            marker=marker,
            edgecolors="black",
            linewidths=1,
            zorder=5,
            label=label,
        )
        # Label offset
        offset_x = 40 if data["ms_per_img"] > 50 else 3
        ax.annotate(
            label,
            (data["ms_per_img"] + offset_x, pct(data["meteor"]) + 0.3),
            fontsize=9,
            color="#333",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference Time (ms/image, log scale)")
    ax.set_ylabel("METEOR (%)")
    ax.set_title(
        "Quality vs Speed Trade-off — All Methods", fontsize=16, fontweight="bold"
    )

    # Add quadrant annotations
    ax.axhline(y=pct(0.45), color="#DDD", linestyle="--", alpha=0.5)
    ax.axvline(x=100, color="#DDD", linestyle="--", alpha=0.5)
    ax.text(
        5,
        50,
        "Fast + High Quality\n(ideal)",
        fontsize=10,
        color="#2E7D32",
        fontweight="bold",
        ha="center",
        alpha=0.6,
    )
    ax.text(
        800,
        50,
        "Slow + High Quality",
        fontsize=10,
        color="#F57C00",
        fontweight="bold",
        ha="center",
        alpha=0.6,
    )

    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "10_method_tradeoff_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  10_method_tradeoff_scatter.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 11: Radar chart of top methods
# ─────────────────────────────────────────────────────────────────────


def plot_metric_profile_radar(out_dir: Path) -> None:
    models = {
        "Multimodal 9B": MULTIMODAL_RESULTS["Qwen3.5-9B"],
        "Multimodal 2B": MULTIMODAL_RESULTS["Qwen3.5-2B"],
        "LoRA 2B": LORA_RESULTS["LoRA(ViT+Qwen3.5-2B)"],
        "LoRA 0.8B": LORA_RESULTS["LoRA(ViT+Qwen3.5-0.8B)"],
        "Previous Week Best": TASK1_BEST["Task1-Best(R50+LSTM+Sub+Attn)"],
    }
    colors = ["#EF5350", "#66BB6A", "#2E7D32", "#1565C0", "#9E9E9E"]

    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for (label, data), color in zip(models.items(), colors):
        vals = [pct(data[m]) for m in METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=label, color=color, markersize=6)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_thetagrids(
        [a * 180 / np.pi for a in angles[:-1]],
        [METRIC_LABELS[m] for m in METRICS],
        fontsize=13,
    )
    ax.set_ylim(0, 75)
    ax.set_rgrids(
        [15, 30, 45, 60, 75], labels=["15%", "30%", "45%", "60%", "75%"], fontsize=9
    )
    ax.set_title("Metric Profile — Top Methods", pad=30, fontsize=16, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "11_metric_profile_radar.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  11_metric_profile_radar.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 12: Qualitative examples comparison
# ─────────────────────────────────────────────────────────────────────


def plot_qualitative_examples(out_dir: Path) -> None:
    n_samples = len(QUALITATIVE_SAMPLES)
    fig, axes = plt.subplots(n_samples, 1, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, sample in enumerate(QUALITATIVE_SAMPLES):
        ax = axes[i]
        ax.axis("off")

        # Build text content
        lines = []
        lines.append(f"Image: {sample['image_name']}")
        lines.append("")
        lines.append(f'Reference: "{sample["references"][0]}"')
        lines.append("")
        lines.append(f'Multimodal 9B: "{sample["multimodal_9b"]}"')
        lines.append(f'Multimodal 0.8B: "{sample["multimodal_0_8b"]}"')
        lines.append(f'LoRA 2B: "{sample["lora_2b"]}"')
        lines.append(f'LoRA 0.8B: "{sample["lora_0_8b"]}"')

        text = "\n".join(lines)

        # Background
        rect = mpatches.FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle="round,pad=0.02",
            facecolor="#FAFAFA",
            edgecolor="#E0E0E0",
            linewidth=1,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        # Image name header
        ax.text(
            0.05,
            0.92,
            f"[IMG] {sample['image_name']}",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            color="#333",
        )

        # Reference
        ax.text(
            0.05,
            0.78,
            "Reference:",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            color="#1565C0",
        )
        ax.text(
            0.15,
            0.78,
            f'"{sample["references"][0]}"',
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            color="#333",
            style="italic",
        )

        # Predictions with color coding
        methods = [
            ("Multimodal 9B:", sample["multimodal_9b"], "#EF5350"),
            ("Multimodal 0.8B:", sample["multimodal_0_8b"], "#42A5F5"),
            ("LoRA 2B:", sample["lora_2b"], "#2E7D32"),
            ("LoRA 0.8B:", sample["lora_0_8b"], "#1565C0"),
        ]
        for j, (method_label, pred, color) in enumerate(methods):
            y_pos = 0.60 - j * 0.16
            ax.text(
                0.05,
                y_pos,
                method_label,
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
                color=color,
            )
            wrapped = textwrap.fill(f'"{pred}"', width=90)
            ax.text(
                0.22,
                y_pos,
                wrapped,
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                color="#333",
            )

    fig.suptitle(
        "Qualitative Prediction Comparison", fontsize=18, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(out_dir / "12_qualitative_examples.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  12_qualitative_examples.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 13: Optimizer impact boxplot (SGD vs Adam vs AdamW)
# ─────────────────────────────────────────────────────────────────────


def plot_hpo_optimizer_impact(
    trials_0_8b_path: Path, trials_2b_path: Path, out_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, trials_path, model_name in [
        (axes[0], trials_0_8b_path, "Qwen3.5-0.8B"),
        (axes[1], trials_2b_path, "Qwen3.5-2B"),
    ]:
        trials = _load_all_trials(trials_path)
        opt_data = {"adam": [], "adamw": [], "sgd": []}
        for trial in trials:
            if trial["state"] != "COMPLETE":
                continue
            opt = trial["params"]["training.optimizer"]
            opt_data[opt].append(pct(trial["value"]))

        positions = []
        box_data = []
        labels = []
        colors = ["#FF9800", "#4CAF50", "#F44336"]

        for i, (opt, vals) in enumerate(opt_data.items()):
            if vals:
                positions.append(i)
                box_data.append(vals)
                labels.append(opt.upper())

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white", markersize=6),
        )
        for patch, color in zip(bp["boxes"], colors[: len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Scatter individual points
        for i, (vals, color) in enumerate(zip(box_data, colors)):
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            ax.scatter(
                [positions[i] + j for j in jitter],
                vals,
                c=color,
                s=30,
                edgecolors="white",
                linewidths=0.5,
                zorder=5,
                alpha=0.8,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("METEOR (%)")
        ax.set_title(model_name)

    fig.suptitle(
        "Optimizer Impact on LoRA Fine-Tuning\n(SGD consistently catastrophic)",
        fontsize=16,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "13_hpo_optimizer_impact.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  13_hpo_optimizer_impact.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 14: Linear vs Q-Former projection comparison
# ─────────────────────────────────────────────────────────────────────


def plot_projection_comparison(
    trials_0_8b_path: Path, trials_2b_path: Path, out_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, trials_path, model_name in [
        (axes[0], trials_0_8b_path, "Qwen3.5-0.8B"),
        (axes[1], trials_2b_path, "Qwen3.5-2B"),
    ]:
        trials = _load_all_trials(trials_path)
        proj_data = {"linear": [], "qformer": []}
        for trial in trials:
            if trial["state"] != "COMPLETE":
                continue
            # Skip SGD trials (they are outliers that mask the projection effect)
            if trial["params"]["training.optimizer"] == "sgd":
                continue
            proj = trial["params"]["projection.type"]
            proj_data[proj].append(pct(trial["value"]))

        positions = [0, 1]
        box_data = [proj_data["linear"], proj_data["qformer"]]
        labels = ["Linear\nProjection", "Q-Former\nBridge"]
        colors = ["#FF9800", "#2196F3"]

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.4,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white", markersize=6),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        for i, (vals, color) in enumerate(zip(box_data, colors)):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(
                [positions[i] + j for j in jitter],
                vals,
                c=color,
                s=40,
                edgecolors="white",
                linewidths=0.5,
                zorder=5,
                alpha=0.8,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("METEOR (%)")
        ax.set_title(model_name)

        # Annotate winner
        mean_linear = np.mean(proj_data["linear"]) if proj_data["linear"] else 0
        mean_qformer = np.mean(proj_data["qformer"]) if proj_data["qformer"] else 0
        winner = "Linear" if mean_linear >= mean_qformer else "Q-Former"
        winner_pos = 0 if winner == "Linear" else 1
        ax.annotate(
            f"✓ {winner} wins",
            xy=(
                winner_pos,
                max(
                    max(proj_data["linear"], default=0),
                    max(proj_data["qformer"], default=0),
                )
                + 1,
            ),
            fontsize=11,
            fontweight="bold",
            color="#2E7D32",
            ha="center",
        )

    fig.suptitle(
        "Projection Type Comparison (excluding SGD outliers)",
        fontsize=16,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "14_projection_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  14_projection_comparison.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 15: Challenges & solutions visual table
# ─────────────────────────────────────────────────────────────────────


def plot_challenges_solutions_table(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    ax.set_title(
        "Challenges Encountered & Solutions Applied",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    challenges = [
        (
            "Verbose multimodal outputs\n→ low evaluation metrics",
            "Prompt engineering:\n'Describe this image in a short sentence.'",
            "METEOR: 0.27 → 0.49 (+81%)",
        ),
        (
            "12-dim hyperparameter space\n→ manual tuning impossible",
            "Optuna TPE sampler + Hyperband\npruning (30 trials per model)",
            "Found optimal configs for\nboth model sizes",
        ),
        (
            "SGD catastrophic failure\nin LoRA fine-tuning",
            "Identified early via HPO;\nexclusively used AdamW",
            "5/5 SGD trials: METEOR < 0.10\nvs AdamW best: 0.414",
        ),
        (
            "Different optimal configs\nper model size",
            "Separate Optuna studies\nper decoder size",
            "0.8B: linear_and_head\n2B: attention",
        ),
        (
            "Vision-language feature\nspace mismatch (768 vs 2048-dim)",
            "Tested Linear vs Q-Former\nprojection; model-size dependent",
            "0.8B: Linear best\n2B: Q-Former best",
        ),
        (
            "Memory constraints for\nlarger decoder models",
            "bfloat16 precision + LoRA\n(only 1-5% params trainable)",
            "Enabled 2B fine-tuning\non single GPU",
        ),
    ]

    headers = ["Challenge", "Solution", "Impact"]
    cell_text = [[c, s, i] for c, s, i in challenges]

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.30, 0.35, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for j in range(3):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=12)

    row_colors = ["#FFEBEE", "#E3F2FD", "#FFEBEE", "#E3F2FD", "#FFEBEE", "#E3F2FD"]
    for i in range(len(challenges)):
        table[i + 1, 0].set_facecolor(row_colors[i])
        table[i + 1, 1].set_facecolor("#E8F5E9")
        table[i + 1, 2].set_facecolor("#FFF9C4")

    fig.tight_layout()
    fig.savefig(out_dir / "15_challenges_solutions_table.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  15_challenges_solutions_table.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 16: Multimodal scaling analysis
# ─────────────────────────────────────────────────────────────────────


def plot_multimodal_scaling(out_dir: Path) -> None:
    sizes = [0.8, 2, 4, 9]
    size_labels = ["0.8B", "2B", "4B", "9B"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Metrics by model size
    ax = axes[0]
    for m in METRICS:
        vals = [pct(MULTIMODAL_RESULTS[f"Qwen3.5-{s}"][m]) for s in size_labels]
        ax.plot(sizes, vals, "o-", label=METRIC_LABELS[m], linewidth=2, markersize=8)

    ax.set_xlabel("Model Size (billions of parameters)")
    ax.set_ylabel("Score (%)")
    ax.set_title("Metrics vs Model Size")
    ax.set_xticks(sizes)
    ax.set_xticklabels(size_labels)
    ax.legend()

    # Highlight 4B dip
    ax.annotate(
        "4B underperforms 2B!\n(verbose output)",
        xy=(4, pct(0.319)),
        xytext=(5.5, pct(0.30)),
        arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.5),
        fontsize=10,
        color="#F44336",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE"),
    )

    # Right: Inference time by model size
    ax2 = axes[1]
    ms_vals = [MULTIMODAL_RESULTS[f"Qwen3.5-{s}"]["ms_per_img"] for s in size_labels]
    bars = ax2.bar(
        range(len(sizes)),
        ms_vals,
        color=[COLORS_MULTIMODAL[s] for s in size_labels],
        edgecolor="white",
    )
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels(size_labels)
    ax2.set_xlabel("Model Size")
    ax2.set_ylabel("Inference Time (ms/image)")
    ax2.set_title("Inference Time vs Model Size")

    for bar, ms in zip(bars, ms_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{ms:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Note about 9B being faster than 4B
    ax2.annotate(
        "9B faster than 4B!\n(better batching/optimization?)",
        xy=(3, 1002),
        xytext=(1.5, 1150),
        arrowprops=dict(arrowstyle="->", color="#FF9800", lw=1.5),
        fontsize=9,
        color="#FF9800",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0"),
    )

    fig.suptitle(
        "Multimodal Model Scaling Analysis", fontsize=16, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_dir / "16_multimodal_scaling.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  16_multimodal_scaling.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 17: LoRA target comparison across trials
# ─────────────────────────────────────────────────────────────────────


def plot_lora_target_comparison(
    trials_0_8b_path: Path, trials_2b_path: Path, out_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    target_order = [
        "attention",
        "full_attention",
        "linear",
        "linear_and_head",
        "all",
        "all_and_head",
    ]
    target_colors = {
        "attention": "#42A5F5",
        "full_attention": "#1E88E5",
        "linear": "#66BB6A",
        "linear_and_head": "#43A047",
        "all": "#FFA726",
        "all_and_head": "#EF5350",
    }

    for ax, trials_path, model_name, best_target in [
        (axes[0], trials_0_8b_path, "Qwen3.5-0.8B", "linear_and_head"),
        (axes[1], trials_2b_path, "Qwen3.5-2B", "attention"),
    ]:
        trials = _load_all_trials(trials_path)
        target_data = {t: [] for t in target_order}
        for trial in trials:
            if trial["state"] != "COMPLETE":
                continue
            if trial["params"]["training.optimizer"] == "sgd":
                continue
            target = trial["params"]["lora.target"]
            if target in target_data:
                target_data[target].append(pct(trial["value"]))

        # Only plot targets with data
        plot_targets = [t for t in target_order if target_data[t]]
        positions = range(len(plot_targets))
        box_data = [target_data[t] for t in plot_targets]
        colors = [target_colors[t] for t in plot_targets]

        if box_data:
            bp = ax.boxplot(
                box_data,
                positions=list(positions),
                widths=0.5,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white", markersize=5),
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            for i, (vals, color) in enumerate(zip(box_data, colors)):
                if vals:
                    jitter = np.random.uniform(-0.12, 0.12, len(vals))
                    ax.scatter(
                        [i + j for j in jitter],
                        vals,
                        c=color,
                        s=30,
                        edgecolors="white",
                        linewidths=0.5,
                        zorder=5,
                        alpha=0.8,
                    )

        ax.set_xticks(list(positions))
        ax.set_xticklabels(plot_targets, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("METEOR (%)")
        ax.set_title(model_name)

        # Mark the best target
        if best_target in plot_targets:
            best_idx = plot_targets.index(best_target)
            ax.annotate(
                "✓ BEST",
                xy=(best_idx, max(target_data[best_target]) + 0.5),
                fontsize=11,
                fontweight="bold",
                color="#2E7D32",
                ha="center",
            )

    fig.suptitle(
        "LoRA Target Module Impact (excluding SGD outliers)",
        fontsize=16,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "17_lora_target_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  17_lora_target_comparison.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 18: Search space overview / HPO setup diagram
# ─────────────────────────────────────────────────────────────────────


def plot_hpo_search_space_table(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_title(
        "Optuna Hyperparameter Search Space", fontsize=18, fontweight="bold", pad=20
    )

    headers = ["Parameter", "Type", "Range / Choices", "Category"]
    rows = [
        ["Learning Rate", "log-float", "1e-5 → 2e-3", "Training"],
        ["LoRA Rank (r)", "categorical", "{4, 8, 16, 32}", "LoRA"],
        ["LoRA Alpha (α)", "categorical", "{2, 4}", "LoRA"],
        [
            "LoRA Target",
            "categorical",
            "{attention, linear, linear_and_head, ...}",
            "LoRA",
        ],
        ["Optimizer", "categorical", "{Adam, AdamW, SGD}", "Training"],
        ["Weight Decay", "categorical", "{0, 1e-4, 1e-3, 1e-2, 0.1}", "Training"],
        ["LR Scheduler", "categorical", "{none, cosine, step}", "Training"],
        ["Projection Type", "categorical", "{linear, qformer}", "Architecture"],
        ["Num Queries", "categorical", "{4, 8, 16, 32}", "Architecture"],
        ["Num Layers", "categorical", "{1, 2}", "Architecture"],
        ["FFN Dim", "categorical", "{1024, 2048}", "Architecture"],
        ["Max Token Length", "categorical", "{128, 256}", "Data"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.22, 0.12, 0.38, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.9)

    for j in range(4):
        table[0, j].set_facecolor("#263238")
        table[0, j].set_text_props(color="white", fontweight="bold")

    category_colors = {
        "Training": "#E3F2FD",
        "LoRA": "#E8F5E9",
        "Architecture": "#FFF3E0",
        "Data": "#F3E5F5",
    }
    for i, row in enumerate(rows):
        cat = row[3]
        color = category_colors.get(cat, "white")
        for j in range(4):
            table[i + 1, j].set_facecolor(color)

    # Footer text
    ax.text(
        0.5,
        -0.02,
        "Sampler: TPE (Tree-structured Parzen Estimator) · Pruner: Hyperband · "
        "Metric: maximize METEOR · Budget: 30 trials per model",
        ha="center",
        va="top",
        fontsize=10,
        color="#666",
        style="italic",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    fig.savefig(out_dir / "18_hpo_search_space_table.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  18_hpo_search_space_table.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 19: Metrics explanation table
# ─────────────────────────────────────────────────────────────────────


def plot_metrics_explanation_table(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.set_title("Evaluation Metrics Used", fontsize=18, fontweight="bold", pad=20)

    headers = ["Metric", "What It Measures", "Strengths", "Range"]
    rows = [
        [
            "BLEU-1",
            "Unigram (word) overlap\nwith references",
            "Simple, fast;\nlexical precision",
            "[0, 1]",
        ],
        [
            "BLEU-2",
            "Bigram (2-word) overlap\nwith references",
            "Captures short\nphrase matches",
            "[0, 1]",
        ],
        [
            "ROUGE-L",
            "Longest common subsequence\nratio (sentence structure)",
            "Rewards correct\nword ordering",
            "[0, 1]",
        ],
        [
            "METEOR",
            "Consensus-based: morphology\n+ stemming + synonyms",
            "Best human correlation;\nour HPO target",
            "[0, 1]",
        ],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.30, 0.22, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for j in range(4):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(len(rows)):
        for j in range(4):
            table[i + 1, j].set_facecolor("#E3F2FD" if i % 2 == 0 else "#BBDEFB")

    # Highlight METEOR row
    for j in range(4):
        table[4, j].set_facecolor("#C8E6C9")

    fig.tight_layout()
    fig.savefig(out_dir / "19_metrics_explanation_table.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  19_metrics_explanation_table.png")


# ─────────────────────────────────────────────────────────────────────
# PLOT 20: Projection architectures side-by-side
# ─────────────────────────────────────────────────────────────────────


def plot_projection_architectures(out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax in axes:
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 7)
        ax.axis("off")

    def draw_box(
        ax, x, y, w, h, text, color, subtext=None, text_color="white", fontsize=11
    ):
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.12",
            facecolor=color,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)
        dy = 0.15 if subtext else 0
        ax.text(
            x + w / 2,
            y + h / 2 + dy,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=text_color,
        )
        if subtext:
            ax.text(
                x + w / 2,
                y + h / 2 - 0.2,
                subtext,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                style="italic",
            )

    def draw_arrow(ax, x1, y1, x2, y2, text=None):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#333", lw=2),
        )
        if text:
            ax.text((x1 + x2) / 2 + 0.3, (y1 + y2) / 2, text, fontsize=8, color="#555")

    # ─── Left: Linear Projection ───
    ax = axes[0]
    ax.set_title("Linear Projection", fontsize=14, fontweight="bold", pad=15)

    draw_box(ax, 1.5, 5.5, 5, 1.0, "ViT Patches", "#2196F3", subtext="197 × 768")
    draw_arrow(ax, 4, 5.5, 4, 4.8)
    draw_box(
        ax,
        1.5,
        3.5,
        5,
        1.0,
        "nn.Linear(768, D)",
        "#FF9800",
        subtext="D = decoder hidden dim",
    )
    draw_arrow(ax, 4, 3.5, 4, 2.8)
    draw_box(ax, 1.5, 1.5, 5, 1.0, "Visual Prefix", "#4CAF50", subtext="197 × D tokens")

    ax.text(
        4,
        0.7,
        "Simple, fast, minimal parameters\n✓ Best for 0.8B model",
        ha="center",
        fontsize=10,
        color="#2E7D32",
        fontweight="bold",
    )

    # ─── Right: Q-Former Bridge ───
    ax = axes[1]
    ax.set_title(
        "Q-Former Bridge (BLIP-2 inspired)", fontsize=14, fontweight="bold", pad=15
    )

    draw_box(
        ax, 1.5, 5.5, 5, 1.0, "ViT Patches", "#2196F3", subtext="197 × 768 (Key/Value)"
    )
    draw_box(
        ax, 0.2, 3.8, 2.5, 0.8, "Learned\nQueries", "#7E57C2", subtext="N × D", fontsize=9
    )
    draw_arrow(ax, 2.7, 4.2, 3.5, 4.2)
    draw_box(
        ax,
        3.5,
        3.5,
        4.0,
        1.2,
        "Cross-Attention\nDecoder Layer",
        "#FF9800",
        subtext="TransformerDecoder",
    )
    draw_arrow(ax, 5.5, 5.5, 5.5, 4.7)
    draw_arrow(ax, 5.5, 3.5, 5.5, 2.8)
    draw_box(
        ax,
        1.5,
        1.5,
        5,
        1.0,
        "Compressed Prefix",
        "#4CAF50",
        subtext="N × D tokens (N=4..32)",
    )

    ax.text(
        4,
        0.7,
        "Learned compression, more parameters\n✓ Best for 2B model",
        ha="center",
        fontsize=10,
        color="#1565C0",
        fontweight="bold",
    )

    fig.suptitle(
        "Projection Module Architectures", fontsize=18, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_dir / "20_projection_architectures.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓  20_projection_architectures.png")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate Task 2 presentation plots & tables"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/task2_presentation",
        help="Directory to save generated plots",
    )
    parser.add_argument(
        "--outputs-root",
        type=str,
        default="outputs",
        help="Root directory containing experiment outputs",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_root = Path(args.outputs_root)

    # Paths to HPO trial data
    trials_0_8b = outputs_root / "optuna_lora_qwen_0.8b_qformer" / "all_trials.json"
    trials_2b = outputs_root / "optuna_lora_qwen_2b" / "all_trials.json"

    setup_style()
    print(f"\nGenerating Task 2 presentation plots → {out_dir}/\n")

    # T2.1 plots
    plot_multimodal_results_bar(out_dir)
    plot_prompt_comparison_9b(out_dir)
    plot_multimodal_scaling(out_dir)

    # T2.2 architecture diagrams
    plot_architecture_diagram(out_dir)
    plot_lora_targets_diagram(out_dir)
    plot_projection_architectures(out_dir)

    # T2.2 HPO plots
    plot_hpo_search_space_table(out_dir)
    if trials_0_8b.exists():
        plot_hpo_trial_scatter(
            trials_0_8b, "Qwen3.5-0.8B", 26, out_dir, "05_hpo_trial_scatter_0.8b.png"
        )
    if trials_2b.exists():
        plot_hpo_trial_scatter(
            trials_2b, "Qwen3.5-2B", 27, out_dir, "06_hpo_trial_scatter_2b.png"
        )
    plot_hpo_best_params_comparison(out_dir)
    if trials_0_8b.exists() and trials_2b.exists():
        plot_hpo_optimizer_impact(trials_0_8b, trials_2b, out_dir)
        plot_projection_comparison(trials_0_8b, trials_2b, out_dir)
        plot_lora_target_comparison(trials_0_8b, trials_2b, out_dir)

    # T2.2 final eval
    plot_lora_results_bar(out_dir)

    # T2.3 comparison table + analysis
    plot_full_comparison_table(out_dir)
    plot_method_tradeoff_scatter(out_dir)
    plot_metric_profile_radar(out_dir)

    # T2.4 Discussion / supporting
    plot_qualitative_examples(out_dir)
    plot_challenges_solutions_table(out_dir)
    plot_metrics_explanation_table(out_dir)

    print(f"\n✅ Done! {len(list(out_dir.glob('*.png')))} plots saved to {out_dir}/")
    print("\nSlide mapping:")
    print("  Slide 5  (Metrics)        → 19_metrics_explanation_table.png")
    print("  Slide 7  (Prompts)        → 02_prompt_comparison_9b.png")
    print(
        "  Slide 8  (T2.1 Results)   → 01_multimodal_results_bar.png, 16_multimodal_scaling.png"
    )
    print("  Slide 10 (Architecture)   → 03_architecture_diagram.png")
    print("  Slide 11 (Projection)     → 20_projection_architectures.png")
    print("  Slide 12 (LoRA Targets)   → 04_lora_targets_diagram.png")
    print("  Slide 13 (HPO Setup)      → 18_hpo_search_space_table.png")
    print("  Slide 14 (HPO 0.8B)       → 05_hpo_trial_scatter_0.8b.png")
    print(
        "  Slide 15 (HPO 2B)         → 06_hpo_trial_scatter_2b.png, 07_hpo_best_params_comparison.png"
    )
    print(
        "  Slide 15 (HPO Analysis)   → 13_hpo_optimizer_impact.png, 14_projection_comparison.png, 17_lora_target_comparison.png"
    )
    print("  Slide 16 (LoRA Eval)      → 08_lora_results_bar.png")
    print(
        "  Slide 17 (Full Table)     → 09_full_comparison_table.png, 10_method_tradeoff_scatter.png"
    )
    print("  Slide 18 (Discussion)     → 11_metric_profile_radar.png")
    print("  Slide 19 (Qualitative)    → 12_qualitative_examples.png")
    print("  Slide 20 (Challenges)     → 15_challenges_solutions_table.png")


if __name__ == "__main__":
    main()
