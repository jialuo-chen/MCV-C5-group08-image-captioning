"""Generate presentation-quality plots for Task 2.1 — Direct Multimodal Evaluation.

Usage:
    python src/generate_t21_plots.py [--output-dir outputs/t21_plots]

Produces 7 plots:
    t21_01_prompt_comparison.png        — Prompt engineering (9B, 3 prompts)
    t21_02_prompt_inference_time.png    — Prompt impact on inference time
    t21_03_model_scaling_metrics.png    — Model size vs metrics (no thinking)
    t21_04_model_scaling_time.png       — Model size vs inference time
    t21_05_thinking_vs_no_thinking.png  — Thinking mode metric comparison
    t21_06_thinking_time_penalty.png    — Thinking mode time penalty
    t21_07_summary_table.png           — Full T2.1 summary table
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────

METRICS = ["bleu1", "bleu2", "rougeL", "meteor"]
METRIC_LABELS = {"bleu1": "BLEU-1", "bleu2": "BLEU-2", "rougeL": "ROUGE-L", "meteor": "METEOR"}

# Prompt engineering (9B, no thinking)
PROMPT_RESULTS = {
    "Describe this image.": {
        "bleu1": 0.1401, "bleu2": 0.0699, "rougeL": 0.1507, "meteor": 0.2720, "ms": 1632.6,
    },
    "Describe this image briefly.": {
        "bleu1": 0.1710, "bleu2": 0.0902, "rougeL": 0.1655, "meteor": 0.3028, "ms": 1646.6,
    },
    "Describe this image in a short sentence.": {
        "bleu1": 0.6115, "bleu2": 0.4117, "rougeL": 0.4744, "meteor": 0.4911, "ms": 1002.3,
    },
}

# Model sizes — no thinking, best prompt
NO_THINK = {
    "0.8B": {"bleu1": 0.5371, "bleu2": 0.3352, "rougeL": 0.4169, "meteor": 0.4580, "ms": 481.0},
    "2B":   {"bleu1": 0.5594, "bleu2": 0.3506, "rougeL": 0.4243, "meteor": 0.4755, "ms": 771.2},
    "4B":   {"bleu1": 0.5192, "bleu2": 0.3191, "rougeL": 0.4029, "meteor": 0.4752, "ms": 1235.8},
    "9B":   {"bleu1": 0.6115, "bleu2": 0.4117, "rougeL": 0.4744, "meteor": 0.4911, "ms": 1002.3},
}

# Thinking mode results
THINK = {
    "0.8B": {"bleu1": 0.2920, "bleu2": 0.1764, "rougeL": 0.3996, "meteor": 0.4420, "ms": 10722.9},
    "2B":   {"bleu1": 0.4137, "bleu2": 0.2533, "rougeL": 0.4028, "meteor": 0.4698, "ms": 8004.5},
    "4B":   {"bleu1": 0.3516, "bleu2": 0.2180, "rougeL": 0.4048, "meteor": 0.4753, "ms": 12939.1},
}

# ─────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────

# Presentation palette
PAL = {
    "blue":    "#2563EB",
    "sky":     "#38BDF8",
    "green":   "#22C55E",
    "emerald": "#10B981",
    "amber":   "#F59E0B",
    "orange":  "#F97316",
    "red":     "#EF4444",
    "rose":    "#FB7185",
    "purple":  "#8B5CF6",
    "slate":   "#64748B",
    "gray":    "#94A3B8",
}

SIZE_COLORS = {
    "0.8B": PAL["sky"],
    "2B":   PAL["emerald"],
    "4B":   PAL["amber"],
    "9B":   PAL["purple"],
}

def pct(v: float) -> float:
    return v * 100.0


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#E2E8F0",
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFBFC",
        "axes.edgecolor": "#CBD5E1",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E2E8F0",
        "grid.alpha": 0.7,
        "grid.linewidth": 0.8,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.3,
        "savefig.bbox": "tight",
    })


def add_bar_labels(ax, bars, fmt="{:.1f}", fontsize=10, offset=0.5, bold=False):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + offset,
            fmt.format(h),
            ha="center", va="bottom", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color="#334155",
        )


# ─────────────────────────────────────────────────────────────────────
# Plot 1: Prompt Comparison
# ─────────────────────────────────────────────────────────────────────

def plot_prompt_comparison(out_dir: Path) -> None:
    prompts = list(PROMPT_RESULTS.keys())
    labels = [
        '"Describe this image."',
        '"Describe this image briefly."',
        '"...in a short sentence."',
    ]
    colors = [PAL["red"], PAL["amber"], PAL["green"]]

    n_prompts = len(prompts)
    n_metrics = len(METRICS)
    x = np.arange(n_metrics)
    width = 0.24

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (prompt, label, color) in enumerate(zip(prompts, labels, colors)):
        vals = [pct(PROMPT_RESULTS[prompt][m]) for m in METRICS]
        bars = ax.bar(
            x + (i - (n_prompts - 1) / 2) * width,
            vals, width * 0.88, label=label,
            color=color, edgecolor="white", linewidth=1.2,
            zorder=3,
        )
        add_bar_labels(ax, bars, fontsize=10, bold=(i == 2))

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 72)
    ax.set_title("Prompt Engineering Impact — Qwen3.5-9B (no thinking)", pad=15)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "t21_01_prompt_comparison.png")
    plt.close(fig)
    print("  -> t21_01_prompt_comparison.png")


# ─────────────────────────────────────────────────────────────────────
# Plot 2: Prompt Inference Time
# ─────────────────────────────────────────────────────────────────────

def plot_prompt_inference_time(out_dir: Path) -> None:
    prompts = list(PROMPT_RESULTS.keys())
    labels = [
        '"Describe this image."',
        '"Describe this image briefly."',
        '"...in a short sentence."',
    ]
    colors = [PAL["red"], PAL["amber"], PAL["green"]]
    ms_vals = [PROMPT_RESULTS[p]["ms"] for p in prompts]

    fig, ax = plt.subplots(figsize=(12, 5))

    bars = ax.barh(
        range(len(prompts)), ms_vals, color=colors,
        edgecolor="white", height=0.55, linewidth=1.2, zorder=3,
    )
    ax.set_yticks(range(len(prompts)))
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel("Inference Time (ms / image)")
    ax.set_title("Prompt Impact on Inference Time — Qwen3.5-9B", pad=15)
    ax.invert_yaxis()
    ax.set_xlim(0, 2000)

    for bar, ms in zip(bars, ms_vals):
        ax.text(
            bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
            f"{ms:.0f} ms", va="center", fontsize=13, fontweight="bold", color="#334155",
        )

    fig.tight_layout()
    fig.savefig(out_dir / "t21_02_prompt_inference_time.png")
    plt.close(fig)
    print("  -> t21_02_prompt_inference_time.png")


# ─────────────────────────────────────────────────────────────────────
# Plot 3: Model Scaling — Metrics
# ─────────────────────────────────────────────────────────────────────

def plot_model_scaling_metrics(out_dir: Path) -> None:
    sizes = list(NO_THINK.keys())
    n_sizes = len(sizes)
    n_metrics = len(METRICS)
    x = np.arange(n_metrics)
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, size in enumerate(sizes):
        vals = [pct(NO_THINK[size][m]) for m in METRICS]
        bars = ax.bar(
            x + (i - (n_sizes - 1) / 2) * width,
            vals, width * 0.88,
            label=f"Qwen3.5-{size}",
            color=SIZE_COLORS[size], edgecolor="white", linewidth=1.2, zorder=3,
        )
        add_bar_labels(ax, bars, fontsize=9, offset=0.4, bold=(size == "9B"))

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 72)
    ax.set_title('Model Size Comparison — "Describe this image in a short sentence." (no thinking)', pad=15)
    ax.legend(loc="upper left", frameon=True, ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / "t21_03_model_scaling_metrics.png")
    plt.close(fig)
    print("  -> t21_03_model_scaling_metrics.png")


# ─────────────────────────────────────────────────────────────────────
# Plot 4: Model Scaling — Inference Time
# ─────────────────────────────────────────────────────────────────────

def plot_model_scaling_time(out_dir: Path) -> None:
    sizes = list(NO_THINK.keys())
    ms_vals = [NO_THINK[s]["ms"] for s in sizes]
    throughput = [1000.0 / ms for ms in ms_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: inference time
    bars1 = ax1.bar(
        sizes, ms_vals,
        color=[SIZE_COLORS[s] for s in sizes],
        edgecolor="white", linewidth=1.2, width=0.55, zorder=3,
    )
    ax1.set_ylabel("Inference Time (ms / image)")
    ax1.set_xlabel("Model Size")
    ax1.set_title("Inference Time", pad=12)
    ax1.set_ylim(0, max(ms_vals) * 1.18)
    for bar, ms in zip(bars1, ms_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            f"{ms:.0f}", ha="center", va="bottom", fontsize=12, fontweight="bold", color="#334155",
        )

    # Right: throughput
    bars2 = ax2.bar(
        sizes, throughput,
        color=[SIZE_COLORS[s] for s in sizes],
        edgecolor="white", linewidth=1.2, width=0.55, zorder=3,
    )
    ax2.set_ylabel("Throughput (images / sec)")
    ax2.set_xlabel("Model Size")
    ax2.set_title("Throughput", pad=12)
    ax2.set_ylim(0, max(throughput) * 1.18)
    for bar, tp in zip(bars2, throughput):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{tp:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold", color="#334155",
        )

    fig.suptitle("Inference Cost vs Model Size (no thinking)", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=4)
    fig.savefig(out_dir / "t21_04_model_scaling_time.png")
    plt.close(fig)
    print("  -> t21_04_model_scaling_time.png")


# ─────────────────────────────────────────────────────────────────────
# Plot 5: Thinking vs No Thinking — Metrics
# ─────────────────────────────────────────────────────────────────────

def plot_thinking_vs_no_thinking(out_dir: Path) -> None:
    sizes = ["0.8B", "2B", "4B"]
    color_no = PAL["blue"]
    color_yes = PAL["rose"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6.5), sharey=False)

    for col, metric in enumerate(METRICS):
        ax = axes[col]
        x = np.arange(len(sizes))
        width = 0.32

        vals_no = [pct(NO_THINK[s][metric]) for s in sizes]
        vals_yes = [pct(THINK[s][metric]) for s in sizes]

        bars_no = ax.bar(x - width / 2, vals_no, width * 0.88, label="No Thinking",
                         color=color_no, edgecolor="white", linewidth=1.2, zorder=3)
        bars_yes = ax.bar(x + width / 2, vals_yes, width * 0.88, label="Thinking",
                          color=color_yes, edgecolor="white", linewidth=1.2, zorder=3)

        # Value labels
        for bar in list(bars_no) + list(bars_yes):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9, color="#334155")

        # Delta annotations
        for j, s in enumerate(sizes):
            delta = pct(THINK[s][metric]) - pct(NO_THINK[s][metric])
            sign = "+" if delta >= 0 else ""
            color_delta = PAL["green"] if delta >= 0 else PAL["red"]
            y_pos = max(vals_no[j], vals_yes[j]) + 3
            ax.text(x[j], y_pos, f"{sign}{delta:.1f}", ha="center", fontsize=9,
                    fontweight="bold", color=color_delta)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}" for s in sizes], fontsize=12)
        ax.set_xlabel("Model Size", fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=15, fontweight="bold")
        ax.set_ylabel("Score (%)" if col == 0 else "")

        if col == 0:
            ax.legend(loc="upper right", fontsize=10)

    # Adjust y-limits per subplot for better readability
    for col, metric in enumerate(METRICS):
        ax = axes[col]
        all_vals = [pct(NO_THINK[s][metric]) for s in sizes] + [pct(THINK[s][metric]) for s in sizes]
        ymin = max(0, min(all_vals) - 10)
        ymax = max(all_vals) + 8
        ax.set_ylim(ymin, ymax)

    fig.suptitle("Thinking Mode vs No Thinking — Metric Comparison", fontsize=18, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "t21_05_thinking_vs_no_thinking.png")
    plt.close(fig)
    print("  -> t21_05_thinking_vs_no_thinking.png")


# ─────────────────────────────────────────────────────────────────────
# Plot 6: Thinking Time Penalty
# ─────────────────────────────────────────────────────────────────────

def plot_thinking_time_penalty(out_dir: Path) -> None:
    sizes = ["0.8B", "2B", "4B"]
    y = np.arange(len(sizes))
    height = 0.32

    ms_no = [NO_THINK[s]["ms"] for s in sizes]
    ms_yes = [THINK[s]["ms"] for s in sizes]
    multipliers = [ms_yes[i] / ms_no[i] for i in range(len(sizes))]

    fig, ax = plt.subplots(figsize=(14, 5.5))

    bars_no = ax.barh(y - height / 2, ms_no, height * 0.88,
                       label="No Thinking", color=PAL["blue"],
                       edgecolor="white", linewidth=1.2, zorder=3)
    bars_yes = ax.barh(y + height / 2, ms_yes, height * 0.88,
                        label="Thinking", color=PAL["rose"],
                        edgecolor="white", linewidth=1.2, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([f"Qwen3.5-{s}" for s in sizes], fontsize=14)
    ax.set_xlabel("Inference Time (ms / image)")
    ax.set_title("Thinking Mode — Inference Time Penalty", pad=15)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=12)

    # Time labels + multiplier badges
    for i, (bar_n, bar_y, mult) in enumerate(zip(bars_no, bars_yes, multipliers)):
        ax.text(bar_n.get_width() + 100, bar_n.get_y() + bar_n.get_height() / 2,
                f"{ms_no[i]:.0f} ms", va="center", fontsize=11, color="#334155")
        ax.text(bar_y.get_width() + 100, bar_y.get_y() + bar_y.get_height() / 2,
                f"{ms_yes[i]:.0f} ms", va="center", fontsize=11, color="#334155")

        # Multiplier label
        badge_x = max(ms_yes[i], ms_no[i]) + 1800
        badge_y = y[i]
        ax.text(badge_x, badge_y, f"{mult:.0f}x",
                ha="center", va="center", fontsize=14, fontweight="bold",
                color=PAL["red"])

    ax.set_xlim(0, max(ms_yes) + 3500)

    fig.tight_layout()
    fig.savefig(out_dir / "t21_06_thinking_time_penalty.png")
    plt.close(fig)
    print("  -> t21_06_thinking_time_penalty.png")


# ─────────────────────────────────────────────────────────────────────
# Plot 7: Summary Table
# ─────────────────────────────────────────────────────────────────────

def plot_summary_table(out_dir: Path) -> None:
    # Build rows: [Model, Mode/Prompt, BLEU-1, BLEU-2, ROUGE-L, METEOR, ms/img]
    rows = []

    # Section 1: Prompt engineering (9B)
    for prompt, data in PROMPT_RESULTS.items():
        short = prompt.replace("Describe this image", "...").replace(".", "")
        if prompt == "Describe this image.":
            short = '"Describe this image."'
        elif prompt == "Describe this image briefly.":
            short = '"...briefly."'
        else:
            short = '"...in a short sentence."'
        rows.append([
            "Qwen3.5-9B", short,
            f"{pct(data['bleu1']):.1f}", f"{pct(data['bleu2']):.1f}",
            f"{pct(data['rougeL']):.1f}", f"{pct(data['meteor']):.1f}",
            f"{data['ms']:.0f}",
        ])

    # Separator
    rows.append(["", "", "", "", "", "", ""])

    # Section 2: No thinking (best prompt)
    for size, data in NO_THINK.items():
        rows.append([
            f"Qwen3.5-{size}", "No Thinking",
            f"{pct(data['bleu1']):.1f}", f"{pct(data['bleu2']):.1f}",
            f"{pct(data['rougeL']):.1f}", f"{pct(data['meteor']):.1f}",
            f"{data['ms']:.0f}",
        ])

    # Separator
    rows.append(["", "", "", "", "", "", ""])

    # Section 3: Thinking mode
    for size, data in THINK.items():
        rows.append([
            f"Qwen3.5-{size}", "Thinking",
            f"{pct(data['bleu1']):.1f}", f"{pct(data['bleu2']):.1f}",
            f"{pct(data['rougeL']):.1f}", f"{pct(data['meteor']):.1f}",
            f"{data['ms']:.0f}",
        ])

    col_labels = ["Model", "Mode / Prompt", "BLEU-1", "BLEU-2", "ROUGE-L", "METEOR", "ms/img"]

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.axis("off")
    ax.set_title("Task 2.1 — Full Results Summary", fontsize=22, fontweight="bold", pad=30)

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 2.0)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1E293B")
        cell.set_text_props(color="white", fontweight="bold", fontsize=12)
        cell.set_edgecolor("#334155")

    # Find best values for highlighting
    metric_cols = {2: "bleu1", 3: "bleu2", 4: "rougeL", 5: "meteor"}
    best_vals = {}
    for col_idx, metric_key in metric_cols.items():
        vals = []
        for i, row in enumerate(rows):
            if row[0] and row[col_idx]:
                try:
                    vals.append((i, float(row[col_idx])))
                except ValueError:
                    pass
        if vals:
            best_i, best_v = max(vals, key=lambda x: x[1])
            best_vals[col_idx] = (best_i, best_v)

    # Style data rows
    section_colors = {
        "prompt": "#F0F9FF",  # blue tint
        "no_think": "#F0FDF4",  # green tint
        "think": "#FFF1F2",  # rose tint
        "sep": "#FFFFFF",
    }

    for i, row in enumerate(rows):
        row_idx = i + 1  # +1 for header
        if not row[0]:  # separator
            for j in range(len(col_labels)):
                table[row_idx, j].set_facecolor("white")
                table[row_idx, j].set_edgecolor("white")
                table[row_idx, j].set_height(0.02)
            continue

        if "Thinking" in row[1]:
            bg = section_colors["think"]
        elif any(p in row[1] for p in ['"', "..."]):
            bg = section_colors["prompt"]
        else:
            bg = section_colors["no_think"]

        for j in range(len(col_labels)):
            cell = table[row_idx, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#E2E8F0")

            # Highlight best values
            if j in best_vals and best_vals[j][0] == i:
                cell.set_text_props(fontweight="bold", color=PAL["green"])
                cell.set_facecolor("#DCFCE7")

    # Highlight best ms/img (lowest, excluding separators)
    ms_vals = []
    for i, row in enumerate(rows):
        if row[0] and row[6]:
            try:
                ms_vals.append((i, float(row[6])))
            except ValueError:
                pass
    if ms_vals:
        best_ms_i, _ = min(ms_vals, key=lambda x: x[1])
        cell = table[best_ms_i + 1, 6]
        cell.set_text_props(fontweight="bold", color=PAL["green"])
        cell.set_facecolor("#DCFCE7")

    fig.tight_layout()
    fig.savefig(out_dir / "t21_07_summary_table.png")
    plt.close(fig)
    print("  -> t21_07_summary_table.png")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate T2.1 presentation plots")
    parser.add_argument("--output-dir", type=str, default="outputs/t21_plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    print(f"\nGenerating T2.1 plots -> {out_dir}/\n")

    plot_prompt_comparison(out_dir)
    plot_prompt_inference_time(out_dir)
    plot_model_scaling_metrics(out_dir)
    plot_model_scaling_time(out_dir)
    plot_thinking_vs_no_thinking(out_dir)
    plot_thinking_time_penalty(out_dir)
    plot_summary_table(out_dir)

    n_plots = len(list(out_dir.glob("t21_*.png")))
    print(f"\nDone! {n_plots} plots saved to {out_dir}/")

    print("\nSlide mapping:")
    print("  Slide 4  (Prompt Eng.)     -> t21_01_prompt_comparison.png")
    print("  Slide 5  (Prompt Time)     -> t21_02_prompt_inference_time.png")
    print("  Slide 6  (Size Metrics)    -> t21_03_model_scaling_metrics.png")
    print("  Slide 7  (Size Time)       -> t21_04_model_scaling_time.png")
    print("  Slide 8  (Thinking Metr.)  -> t21_05_thinking_vs_no_thinking.png")
    print("  Slide 9  (Thinking Time)   -> t21_06_thinking_time_penalty.png")
    print("  Slide 10 (Summary Table)   -> t21_07_summary_table.png")


if __name__ == "__main__":
    main()
