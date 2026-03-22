from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

PHASE_A = ["A0", "A1", "A2", "A3", "A4", "A5"]
PHASE_B = ["B1", "B2", "B3", "B4", "B5", "B6"]
PHASE_GRU = ["GRU1", "GRU2", "GRU3", "GRU4", "GRU5"]
ALL_EXPERIMENTS = PHASE_A + PHASE_B

METRICS = ["bleu1", "bleu2", "rougeL", "meteor"]
METRIC_LABELS = {
    "bleu1": "BLEU-1",
    "bleu2": "BLEU-2",
    "rougeL": "ROUGE-L",
    "meteor": "METEOR",
}

PHASE_A_LABELS = {
    "A0": "Baseline\n(R18+GRU+char)",
    "A1": "R18\u2192R50\n(encoder)",
    "A2": "GRU\u2192LSTM\n(decoder)",
    "A3": "char\u2192subword\n(tokenizer)",
    "A4": "char\u2192word\n(tokenizer)",
    "A5": "+Bahdanau\n(attention)",
}

PHASE_B_LABELS = {
    "B1": "char",
    "B2": "word",
    "B3": "subword",
    "B4": "char + attn",
    "B5": "word + attn",
    "B6": "subword + attn",
}

PHASE_GRU_LABELS = {
    "GRU1": "word",
    "GRU2": "subword",
    "GRU3": "char + attn",
    "GRU4": "word + attn",
    "GRU5": "subword + attn",
}

EXP_LABELS_SHORT = {
    "A0": "Baseline (R18+GRU+char)",
    "A1": "R18\u2192R50",
    "A2": "GRU\u2192LSTM",
    "A3": "char\u2192subword",
    "A4": "char\u2192word",
    "A5": "+Bahdanau attn",
    "B1": "char",
    "B2": "word",
    "B3": "subword",
    "B4": "char + attn",
    "B5": "word + attn",
    "B6": "subword + attn",
}

FINAL_METRICS: dict[str, dict[str, float]] = {
    "A0": {"bleu1": 0.4505, "bleu2": 0.2242, "rougeL": 0.3314, "meteor": 0.2836},
    "A1": {"bleu1": 0.4543, "bleu2": 0.2526, "rougeL": 0.3342, "meteor": 0.3086},
    "A2": {"bleu1": 0.3882, "bleu2": 0.1759, "rougeL": 0.2721, "meteor": 0.2817},
    "A3": {"bleu1": 0.5588, "bleu2": 0.3078, "rougeL": 0.3656, "meteor": 0.3132},
    "A4": {"bleu1": 0.5046, "bleu2": 0.2853, "rougeL": 0.3640, "meteor": 0.3129},
    "A5": {"bleu1": 0.4589, "bleu2": 0.2657, "rougeL": 0.3519, "meteor": 0.3195},
    "B1": {"bleu1": 0.5289, "bleu2": 0.3064, "rougeL": 0.3600, "meteor": 0.3283},
    "B2": {"bleu1": 0.5585, "bleu2": 0.3465, "rougeL": 0.4070, "meteor": 0.3554},
    "B3": {"bleu1": 0.6205, "bleu2": 0.3795, "rougeL": 0.4104, "meteor": 0.3622},
    "B4": {"bleu1": 0.5000, "bleu2": 0.3025, "rougeL": 0.3740, "meteor": 0.3454},
    "B5": {"bleu1": 0.5609, "bleu2": 0.3496, "rougeL": 0.4131, "meteor": 0.3696},
    "B6": {"bleu1": 0.6088, "bleu2": 0.3782, "rougeL": 0.4116, "meteor": 0.3718},
    "GRU1": {"bleu1": 0.5554, "bleu2": 0.3400, "rougeL": 0.4037, "meteor": 0.3513},
    "GRU2": {"bleu1": 0.6105, "bleu2": 0.3685, "rougeL": 0.4017, "meteor": 0.3580},
    "GRU3": {"bleu1": 0.4544, "bleu2": 0.2765, "rougeL": 0.3689, "meteor": 0.3464},
    "GRU4": {"bleu1": 0.5419, "bleu2": 0.3389, "rougeL": 0.4095, "meteor": 0.3579},
    "GRU5": {"bleu1": 0.5999, "bleu2": 0.3716, "rougeL": 0.4124, "meteor": 0.3707},
}

PARAMS_M: dict[str, float] = {
    "A0": 13.0,
    "A1": 26.0,
    "A2": 13.0,
    "A3": 17.0,
    "A4": 22.0,
    "A5": 14.0,
    "B1": 27.8,
    "B2": 38.0,
    "B3": 31.8,
    "B4": 32.6,
    "B5": 42.8,
    "B6": 36.7,
    "GRU1": 36.4,
    "GRU2": 30.2,
    "GRU3": 30.0,
    "GRU4": 40.2,
    "GRU5": 34.0,
}

FLOPS_G: dict[str, float] = {
    "A0": 3.63,
    "A1": 8.18,
    "A2": 3.63,
    "A3": 3.71,
    "A4": 3.81,
    "A5": 7.51,
    "B1": 8.18,
    "B2": 8.37,
    "B3": 8.26,
    "B4": 17.34,
    "B5": 17.53,
    "B6": 17.41,
    "GRU1": 8.37,
    "GRU2": 8.25,
    "GRU3": 17.33,
    "GRU4": 17.53,
    "GRU5": 17.41,
}

COLORS: dict[str, str] = {
    "A0": "#6C757D",  # gray — baseline
    "A1": "#2196F3",  # blue — encoder
    "A2": "#F44336",  # red — decoder
    "A3": "#4CAF50",  # green — subword
    "A4": "#8BC34A",  # light green — word
    "A5": "#FF9800",  # orange — attention
    "B1": "#1565C0",  # dark blue
    "B2": "#2E7D32",  # dark green
    "B3": "#00897B",  # teal
    "B4": "#E65100",  # dark orange
    "B5": "#6A1B9A",  # purple
    "B6": "#C62828",  # dark red — best
    "GRU1": "#42A5F5",  # blue — word
    "GRU2": "#66BB6A",  # green — subword
    "GRU3": "#FFA726",  # orange — char+attn
    "GRU4": "#AB47BC",  # purple — word+attn
    "GRU5": "#EF5350",  # red — subword+attn
}

TOK_TYPE: dict[str, str] = {
    "A0": "char",
    "A1": "char",
    "A2": "char",
    "A3": "subword",
    "A4": "word",
    "A5": "char",
    "B1": "char",
    "B2": "word",
    "B3": "subword",
    "B4": "char",
    "B5": "word",
    "B6": "subword",
}


def pct(v: float) -> float:
    """Convert 0-1 metric to percentage."""
    return v * 100.0


def setup_style() -> None:
    """Configure matplotlib for presentation-quality output."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
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
    # Note: savefig.bbox_inches is not a valid rcParam — we pass
    # bbox_inches="tight" directly in each fig.savefig() call instead.


def load_all_data(outputs_dir: Path) -> dict[str, dict]:
    """Load experiment_log.json for every experiment. Returns {exp_id: data}."""
    data = {}
    for exp in ALL_EXPERIMENTS:
        log_path = outputs_dir / exp / "experiment_log.json"
        if log_path.exists():
            with open(log_path) as f:
                data[exp] = json.load(f)
        else:
            print(f"  [WARN] Missing {log_path}")
    return data


def plot_phase_a_ablation_bars(data: dict, out_dir: Path) -> None:
    exps = PHASE_A
    n_metrics = len(METRICS)
    n_exps = len(exps)
    x = np.arange(n_metrics)
    width = 0.12
    offsets = np.arange(n_exps) - (n_exps - 1) / 2

    fig, ax = plt.subplots(figsize=(14, 7))

    baseline_vals = [pct(FINAL_METRICS["A0"][m]) for m in METRICS]

    for i, exp in enumerate(exps):
        vals = [pct(FINAL_METRICS[exp][m]) for m in METRICS]
        bars = ax.bar(
            x + offsets[i] * width,
            vals,
            width * 0.9,
            label=PHASE_A_LABELS[exp],
            color=COLORS[exp],
            edgecolor="white",
            linewidth=0.5,
        )
        if exp != "A0":
            for j, (bar, val) in enumerate(zip(bars, vals)):
                delta = val - baseline_vals[j]
                sign = "+" if delta >= 0 else ""
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{sign}{delta:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#2E7D32" if delta > 0 else "#C62828",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=14)
    ax.set_ylabel("Score (%)")
    ax.set_title(
        "Phase A — One Change at a Time\n(each changes ONE component from baseline R18+GRU+char)",
        fontsize=16,
    )
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0, 70)

    fig.tight_layout()
    fig.savefig(out_dir / "01_phase_a_ablation_bars.png", bbox_inches="tight")
    plt.close(fig)
    print("  [1/10] 01_phase_a_ablation_bars.png")


def plot_ablation_delta_heatmap(data: dict, out_dir: Path) -> None:
    ablations = ["A1", "A2", "A3", "A4", "A5"]
    labels_row = [
        "Encoder → ResNet-50",
        "Decoder → LSTM",
        "Tokenizer → Subword",
        "Tokenizer → Word",
        "+ Bahdanau Attention",
    ]
    labels_col = [METRIC_LABELS[m] for m in METRICS]

    matrix = np.zeros((len(ablations), len(METRICS)))
    for i, exp in enumerate(ablations):
        for j, m in enumerate(METRICS):
            matrix[i, j] = pct(FINAL_METRICS[exp][m] - FINAL_METRICS["A0"][m])

    vmax = max(abs(matrix.min()), abs(matrix.max()))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="+.1f",
        cmap="RdYlGn",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=labels_col,
        yticklabels=labels_row,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Δ from Baseline (pp)"},
        ax=ax,
    )
    ax.set_title("Impact of Each Component (Δ percentage points from Baseline)")
    fig.tight_layout()
    fig.savefig(out_dir / "02_ablation_delta_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  [2/10] 02_ablation_delta_heatmap.png")


def plot_phase_b_combined_bars(data: dict, out_dir: Path) -> None:
    exps = ["A0"] + PHASE_B
    n_metrics = len(METRICS)
    n_exps = len(exps)
    x = np.arange(n_metrics)
    width = 0.11
    offsets = np.arange(n_exps) - (n_exps - 1) / 2

    fig, ax = plt.subplots(figsize=(16, 7))

    labels = {}
    labels["A0"] = "A0: Baseline (R18+GRU+char)"
    for exp in PHASE_B:
        labels[exp] = f"{exp}: {PHASE_B_LABELS[exp]}"

    for i, exp in enumerate(exps):
        vals = [pct(FINAL_METRICS[exp][m]) for m in METRICS]
        edgecolor = "gold" if exp == "B6" else "white"
        linewidth = 2.0 if exp == "B6" else 0.5
        ax.bar(
            x + offsets[i] * width,
            vals,
            width * 0.9,
            label=labels[exp],
            color=COLORS[exp],
            edgecolor=edgecolor,
            linewidth=linewidth,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=14)
    ax.set_ylabel("Score (%)")
    ax.set_title(
        "Phase B — Combined Architectures (all use R50+LSTM, gold border = best)",
        fontsize=16,
    )
    ax.legend(loc="upper left", ncol=4)
    ax.set_ylim(0, 75)

    fig.tight_layout()
    fig.savefig(out_dir / "03_phase_b_combined_bars.png", bbox_inches="tight")
    plt.close(fig)
    print("  [3/10] 03_phase_b_combined_bars.png")


def plot_baseline_to_best_progression(data: dict, out_dir: Path) -> None:
    path = ["A0", "A1", "B1", "B3", "B6"]
    step_labels = [
        "Baseline\n(R18+GRU+char)",
        "+ResNet-50\n(encoder)",
        "+LSTM\n(decoder)",
        "+Subword\n(tokenizer)",
        "+Attention\n(Bahdanau)",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=False)

    all_vals = [pct(FINAL_METRICS[e][m]) for m in METRICS for e in path]
    y_min = min(all_vals)
    y_max = max(all_vals)
    y_pad = (y_max - y_min) * 0.12
    global_ylim = (y_min - y_pad - 2, y_max + y_pad + 2)

    for ax_i, m in enumerate(METRICS):
        ax = axes[ax_i]
        vals = [pct(FINAL_METRICS[e][m]) for e in path]

        ax.plot(
            range(len(path)),
            vals,
            "o-",
            color="#1565C0",
            markersize=10,
            linewidth=2.5,
            zorder=3,
        )

        ax.fill_between(range(len(path)), vals, alpha=0.1, color="#1565C0")

        for j in range(1, len(path)):
            delta = vals[j] - vals[j - 1]
            sign = "+" if delta >= 0 else ""
            color = "#2E7D32" if delta >= 0 else "#C62828"
            y_pos = (vals[j] + vals[j - 1]) / 2
            ax.annotate(
                f"{sign}{delta:.1f}pp",
                xy=(j - 0.5, y_pos),
                fontsize=9,
                fontweight="bold",
                color=color,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.8,
                ),
            )

        ax.text(
            0,
            vals[0] - 1.5,
            f"{vals[0]:.1f}%",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#6C757D",
        )
        ax.text(
            len(path) - 1,
            vals[-1] + 1.0,
            f"{vals[-1]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#C62828",
        )

        ax.set_ylim(global_ylim)
        ax.set_xticks(range(len(path)))
        ax.set_xticklabels(step_labels, fontsize=9, ha="center")
        ax.set_title(METRIC_LABELS[m], fontsize=14, fontweight="bold")
        ax.set_ylabel("Score (%)" if ax_i == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", alpha=0.0)

    fig.suptitle(
        "Progression from Baseline to Best Model",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "04_baseline_to_best_progression.png", bbox_inches="tight")
    plt.close(fig)
    print("  [4/10] 04_baseline_to_best_progression.png")


def plot_radar_top_models(data: dict, out_dir: Path) -> None:
    models = ["A0", "A3", "B3", "B5", "B6"]
    labels = [
        "A0: Baseline (R18+GRU+char)",
        "A3: char\u2192subword",
        "B3: R50+LSTM+subword",
        "B5: R50+LSTM+word+attn",
        "B6: R50+LSTM+subword+attn",
    ]

    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for exp, label in zip(models, labels):
        vals = [pct(FINAL_METRICS[exp][m]) for m in METRICS]
        vals += vals[:1]
        ax.plot(
            angles,
            vals,
            "o-",
            linewidth=2,
            label=label,
            color=COLORS[exp],
            markersize=6,
        )
        ax.fill(angles, vals, alpha=0.1, color=COLORS[exp])

    ax.set_thetagrids(
        [a * 180 / np.pi for a in angles[:-1]],
        [METRIC_LABELS[m] for m in METRICS],
        fontsize=13,
    )
    ax.set_ylim(0, 70)
    ax.set_rgrids(
        [10, 20, 30, 40, 50, 60],
        labels=["10%", "20%", "30%", "40%", "50%", "60%"],
        fontsize=9,
    )
    ax.set_title("Multi-Metric Comparison — Top Models", pad=30, fontsize=18)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=10,
        frameon=True,
        fancybox=True,
    )

    fig.tight_layout()
    fig.savefig(out_dir / "05_radar_top_models.png", bbox_inches="tight")
    plt.close(fig)
    print("  [5/10] 05_radar_top_models.png")


def _plot_training_curves(
    data: dict,
    exps: list[str],
    title: str,
    filename: str,
    out_dir: Path,
) -> None:
    char_exps = [e for e in exps if TOK_TYPE[e] == "char" and e in data]
    text_exps = [e for e in exps if TOK_TYPE[e] in ("subword", "word") and e in data]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey="row")

    for col, (group, group_label) in enumerate(
        [
            (char_exps, "Char-based"),
            (text_exps, "Subword / Word"),
        ]
    ):
        for exp in group:
            epochs_data = data[exp]["training"]["epochs"]
            ep = [e["epoch"] for e in epochs_data]
            val_loss = [e["val_loss"] for e in epochs_data]
            bleu1 = [pct(e["bleu1"]) for e in epochs_data]
            best_ep = data[exp]["summary"]["best_epoch"]

            ax_loss = axes[0][col]
            ax_loss.plot(
                ep,
                val_loss,
                "-",
                color=COLORS[exp],
                label=EXP_LABELS_SHORT[exp],
                linewidth=2,
            )
            idx_best = next(
                (i for i, e in enumerate(epochs_data) if e["epoch"] == best_ep), None
            )
            if idx_best is not None:
                ax_loss.plot(
                    ep[idx_best],
                    val_loss[idx_best],
                    "*",
                    color=COLORS[exp],
                    markersize=14,
                    zorder=5,
                )

            ax_bleu = axes[1][col]
            ax_bleu.plot(
                ep,
                bleu1,
                "-",
                color=COLORS[exp],
                label=EXP_LABELS_SHORT[exp],
                linewidth=2,
            )
            if idx_best is not None:
                ax_bleu.plot(
                    ep[idx_best],
                    bleu1[idx_best],
                    "*",
                    color=COLORS[exp],
                    markersize=14,
                    zorder=5,
                )

        axes[0][col].set_title(f"Validation Loss — {group_label}", fontsize=14)
        axes[0][col].set_xlabel("Epoch")
        axes[0][col].set_ylabel("Val Loss" if col == 0 else "")
        axes[0][col].legend(fontsize=10)

        axes[1][col].set_title(f"BLEU-1 — {group_label}", fontsize=14)
        axes[1][col].set_xlabel("Epoch")
        axes[1][col].set_ylabel("BLEU-1 (%)" if col == 0 else "")
        axes[1][col].legend(fontsize=10)

        if not group:
            axes[0][col].text(
                0.5,
                0.5,
                "No experiments",
                transform=axes[0][col].transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="#999",
            )
            axes[1][col].text(
                0.5,
                0.5,
                "No experiments",
                transform=axes[1][col].transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="#999",
            )

    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves_phase_a(data: dict, out_dir: Path) -> None:
    _plot_training_curves(
        data,
        PHASE_A,
        "Phase A — Training Dynamics",
        "06_training_curves_phase_a.png",
        out_dir,
    )
    print("  [6/10] 06_training_curves_phase_a.png")


def plot_training_curves_phase_b(data: dict, out_dir: Path) -> None:
    _plot_training_curves(
        data,
        PHASE_B,
        "Phase B — Training Dynamics",
        "07_training_curves_phase_b.png",
        out_dir,
    )
    print("  [7/10] 07_training_curves_phase_b.png")


def plot_efficiency_scatter(data: dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for exp in ALL_EXPERIMENTS:
        bleu1 = pct(FINAL_METRICS[exp]["bleu1"])
        params = PARAMS_M[exp]
        flops = FLOPS_G[exp]

        is_phase_a = exp in PHASE_A
        marker = "o" if is_phase_a else "s"

        ax1.scatter(
            params,
            bleu1,
            s=flops * 15,
            color=COLORS[exp],
            marker=marker,
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )
        ax1.annotate(
            exp,
            (params, bleu1),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            fontweight="bold",
        )

        ax2.scatter(
            flops,
            bleu1,
            s=params * 8,
            color=COLORS[exp],
            marker=marker,
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )
        ax2.annotate(
            exp,
            (flops, bleu1),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_xlabel("Parameters (M)")
    ax1.set_ylabel("BLEU-1 (%)")
    ax1.set_title("BLEU-1 vs Model Size")
    ax1.set_ylim(25, 70)

    ax2.set_xlabel("FLOPs (G)")
    ax2.set_ylabel("BLEU-1 (%)")
    ax2.set_title("BLEU-1 vs Compute Cost")
    ax2.set_ylim(25, 70)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#6C757D",
            markersize=10,
            label="Phase A",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#1565C0",
            markersize=10,
            label="Phase B",
        ),
    ]
    ax1.legend(handles=legend_elements, loc="lower right")
    ax2.legend(handles=legend_elements, loc="lower right")

    fig.suptitle(
        "Efficiency Analysis — Performance vs Complexity",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "08_efficiency_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print("  [8/10] 08_efficiency_scatter.png")


def plot_phase_b_meteor_scatter(data: dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, x_key, xlabel, title in [
        (ax1, "params", "Parameters (M)", "METEOR vs Model Size"),
        (ax2, "flops", "FLOPs (G)", "METEOR vs Compute Cost"),
    ]:
        for exp in PHASE_B:
            meteor = pct(FINAL_METRICS[exp]["meteor"])
            x = PARAMS_M[exp] if x_key == "params" else FLOPS_G[exp]
            ax.scatter(
                x,
                meteor,
                s=200,
                color=COLORS[exp],
                marker="s",
                edgecolors="white",
                linewidth=0.8,
                zorder=3,
                label=f"{PHASE_B_LABELS[exp]} (LSTM)" if ax is ax1 else "_nolegend_",
            )
            ax.annotate(
                PHASE_B_LABELS[exp],
                (x, meteor),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
                fontweight="bold",
                color=COLORS[exp],
            )

        for exp in PHASE_GRU:
            meteor = pct(FINAL_METRICS[exp]["meteor"])
            x = PARAMS_M[exp] if x_key == "params" else FLOPS_G[exp]
            ax.scatter(
                x,
                meteor,
                s=200,
                color=COLORS[exp],
                marker="o",
                edgecolors="white",
                linewidth=0.8,
                zorder=3,
                label=f"{PHASE_GRU_LABELS[exp]} (GRU)" if ax is ax1 else "_nolegend_",
            )
            ax.annotate(
                PHASE_GRU_LABELS[exp],
                (x, meteor),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
                fontweight="bold",
                color=COLORS[exp],
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("METEOR (%)")
        ax.set_title(title)
        ax.set_ylim(28, 42)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#555",
            markersize=10,
            label="LSTM (Phase B)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#555",
            markersize=10,
            label="GRU",
        ),
    ]
    ax1.legend(handles=legend_elements, loc="lower right")
    ax2.legend(handles=legend_elements, loc="lower right")

    fig.suptitle(
        "Phase B — METEOR Efficiency Analysis (R50 + LSTM vs GRU)",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "11_phase_b_meteor_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print("  [11] 11_phase_b_meteor_scatter.png")


def plot_component_impact_bars(data: dict, out_dir: Path) -> None:
    components = [
        ("Encoder → ResNet-50", "A1", "#2196F3"),
        ("Decoder → LSTM", "A2", "#F44336"),
        ("Tokenizer → Subword", "A3", "#4CAF50"),
        ("Tokenizer → Word", "A4", "#8BC34A"),
        ("+ Bahdanau Attention", "A5", "#FF9800"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    for ax_i, m in enumerate(METRICS):
        names = []
        deltas = []
        colors = []
        for comp_name, exp, color in components:
            d = pct(FINAL_METRICS[exp][m] - FINAL_METRICS["A0"][m])
            names.append(comp_name)
            deltas.append(d)
            colors.append(color)

        # Sort by delta descending
        order = np.argsort(deltas)[::-1]
        names = [names[i] for i in order]
        deltas = [deltas[i] for i in order]
        colors = [colors[i] for i in order]

        y_pos = np.arange(len(names))

        bars = axes[ax_i].barh(
            y_pos, deltas, color=colors, edgecolor="white", height=0.6
        )
        axes[ax_i].set_yticks(y_pos)
        axes[ax_i].set_yticklabels(
            names if ax_i == 0 else [""] * len(names), fontsize=11
        )
        axes[ax_i].set_xlabel("Δ from Baseline (pp)")
        axes[ax_i].set_title(METRIC_LABELS[m], fontsize=14, fontweight="bold")
        axes[ax_i].axvline(x=0, color="black", linewidth=0.8)

        # Annotate values
        for bar, d in zip(bars, deltas):
            sign = "+" if d >= 0 else ""
            axes[ax_i].text(
                bar.get_width() + (0.3 if d >= 0 else -0.3),
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{d:.1f}",
                va="center",
                ha="left" if d >= 0 else "right",
                fontsize=10,
                fontweight="bold",
            )

    fig.suptitle(
        "Individual Component Impact (Δ from Baseline)",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "09_component_impact_bars.png", bbox_inches="tight")
    plt.close(fig)
    print("  [9/10] 09_component_impact_bars.png")


def plot_summary_table(data: dict, out_dir: Path) -> None:
    col_labels = [
        "Exp",
        "Encoder",
        "Decoder",
        "Tok",
        "Attn",
        "BLEU-1 (%)",
        "BLEU-2 (%)",
        "ROUGE-L (%)",
        "METEOR (%)",
    ]

    configs = {
        "A0": ("R18", "GRU", "char", "—"),
        "A1": ("R50", "GRU", "char", "—"),
        "A2": ("R18", "LSTM", "char", "—"),
        "A3": ("R18", "GRU", "sub", "—"),
        "A4": ("R18", "GRU", "word", "—"),
        "A5": ("R18", "GRU", "char", "Bah"),
        "B1": ("R50", "LSTM", "char", "—"),
        "B2": ("R50", "LSTM", "word", "—"),
        "B3": ("R50", "LSTM", "sub", "—"),
        "B4": ("R50", "LSTM", "char", "Bah"),
        "B5": ("R50", "LSTM", "word", "Bah"),
        "B6": ("R50", "LSTM", "sub", "Bah"),
    }

    table_data = []
    metric_values = []  # for coloring
    for exp in ALL_EXPERIMENTS:
        enc, dec, tok, attn = configs[exp]
        m = FINAL_METRICS[exp]
        row = [
            exp,
            enc,
            dec,
            tok,
            attn,
            f"{pct(m['bleu1']):.1f}",
            f"{pct(m['bleu2']):.1f}",
            f"{pct(m['rougeL']):.1f}",
            f"{pct(m['meteor']):.1f}",
        ]
        table_data.append(row)
        metric_values.append([m["bleu1"], m["bleu2"], m["rougeL"], m["meteor"]])

    metric_arr = np.array(metric_values)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    ax.set_title(
        "Final Results Summary — All Experiments",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(len(ALL_EXPERIMENTS)):
        is_b = ALL_EXPERIMENTS[i].startswith("B")
        base_color = "#F5F5F5" if is_b else "white"

        for j in range(5):  # config columns
            table[i + 1, j].set_facecolor(base_color)

        for j_metric in range(4):
            val = metric_arr[i, j_metric]
            col_min = metric_arr[:, j_metric].min()
            col_max = metric_arr[:, j_metric].max()
            if col_max > col_min:
                norm = (val - col_min) / (col_max - col_min)
            else:
                norm = 0.5
            r = int(200 - norm * 120)
            g = int(230 - norm * 50)
            b = int(200 - norm * 120)
            table[i + 1, j_metric + 5].set_facecolor(f"#{r:02x}{g:02x}{b:02x}")

        if ALL_EXPERIMENTS[i] == "B6":
            for j in range(len(col_labels)):
                cell = table[i + 1, j]
                cell.set_edgecolor("#C62828")
                cell.set_text_props(fontweight="bold")
                cell.set_linewidth(2)

    fig.tight_layout()
    fig.savefig(out_dir / "10_final_summary_table.png", bbox_inches="tight")
    plt.close(fig)
    print("  [10/10] 10_final_summary_table.png")


def generate_all_plots(
    outputs_dir: str = "outputs", out_dir: str = "outputs/quantitative_plots"
) -> None:
    outputs_path = Path(outputs_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    setup_style()
    print(f"Loading experiment data from {outputs_path} ...")
    data = load_all_data(outputs_path)
    print(f"Loaded {len(data)} experiments.\n")
    print("Generating presentation plots ...")

    plot_phase_a_ablation_bars(data, out_path)
    plot_ablation_delta_heatmap(data, out_path)
    plot_phase_b_combined_bars(data, out_path)
    plot_baseline_to_best_progression(data, out_path)
    plot_radar_top_models(data, out_path)
    plot_training_curves_phase_a(data, out_path)
    plot_training_curves_phase_b(data, out_path)
    plot_efficiency_scatter(data, out_path)
    plot_component_impact_bars(data, out_path)
    plot_summary_table(data, out_path)
    plot_phase_b_meteor_scatter(data, out_path)

    print(f"\nAll plots saved to {out_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate presentation plots.")
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Directory containing experiment outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/quantitative_plots",
        help="Output directory for generated plots.",
    )
    args = parser.parse_args()
    generate_all_plots(args.outputs_dir, args.out_dir)
