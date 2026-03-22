"""Generate plots for Phase C hyperparameter optimisation (C1 and C2 sweeps).

Produces 8 publication-quality PNGs:
  01_unoptimized_comparison.png   — A0 vs B6 (no HPO)
  02_optimized_comparison.png     — C1★ vs C2★ (after HPO)
  03_baseline_hpo_gain.png        — A0 → C1★ before/after per metric
  04_b6_hpo_gain.png              — B6 → C2★ before/after per metric
  05_c1_hpo_history.png           — C1 trial METEOR over time + running best
  06_c2_hpo_history.png           — C2 trial METEOR over time + running best
  07_c1_param_analysis.png        — C1 hyperparameter distributions coloured by METEOR
  08_c2_param_analysis.png        — C2 hyperparameter distributions coloured by METEOR

Usage
-----
    uv run python src/generate_hpo_plots.py [--outputs-dir outputs] [--out-dir outputs/quantitative_plots]
    uv run python main.py hpo-plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRICS = ["bleu1", "bleu2", "rougeL", "meteor"]
METRIC_LABELS = {"bleu1": "BLEU-1", "bleu2": "BLEU-2", "rougeL": "ROUGE-L", "meteor": "METEOR"}

# Final eval metrics (from eval_results.json / EXPERIMENTS.md)
METRICS_A0 = {"bleu1": 0.4505, "bleu2": 0.2242, "rougeL": 0.3314, "meteor": 0.2836}
METRICS_B6 = {"bleu1": 0.6088, "bleu2": 0.3782, "rougeL": 0.4116, "meteor": 0.3718}
METRICS_C1 = {"bleu1": 0.5020, "bleu2": 0.2885, "rougeL": 0.3572, "meteor": 0.3254}
METRICS_C2 = {"bleu1": 0.6382, "bleu2": 0.4078, "rougeL": 0.4287, "meteor": 0.3826}

# Colours — blue family = baseline, green family = best architecture
# Muted shades = no HPO, saturated shades = with HPO
COL_A0 = "#90CAF9"   # light blue  — baseline, no HPO
COL_C1 = "#1565C0"   # dark blue   — baseline, with HPO
COL_B6 = "#A5D6A7"   # light green — best arch, no HPO
COL_C2 = "#2E7D32"   # dark green  — best arch, with HPO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct(v: float) -> float:
    return v * 100.0


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 200,
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.2,
    })


def load_trials(sweep_dir: Path) -> list[dict]:
    with open(sweep_dir / "all_trials.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1 — Unoptimized comparison: A0 vs B6
# ---------------------------------------------------------------------------

def plot_unoptimized_comparison(out_dir: Path) -> None:
    models = [
        ("A0 — Baseline (R18 + GRU + char)", METRICS_A0, COL_A0),
        ("B6 — Best Phase B (R50 + LSTM + subword + attn)", METRICS_B6, COL_B6),
    ]
    x = np.arange(len(METRICS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (label, m, color) in enumerate(models):
        vals = [pct(m[k]) for k in METRICS]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=color)

    # Delta annotations
    for j, k in enumerate(METRICS):
        delta = pct(METRICS_B6[k] - METRICS_A0[k])
        txt = ax.annotate(f"Δ+{delta:.1f}pp",
                    xy=(j + width / 2, pct(METRICS_B6[k]) + 5.5),
                    ha="center", va="bottom", fontsize=11, color="black", fontweight="bold")
        txt.set_path_effects([withStroke(linewidth=3, foreground="white")])

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[k] for k in METRICS], fontsize=13)
    ax.set_ylabel("Score (%)")
    ax.set_title("Baseline vs Best Architecture — Before Hyperparameter Optimisation")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "01_unoptimized_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  [1/8] 01_unoptimized_comparison.png")


# ---------------------------------------------------------------------------
# Plot 2 — Optimized comparison: C1★ vs C2★
# ---------------------------------------------------------------------------

def plot_optimized_comparison(out_dir: Path) -> None:
    models = [
        ("C1★ — Optimised Baseline (R18 + GRU + char)", METRICS_C1, COL_C1),
        ("C2★ — Optimised Best (R50 + LSTM + subword + attn)", METRICS_C2, COL_C2),
    ]
    x = np.arange(len(METRICS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (label, m, color) in enumerate(models):
        vals = [pct(m[k]) for k in METRICS]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=color)

    # Delta annotations
    for j, k in enumerate(METRICS):
        delta = pct(METRICS_C2[k] - METRICS_C1[k])
        txt = ax.annotate(f"Δ+{delta:.1f}pp",
                    xy=(j + width / 2, pct(METRICS_C2[k]) + 5.5),
                    ha="center", va="bottom", fontsize=11, color="black", fontweight="bold")
        txt.set_path_effects([withStroke(linewidth=3, foreground="white")])

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[k] for k in METRICS], fontsize=13)
    ax.set_ylabel("Score (%)")
    ax.set_title("Optimised Baseline vs Optimised Best Architecture — After HPO (50 trials)")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "02_optimized_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  [2/8] 02_optimized_comparison.png")


# ---------------------------------------------------------------------------
# Plot 3 — Before/After HPO for Baseline (A0 → C1★)
# ---------------------------------------------------------------------------

def plot_baseline_hpo_gain(out_dir: Path) -> None:
    _plot_before_after(
        before_metrics=METRICS_A0,
        after_metrics=METRICS_C1,
        before_label="A0 — No HPO",
        after_label="C1★ — After HPO",
        before_color=COL_A0,
        after_color=COL_C1,
        title="Baseline: Effect of Hyperparameter Optimisation\n(R18 + GRU + char-level tokenizer)",
        out_path=out_dir / "03_baseline_hpo_gain.png",
        plot_id="3/8",
    )


# ---------------------------------------------------------------------------
# Plot 4 — Before/After HPO for B6 (B6 → C2★)
# ---------------------------------------------------------------------------

def plot_b6_hpo_gain(out_dir: Path) -> None:
    _plot_before_after(
        before_metrics=METRICS_B6,
        after_metrics=METRICS_C2,
        before_label="B6 — No HPO",
        after_label="C2★ — After HPO",
        before_color=COL_B6,
        after_color=COL_C2,
        title="Best Architecture (B6): Effect of Hyperparameter Optimisation\n(R50 + LSTM + subword + Bahdanau attention)",
        out_path=out_dir / "04_b6_hpo_gain.png",
        plot_id="4/8",
    )


def _plot_before_after(
    before_metrics: dict, after_metrics: dict,
    before_label: str, after_label: str,
    before_color: str, after_color: str,
    title: str, out_path: Path, plot_id: str,
) -> None:
    x = np.arange(len(METRICS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (label, m, color) in enumerate([
        (before_label, before_metrics, before_color),
        (after_label,  after_metrics,  after_color),
    ]):
        vals = [pct(m[k]) for k in METRICS]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=label, color=color,
                      edgecolor="white", linewidth=0.5, alpha=0.9 if i == 0 else 1.0)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=color)

    for j, k in enumerate(METRICS):
        delta = pct(after_metrics[k] - before_metrics[k])
        sign = "+" if delta >= 0 else ""
        txt = ax.annotate(f"{sign}{delta:.1f}pp",
                    xy=(j + width / 2, pct(after_metrics[k]) + 5.5),
                    ha="center", va="bottom", fontsize=11, color="black", fontweight="bold")
        txt.set_path_effects([withStroke(linewidth=3, foreground="white")])

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[k] for k in METRICS], fontsize=13)
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.set_ylim(0, 80)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{plot_id}] {out_path.name}")


# ---------------------------------------------------------------------------
# Plots 5 & 6 — HPO optimisation history
# ---------------------------------------------------------------------------

def plot_hpo_history(trials: list[dict], study_name: str, best_trial: int,
                     color: str, out_path: Path, plot_id: str) -> None:
    nums = [t["number"] for t in trials]
    vals = [t["value"] for t in trials]

    running_best: list[float] = []
    cur_best = -np.inf
    for v in vals:
        if v > cur_best:
            cur_best = v
        running_best.append(cur_best)

    fig, ax = plt.subplots(figsize=(12, 5))

    # All trials as scatter
    scatter_colors = [color if v == max(vals) else "#BDBDBD" for v in vals]
    ax.scatter(nums, [pct(v) for v in vals], c=scatter_colors, s=50, zorder=3,
               alpha=0.7, label="Trial METEOR")

    # Running best line
    ax.plot(nums, [pct(v) for v in running_best], color=color, linewidth=2.5,
            label="Running best", zorder=4)

    # Mark best trial
    best_val = pct(vals[best_trial])
    ax.axvline(best_trial, color=color, linestyle="--", linewidth=1.5, alpha=0.6)
    ax.annotate(
        f"Best: trial {best_trial}\nMETEOR={best_val:.2f}%",
        xy=(best_trial, best_val),
        xytext=(best_trial + 2, best_val - 3),
        arrowprops=dict(arrowstyle="->", color=color),
        fontsize=11, color=color, fontweight="bold",
    )

    ax.set_xlabel("Trial number")
    ax.set_ylabel("METEOR (%)")
    ax.set_title(f"{study_name} — Optimisation History")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-1, len(trials))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{plot_id}] {out_path.name}")


# ---------------------------------------------------------------------------
# Plots 7 & 8 — Parameter importance (Optuna fANOVA, parsed from HTML)
# ---------------------------------------------------------------------------

PARAM_LABELS = {
    "training.lr": "Learning rate",
    "training.batch_size": "Batch size",
    "decoder.hidden_size": "Hidden size",
    "decoder.embed_size": "Embed size",
    "decoder.num_layers": "Num layers",
    "decoder.dropout": "Dropout",
    "training.optimizer": "Optimizer",
    "training.scheduler": "Scheduler",
    "training.weight_decay": "Weight decay",
}


def load_param_importance(sweep_dir: Path) -> tuple[list[str], list[float]]:
    """Parse fANOVA importance values directly from Optuna's param_importances.html."""
    import re
    html = (sweep_dir / "plots" / "param_importances.html").read_text()
    x_match = re.search(r'"x":\[([0-9.,e+\-]+)\]', html)
    y_match = re.search(r'"y":\[([^\]]+)\]', html)
    scores = [float(v) for v in x_match.group(1).split(",")]
    keys = [v.strip('"') for v in y_match.group(1).split(",")]
    labels = [PARAM_LABELS.get(k, k) for k in keys]
    return labels, scores


def plot_param_importance(sweep_dir: Path, study_name: str, color: str,
                          out_path: Path, plot_id: str) -> None:
    labels, scores = load_param_importance(sweep_dir)
    # already sorted ascending from Optuna
    labels, scores = zip(*sorted(zip(labels, scores), key=lambda x: x[1]))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, [s * 100 for s in scores], color=color, edgecolor="white")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{score * 100:.1f}%", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Relative importance (%)")
    ax.set_title(f"{study_name}\nHyperparameter Importance")
    ax.set_xlim(0, max(s * 100 for s in scores) * 1.25)
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{plot_id}] {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(outputs_dir: str = "outputs", out_dir: str = "outputs/quantitative_plots") -> None:
    setup_style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sweep_c1 = Path(outputs_dir) / "optuna_sweep_c1"
    sweep_c2 = Path(outputs_dir) / "optuna_sweep_c2"

    print("Generating HPO plots...")
    plot_unoptimized_comparison(out)
    plot_optimized_comparison(out)
    plot_baseline_hpo_gain(out)
    plot_b6_hpo_gain(out)

    trials_c1 = load_trials(sweep_c1)
    plot_hpo_history(trials_c1, "C1 (Baseline HPO)", best_trial=37,
                     color=COL_C1, out_path=out / "05_c1_hpo_history.png", plot_id="5/6")

    trials_c2 = load_trials(sweep_c2)
    plot_hpo_history(trials_c2, "C2 (Best Architecture HPO)", best_trial=44,
                     color=COL_C2, out_path=out / "06_c2_hpo_history.png", plot_id="6/8")

    plot_param_importance(sweep_c1, "C1 — Baseline HPO", COL_C1,
                          out / "07_c1_param_importance.png", "7/8")
    plot_param_importance(sweep_c2, "C2 — Best Architecture HPO", COL_C2,
                          out / "08_c2_param_importance.png", "8/8")

    print(f"Done. All 8 plots saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HPO analysis plots.")
    parser.add_argument("--outputs-dir", type=str, default="outputs",
                        help="Root directory containing experiment outputs")
    parser.add_argument("--out-dir", type=str, default="outputs/quantitative_plots",
                        help="Output directory for PNG files")
    args = parser.parse_args()
    main(args.outputs_dir, args.out_dir)
