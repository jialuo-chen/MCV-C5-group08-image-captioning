from __future__ import annotations

from pathlib import Path

import joblib
import optuna


def generate_optuna_plots(study: optuna.Study, output_dir: str | Path) -> None:
    output_dir = Path(output_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        print(f"Only {len(completed)} completed trial(s); skipping most plots.")
        if len(completed) == 1:
            _save_optimization_history(study, output_dir)
        return

    _save_optimization_history(study, output_dir)
    _save_param_importances(study, output_dir)
    _save_parallel_coordinate(study, output_dir)
    _save_slice_plot(study, output_dir)
    _save_contour_plot(study, output_dir)
    _save_timeline(study, output_dir)
    _save_edf(study, output_dir)
    _save_summary_table(study, output_dir)

    print(f"Optuna plots saved to: {output_dir}")


def _save_plot(fig, path: Path) -> None:
    fig.write_html(str(path.with_suffix(".html")))
    try:
        fig.write_image(str(path.with_suffix(".png")), width=1200, height=700, scale=2)
    except (ValueError, ImportError):
        pass


def _save_optimization_history(study: optuna.Study, output_dir: Path) -> None:
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.update_layout(title="Optimization History", template="plotly_white")
        _save_plot(fig, output_dir / "optimization_history")
    except Exception as e:
        print(f"  [skip] optimization_history: {e}")


def _save_param_importances(study: optuna.Study, output_dir: Path) -> None:
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.update_layout(title="Hyperparameter Importances", template="plotly_white")
        _save_plot(fig, output_dir / "param_importances")
    except Exception as e:
        print(f"  [skip] param_importances: {e}")


def _save_parallel_coordinate(study: optuna.Study, output_dir: Path) -> None:
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.update_layout(title="Parallel Coordinate Plot", template="plotly_white")
        _save_plot(fig, output_dir / "parallel_coordinate")
    except Exception as e:
        print(f"  [skip] parallel_coordinate: {e}")


def _save_slice_plot(study: optuna.Study, output_dir: Path) -> None:
    try:
        fig = optuna.visualization.plot_slice(study)
        fig.update_layout(title="Slice Plot", template="plotly_white")
        _save_plot(fig, output_dir / "slice_plot")
    except Exception as e:
        print(f"  [skip] slice_plot: {e}")


def _save_contour_plot(study: optuna.Study, output_dir: Path) -> None:
    try:
        fig = optuna.visualization.plot_contour(study)
        fig.update_layout(title="Contour Plot", template="plotly_white")
        _save_plot(fig, output_dir / "contour_plot")
    except Exception as e:
        print(f"  [skip] contour_plot: {e}")


def _save_timeline(study: optuna.Study, output_dir: Path) -> None:
    try:
        fig = optuna.visualization.plot_timeline(study)
        fig.update_layout(title="Trial Timeline", template="plotly_white")
        _save_plot(fig, output_dir / "timeline")
    except Exception as e:
        print(f"  [skip] timeline: {e}")


def _save_edf(study: optuna.Study, output_dir: Path) -> None:
    try:
        fig = optuna.visualization.plot_edf(study)
        fig.update_layout(
            title="Empirical Distribution Function", template="plotly_white"
        )
        _save_plot(fig, output_dir / "edf")
    except Exception as e:
        print(f"  [skip] edf: {e}")


def _save_summary_table(study: optuna.Study, output_dir: Path) -> None:
    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

    lines = [
        f"# Optuna Study: {study.study_name}",
        "",
        "## Summary",
        "",
        f"- **Total trials**: {len(trials)}",
        f"- **Completed**: {len(completed)}",
        f"- **Pruned**: {len(pruned)}",
        f"- **Failed**: {len(failed)}",
        f"- **Best trial**: #{study.best_trial.number}",
        f"- **Best value**: {study.best_value:.6f}",
        "",
        "## Best Parameters",
        "",
    ]
    for k, v in study.best_trial.params.items():
        lines.append(f"- `{k}`: {v}")

    lines += [
        "",
        "## All Completed Trials",
        "",
        "| Trial | Value | " + " | ".join(study.best_trial.params.keys()) + " |",
        "| ----- | ----- | "
        + " | ".join(["-----"] * len(study.best_trial.params))
        + " |",
    ]
    for t in sorted(
        completed,
        key=lambda t: t.value if t.value is not None else float("-inf"),
        reverse=True,
    ):
        vals = " | ".join(
            str(t.params.get(k, "")) for k in study.best_trial.params.keys()
        )
        value_str = f"{t.value:.6f}" if t.value is not None else "N/A"
        lines.append(f"| {t.number} | {value_str} | {vals} |")

    (output_dir / "study_summary.md").write_text("\n".join(lines))


def load_and_visualize(study_dir: str) -> None:
    study_dir = Path(study_dir)
    study_path = study_dir / "study.pkl"

    if not study_path.exists():
        raise FileNotFoundError(
            f"No study.pkl found in {study_dir}. "
            "Run an optuna-sweep first, or provide the correct --study-dir."
        )

    study: optuna.Study = joblib.load(study_path)
    print(f"Loaded study '{study.study_name}' with {len(study.trials)} trials.")

    generate_optuna_plots(study, study_dir)
