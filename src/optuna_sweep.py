from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import optuna
import yaml

from src.optuna_visualize import generate_optuna_plots
from src.train import train
from src.utils.config import load_config


def _build_sampler(cfg: dict) -> optuna.samplers.BaseSampler:
    name = cfg.get("name", "tpe").lower()
    seed = cfg.get("seed", 42)
    n_startup = cfg.get("n_startup_trials", 10)

    if name == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup,
            multivariate=True,
        )
    elif name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    elif name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler: {name}")


def _build_pruner(cfg: dict) -> optuna.pruners.BasePruner:
    name = cfg.get("name", "hyperband").lower()

    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=cfg.get("min_resource", 3),
            max_resource=cfg.get("max_resource", 50),
            reduction_factor=cfg.get("reduction_factor", 3),
        )
    elif name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=cfg.get("n_startup_trials", 5),
            n_warmup_steps=cfg.get("n_warmup_steps", 3),
        )
    elif name == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=cfg.get("percentile", 25.0),
            n_startup_trials=cfg.get("n_startup_trials", 5),
            n_warmup_steps=cfg.get("n_warmup_steps", 3),
        )
    elif name == "none":
        return optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {name}")


def _suggest_param(trial: optuna.Trial, name: str, spec: dict) -> Any:
    ptype = spec["type"]

    if ptype == "float":
        return trial.suggest_float(name, spec["low"], spec["high"])
    elif ptype == "log_float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    elif ptype == "int":
        return trial.suggest_int(
            name, spec["low"], spec["high"], step=spec.get("step", 1)
        )
    elif ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    else:
        raise ValueError(f"Unknown parameter type '{ptype}' for {name}")


def _make_objective(
    sweep_cfg: dict,
    base_config_path: str,
    output_dir: Path,
    metric_name: str,
):
    param_specs = sweep_cfg["parameters"]

    def objective(trial: optuna.Trial) -> float:

        overrides: list[str] = []
        for param_name, spec in param_specs.items():
            value = _suggest_param(trial, param_name, spec)
            overrides.append(f"{param_name}={value}")

        cfg = load_config(base_config_path, overrides=overrides)

        trial_name = f"optuna_trial_{trial.number:04d}"
        cfg["run_name"] = trial_name
        cfg["output_dir"] = str(output_dir / "trials")

        wandb_cfg = sweep_cfg.get("wandb", {})
        if wandb_cfg.get("enabled", False):
            cfg.wandb.enabled = True
            cfg.wandb.project = wandb_cfg.get("project", "c5-image-caption-optuna")
            cfg.wandb["tags"] = [f"optuna-trial-{trial.number}"]

        def epoch_callback(metric_value: float, epoch: int) -> bool:
            trial.report(metric_value, step=epoch)
            return trial.should_prune()

        best_metric = train(cfg, epoch_callback=epoch_callback)

        return best_metric

    return objective


def run_optuna_sweep(config_path: str) -> optuna.Study:
    with open(config_path) as f:
        sweep_cfg = yaml.safe_load(f)

    base_config_path = sweep_cfg["base_config"]
    study_name = sweep_cfg.get("study_name", "c5-caption-optuna")
    n_trials = sweep_cfg.get("n_trials", 50)
    storage = sweep_cfg.get("storage")
    output_dir = Path(sweep_cfg.get("output_dir", "outputs/optuna_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_cfg = sweep_cfg.get("metric", {})
    metric_name = metric_cfg.get("name", "meteor")
    direction = metric_cfg.get("direction", "maximize")

    sampler = _build_sampler(sweep_cfg.get("sampler", {}))
    pruner = _build_pruner(sweep_cfg.get("pruner", {}))

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = _make_objective(sweep_cfg, base_config_path, output_dir, metric_name)

    print(f"Starting Optuna sweep: {study_name}")
    print(f"  Trials: {n_trials} | Metric: {metric_name} ({direction})")
    print(f"  Sampler: {type(sampler).__name__} | Pruner: {type(pruner).__name__}")
    print(f"  Output: {output_dir}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    _save_study_results(study, output_dir, sweep_cfg)

    generate_optuna_plots(study, output_dir)

    return study


def _save_study_results(study: optuna.Study, output_dir: Path, sweep_cfg: dict) -> None:
    best = study.best_trial
    best_info = {
        "study_name": study.study_name,
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
        "n_complete": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "n_pruned": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
        "n_failed": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        ),
        "sweep_config": sweep_cfg,
    }
    with open(output_dir / "best_trial.json", "w") as f:
        json.dump(best_info, f, indent=2, default=str)

    trials_data = []
    for t in study.trials:
        trials_data.append(
            {
                "number": t.number,
                "value": t.value,
                "state": t.state.name,
                "params": t.params,
                "duration_s": (
                    (t.datetime_complete - t.datetime_start).total_seconds()
                    if t.datetime_complete and t.datetime_start
                    else None
                ),
            }
        )
    with open(output_dir / "all_trials.json", "w") as f:
        json.dump(trials_data, f, indent=2, default=str)

    joblib.dump(study, output_dir / "study.pkl")

    print("\nOptuna sweep complete!")
    print(f"  Best trial: #{best.number} — {study.best_value:.6f}")
    print(f"  Best params: {best.params}")
    print(f"  Results saved to: {output_dir}")
