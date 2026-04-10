from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import optuna
import yaml
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import FrozenTrial, TrialState

from src.models.vit_qwen_lora import LORA_TARGETS
from src.optuna_visualize import generate_optuna_plots
from src.train import train
from src.train_lora import train_lora
from src.utils.config import load_config

_TASK_FUNCTIONS = {
    "train": train,
    "train_lora": train_lora,
}


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


def _spec_to_distribution(name: str, spec: dict):
    """Convert a sweep parameter spec to an Optuna distribution."""
    ptype = spec["type"]
    if ptype == "float":
        return FloatDistribution(spec["low"], spec["high"])
    elif ptype == "log_float":
        return FloatDistribution(spec["low"], spec["high"], log=True)
    elif ptype == "int":
        return IntDistribution(spec["low"], spec["high"], step=spec.get("step", 1))
    elif ptype == "categorical":
        return CategoricalDistribution(spec["choices"])
    else:
        raise ValueError(f"Unknown parameter type '{ptype}' for {name}")


# Reverse lookup: sorted target_modules tuple → preset name
_REVERSE_LORA_TARGETS: dict[tuple[str, ...], str] = {
    tuple(sorted(v)): k for k, v in LORA_TARGETS.items()
}


def _extract_trial_params_from_disk(
    trial_dir: Path,
    param_specs: dict[str, dict],
) -> tuple[dict[str, Any] | None, float | None, list[tuple[int, float]]]:
    """Extract sweep param values and metric from a completed trial directory.

    Returns (params_dict, best_metric, intermediate_values) or (None, None, [])
    if the trial data cannot be read.
    """
    # First try the easy path: optuna_params.json (saved by newer runs)
    optuna_params_file = trial_dir / "optuna_params.json"
    if optuna_params_file.exists():
        with open(optuna_params_file) as f:
            saved = json.load(f)
        return saved["params"], saved["metric"], saved.get("intermediates", [])

    # Fall back to reconstructing from training artifacts
    exp_log = trial_dir / "experiment_log.json"
    if not exp_log.exists():
        return None, None, []

    with open(exp_log) as f:
        log_data = json.load(f)

    hp = log_data.get("hyperparameters", {})
    summary = log_data.get("summary", {})
    best_metrics = summary.get("best_metrics", {})

    # Read lora adapter config (if exists)
    adapter_data: dict = {}
    for ckpt in ("best", "last"):
        adapter_cfg = (
            trial_dir / "checkpoints" / ckpt / "lora_adapter" / "adapter_config.json"
        )
        if adapter_cfg.exists():
            with open(adapter_cfg) as f:
                adapter_data = json.load(f)
            break

    # Read projection metadata (if exists)
    proj_data: dict = {}
    for ckpt in ("best", "last"):
        proj_meta = trial_dir / "checkpoints" / ckpt / "projection_meta.json"
        if proj_meta.exists():
            with open(proj_meta) as f:
                proj_data = json.load(f)
            break

    # Build a lookup for known param extraction paths
    r = adapter_data.get("r")
    lora_alpha_raw = adapter_data.get("lora_alpha", 0)

    target_modules = adapter_data.get("target_modules", [])
    lora_target = _REVERSE_LORA_TARGETS.get(tuple(sorted(target_modules)))

    scheduler_val = hp.get("training", {}).get("scheduler")
    if scheduler_val is None:
        scheduler_val = "none"

    source: dict[str, Any] = {
        "training.lr": hp.get("training", {}).get("lr"),
        "tokenizer.max_length": hp.get("tokenizer", {}).get("max_length"),
        "training.optimizer": hp.get("training", {}).get("optimizer"),
        "training.weight_decay": hp.get("training", {}).get("weight_decay"),
        "training.scheduler": scheduler_val,
        "lora.r": r,
        "lora.alpha": int(lora_alpha_raw / r) if r else None,
        "lora.target": lora_target,
        "projection.type": proj_data.get("type"),
        "projection.num_queries": proj_data.get("num_queries"),
        "projection.num_layers": proj_data.get("num_layers"),
        "projection.ffn_dim": proj_data.get("ffn_dim"),
    }

    params: dict[str, Any] = {}
    for param_name in param_specs:
        value = source.get(param_name)
        if value is None:
            print(
                f"  [recovery] WARNING: could not extract '{param_name}' for {trial_dir.name}, skipping trial"
            )
            return None, None, []
        params[param_name] = value

    # Extract metric
    metric = best_metrics.get("meteor")  # default sweep metric

    # Extract intermediate values (per-epoch reports)
    intermediates: list[tuple[int, float]] = []
    for ep in log_data.get("training", {}).get("epochs", []):
        epoch_num = ep.get("epoch")
        meteor_val = ep.get("meteor")
        if epoch_num is not None and meteor_val is not None:
            intermediates.append((epoch_num, meteor_val))

    return params, metric, intermediates


def _recover_trials_from_disk(
    study: optuna.Study,
    output_dir: Path,
    param_specs: dict[str, dict],
    metric_name: str,
) -> int:
    """Scan trial directories and add completed trials to an empty study.

    Returns the number of recovered trials.
    """
    trials_dir = output_dir / "trials"
    if not trials_dir.exists():
        return 0

    # Find trial directories matching the naming pattern
    trial_dirs = sorted(
        d
        for d in trials_dir.iterdir()
        if d.is_dir() and re.match(r"optuna_trial_\d+", d.name)
    )

    if not trial_dirs:
        return 0

    # Build distributions from param specs
    distributions = {
        name: _spec_to_distribution(name, spec) for name, spec in param_specs.items()
    }

    recovered = 0
    for trial_dir in trial_dirs:
        trial_num = int(trial_dir.name.split("_")[-1])

        # Skip if this trial already exists in the study
        existing_numbers = {t.number for t in study.trials}
        if trial_num in existing_numbers:
            continue

        params, metric, intermediates = _extract_trial_params_from_disk(
            trial_dir, param_specs
        )
        if params is None or metric is None:
            print(f"  [recovery] Skipping {trial_dir.name} (incomplete data)")
            continue

        # Cast param values to match distribution types
        casted_params: dict[str, Any] = {}
        for pname, dist in distributions.items():
            val = params[pname]
            if isinstance(dist, IntDistribution):
                casted_params[pname] = int(val)
            elif isinstance(dist, FloatDistribution):
                casted_params[pname] = float(val)
            elif isinstance(dist, CategoricalDistribution):
                # Ensure the value is one of the valid choices
                if val not in dist.choices:
                    # Try type coercion
                    for choice in dist.choices:
                        if str(choice) == str(val):
                            val = choice
                            break
                casted_params[pname] = val
            else:
                casted_params[pname] = val

        # Build intermediate values dict
        intermediate_values = {step: value for step, value in intermediates}

        frozen = FrozenTrial(
            number=trial_num,
            state=TrialState.COMPLETE,
            value=metric,
            datetime_start=datetime.now(),
            datetime_complete=datetime.now(),
            params=casted_params,
            distributions=distributions,
            user_attrs={},
            system_attrs={},
            intermediate_values=intermediate_values,
            trial_id=trial_num,
        )
        study.add_trial(frozen)
        recovered += 1

    return recovered


def _make_objective(
    sweep_cfg: dict,
    base_config_path: str,
    output_dir: Path,
    metric_name: str,
):
    param_specs = sweep_cfg["parameters"]
    task = sweep_cfg.get("task", "train")
    train_fn = _TASK_FUNCTIONS.get(task)
    if train_fn is None:
        valid = ", ".join(_TASK_FUNCTIONS)
        raise ValueError(f"Unknown task '{task}'. Valid options: {valid}")

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

        reported_intermediates: list[tuple[int, float]] = []

        def epoch_callback(metric_value: float, epoch: int) -> bool:
            trial.report(metric_value, step=epoch)
            reported_intermediates.append((epoch, metric_value))
            return trial.should_prune()

        best_metric = train_fn(cfg, epoch_callback=epoch_callback)

        # Save params + metric for easy recovery on future crashes
        trial_dir = output_dir / "trials" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        intermediates = sorted(reported_intermediates)
        with open(trial_dir / "optuna_params.json", "w") as f:
            json.dump(
                {
                    "params": trial.params,
                    "metric": best_metric,
                    "intermediates": intermediates,
                },
                f,
                indent=2,
                default=str,
            )

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

    # Default to SQLite so trials persist across crashes
    if not storage:
        db_path = output_dir / "optuna_study.db"
        storage = f"sqlite:///{db_path}"

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

    # Count already-completed trials (from SQLite or previous run)
    done_states = {TrialState.COMPLETE, TrialState.PRUNED}
    n_done = len([t for t in study.trials if t.state in done_states])

    # If study is empty, attempt to recover completed trials from disk
    if n_done == 0:
        param_specs = sweep_cfg["parameters"]
        n_recovered = _recover_trials_from_disk(
            study, output_dir, param_specs, metric_name
        )
        if n_recovered > 0:
            n_done = n_recovered
            print(f"  Recovered {n_recovered} completed trials from disk")

    remaining = n_trials - n_done
    if remaining <= 0:
        print(f"All {n_trials} trials already completed ({n_done} done). Nothing to do.")
        _save_study_results(study, output_dir, sweep_cfg)
        generate_optuna_plots(study, output_dir)
        return study

    objective = _make_objective(sweep_cfg, base_config_path, output_dir, metric_name)

    print(f"{'Resuming' if n_done > 0 else 'Starting'} Optuna sweep: {study_name}")
    print(f"  Trials: {remaining} remaining ({n_done}/{n_trials} done)")
    print(f"  Metric: {metric_name} ({direction})")
    print(f"  Sampler: {type(sampler).__name__} | Pruner: {type(pruner).__name__}")
    print(f"  Storage: {storage}")
    print(f"  Output: {output_dir}")

    study.optimize(objective, n_trials=remaining, show_progress_bar=True)

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
