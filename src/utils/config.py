from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "device": "cuda",
    "run_name": None,  # auto-generated if None
    "output_dir": "outputs",
    "encoder": {
        "name": "resnet18",
        "pretrained": "microsoft/resnet-18",
        "freeze": False,
        "feature_dim": 512,  # overridden per model
    },
    "decoder": {
        "type": "rnn",  # "rnn" or "hf_lm"
        "name": "gru",  # "gru", "lstm" (for rnn) or HF model id (for hf_lm)
        "pretrained": None,  # HF model id for hf_lm decoders
        "hidden_size": 512,
        "num_layers": 1,
        "embed_size": 512,
        "dropout": 0.0,
    },
    "attention": {
        "enabled": False,
        "type": "bahdanau",  # "bahdanau" or "luong"
        "attention_dim": 256,
    },
    "tokenizer": {
        "type": "char",  # "char", "subword", "word"
        "vocab_size": None,  # auto for char; set for subword/word
        "max_length": 201,
        "pretrained": None,  # path or HF tokenizer name for subword
    },
    "dataset": {
        "root": "/data/123-1/users/jchen/VizWiz",
        "train_ann": "annotations/train.json",
        "val_ann": "annotations/val.json",
        "train_img_dir": "train",
        "val_img_dir": "val",
        "ignore_rejected": True,
        "ignore_precanned": True,
        "val_split_ratio": 0.1,  # fraction of training data used as validation
    },
    "training": {
        "epochs": 20,
        "batch_size": 64,
        "lr": 1e-3,
        "freeze_encoder": False,
        "freeze_decoder": False,
        "optimizer": "adam",  # "adam", "adamw", "sgd"
        "weight_decay": 0.0,
        "scheduler": None,  # "cosine", "step", None
        "scheduler_params": {},
        "grad_clip": 5.0,
        "teacher_forcing_ratio": 1.0,
        "mixed_precision": False,
        "num_workers": 4,
        "early_stopping_patience": None,
    },
    "inference": {
        "beam_size": 1,  # 1 = greedy
        "max_length": 201,
        "temperature": 1.0,
    },
    "wandb": {
        "enabled": False,
        "project": "c5-image-caption",
        "entity": None,
        "tags": [],
    },
}

ENCODER_REGISTRY: dict[str, dict[str, Any]] = {
    "resnet18": {"pretrained": "microsoft/resnet-18", "feature_dim": 512},
    "resnet34": {"pretrained": "microsoft/resnet-34", "feature_dim": 512},
    "resnet50": {"pretrained": "microsoft/resnet-50", "feature_dim": 2048},
    "vgg16": {"pretrained": "google/vgg-16", "feature_dim": 512},
    "vgg19": {"pretrained": "google/vgg-19", "feature_dim": 512},
}


class Config(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")
        if isinstance(val, dict) and not isinstance(val, Config):
            val = Config(val)
            self[key] = val
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _set_nested(d: dict, key: str, value: str) -> None:
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    # Try to parse value to correct type
    parsed = _parse_value(value)
    d[parts[-1]] = parsed


def _parse_value(value: str) -> Any:
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    path = Path(path)
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}

    enc_name = user_cfg.get("encoder", {}).get("name")
    if enc_name and enc_name in ENCODER_REGISTRY:
        registry_defaults = ENCODER_REGISTRY[enc_name]
        enc_section = user_cfg.get("encoder", {})
        for k, v in registry_defaults.items():
            enc_section.setdefault(k, v)
        user_cfg["encoder"] = enc_section

    merged = _deep_merge(DEFAULTS, user_cfg)

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be key=value, got: {item}")
            key, value = item.split("=", 1)
            _set_nested(merged, key, value)

    return Config(merged)
