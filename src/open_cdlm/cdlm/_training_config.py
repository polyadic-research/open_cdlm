"""Configuration shared by the native PyTorch training recipes."""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    directory: str = "data/owt_gpt2"


@dataclass
class TrainConfig:
    output_dir: str = "outputs/owt_sc"
    total_steps: int = 1_000_000
    batch_size: int = 8
    global_batch_size: int = 512
    learning_rate: float = 3e-4
    warmup_steps: int = 2500
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    ema_decay: float = 0.9999
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = False
    num_workers: int = 4
    seed: int = 42
    log_every: int = 100
    save_every: int = 10_000
    save_total_limit: int = 5
    checkpoint_keep_steps: list[int] = field(default_factory=lambda: [250_000, 1_000_000])
    eval_every: int = 10_000
    eval_max_batches: int = 64
    eval_batch_size: int = 8
    tf32: bool = True

    def validate(self) -> None:
        validate_train(self)


def validate_train(settings) -> None:
    for name in (
        "total_steps",
        "batch_size",
        "global_batch_size",
        "log_every",
        "eval_batch_size",
        "eval_max_batches",
    ):
        if getattr(settings, name) <= 0:
            raise ValueError(f"train.{name} must be positive")
    for name in ("warmup_steps", "save_every", "eval_every", "num_workers", "save_total_limit"):
        if getattr(settings, name) < 0:
            raise ValueError(f"train.{name} cannot be negative")
    if any(step < 0 for step in settings.checkpoint_keep_steps):
        raise ValueError("checkpoint retention steps cannot be negative")
    if settings.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if (
        not 0 <= settings.ema_decay < 1
        or not 0 < settings.adam_beta1 < 1
        or not 0 < settings.adam_beta2 < 1
    ):
        raise ValueError("invalid EMA decay or Adam moments")
    if settings.mixed_precision not in ("bf16", "no"):
        raise ValueError("mixed_precision must be bf16 or no")


def read_config(path: str | Path) -> dict:
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    raw = json.loads(content) if path.suffix == ".json" else yaml.safe_load(content)
    if not isinstance(raw, dict):
        raise TypeError(f"expected a configuration mapping in {path}")
    return raw


def construct(cls, values: dict):
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)
