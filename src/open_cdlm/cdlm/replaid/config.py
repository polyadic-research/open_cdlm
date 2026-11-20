from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from .._training_config import DataConfig, construct, read_config, validate_train


@dataclass
class ModelConfig:
    tokenizer_name_or_path: str = "openai-community/gpt2"
    vocab_size: int = 50257
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_seq_length: int = 1024
    cond_dim: int = 128
    dropout: float = 0.1
    self_condition_probability: float = 0.25
    bias_warmup_steps: int = 5000
    replaid_embedding_dim: int = 16
    replaid_schedule_hidden_size: int = 1024
    replaid_gamma_0_init: float = -3.0
    replaid_gamma_1_init: float = 6.0
    replaid_reconstruction_ema_decay: float = 0.997

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.num_attention_heads

    def validate(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "max_seq_length",
            "cond_dim",
            "replaid_embedding_dim",
            "replaid_schedule_hidden_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if self.hidden_size % self.num_attention_heads or self.head_dim % 2:
            raise ValueError("hidden_size must divide into heads with even head_dim for RoPE")
        if not 0 <= self.dropout < 1:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.self_condition_probability not in (0.0, 0.25):
            raise ValueError("paper configurations use self_condition_probability=0 or 0.25")
        if self.bias_warmup_steps < 0:
            raise ValueError("bias_warmup_steps cannot be negative")
        if self.replaid_gamma_0_init >= self.replaid_gamma_1_init:
            raise ValueError("gamma_0 must be smaller than gamma_1")
        if not 0 < self.replaid_reconstruction_ema_decay < 1:
            raise ValueError("reconstruction EMA decay must be in (0, 1)")


@dataclass
class TrainConfig:
    output_dir: str = "outputs/owt_sc"
    total_steps: int = 1_000_000
    batch_size: int = 8
    global_batch_size: int = 512
    learning_rate: float = 3e-4
    special_learning_rate: float = 1e-2
    warmup_steps: int = 2500
    weight_decay: float = 0.0
    gamma_bounds_weight_decay: float = 0.1
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
        if self.special_learning_rate <= 0:
            raise ValueError("special_learning_rate must be positive")


@dataclass
class ExperimentConfig:
    experiment_id: str = "REPLAID_ORIGINAL_OWT_SC"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def validate(self) -> None:
        self.model.validate()
        self.train.validate()


def load_config(path: str | Path) -> ExperimentConfig:
    raw = read_config(path)
    for name, cls in (("model", ModelConfig), ("data", DataConfig), ("train", TrainConfig)):
        raw[name] = construct(cls, raw.get(name, {}))
    config = construct(ExperimentConfig, raw)
    config.validate()
    return config
