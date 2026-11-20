"""LangFlow's model and objective for the shared training loop."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn import functional as F

from .._training_config import DataConfig, construct, read_config
from .._training_config import TrainConfig as BaseTrainConfig
from ..replaid.data import validation_loader
from .config import LangFlowConfig
from .model import LangFlow
from .objective import sample_gamma, training_objective

DEFAULT_CONFIG = Path(__file__).parent / "configs/owt_pytorch.yaml"
RESUME_FIELDS = (
    "proposal_learning_rate",
    "self_condition_probability",
    "bias_warmup_steps",
    "scheduler_loss_weight",
    "total_steps",
)
LOSS_NAMES = ("loss", "ce_loss", "scheduler_loss")
VALIDATION_METRIC = "denoising_ce_nats_per_token"
SOURCE = "LangFlow denoising and Gumbel schedule fitting on prepared blocks"


@dataclass
class TrainConfig(BaseTrainConfig):
    output_dir: str = "outputs/langflow_owt"
    warmup_steps: int = 5000
    weight_decay: float = 0.01
    proposal_learning_rate: float | None = None
    self_condition_probability: float = 0.25
    bias_warmup_steps: int = 5000
    scheduler_loss_weight: float = 1.0

    def validate(self) -> None:
        super().validate()
        if self.proposal_learning_rate is not None and not self.proposal_learning_rate > 0:
            raise ValueError("proposal_learning_rate must be positive")
        if not 0 <= self.self_condition_probability <= 1:
            raise ValueError("self_condition_probability must be in [0, 1]")
        if self.bias_warmup_steps < 0 or not self.scheduler_loss_weight >= 0:
            raise ValueError("bias_warmup_steps and scheduler_loss_weight cannot be negative")


@dataclass
class ExperimentConfig:
    experiment_id: str = "LANGFLOW_OWT"
    model: LangFlowConfig = field(default_factory=LangFlowConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "model": self.model.to_dict(),
            "data": asdict(self.data),
            "train": asdict(self.train),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def validate(self) -> None:
        self.model.validate()
        self.train.validate()


def load_config(path: str | Path) -> ExperimentConfig:
    raw = read_config(path)
    model = dict(raw.get("model", {}))
    if model.pop("model_type", LangFlowConfig.model_type) != LangFlowConfig.model_type:
        raise ValueError("the configuration belongs to a different model")
    raw["model"] = LangFlowConfig(**model)
    raw["data"] = construct(DataConfig, raw.get("data", {}))
    raw["train"] = construct(TrainConfig, raw.get("train", {}))
    config = construct(ExperimentConfig, raw)
    config.validate()
    return config


def sequence_length(config):
    return config.model.model_length


def validate_rank_batch(config, size):
    """Any positive per-rank batch size is supported."""


def build_model(config):
    model = LangFlow(config.model)
    model.backbone.use_bfloat16 = config.train.mixed_precision == "bf16"
    model.backbone.gradient_checkpointing = config.train.gradient_checkpointing
    return model


def build_optimizer(model, config):
    t = config.train
    proposal = list(model.proposal.parameters())
    proposal_ids = {id(parameter) for parameter in proposal}
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in proposal_ids:
            continue
        if any(part in name for part in ("bias", "norm", "embedding", "ln_", "LayerNorm")):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {
            "params": decay,
            "lr": t.learning_rate,
            "weight_decay": t.weight_decay,
            "name": "backbone",
        },
        {"params": no_decay, "lr": t.learning_rate, "weight_decay": 0.0, "name": "no_decay"},
        {
            "params": proposal,
            "lr": t.proposal_learning_rate
            if t.proposal_learning_rate is not None
            else t.learning_rate * 0.1,
            "weight_decay": 0.0,
            "name": "proposal",
        },
    ]
    return torch.optim.AdamW(
        groups,
        betas=(t.adam_beta1, t.adam_beta2),
        fused=model.proposal.loc.device.type == "cuda",
    )


def build_scheduler(optimizer, config):
    t = config.train

    def multiplier(step):
        if step < t.warmup_steps:
            return step / t.warmup_steps
        progress = min(1.0, (step - t.warmup_steps) / max(1, t.total_steps - t.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class Update:
    def __init__(self, core, config, step, accumulation, generator):
        self.core = core
        self.settings = config.train
        self.use_bias = config.model.use_bias and step >= self.settings.bias_warmup_steps
        self.generator = generator

    def loss(self, model, rows, microbatch):
        gamma = sample_gamma(
            self.core,
            len(rows),
            device=rows.device,
            generator=self.generator,
        )
        return training_objective(
            model,
            rows,
            gamma,
            self_condition_probability=self.settings.self_condition_probability,
            scheduler_loss_weight=self.settings.scheduler_loss_weight,
            use_bias=self.use_bias,
            generator=self.generator,
        )

    def finish(self):
        """LangFlow's proposal is updated by the optimizer."""

    def diagnostics(self):
        return {
            "gumbel_loc": float(self.core.proposal.loc.detach()),
            "gumbel_scale": float(self.core.proposal.scale.detach()),
            "gumbel_entropy": float(self.core.proposal.entropy.detach()),
            "use_bias": self.use_bias,
        }


@torch.no_grad()
def evaluate(model, dataset, *, batch_size, device, max_batches, seed):
    """Measure denoising cross entropy on a fixed gamma grid."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    loader, size = validation_loader(
        dataset,
        batch_size=batch_size,
        max_batches=max_batches,
        rank=rank,
        world=world,
    )
    gamma_levels = tuple(range(-2, 9))
    totals = torch.zeros(len(gamma_levels) + 1, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + rank * 100_003)
    was_training = model.training
    model.eval()
    try:
        for rows in loader:
            rows = rows.to(device)
            clean = model._embed_tokens(rows)
            for index, level in enumerate(gamma_levels):
                gamma = torch.full((len(rows),), float(level), device=device)
                noisy = model._forward_diffusion(clean, gamma, generator=generator)
                logits = model(noisy_embeds=noisy, timesteps=gamma, return_dict=False)
                totals[index] += F.cross_entropy(
                    logits.float().flatten(0, 1),
                    rows.flatten(),
                    reduction="sum",
                )
                del logits
            totals[-1] += rows.numel()
        if world > 1:
            dist.all_reduce(totals)
    finally:
        model.train(was_training)
    if not totals[-1]:
        raise ValueError("no validation tokens were evaluated")
    means = (totals[:-1] / totals[-1]).tolist()
    if not all(math.isfinite(value) for value in means):
        raise FloatingPointError("non-finite validation denoising cross entropy")
    return {
        VALIDATION_METRIC: sum(means) / len(means),
        "gamma_grid": gamma_levels,
        "ce_by_gamma": means,
        "num_sequences": size,
        "seed": seed,
    }
