"""RePlaid's model and objective for the shared training loop."""

from pathlib import Path

import torch

from .config import load_config
from .evaluate import evaluate
from .model import RePlaid
from .objective import make_batch_plan, nelbo_objective, update_loss_moments
from .optim import build_optimizer, build_scheduler

__all__ = ["build_optimizer", "build_scheduler", "evaluate", "load_config"]

DEFAULT_CONFIG = Path(__file__).parent / "configs/owt_sc.yaml"
RESUME_FIELDS = ("special_learning_rate", "gamma_bounds_weight_decay")
LOSS_NAMES = ("loss", "reconstruction", "diffusion", "prior")
VALIDATION_METRIC = "ppl_upper_bound"
SOURCE = "paper v3, OWT benchmark Table 9"


def sequence_length(config):
    return config.model.max_seq_length


def validate_rank_batch(config, size):
    if size < 2 or (config.model.self_condition_probability and size % 4):
        raise ValueError("effective per-rank batch must be >=2 and divisible by 4 for s.c.")


def build_model(config):
    model = RePlaid(config.model)
    model.use_bfloat16 = config.train.mixed_precision == "bf16"
    if config.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


class Update:
    def __init__(self, core, config, step, accumulation, generator):
        self.core = core
        self.config = config
        self.step = step
        self.accumulation = accumulation
        self.generator = generator
        size = config.train.batch_size * accumulation
        self.plan = make_batch_plan(core, size, generator=generator)
        self.moments = torch.zeros(6, dtype=torch.float64, device=generator.device)

    def loss(self, model, rows, microbatch):
        size = self.config.train.batch_size
        loss, values = nelbo_objective(
            model,
            rows,
            torch.ones_like(rows),
            self.config,
            self.step,
            plan=self.plan.slice(microbatch * size, size),
            generator=self.generator,
            accumulation_steps=self.accumulation,
        )
        self.moments += values["moments"]
        return loss, values

    def finish(self):
        update_loss_moments(self.core, self.moments)

    def diagnostics(self):
        return {
            "reconstruction_batch_size": self.plan.reconstruction_count,
            "gamma_0": float(self.core.gamma_0.detach()),
            "gamma_1": float(self.core.gamma_1.detach()),
        }
