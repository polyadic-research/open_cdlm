from __future__ import annotations

import torch

from .config import ExperimentConfig


def build_optimizer(model, config: ExperimentConfig):
    """Paper Appendix L, Table 9: NOT the scaling-law cosine setup of Table 7."""
    t = config.train
    special = {id(p) for m in (model.embedding, model.noise_schedule) for p in m.parameters()}
    bounds = {id(model.gamma_0), id(model.gamma_1)}
    regular = [p for p in model.parameters() if id(p) not in special | bounds]
    groups = [
        dict(params=regular, lr=t.learning_rate, weight_decay=t.weight_decay, name="backbone"),
        dict(
            params=list(model.embedding.parameters()),
            lr=t.special_learning_rate,
            weight_decay=0.0,
            name="embedding",
        ),
        dict(
            params=list(model.noise_schedule.parameters()),
            lr=t.special_learning_rate,
            weight_decay=0.0,
            name="schedule",
        ),
        dict(
            params=[model.gamma_0, model.gamma_1],
            lr=t.special_learning_rate,
            weight_decay=t.gamma_bounds_weight_decay,
            name="bounds",
        ),
    ]
    return torch.optim.AdamW(
        groups,
        betas=(t.adam_beta1, t.adam_beta2),
        eps=1e-8,
        fused=model.embedding.weight.device.type == "cuda",
    )


def build_scheduler(optimizer, config: ExperimentConfig):
    warmup = config.train.warmup_steps
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / warmup) if warmup else 1.0,
    )
