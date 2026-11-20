"""Common-API adapters for native RePlaid checkpoints and samplers."""
from __future__ import annotations

import torch


def from_pretrained(checkpoint, *, device="cpu", weights="ema"):
    """Load a reproduction checkpoint; use ``load_checkpoint`` for tokenizer/config too."""
    from .checkpoint import load_model

    model, _, _ = load_model(checkpoint, device=device, weights=weights)
    return model


@torch.no_grad()
def sample_tokens(
    model, *, num_samples, length, steps, seed, sampler, temperature, **kwargs,
):
    length = model.config.max_seq_length if length is None else length
    if seed is not None:
        if any(key in kwargs for key in ("generator", "initial_noise", "sample_seeds")):
            raise ValueError("set seed=None when passing native generator/initial_noise/sample_seeds")
        kwargs["sample_seeds"] = tuple(seed + index for index in range(num_samples))
    output = model.sample(
        num_samples, length, steps=steps, sampler=sampler, temperature=temperature, **kwargs,
    )
    return output.token_ids

