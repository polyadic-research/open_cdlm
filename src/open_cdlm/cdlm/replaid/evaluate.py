"""Held-out per-token NELBO and perplexity upper bound, separate from GenPPL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import distributed as dist

from .checkpoint import load_model
from .data import PreparedDataset, tokenizer_fingerprint, validation_loader
from .distributed import finish, initialize
from .objective import validation_terms


@torch.no_grad()
def evaluate(
    model,
    dataset,
    *,
    batch_size: int,
    device,
    max_batches: int | None = None,
    mc_samples: int = 1,
    seed: int = 0,
) -> dict:
    if batch_size <= 0 or mc_samples <= 0 or (max_batches is not None and max_batches <= 0):
        raise ValueError("batch size, MC samples, and max batches must be positive")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    loader, size = validation_loader(
        dataset, batch_size=batch_size, max_batches=max_batches, rank=rank, world=world,
    )
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + rank * 100_003)
    was_training = model.training
    model.eval()
    try:
        for rows in loader:
            rows = rows.to(device)
            for _ in range(mc_samples):
                terms = validation_terms(model, rows, generator=generator)
                for i, key in enumerate(("reconstruction", "diffusion", "prior", "nelbo")):
                    totals[i] += terms[key].sum()
                totals[4] += terms["nelbo"].square().sum()
                totals[5] += rows.shape[0]
        if world > 1:
            dist.all_reduce(totals)
    finally:
        model.train(was_training)
    count = int(totals[5].item())
    if not count:
        raise ValueError("no validation sequences were evaluated")
    means = (totals[:4] / count).tolist()
    if not all(math.isfinite(value) for value in means):
        raise FloatingPointError("non-finite validation NELBO")
    return {
        "reconstruction_nats_per_token": means[0],
        "diffusion_nats_per_token": means[1],
        "prior_nats_per_token": means[2],
        "nelbo_nats_per_token": means[3],
        "ppl_upper_bound": math.exp(means[3]) if means[3] < 709 else None,
        "num_sequences": size,
        "mc_samples": mc_samples,
        "num_estimates": count,
        "self_conditioning": model.config.self_condition_probability > 0,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--mc-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device, rank, _ = initialize(args.device)
    try:
        model, tokenizer, config = load_model(args.checkpoint, device=device, weights=args.weights)
        dataset = PreparedDataset(
            args.data_dir or config.data.directory,
            "validation",
            sequence_length=config.model.max_seq_length,
            vocab_size=config.model.vocab_size,
        )
        if tokenizer_fingerprint(tokenizer) != dataset.manifest["tokenizer_fingerprint"]:
            raise ValueError("validation tokenizer differs from checkpoint tokenizer")
        saved_manifest = json.loads((args.checkpoint / "data_manifest.json").read_text())
        if saved_manifest["splits"]["validation"] != dataset.manifest["splits"]["validation"]:
            raise ValueError(
                "validation split differs from the training checkpoint's held-out split"
            )
        metrics = evaluate(
            model,
            dataset,
            batch_size=args.batch_size,
            device=device,
            max_batches=args.max_batches,
            mc_samples=args.mc_samples,
            seed=args.seed,
        )
        metrics.update(
            checkpoint=str(args.checkpoint.resolve()),
            weights=args.weights,
            validation_sha256=dataset.manifest["splits"]["validation"]["sha256"],
        )
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics, indent=2))
    finally:
        finish()


if __name__ == "__main__":
    main()
