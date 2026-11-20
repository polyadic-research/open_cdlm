"""Unconditional RePlaid generation; one seed per sample, independent of rank count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import load_model
from .distributed import barrier, finish, initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
    parser.add_argument("--num-samples", type=int, default=5120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="integration intervals; final decoder evaluation is additional",
    )
    parser.add_argument("--sampler", choices=("ddpm", "ddim", "heun"), default="ddpm")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-self-conditioning",
        action="store_true",
        help="explicit inference ablation, not the separately trained no-s.c. model",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if min(args.num_samples, args.batch_size, args.length, args.steps) <= 0:
        raise ValueError("sample count, batch size, length, and steps must be positive")
    device, rank, world = initialize(args.device)
    try:
        model, tokenizer, config = load_model(args.checkpoint, device=device, weights=args.weights)
        sc = config.model.self_condition_probability > 0 and not args.no_self_conditioning
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shard = args.output.with_name(args.output.name + f".rank-{rank:04d}.part")
        # Exclusive creation protects an earlier run from accidental overwrite.
        if args.output.exists():
            raise FileExistsError(args.output)
        identifiers = list(range(rank, args.num_samples, world))
        with shard.open("x", encoding="utf-8") as stream:
            for start in range(0, len(identifiers), args.batch_size):
                indices = identifiers[start : start + args.batch_size]
                seeds = tuple((args.seed + index * 1_000_003) % (2**63 - 1) for index in indices)
                result = model.sample(
                    len(indices),
                    args.length,
                    steps=args.steps,
                    temperature=args.temperature,
                    sample_seeds=seeds,
                    sampler=args.sampler,
                    self_conditioning=sc,
                )
                for index, seed, ids, entropy in zip(
                    indices,
                    seeds,
                    result.token_ids.cpu().tolist(),
                    result.token_entropy.mean(-1).cpu().tolist(),
                    strict=True,
                ):
                    record = {
                        "sample_id": index,
                        "seed": seed,
                        "token_ids": ids,
                        "text": tokenizer.decode(ids, skip_special_tokens=False),
                        "clean_text": tokenizer.decode(ids, skip_special_tokens=True),
                        "decoder_entropy": entropy,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        barrier()
        if rank == 0:
            shards = [
                args.output.with_name(args.output.name + f".rank-{i:04d}.part")
                for i in range(world)
            ]
            records = [
                json.loads(line) for path in shards for line in path.read_text().splitlines()
            ]
            records.sort(key=lambda row: row["sample_id"])
            if [row["sample_id"] for row in records] != list(range(args.num_samples)):
                raise RuntimeError("missing or duplicate distributed sample IDs")
            with args.output.open("x", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            metadata = {
                "checkpoint": str(args.checkpoint.resolve()),
                "weights": args.weights,
                "num_samples": args.num_samples,
                "length": args.length,
                "sampler": args.sampler,
                "integration_steps": args.steps,
                "denoising_nfe": args.steps * (2 if args.sampler == "heun" else 1),
                "final_decoder_nfe": 1,
                "temperature": args.temperature,
                "self_conditioning": sc,
                "training_self_conditioning": config.model.self_condition_probability > 0,
                "seed": args.seed,
            }
            args.output.with_suffix(".metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            for path in shards:
                path.unlink()
            print(json.dumps(metadata, indent=2))
        barrier()
    finally:
        finish()


if __name__ == "__main__":
    main()
