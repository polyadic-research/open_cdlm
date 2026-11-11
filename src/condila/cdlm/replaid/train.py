"""Train the paper's OWT RePlaid benchmark with torchrun or a single process."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from importlib.metadata import version
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
from safetensors.torch import load_file
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .checkpoint import (
    initialize_ema,
    latest_checkpoint,
    restore_rng,
    save_checkpoint,
    update_ema,
    use_ema,
)
from .config import load_config
from .data import PreparedDataset, RandomBatchSampler, tokenizer_fingerprint
from .distributed import barrier, finish, initialize
from .evaluate import evaluate
from .model import RePlaid
from .objective import make_batch_plan, nelbo_objective, update_loss_moments
from .optim import build_optimizer, build_scheduler

LOGGER = logging.getLogger(__name__)


def accumulation_steps(config, world):
    divisor = config.train.batch_size * world
    if config.train.global_batch_size % divisor:
        raise ValueError("global_batch_size must be divisible by batch_size * world_size")
    steps = config.train.global_batch_size // divisor
    if steps < 1:
        raise ValueError("global batch is smaller than one microbatch across ranks")
    effective = steps * config.train.batch_size
    if effective < 2 or (config.model.self_condition_probability and effective % 4):
        raise ValueError("effective per-rank batch must be >=2 and divisible by 4 for s.c.")
    return steps


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config, *, device, rank, world, resume=None, stop_after_steps=None):
    t = config.train
    accum = accumulation_steps(config, world)
    stop = min(t.total_steps, stop_after_steps or t.total_steps)
    if stop < 1:
        raise ValueError("stop-after-steps must be positive")
    root = Path(t.output_dir)
    resume_path = (
        latest_checkpoint(root) if resume == "latest" else Path(resume) if resume else None
    )
    if resume and (resume_path is None or not resume_path.is_dir()):
        raise FileNotFoundError(f"resume checkpoint not found: {resume}")
    if resume_path is None and (root / "run_config.json").exists():
        raise FileExistsError(
            f"existing run in {root}; use --resume latest or another output directory"
        )
    train_data = PreparedDataset(
        config.data.directory,
        "train",
        sequence_length=config.model.max_seq_length,
        vocab_size=config.model.vocab_size,
    )
    valid_data = None
    if t.eval_every:
        valid_data = PreparedDataset(
            config.data.directory,
            "validation",
            sequence_length=config.model.max_seq_length,
            vocab_size=config.model.vocab_size,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        Path(config.data.directory) / "tokenizer", use_fast=True
    )
    if tokenizer_fingerprint(tokenizer) != train_data.manifest["tokenizer_fingerprint"]:
        raise ValueError("prepared tokenizer fingerprint mismatch")
    if len(tokenizer) != config.model.vocab_size:
        raise ValueError("prepared tokenizer vocabulary differs from the model")
    torch.backends.cuda.matmul.allow_tf32 = t.tf32
    torch.backends.cudnn.allow_tf32 = t.tf32
    _seed(t.seed)
    core = RePlaid(config.model).to(device)
    core.use_bfloat16 = t.mixed_precision == "bf16"
    if t.gradient_checkpointing:
        core.gradient_checkpointing_enable()
    optimizer = build_optimizer(core, config)
    scheduler = build_scheduler(optimizer, config)
    ema = initialize_ema(core)
    topology = {"world_size": world, "batch_size": t.batch_size, "accumulation_steps": accum}
    completed = 0
    restored = None
    if resume_path is not None:
        saved = load_config(resume_path / "experiment_config.json")
        if saved.model != config.model:
            raise ValueError("resume model configuration differs from checkpoint")
        for name in (
            "global_batch_size",
            "learning_rate",
            "special_learning_rate",
            "warmup_steps",
            "weight_decay",
            "gamma_bounds_weight_decay",
            "adam_beta1",
            "adam_beta2",
            "ema_decay",
            "max_grad_norm",
            "mixed_precision",
            "seed",
            "tf32",
            "gradient_checkpointing",
        ):
            if getattr(saved.train, name) != getattr(t, name):
                raise ValueError(
                    f"resume changes train.{name}; exact continuation requires same recipe"
                )
        saved_manifest = json.loads((resume_path / "data_manifest.json").read_text())
        if saved_manifest != train_data.manifest:
            raise ValueError("resume data manifest differs from the original run")
        # This file is created locally by save_checkpoint and contains optimizer
        # and Python/NumPy RNG objects; it is not a model weights download.
        restored = torch.load(resume_path / "training.pt", map_location="cpu", weights_only=False)
        if restored["topology"] != topology:
            raise ValueError(
                "exact resume requires identical world size, microbatch, and accumulation"
            )
        completed = restored["completed_steps"]
        if completed > stop:
            raise ValueError("checkpoint step exceeds requested stop step")
        core.load_state_dict(load_file(str(resume_path / "model.safetensors")), strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        loaded_ema = load_file(str(resume_path / "ema.safetensors"))
        if set(loaded_ema) != set(ema):
            raise ValueError("EMA parameter mismatch during resume")
        ema = {k: v.to(device) for k, v in loaded_ema.items()}
    model = core
    if world > 1:
        model = DistributedDataParallel(
            core,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    _seed(t.seed + rank * 100_003)
    generator = torch.Generator(device=device).manual_seed(t.seed + rank * 100_003)
    sampler = RandomBatchSampler(
        len(train_data),
        t.batch_size,
        seed=t.seed,
        rank=rank,
        start_batch=completed * accum,
        stop_batch=stop * accum,
    )
    loader = DataLoader(
        train_data,
        batch_sampler=sampler,
        num_workers=t.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=t.num_workers > 0,
        generator=torch.Generator().manual_seed(t.seed),
    )
    iterator = iter(loader)
    if restored is not None:
        restore_rng(restored["rng_by_rank"][rank], generator)
        del restored
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        config.save(root / "run_config.json")
        environment = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": version("transformers"),
            "numpy": np.__version__,
            "topology": topology,
            "model_parameters": sum(p.numel() for p in core.parameters()),
            "device": str(device),
            "source": "paper v3, OWT benchmark Table 9",
        }
        (root / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        LOGGER.info(
            "parameters=%s world=%d global_batch=%d accumulation=%d resume_step=%d",
            f"{environment['model_parameters']:,}",
            world,
            t.global_batch_size,
            accum,
            completed,
        )
    barrier()
    last_saved = completed if resume_path is not None else -1
    tick = time.monotonic()
    core.train()
    for step in range(completed, stop):
        plan = make_batch_plan(core, t.batch_size * accum, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(6, dtype=torch.float64, device=device)
        metrics = torch.zeros(4, dtype=torch.float64, device=device)
        for micro in range(accum):
            rows = next(iterator).to(device, non_blocking=True)
            sync = model.no_sync() if world > 1 and micro < accum - 1 else nullcontext()
            with sync:
                loss, values = nelbo_objective(
                    model,
                    rows,
                    torch.ones_like(rows),
                    config,
                    step,
                    plan=plan.slice(micro * t.batch_size, t.batch_size),
                    generator=generator,
                    accumulation_steps=accum,
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite loss at update {step}")
                (loss / accum).backward()
            totals += values["moments"]
            metrics += (
                torch.stack([values[k] for k in ("loss", "reconstruction", "diffusion", "prior")])
                / accum
            )
        norm = torch.nn.utils.clip_grad_norm_(
            core.parameters(),
            t.max_grad_norm if t.max_grad_norm > 0 else float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        update_loss_moments(core, totals)
        update_ema(core, ema, t.ema_decay)
        completed = step + 1
        if completed % t.log_every == 0 or completed == 1:
            if world > 1:
                dist.all_reduce(metrics)
                metrics /= world
            if rank == 0:
                entry = dict(
                    zip(("loss", "reconstruction", "diffusion", "prior"), metrics.tolist())
                )
                entry.update(
                    step=completed,
                    grad_norm=float(norm),
                    reconstruction_batch_size=plan.reconstruction_count,
                    gamma_0=float(core.gamma_0),
                    gamma_1=float(core.gamma_1),
                    backbone_lr=optimizer.param_groups[0]["lr"],
                    elapsed_seconds=time.monotonic() - tick,
                )
                with (root / "train_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
                LOGGER.info(
                    "step=%d nelbo=%.5f recon=%.5f diffusion=%.5f prior=%.5f",
                    completed,
                    *metrics.tolist(),
                )
        if t.eval_every and completed % t.eval_every == 0:
            with use_ema(core, ema):
                evaluation = evaluate(
                    core,
                    valid_data,
                    batch_size=t.eval_batch_size,
                    device=device,
                    max_batches=t.eval_max_batches,
                    seed=t.seed,
                )
            if rank == 0:
                evaluation.update(step=completed, weights="ema", preliminary=True)
                with (root / "validation_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(evaluation) + "\n")
                LOGGER.info(
                    "validation step=%d EMA PPL bound=%.5f",
                    completed,
                    evaluation["ppl_upper_bound"],
                )
        if t.save_every and completed % t.save_every == 0:
            save_checkpoint(
                root,
                core,
                tokenizer,
                config,
                optimizer,
                scheduler,
                ema,
                completed,
                generator,
                topology,
                train_data.manifest,
            )
            last_saved = completed
    if completed != last_saved:
        save_checkpoint(
            root,
            core,
            tokenizer,
            config,
            optimizer,
            scheduler,
            ema,
            completed,
            generator,
            topology,
            train_data.manifest,
        )
    barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/owt_sc.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate topology/model without data"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    if args.data_dir:
        config.data.directory = args.data_dir
    for name in ("output_dir", "batch_size", "global_batch_size", "total_steps"):
        value = getattr(args, name)
        if value is not None:
            setattr(config.train, name, value)
    if args.gradient_checkpointing:
        config.train.gradient_checkpointing = True
    config.data.directory = str(Path(config.data.directory).resolve())
    config.train.output_dir = str(Path(config.train.output_dir).resolve())
    config.validate()
    device, rank, world = initialize(args.device)
    try:
        if args.dry_run:
            accum = accumulation_steps(config, world)
            with torch.device("meta"):
                model = RePlaid(config.model)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "config": config.to_dict(),
                            "world_size": world,
                            "accumulation_steps": accum,
                            "parameters": sum(p.numel() for p in model.parameters()),
                        },
                        indent=2,
                    )
                )
        else:
            train(
                config,
                device=device,
                rank=rank,
                world=world,
                resume=args.resume,
                stop_after_steps=args.stop_after_steps,
            )
    finally:
        finish()


if __name__ == "__main__":
    main()
