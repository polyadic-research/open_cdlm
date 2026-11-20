"""Train LangFlow or RePlaid on prepared token blocks with native PyTorch and torchrun."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .registry import from_pretrained
from .replaid.checkpoint import (
    initialize_ema,
    latest_checkpoint,
    restore_rng,
    save_checkpoint,
    update_ema,
    use_ema,
)
from .replaid.data import PreparedDataset, RandomBatchSampler, tokenizer_fingerprint
from .replaid.distributed import barrier, finish, initialize

LOGGER = logging.getLogger(__name__)


def _recipe(model_name):
    if model_name == "langflow":
        from .langflow import _training
    elif model_name == "replaid":
        from .replaid import _training
    else:
        raise ValueError("training supports langflow and replaid")
    return _training


def accumulation_steps(config, world, *, model_name):
    divisor = config.train.batch_size * world
    if config.train.global_batch_size % divisor:
        raise ValueError("global_batch_size must be divisible by batch_size * world_size")
    steps = config.train.global_batch_size // divisor
    if steps < 1:
        raise ValueError("global batch is smaller than one microbatch across ranks")
    _recipe(model_name).validate_rank_batch(config, steps * config.train.batch_size)
    return steps


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    config,
    *,
    model_name,
    device,
    rank,
    world,
    resume=None,
    init_from=None,
    stop_after_steps=None,
):
    """Train with the chosen model recipe and the same data, checkpoint and DDP protocol."""
    recipe = _recipe(model_name)
    config.validate()
    t = config.train
    accum = accumulation_steps(config, world, model_name=model_name)
    stop = t.total_steps if stop_after_steps is None else min(t.total_steps, stop_after_steps)
    if stop < 1:
        raise ValueError("stop-after-steps must be positive")
    if resume and init_from:
        raise ValueError("choose either --resume or --init-from")
    root = Path(t.output_dir)
    resume_path = (
        latest_checkpoint(root) if resume == "latest" else Path(resume) if resume else None
    )
    if resume_path is not None and resume_path.is_file():
        raise ValueError(
            "--resume expects a training checkpoint directory. Use --init-from for weights."
        )
    if resume and (resume_path is None or not resume_path.is_dir()):
        raise FileNotFoundError(f"resume checkpoint not found: {resume}")
    if resume_path is None and (root / "run_config.json").exists():
        raise FileExistsError(
            f"existing run in {root}; use --resume latest or another output directory"
        )
    shape = {
        "sequence_length": recipe.sequence_length(config),
        "vocab_size": config.model.vocab_size,
    }
    train_data = PreparedDataset(config.data.directory, "train", **shape)
    valid_data = (
        PreparedDataset(config.data.directory, "validation", **shape) if t.eval_every else None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        Path(config.data.directory) / "tokenizer", use_fast=True
    )
    if tokenizer_fingerprint(tokenizer) != train_data.manifest["tokenizer_fingerprint"]:
        raise ValueError("prepared tokenizer fingerprint mismatch")
    if len(tokenizer) != config.model.vocab_size:
        raise ValueError("prepared tokenizer vocabulary differs from the model")
    _seed(t.seed)
    core = recipe.build_model(config)
    if init_from:
        model_options = {"config": config.model} if model_name == "langflow" else {}
        pretrained = from_pretrained(model_name, init_from, **model_options)
        core.load_state_dict(pretrained.state_dict(), strict=True)
        del pretrained
    torch.backends.cuda.matmul.allow_tf32 = t.tf32
    torch.backends.cudnn.allow_tf32 = t.tf32
    core = core.to(device)
    optimizer = recipe.build_optimizer(core, config)
    scheduler = recipe.build_scheduler(optimizer, config)
    ema = initialize_ema(core)
    topology = {"world_size": world, "batch_size": t.batch_size, "accumulation_steps": accum}
    completed = 0
    restored = None
    if resume_path is not None:
        saved = recipe.load_config(resume_path / "experiment_config.json")
        if saved.to_dict()["model"] != config.to_dict()["model"]:
            raise ValueError("resume model configuration differs from checkpoint")
        for name in (
            "global_batch_size",
            "learning_rate",
            "warmup_steps",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "ema_decay",
            "max_grad_norm",
            "mixed_precision",
            "seed",
            "tf32",
            "gradient_checkpointing",
            *recipe.RESUME_FIELDS,
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
            "model": model_name,
            "source": recipe.SOURCE,
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
        objective = recipe.Update(core, config, step, accum, generator)
        optimizer.zero_grad(set_to_none=True)
        metrics = torch.zeros(len(recipe.LOSS_NAMES), dtype=torch.float64, device=device)
        for micro in range(accum):
            rows = next(iterator).to(device, non_blocking=True)
            sync = model.no_sync() if world > 1 and micro < accum - 1 else nullcontext()
            with sync:
                loss, values = objective.loss(model, rows, micro)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite loss at update {step}")
                (loss / accum).backward()
            metrics += torch.stack([values[key] for key in recipe.LOSS_NAMES]) / accum
        norm = torch.nn.utils.clip_grad_norm_(
            core.parameters(),
            t.max_grad_norm if t.max_grad_norm > 0 else float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        objective.finish()
        update_ema(core, ema, t.ema_decay)
        completed = step + 1
        if completed % t.log_every == 0 or completed == 1:
            if world > 1:
                dist.all_reduce(metrics)
                metrics /= world
            if rank == 0:
                entry = dict(zip(recipe.LOSS_NAMES, metrics.tolist()))
                entry.update(objective.diagnostics())
                entry.update(
                    step=completed,
                    grad_norm=float(norm),
                    backbone_lr=optimizer.param_groups[0]["lr"],
                    elapsed_seconds=time.monotonic() - tick,
                )
                with (root / "train_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
                LOGGER.info(
                    "%s step=%d %s",
                    model_name,
                    completed,
                    " ".join(f"{key}={entry[key]:.5f}" for key in recipe.LOSS_NAMES),
                )
        if t.eval_every and completed % t.eval_every == 0:
            with use_ema(core, ema):
                evaluation = recipe.evaluate(
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
                    "%s validation step=%d EMA %s=%s",
                    model_name,
                    completed,
                    recipe.VALIDATION_METRIC,
                    evaluation[recipe.VALIDATION_METRIC],
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


def main(argv=None, *, model_name=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if model_name is None:
        parser.add_argument("--model", choices=("langflow", "replaid"), required=True)
    else:
        parser.set_defaults(model=model_name)
    parser.add_argument("--config", type=Path, help="experiment YAML or JSON")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", nargs="?", const="latest", help="restore the training state")
    start.add_argument("--init-from", help="start a new run with pretrained weights")
    parser.add_argument("--mixed-precision", choices=("bf16", "no"))
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate topology/model without data"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    recipe = _recipe(args.model)
    config = recipe.load_config(args.config or recipe.DEFAULT_CONFIG)
    if args.data_dir:
        config.data.directory = args.data_dir
    for name in ("output_dir", "batch_size", "global_batch_size", "total_steps", "mixed_precision"):
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
            accum = accumulation_steps(config, world, model_name=args.model)
            with torch.device("meta"):
                model = recipe.build_model(config)
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
                model_name=args.model,
                device=device,
                rank=rank,
                world=world,
                resume=args.resume,
                init_from=args.init_from,
                stop_after_steps=args.stop_after_steps,
            )
    finally:
        finish()


if __name__ == "__main__":
    main()
