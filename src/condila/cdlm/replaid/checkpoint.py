"""Self-contained checkpoints, including optimizer, EMA, data position, and rank RNGs."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import random
import shutil

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import distributed as dist
from transformers import AutoTokenizer

from .config import load_config
from .distributed import barrier
from .model import RePlaid


def initialize_ema(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


@torch.no_grad()
def update_ema(model, ema, decay):
    for name, parameter in model.named_parameters():
        ema[name].lerp_(parameter.detach(), 1 - decay)


@contextmanager
def use_ema(model, ema):
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    try:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(ema[name])
        yield
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(original[name])


def capture_rng(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "generator": generator.get_state(),
        "cuda": torch.cuda.get_rng_state() if generator.device.type == "cuda" else None,
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["generator"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device=generator.device)


def save_checkpoint(
    output_dir,
    model,
    tokenizer,
    config,
    optimizer,
    scheduler,
    ema,
    completed_steps,
    generator,
    topology,
    data_manifest,
):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    states = [None] * world
    state = capture_rng(generator)
    if world > 1:
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    checkpoint = Path(output_dir) / "checkpoints" / f"checkpoint-{completed_steps}"
    # All ranks must fail consistently, so rank 0 broadcasts any local I/O error.
    error = [None]
    if rank == 0:
        try:
            temporary = checkpoint.with_name(checkpoint.name + ".incomplete")
            if checkpoint.exists() or temporary.exists():
                raise FileExistsError(f"checkpoint already exists: {checkpoint}")
            temporary.mkdir(parents=True)
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                str(temporary / "model.safetensors"),
            )
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in ema.items()},
                str(temporary / "ema.safetensors"),
            )
            config.save(temporary / "experiment_config.json")
            tokenizer.save_pretrained(temporary / "tokenizer")
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "completed_steps": completed_steps,
                    "rng_by_rank": states,
                    "topology": topology,
                },
                temporary / "training.pt",
            )
            (temporary / "data_manifest.json").write_text(
                json.dumps(data_manifest, indent=2) + "\n"
            )
            temporary.rename(checkpoint)
            if config.train.save_total_limit:
                protected = set(config.train.checkpoint_keep_steps) | {completed_steps}
                candidates = [
                    p
                    for p in checkpoint.parent.glob("checkpoint-*")
                    if p.name.split("-")[-1].isdigit()
                    and not p.is_symlink()
                    and (p / "training.pt").is_file()
                ]
                candidates.sort(key=lambda p: int(p.name.split("-")[-1]))
                recent = set(candidates[-config.train.save_total_limit :])
                for previous in candidates:
                    if (
                        previous not in recent
                        and int(previous.name.split("-")[-1]) not in protected
                    ):
                        shutil.rmtree(previous)
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"could not save checkpoint: {error[0]}")
    barrier()
    return checkpoint


def latest_checkpoint(output_dir):
    paths = list((Path(output_dir) / "checkpoints").glob("checkpoint-*"))
    paths = [p for p in paths if p.name.split("-")[-1].isdigit() and (p / "training.pt").is_file()]
    return max(paths, key=lambda p: int(p.name.split("-")[-1])) if paths else None


def load_model(checkpoint, *, device="cpu", weights="ema"):
    checkpoint = Path(checkpoint)
    config = load_config(checkpoint / "experiment_config.json")
    torch.backends.cuda.matmul.allow_tf32 = config.train.tf32
    torch.backends.cudnn.allow_tf32 = config.train.tf32
    model = RePlaid(config.model)
    model.use_bfloat16 = config.train.mixed_precision == "bf16"
    state = load_file(str(checkpoint / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    if weights == "ema":
        ema = load_file(str(checkpoint / "ema.safetensors"))
        if set(ema) != {name for name, _ in model.named_parameters()}:
            raise ValueError("EMA checkpoint does not contain exactly every learned parameter")
        model.load_state_dict({**state, **ema}, strict=True)
    elif weights != "raw":
        raise ValueError("weights must be raw or ema")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", use_fast=True)
    return model.to(device).eval(), tokenizer, config
