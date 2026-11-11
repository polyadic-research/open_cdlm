from __future__ import annotations

from datetime import timedelta
import os

import torch
from torch import distributed as dist


def initialize(device: str = "auto"):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.device(f"cuda:{local_rank}" if device == "cuda" else device)
    if target.type == "cuda":
        torch.cuda.set_device(target)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(
            "nccl" if target.type == "cuda" else "gloo",
            timeout=timedelta(minutes=30),
        )
    return target, rank, world


def barrier():
    if dist.is_initialized():
        dist.barrier()


def finish():
    if dist.is_initialized():
        dist.destroy_process_group()
