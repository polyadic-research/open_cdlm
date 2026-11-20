"""The judge language model shared by generative perplexity and the feature metrics (default ``gpt2-large``, fp32).

One instance per ``(name, device)`` is cached, so ``GenerativePerplexity``, ``KTD``, ``MTD`` and ``MAUVE`` on one device
load the weights once. Pass ``model=`` / ``tokenizer=`` to use an already loaded model.
"""
from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from .inputs import default_device


class JudgeLM:
    """A causal LM with ``pad = eos`` (the convention of LangFlow's gen_ppl.py and the MDLM / DUO / FLM metrics)."""

    def __init__(self, name: str = "gpt2-large", device: Optional[str] = None, model: Any = None,
                 tokenizer: Any = None, dtype: torch.dtype = torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.device = torch.device(device or default_device())
        self.tok = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
            self.tok.pad_token_id = self.tok.eos_token_id
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(name)
            if dtype is not None:
                model = model.to(dtype)
        self.lm = model.to(self.device).eval().requires_grad_(False)
        self.vocab_size = int(self.lm.config.vocab_size)

    def __repr__(self) -> str:
        return f"JudgeLM({self.name!r}, device={str(self.device)!r})"


@functools.lru_cache(maxsize=None)
def _cached(name: str, device: str) -> JudgeLM:
    return JudgeLM(name, device)


def get_judge(judge: Any = "gpt2-large", device: Optional[str] = None) -> JudgeLM:
    """``JudgeLM`` from a name (cached per device) or pass through an existing ``JudgeLM``."""
    if isinstance(judge, JudgeLM):
        return judge
    return _cached(str(judge), str(torch.device(device or default_device())))


def autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    import contextlib

    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)
