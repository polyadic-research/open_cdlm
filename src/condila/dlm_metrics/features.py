"""Sentence-level features of a causal LM for the two-sample metrics (MMD, PRDC, Frechet distance, Vendi, topics, MAUVE).

``TextFeatureExtractor(pooling="mean")`` (default; MMD and friends): the last-layer hidden states of the judge LM
(``gpt2-large``), averaged over all positions of the row. Texts keep their special tokens (the same text the generative
perplexity scores), are re-tokenized by the judge with truncation at ``max_length`` tokens, and run under bf16 autocast
on CUDA. Features are returned un-normalized (float64); the metrics L2-normalize them by default.

``TextFeatureExtractor(pooling="last")`` (MAUVE): the hidden state of the LAST token, fp32, texts with special tokens
removed, each text tokenized on its own with truncation at 1024 tokens; empty texts are dropped. This reproduces the
featurization of the ``mauve-text`` package (``get_features_from_input``) with a shared model instance.

Pass one extractor object to several metrics: features of a batch are computed once and served from a small cache.
``cache_dir`` additionally stores features on disk (keyed by the rows' content hash and the extractor settings), which
saves the reference features across runs.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import torch

from .inputs import Samples, TokenizerLike
from .models import JudgeLM, autocast_context, get_judge

_DTYPES = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16,
           None: None, "none": None, "float32": None, "fp32": None}


class TextFeatureExtractor:
    """Hidden-state features of a judge LM.

    Args:
        model: judge name (``"gpt2-large"``) or a :class:`~condila.dlm_metrics.models.JudgeLM`.
        device: device of the judge (default: CUDA if available).
        pooling: ``"mean"`` (masked mean over positions) or ``"last"`` (last token, MAUVE's featurization).
        layer: hidden-state index (``-1`` = after the final layer norm).
        max_length: judge tokens per row (truncation).
        batch_size: rows per forward pass.
        autocast: ``"bfloat16"`` / ``"float16"`` / ``None``; default bf16 for ``mean``, fp32 for ``last``.
        keep_special_tokens: decode ids with special tokens kept (default: True for ``mean``, False for ``last``).
        cache_dir: optional directory for an on-disk feature cache.
        memory_cache: number of feature sets kept in memory.
    """

    def __init__(self, model: Any = "gpt2-large", device: Optional[str] = None, pooling: str = "mean", layer: int = -1,
                 max_length: int = 1024, batch_size: int = 8, autocast: Optional[str] = "default",
                 keep_special_tokens: Optional[bool] = None, cache_dir: Optional[str] = None, memory_cache: int = 16):
        if pooling not in ("mean", "last"):
            raise ValueError("pooling must be 'mean' or 'last'")
        self._judge_spec, self._device = model, device
        self._judge: Optional[JudgeLM] = model if isinstance(model, JudgeLM) else None
        self.pooling, self.layer, self.max_length, self.batch_size = pooling, layer, max_length, batch_size
        if autocast == "default":
            autocast = "bfloat16" if pooling == "mean" else None
        self.autocast = _DTYPES[autocast]
        self.keep_special_tokens = (pooling == "mean") if keep_special_tokens is None else keep_special_tokens
        self.cache_dir = cache_dir
        self._mem: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._mem_size = memory_cache

    @property
    def judge(self) -> JudgeLM:
        if self._judge is None:
            self._judge = get_judge(self._judge_spec, self._device)
        return self._judge

    @property
    def model_name(self) -> str:
        return self._judge.name if self._judge is not None else str(self._judge_spec)

    def config_key(self) -> str:
        return (f"{self.model_name}|{self.pooling}|{self.layer}|{self.max_length}|{self.autocast}|"
                f"{self.keep_special_tokens}")

    def __repr__(self) -> str:
        return (f"TextFeatureExtractor(model={self.model_name!r}, pooling={self.pooling!r}, layer={self.layer}, "
                f"max_length={self.max_length}, autocast={self.autocast})")

    # ------------------------------------------------------------------ public
    def __call__(self, samples: Any, tokenizer: TokenizerLike = None) -> np.ndarray:
        s = Samples.wrap(samples, tokenizer)
        key = hashlib.sha1((s.digest() + "|" + self.config_key()).encode()).hexdigest()
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]
        path = os.path.join(self.cache_dir, f"feat_{key[:24]}.npy") if self.cache_dir else None
        if path and os.path.exists(path):
            F = np.load(path)
        else:
            texts = s.texts(skip_special_tokens=not self.keep_special_tokens)
            F = self._mean(texts) if self.pooling == "mean" else self._last(texts)
            if path:
                os.makedirs(self.cache_dir, exist_ok=True)
                np.save(path + ".tmp.npy", F)
                os.replace(path + ".tmp.npy", path)
        self._mem[key] = F
        while len(self._mem) > self._mem_size:
            self._mem.popitem(last=False)
        return F

    # ------------------------------------------------------------------ poolings
    @torch.no_grad()
    def _mean(self, texts):
        j = self.judge
        feats = []
        for s in range(0, len(texts), self.batch_size):
            enc = j.tok(texts[s:s + self.batch_size], return_tensors="pt", padding=True, truncation=True,
                        max_length=self.max_length).to(j.device)
            with autocast_context(j.device, self.autocast):
                h = j.lm(**enc, output_hidden_states=True).hidden_states[self.layer].float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            feats.append(((h * m).sum(1) / m.sum(1).clamp_min(1)).cpu())
        return torch.cat(feats).numpy().astype(np.float64)

    @torch.no_grad()
    def _last(self, texts):
        j = self.judge
        kept = [t for t in texts if len(t) > 0]
        if len(kept) < len(texts):
            warnings.warn(f"{len(texts) - len(kept)} empty texts dropped (as mauve-text does)")
        toks = [torch.as_tensor(j.tok.encode(t, truncation=True, max_length=self.max_length)).view(-1) for t in kept]
        feats = []
        for s in range(0, len(toks), self.batch_size):
            chunk = toks[s:s + self.batch_size]
            lens = [len(c) for c in chunk]
            x = torch.nn.utils.rnn.pad_sequence(chunk, batch_first=True, padding_value=0).to(j.device)
            m = torch.nn.utils.rnn.pad_sequence([torch.ones(n, dtype=torch.long) for n in lens], batch_first=True,
                                                padding_value=0).to(j.device)
            with autocast_context(j.device, self.autocast):
                hs = j.lm(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states[self.layer].float()
            feats.append(torch.stack([hs[i, n - 1] for i, n in enumerate(lens)]).cpu())
        return torch.cat(feats).numpy().astype(np.float32)


def resolve_extractor(extractor: Any, device: Optional[str], pooling: str = "mean"):
    """The feature extractor of a metric: a ``TextFeatureExtractor`` (or any callable ``f(Samples) -> [n, d] array``,
    e.g. a sentence-embedding model of your choice) is used as is; a judge name / ``JudgeLM`` / ``None`` builds a
    ``TextFeatureExtractor`` with the given pooling (default judge gpt2-large)."""
    if isinstance(extractor, TextFeatureExtractor):
        return extractor
    if extractor is None or isinstance(extractor, (str, JudgeLM)):
        return TextFeatureExtractor(extractor or "gpt2-large", device=device, pooling=pooling)
    if callable(extractor):
        return extractor
    raise TypeError("extractor must be a TextFeatureExtractor, a callable, a judge name or None")
