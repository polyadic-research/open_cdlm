"""Generative perplexity: the NLL of decoded samples under a judge LM (default ``gpt2-large``, fp32).

Two protocols, each a faithful copy of one code base. They give different numbers for the same samples: never mix them.

``"langflow"`` (LangFlow ``gen_ppl.py``)
    decode with the generator tokenizer keeping special tokens (``[CLS]`` / ``[SEP]`` / ``<|endoftext|>`` become text),
    re-tokenize for the judge with truncation at ``max_length`` (1024) judge tokens and padding per batch, score every
    judge token except the first; ``gen_ppl = exp(sum NLL / sum tokens)``.
``"flm"`` (``Metrics.record_generative_perplexity`` of MDLM / DUO / FLM)
    decode with their ``get_tokenizer``, re-tokenize ALL rows at once (padding to the longest row of the set) with
    truncation at ``max_length`` = the generator's row length, score the judge tokens that are not EOS plus the first EOS
    of a row (so one padding position counts when a row has no EOS of its own). The original drops the last
    ``len % batch_size`` rows; here every row counts.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..inputs import Samples, flm_tokenizer
from ..models import JudgeLM

PROTOCOLS = ("langflow", "flm")


@torch.no_grad()
def judge_nll_langflow(texts: List[str], judge: JudgeLM, batch_size: int = 16, max_length: int = 1024
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Per row: NLL summed over the scored judge tokens, and their number."""
    row_nll, row_tok = [], []
    for s in range(0, len(texts), batch_size):
        enc = judge.tok(texts[s:s + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        x, m = enc["input_ids"].to(judge.device), enc["attention_mask"].to(judge.device)
        logits = judge.lm(input_ids=x, attention_mask=m).logits[:, :-1].float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none").view(x.shape[0], -1)
        mm = m[:, 1:].bool()
        row_nll.append((ce * mm).sum(1).double().cpu())
        row_tok.append(mm.sum(1).double().cpu())
    return torch.cat(row_nll).numpy(), torch.cat(row_tok).numpy()


def _flm_valid_mask(chunk: torch.Tensor, eos_id: int) -> torch.Tensor:
    first_eos = (chunk == eos_id).cumsum(-1) == 1
    token_mask = chunk != eos_id
    return first_eos[..., 1:] + token_mask[..., 1:]


@torch.no_grad()
def judge_nll_flm(id_lists: List[List[int]], tokenizer: str, judge: JudgeLM, batch_size: int, max_length: int,
                  context: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    texts = flm_tokenizer(tokenizer).batch_decode(id_lists) if isinstance(tokenizer, str) else tokenizer.batch_decode(id_lists)
    enc = judge.tok(texts, return_tensors="pt", return_token_type_ids=False, return_attention_mask=True,
                    truncation=True, padding=True, max_length=max_length)
    samples, attn = enc["input_ids"], enc["attention_mask"]
    eos = judge.tok.eos_token_id
    row_nll, row_tok = [], []
    for s in range(0, samples.shape[0], batch_size):
        nll_b = torch.zeros(min(batch_size, samples.shape[0] - s), dtype=torch.float64)
        tok_b = torch.zeros_like(nll_b)
        for chunk, mchunk in zip(torch.split(samples[s:s + batch_size], context, dim=-1),
                                 torch.split(attn[s:s + batch_size], context, dim=-1)):
            chunk, mchunk = chunk.to(judge.device), mchunk.to(judge.device)
            logits = judge.lm(chunk, attention_mask=mchunk)[0].transpose(-1, -2)
            nlls = F.cross_entropy(logits[..., :-1], chunk[..., 1:], reduction="none")
            valid = _flm_valid_mask(chunk, eos)
            nll_b += (nlls * valid).sum(-1).double().cpu()
            tok_b += valid.sum(-1).double().cpu()
        row_nll.append(nll_b)
        row_tok.append(tok_b)
    return torch.cat(row_nll).numpy(), torch.cat(row_tok).numpy()


def row_nll(samples: Samples, judge: JudgeLM, protocol: str = "langflow", batch_size: int = None,
            max_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """(per-row NLL summed over the row, judge tokens per row) under the chosen protocol."""
    L = max(samples.lengths()) if samples.has_ids else 1024
    if protocol == "langflow":
        bs = batch_size or (64 if L <= 256 else 16)
        return judge_nll_langflow(samples.texts(skip_special_tokens=False), judge, bs, max_length or 1024)
    if protocol == "flm":
        if not samples.has_ids:
            raise ValueError("protocol 'flm' needs token-id samples")
        bs = batch_size or (32 if L <= 256 else 16)
        return judge_nll_flm([r.tolist() for r in samples.ids()], samples.tokenizer, judge, bs, max_length or L)
    raise ValueError(f"protocol must be one of {PROTOCOLS}")


def generative_perplexity(samples, judge="gpt2-large", tokenizer=None, protocol="langflow", batch_size=None,
                          max_length=None, device=None) -> float:
    """Functional one-shot gen ppl: ``exp(sum NLL / sum judge tokens)``."""
    from ..models import get_judge

    s = Samples.wrap(samples, tokenizer)
    nll, tok = row_nll(s, get_judge(judge, device), protocol, batch_size, max_length)
    return float(np.exp(nll.sum() / tok.sum()))
