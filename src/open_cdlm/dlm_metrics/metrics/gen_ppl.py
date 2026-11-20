"""Generative perplexity as a stateful metric."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from ..base import Metric
from ..functional import gen_ppl as G
from ..functional.entropy import split_halves
from ..inputs import Samples, TokenizerLike
from ..models import get_judge


class GenerativePerplexity(Metric):
    """``exp(sum NLL / sum judge tokens)`` of the samples under a judge LM.

    Args:
        judge: judge model name (default ``gpt2-large``, fp32) or a :class:`~open_cdlm.dlm_metrics.models.JudgeLM`.
        tokenizer: the generator tokenizer (``gpt2`` for OWT models, ``bert-base-uncased`` for LM1B models).
        protocol: ``"langflow"`` (LangFlow gen_ppl.py; default) or ``"flm"`` (MDLM / DUO / FLM metrics.py).
            See :mod:`open_cdlm.dlm_metrics.functional.gen_ppl`; never mix protocols in one table.
        max_length: judge tokens per row (default 1024 for ``langflow``, the generator row length for ``flm``).
        batch_size: judge batch (default 16 for rows > 256 tokens, else 64 / 32).
        device: judge device.
        split_half: random half-splits for ``gen_ppl_sd_half`` (0 = off).
        return_rows: also return the per-row NLL per token.

    Gen ppl alone rewards narrow, repetitive samples: always report it next to ``H_row`` and ``JSD_rows``
    (:class:`EntropyDecomposition`) and compare samplers at matched entropy, or use a distribution distance (``MMD``).
    """
    higher_is_better = False

    def __init__(self, judge: Any = "gpt2-large", tokenizer: TokenizerLike = None, protocol: str = "langflow",
                 max_length: Optional[int] = None, batch_size: Optional[int] = None, device: Optional[str] = None,
                 split_half: int = 0, seed: int = 0, return_rows: bool = False, **kw):
        super().__init__(tokenizer=tokenizer, **kw)
        if protocol not in G.PROTOCOLS:
            raise ValueError(f"protocol must be one of {G.PROTOCOLS}")
        self.judge_spec, self.device = judge, device
        self.protocol, self.max_length, self.batch_size = protocol, max_length, batch_size
        self.split_half, self.seed, self.return_rows = split_half, seed, return_rows
        self._hparams = dict(judge=getattr(judge, "name", judge), protocol=protocol, max_length=max_length)
        self.add_state("row_nll")
        self.add_state("row_tokens")
        self.add_state("pending")         # flm protocol: rows are scored together in compute()

    @property
    def judge(self):
        return get_judge(self.judge_spec, self.device)

    def update(self, samples: Any) -> None:
        s = self._wrap(samples)
        if self.protocol == "flm":
            if not s.has_ids:
                raise ValueError("protocol 'flm' needs token-id samples")
            self.pending.append(s)
            return
        nll, tok = G.row_nll(s, self.judge, "langflow", self.batch_size, self.max_length)
        self.row_nll.extend(nll.tolist())
        self.row_tokens.extend(tok.tolist())

    def _compute(self) -> Dict[str, Any]:
        nll, tok = list(self.row_nll), list(self.row_tokens)
        if self.pending:
            s = Samples.concat(self.pending)
            a, b = G.row_nll(s, self.judge, "flm", self.batch_size, self.max_length)
            nll, tok = nll + a.tolist(), tok + b.tolist()
        nll, tok = np.asarray(nll, np.float64), np.asarray(tok, np.float64)
        if tok.sum() == 0:
            raise RuntimeError("no samples were scored")
        val = nll.sum() / tok.sum()
        out = dict(gen_ppl=math.exp(val), nll_per_token=float(val), n_rows=int(len(nll)), n_tokens=int(tok.sum()),
                   protocol=self.protocol)
        if self.split_half:
            v = [math.exp(nll[h].sum() / tok[h].sum()) for h in split_halves(len(nll), self.split_half, self.seed)]
            out["gen_ppl_sd_half"] = float(np.std(v))
        if self.return_rows:
            out["row_nll_per_token"] = nll / np.maximum(tok, 1)
        return out
