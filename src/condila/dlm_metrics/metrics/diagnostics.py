"""Diagnostics that localise a loss: which attributes / real rows a sample set under-generates."""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from ..base import Metric
from ..functional.attributes import DEFAULT_ATTRIBUTES, attribute_retention, attribute_table
from ..functional.kn import reverse_ce
from ..inputs import Samples, TokenizerLike, get_tokenizer


def _eos_id(tokenizer) -> Optional[int]:
    tok = get_tokenizer(tokenizer)
    for name in ("eos_token_id", "sep_token_id"):
        v = getattr(tok, name, None)
        if v is not None:
            return int(v)
    return None


class AttributePanel(Metric):
    """Per-row text attributes (digits, non-ASCII, code characters, URLs, quotes, line breaks, capitalised words,
    word length, documents per row) and their retention against real rows.

    ``retention[attr]`` = share of generated rows above the real ``percentile``-th percentile / share of real rows
    above it (1 = kept, < 1 = lost, > 1 = over-produced). Sparse failures (a few percent of tokens or rows) are
    visible here and nowhere else.

    Args:
        attributes: ``{name: f(text) -> float}`` (default :data:`DEFAULT_ATTRIBUTES`); texts have special tokens removed.
        percentile: threshold percentile of the real distribution.
        count_documents: add ``docs_per_row`` = number of EOS / SEP tokens in the row.
    """
    is_two_sample = True

    def __init__(self, attributes: Optional[Mapping[str, Callable[[str], float]]] = None, percentile: float = 90.0,
                 count_documents: bool = True, reset_real_features: bool = True, tokenizer: TokenizerLike = None, **kw):
        super().__init__(tokenizer=tokenizer, **kw)
        self.attributes = dict(DEFAULT_ATTRIBUTES if attributes is None else attributes)
        self.percentile, self.count_documents = percentile, count_documents
        self._hparams = dict(attributes=list(self.attributes), percentile=percentile)
        self.add_state("real_tables", keep_on_reset=not reset_real_features)
        self.add_state("fake_tables")

    def _table(self, s: Samples) -> Dict[str, np.ndarray]:
        docs = None
        if self.count_documents and s.tokenizer is not None:
            eos = _eos_id(s.tokenizer)
            if eos is not None:
                docs = [float((r == eos).sum()) for r in s.ids()]
        return attribute_table(s.texts(skip_special_tokens=True), self.attributes, docs)

    def update(self, samples: Any, real: bool = False) -> None:
        (self.real_tables if real else self.fake_tables).append(self._table(self._wrap(samples)))

    @staticmethod
    def _cat(tables):
        return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}

    def _compute(self) -> Dict[str, Any]:
        if not self.real_tables or not self.fake_tables:
            raise RuntimeError("AttributePanel needs real and generated rows")
        R, G = self._cat(self.real_tables), self._cat(self.fake_tables)
        per = attribute_retention(R, G, self.percentile)
        ret = {k: v["retention"] for k, v in per.items()}
        finite = {k: v for k, v in ret.items() if np.isfinite(v)}
        lost = min(finite, key=finite.get) if finite else None
        return dict(retention=ret, mean_ratio={k: v["mean_ratio"] for k, v in per.items()}, detail=per,
                    min_retention=float(finite[lost]) if lost else float("nan"), min_retention_attribute=lost,
                    max_retention=float(max(finite.values())) if finite else float("nan"),
                    n_real=len(next(iter(R.values()))), n_fake=len(next(iter(G.values()))))


class ReverseNGramCE(Metric):
    """Coverage in token space: KN n-gram LM fit on the generated rows vs one fit on as many real rows, both scored on
    held-out real rows. ``reverse_ce_excess`` > 0 = real text is worse explained by the samples than by real text.

    Real rows passed with ``update(..., real=True)`` are split: the first ``n_fit`` (= min(#generated, #real // 2))
    fit the real LM, the rest are the test rows. ``row_excess`` (per test row) points at the lost modes.
    """
    is_two_sample = True
    higher_is_better = False

    def __init__(self, order: int = 4, discount: float = 0.75, vocab_size: Optional[int] = None,
                 reset_real_features: bool = True, tokenizer: TokenizerLike = None, **kw):
        super().__init__(tokenizer=tokenizer, **kw)
        self.order, self.discount, self.vocab_size = order, discount, vocab_size
        self._hparams = dict(order=order, discount=discount)
        self.add_state("real_rows", keep_on_reset=not reset_real_features)
        self.add_state("fake_rows")

    def update(self, samples: Any, real: bool = False) -> None:
        (self.real_rows if real else self.fake_rows).extend(self._wrap(samples).ids())

    def _compute(self) -> Dict[str, Any]:
        n_fit = min(len(self.fake_rows), len(self.real_rows) // 2)
        if n_fit < 1:
            raise RuntimeError("ReverseNGramCE needs generated rows and at least 2 real rows")
        V = self.vocab_size
        if V is None:
            V = (len(get_tokenizer(self.tokenizer)) if self.tokenizer is not None
                 else int(max(max(int(r.max()) for r in self.real_rows), max(int(r.max()) for r in self.fake_rows))) + 1)
        out = reverse_ce([r.tolist() for r in self.fake_rows[:n_fit]], [r.tolist() for r in self.real_rows[:n_fit]],
                         [r.tolist() for r in self.real_rows[n_fit:]], V, self.order, self.discount)
        out.update(n_fit=n_fit, n_test=len(self.real_rows) - n_fit)
        return out
