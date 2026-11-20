"""Token-level metrics: unigram entropies, repetition, n-gram diversity, duplicates, compression. No model needed."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..base import Metric
from ..functional import entropy as E
from ..functional import ngram as NG
from ..inputs import TokenizerLike


class EntropyDecomposition(Metric):
    """``H_row`` (sample entropy), ``H_pool`` (pooled entropy) and ``JSD_rows = H_pool - sum_r w_r H_r`` in nats.

    Args:
        ignore_ids: token ids left out of all counts (e.g. a padding id). Default: count every id, as LangFlow / FLM do.
        split_half: number of random half-splits for ``*_sd_half`` (SD of the value on n/2 rows; the SD at n is about
            ``1/sqrt(2)`` of it). 0 = off.
        seed: seed of the half-splits.
        tokenizer: generator tokenizer, needed only when you pass texts.
    """

    def __init__(self, ignore_ids: Optional[Sequence[int]] = None, split_half: int = 0, seed: int = 0,
                 tokenizer: TokenizerLike = None, **kw):
        super().__init__(tokenizer=tokenizer, **kw)
        self.ignore_ids, self.split_half, self.seed = ignore_ids, split_half, seed
        self._hparams = dict(ignore_ids=ignore_ids, split_half=split_half)
        self.add_state("counts")

    def update(self, samples: Any) -> None:
        self.counts.extend(E.row_counts(self._wrap(samples).ids(), self.ignore_ids))

    def _compute(self) -> Dict[str, float]:
        out = E.entropy_from_row_counts(self.counts)
        out["n_rows"] = len(self.counts)
        if self.split_half:
            vals = [E.entropy_from_row_counts(self.counts, h) for h in E.split_halves(len(self.counts), self.split_half, self.seed)]
            for k in ("H_row", "H_pool", "JSD_rows"):
                out[f"{k}_sd_half"] = float(np.std([v[k] for v in vals]))
        return out


class SampleEntropy(EntropyDecomposition):
    """``H_row``: mean per-row unigram entropy (LangFlow / FLM "sample entropy"). Within-row diversity only."""
    higher_is_better = None

    def _compute(self):
        out = super()._compute()
        return {k: v for k, v in out.items() if k.startswith("H_row") or k == "n_rows"}


class PooledEntropy(EntropyDecomposition):
    """``H_pool``: entropy of the pooled unigram distribution (depends on n and L)."""

    def _compute(self):
        out = super()._compute()
        return {k: v for k, v in out.items() if k.startswith("H_pool") or k == "n_rows"}


class JSDRows(EntropyDecomposition):
    """``JSD_rows``: between-row diversity, the generalized JS divergence of the per-row unigram distributions."""
    higher_is_better = None

    def _compute(self):
        out = super()._compute()
        return {k: v for k, v in out.items() if k.startswith("JSD_rows") or k == "n_rows"}


class _IdRowsMetric(Metric):
    def __init__(self, tokenizer: TokenizerLike = None, **kw):
        super().__init__(tokenizer=tokenizer, **kw)
        self.add_state("rows")

    def update(self, samples: Any) -> None:
        self.rows.extend(self._wrap(samples).ids())


class RepetitionRate(_IdRowsMetric):
    """``rep_n``: mean within-row share of repeated n-grams, ``1 - distinct/total`` (lower = less repetition)."""
    higher_is_better = False

    def __init__(self, n: int = 4, **kw):
        super().__init__(**kw)
        self.n = n
        self._hparams = dict(n=n)

    def _compute(self):
        return {f"rep_{self.n}": NG.rep_n(self.rows, self.n), "n_rows": len(self.rows)}


class DistinctN(_IdRowsMetric):
    """``distinct_n``: corpus-level distinct n-grams / total n-grams."""
    higher_is_better = True

    def __init__(self, n: int = 4, **kw):
        super().__init__(**kw)
        self.n = n
        self._hparams = dict(n=n)

    def _compute(self):
        return {f"distinct_{self.n}": NG.distinct_n(self.rows, self.n), "n_rows": len(self.rows)}


class NGramDiversity(_IdRowsMetric):
    """Cross-row diversity of the rows' n-gram sets: ``VS`` (Vendi of the Jaccard kernel / n), ``xD`` (cross-row
    novelty), ``NDR`` (near-duplicate rate at ``near_dup_threshold``), ``MPJ`` (mean pairwise Jaccard). O(n^2) memory."""

    def __init__(self, n: int = 4, near_dup_threshold: float = 0.5, **kw):
        super().__init__(**kw)
        self.n, self.near_dup_threshold = n, near_dup_threshold
        self._hparams = dict(n=n, near_dup_threshold=near_dup_threshold)

    def _compute(self):
        out = NG.ngram_diversity(self.rows, self.n, self.near_dup_threshold)
        out = {f"{k}_{self.n}": v for k, v in out.items()}
        out["n_rows"] = len(self.rows)
        return out


class ExactDuplicateRate(_IdRowsMetric):
    """Share of rows that are token-for-token copies of another row."""
    higher_is_better = False

    def _compute(self):
        return {"exact_dup_rate": NG.exact_duplicate_rate(self.rows), "n_rows": len(self.rows)}


class CompressionRate(_IdRowsMetric):
    """lzma bits per token of all rows as one stream (duplicates and shared templates compress away)."""

    def _compute(self):
        return {"lzma_bits_per_token": NG.lzma_bits_per_token(self.rows), "n_rows": len(self.rows)}
