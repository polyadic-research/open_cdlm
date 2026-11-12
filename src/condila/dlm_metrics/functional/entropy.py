"""Unigram entropies of token-id rows (nats).

* ``H_row`` ("sample entropy"): mean over rows of the entropy of each row's own unigram distribution. This is
  LangFlow's ``gen_ppl.sample_entropy`` and FLM's ``record_entropy``. It is a WITHIN-row statistic: ``n`` copies of one
  row keep it unchanged.
* ``H_pool``: entropy of the unigram distribution of all rows pooled.
* ``JSD_rows`` = ``H_pool - sum_r w_r H(p_r)`` with ``w_r`` = row length / total length: the generalized Jensen-Shannon
  divergence of the per-row distributions, ``sum_r w_r KL(p_r || p_pool)`` (= ``H_pool - H_row`` for equal-length rows).
  BETWEEN-row diversity: how differently the rows use the vocabulary.

``H_pool`` and ``JSD_rows`` grow with the number of rows and the row length; compare them only at equal ``n`` and ``L``.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def _filter(r: np.ndarray, ignore_ids: Optional[Sequence[int]]) -> np.ndarray:
    r = np.asarray(r, dtype=np.int64).reshape(-1)
    if ignore_ids:
        r = r[~np.isin(r, np.asarray(list(ignore_ids), dtype=np.int64))]
    return r


def row_counts(rows: Iterable[np.ndarray], ignore_ids: Optional[Sequence[int]] = None):
    """Per row: (unique ids, counts)."""
    return [np.unique(_filter(r, ignore_ids), return_counts=True) for r in rows]


def _entropy_from_counts(c: np.ndarray) -> float:
    c = np.asarray(c, dtype=np.float64)
    c = c[c > 0]
    if c.size == 0:
        return float("nan")
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def entropy_from_row_counts(counts: List, idx: Optional[np.ndarray] = None) -> Dict[str, float]:
    """``H_row``, ``H_pool``, ``JSD_rows`` from per-row (unique, counts) pairs, optionally on a subset of rows."""
    sel = counts if idx is None else [counts[i] for i in idx]
    H = np.array([_entropy_from_counts(c) for _, c in sel])
    lens = np.array([c.sum() for _, c in sel], dtype=np.float64)
    if len(sel) == 0 or lens.sum() == 0:
        return dict(H_row=float("nan"), H_pool=float("nan"), JSD_rows=float("nan"))
    u = np.concatenate([u for u, _ in sel])
    c = np.concatenate([c for _, c in sel]).astype(np.float64)
    pooled = np.bincount(u - u.min(), weights=c)
    H_pool = _entropy_from_counts(pooled)
    ok = lens > 0
    w = lens[ok] / lens[ok].sum()
    return dict(H_row=float(np.nanmean(H)), H_pool=H_pool, JSD_rows=float(H_pool - (w * H[ok]).sum()))


def row_entropy(rows: Iterable[np.ndarray], ignore_ids: Optional[Sequence[int]] = None) -> np.ndarray:
    """Entropy (nats) of each row's unigram distribution."""
    return np.array([_entropy_from_counts(c) for _, c in row_counts(rows, ignore_ids)])


def sample_entropy(rows: Iterable[np.ndarray], ignore_ids: Optional[Sequence[int]] = None) -> float:
    """``H_row``: the mean per-row unigram entropy (LangFlow / FLM "sample entropy")."""
    return float(np.nanmean(row_entropy(rows, ignore_ids)))


def pooled_entropy(rows: Iterable[np.ndarray], ignore_ids: Optional[Sequence[int]] = None) -> float:
    """``H_pool``: entropy of the pooled unigram distribution of all rows."""
    return entropy_from_row_counts(row_counts(rows, ignore_ids))["H_pool"]


def entropy_decomposition(rows: Iterable[np.ndarray], ignore_ids: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """``dict(H_row, H_pool, JSD_rows)``; see the module docstring."""
    return entropy_from_row_counts(row_counts(rows, ignore_ids))


def split_halves(n: int, repeats: int = 8, seed: int = 0) -> List[np.ndarray]:
    """The random half-splits used for the +- of every split-half SD (same halves for all metrics of one set)."""
    g = np.random.default_rng(seed)
    return [np.sort(g.permutation(n)[: n // 2]) for _ in range(repeats)]
