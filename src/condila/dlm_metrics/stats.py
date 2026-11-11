"""Uncertainty helpers. A metric difference is readable only against the noise of the metric at the same n.

* :func:`split_half_sd` -- SD of a statistic over random half-splits of the rows (describes n/2 rows; the SD at n is
  about ``1/sqrt(2)`` of it for statistics that average over rows).
* :func:`mmd_permutation_null` -- MMD of random re-splits of the pooled real + generated rows (noise floor of MMD).
* :func:`bootstrap_se` -- bootstrap SE of a (weighted) mean of per-row values.
* :func:`real_floor` -- draw two disjoint random sets of real rows: one reference and one "real" set scored like a
  generator. Every metric of the real set is the value a perfect generator of that size gets.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .functional.entropy import split_halves
from .functional.kernel import mmd_permutation_null
from .inputs import Samples

__all__ = ["split_half_sd", "mmd_permutation_null", "bootstrap_se", "real_floor", "split_halves"]


def split_half_sd(fn: Callable[[np.ndarray], Dict[str, float]], n: int, repeats: int = 8, seed: int = 0) -> Dict[str, float]:
    """``fn(row_indices) -> dict of floats``; returns the SD of every key over ``repeats`` random halves."""
    acc: Dict[str, list] = {}
    for h in split_halves(n, repeats, seed):
        for k, v in fn(h).items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.std(v)) for k, v in acc.items()}


def bootstrap_se(values: Sequence[float], weights: Optional[Sequence[float]] = None, repeats: int = 1000,
                 seed: int = 0) -> float:
    v = np.asarray(values, np.float64)
    w = np.ones_like(v) if weights is None else np.asarray(weights, np.float64)
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(v), size=(repeats, len(v)))
    return float(((v[idx] * w[idx]).sum(1) / w[idx].sum(1)).std(ddof=1))


def real_floor(reference: Samples, n: int, seed: int = 0, shuffle: bool = True) -> Tuple[Samples, Samples]:
    """(reference rows, real rows): two disjoint sets of ``n`` rows drawn at random from ``reference``.

    Use RANDOM rows of the evaluation split: consecutive wrapped rows often come from the same document and share
    vocabulary, which understates the cross-row diversity of real text."""
    if len(reference) < 2 * n:
        raise ValueError(f"the reference has {len(reference)} rows; {2 * n} are needed for n = {n} "
                         f"(n reference + n real rows)")
    order = np.random.default_rng(seed).permutation(len(reference)) if shuffle else np.arange(len(reference))
    return reference.subset(order[:n]), reference.subset(order[n:2 * n])
