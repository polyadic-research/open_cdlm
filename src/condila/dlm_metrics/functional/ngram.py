"""Token-level n-gram statistics of id rows: repetition, cross-row novelty, near-duplicates, n-gram Vendi score.

* ``rep_n``: mean over rows of ``1 - |distinct n-grams| / |n-grams|`` (within-row repetition; "rep-4").
* ``distinct_n``: corpus-level ``|distinct n-grams| / |n-grams|`` over all rows (Li et al. 2016).
* ``ngram_diversity``: on each row's SET of n-grams,
    - ``VS``  Vendi score (Friedman & Dieng 2023) of the Jaccard kernel, divided by n: the effective number of
      distinct rows as a fraction of n (``k`` copies of ``n/k`` rows give exactly ``1/k``),
    - ``xD``  cross-row novelty ``|union of the rows' n-gram sets| / sum of their sizes`` (k-fold duplication -> 1/k),
    - ``NDR`` share of rows whose most similar other row has Jaccard >= ``near_dup_threshold``,
    - ``MPJ`` mean pairwise Jaccard.
* ``exact_duplicate_rate``: share of rows identical to another row.
* ``lzma_bits_per_token``: lzma-compressed size of all rows as one stream, in bits per token.

n-grams are matched exactly (vectorised ``np.unique`` on the windows, no hashing).
"""
from __future__ import annotations

import lzma
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:  # pragma: no cover - numpy < 1.20
    sliding_window_view = None


def _rows(rows: Iterable) -> List[np.ndarray]:
    return [np.asarray(r, dtype=np.int64).reshape(-1) for r in rows]


def gram_ids(rows: Iterable, n: int) -> Tuple[List[np.ndarray], int]:
    """Per row: the ids of its n-grams (with repetition) in a shared exact n-gram index; and the index size."""
    rows = _rows(rows)
    wins, owner = [], []
    for i, r in enumerate(rows):
        if len(r) >= n:
            w = sliding_window_view(r, n)
            wins.append(w)
            owner.append(np.full(len(w), i))
    if not wins:
        return [np.zeros(0, dtype=np.int64) for _ in rows], 0
    W = np.concatenate(wins)
    _, inv = np.unique(W, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    owner = np.concatenate(owner)
    out = [np.zeros(0, dtype=np.int64) for _ in rows]
    order = np.argsort(owner, kind="stable")
    bounds = np.searchsorted(owner[order], np.arange(len(rows) + 1))
    for i in range(len(rows)):
        out[i] = inv[order[bounds[i]:bounds[i + 1]]]
    return out, int(inv.max()) + 1


def rep_n(rows: Iterable, n: int = 4) -> float:
    """Mean within-row n-gram repetition rate ``1 - distinct/total`` (rows shorter than n are skipped)."""
    g, _ = gram_ids(rows, n)
    vals = [1.0 - len(np.unique(x)) / len(x) for x in g if len(x)]
    return float(np.mean(vals)) if vals else float("nan")


def distinct_n(rows: Iterable, n: int = 4) -> float:
    """Corpus-level distinct-n: distinct n-grams over all rows / total n-grams."""
    g, size = gram_ids(rows, n)
    total = sum(len(x) for x in g)
    return float(size / total) if total else float("nan")


def vendi(K: np.ndarray) -> float:
    """exp(von Neumann entropy of K / n) for a PSD kernel matrix K with unit diagonal."""
    lam = np.linalg.eigvalsh(K / K.shape[0])
    lam = lam[lam > 1e-12]
    return float(math.exp(-(lam * np.log(lam)).sum()))


def gram_matrix(rows: Iterable, n: int = 4):
    """Binary rows x distinct-n-gram matrix (csr) and the per-row set sizes."""
    g, size = gram_ids(rows, n)
    r_idx, c_idx = [], []
    for i, x in enumerate(g):
        u = np.unique(x)
        r_idx.append(np.full(len(u), i))
        c_idx.append(u)
    r_idx = np.concatenate(r_idx) if r_idx else np.zeros(0, dtype=np.int64)
    c_idx = np.concatenate(c_idx) if c_idx else np.zeros(0, dtype=np.int64)
    B = sp.csr_matrix((np.ones(len(r_idx), dtype=np.float32), (r_idx, c_idx)), shape=(len(g), size))
    return B, np.asarray(B.sum(1)).ravel()


def ngram_diversity(rows: Iterable, n: int = 4, near_dup_threshold: float = 0.5) -> Dict[str, float]:
    """``dict(VS, xD, NDR, MPJ)`` of the rows' n-gram sets; see the module docstring. O(rows^2) memory."""
    B, sizes = gram_matrix(rows, n)
    m = B.shape[0]
    inter = (B @ B.T).toarray().astype(np.float64)
    union = sizes[:, None] + sizes[None, :] - inter
    J = inter / np.maximum(union, 1.0)
    off = J[~np.eye(m, dtype=bool)]
    nn = (J - np.eye(m)).max(1)
    return dict(VS=vendi(J) / m, xD=float(B.shape[1] / max(sizes.sum(), 1)), NDR=float((nn >= near_dup_threshold).mean()),
                MPJ=float(off.mean()) if off.size else float("nan"))


def exact_duplicate_rate(rows: Iterable) -> float:
    """Share of rows that are token-for-token identical to at least one other row."""
    keys = [np.asarray(r, dtype=np.int64).tobytes() for r in rows]
    counts: Dict[bytes, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return float(np.mean([counts[k] > 1 for k in keys])) if keys else float("nan")


def lzma_bits_per_token(rows: Iterable) -> float:
    """lzma (preset 9 extreme) size of all rows concatenated, in bits per token: a universal-coding proxy for the entropy
    rate of the sample distribution (duplicates and shared templates compress away)."""
    a = np.concatenate(_rows(rows))
    dt = np.uint16 if a.max() < 65536 else np.uint32
    raw = a.astype(dt).tobytes()
    return 8.0 * len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)) / a.size
