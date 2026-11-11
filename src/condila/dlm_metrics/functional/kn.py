"""Interpolated Kneser-Ney n-gram LMs on token-id rows and the reverse (coverage) cross-entropy.

``reverse_ce``: fit one KN model on the generated rows and one on the same number of real rows, score held-out real
rows with both. ``excess = CE(real_test | LM_generated) - CE(real_test | LM_real)`` estimates how much worse the
generated distribution covers real text (duplication, phrase narrowing and missing topics all raise it); the per-row
excess points at the real rows (modes) the generated set explains worst. A diagnostic: pure Python, roughly a minute
per 500k tokens.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np

BOS = -1


class KneserNeyLM:
    """Interpolated Kneser-Ney model of order ``order`` with absolute discount ``discount`` and a uniform 1/V floor."""

    def __init__(self, rows: Iterable[Sequence[int]], vocab_size: int, order: int = 4, discount: float = 0.75):
        self.V, self.N, self.D = int(vocab_size), int(order), float(discount)
        N = self.N
        grams = [set() for _ in range(N + 1)]
        raw = defaultdict(int)
        for r in rows:
            seq = [BOS] * (N - 1) + [int(t) for t in r]
            for i in range(N - 1, len(seq)):
                for k in range(1, N + 1):
                    grams[k].add(tuple(seq[i - k + 1:i + 1]))
                raw[(tuple(seq[i - N + 1:i]), seq[i])] += 1
        self.num: List = [None] * (N + 1)
        self.den: List = [None] * (N + 1)
        self.types: List = [None] * (N + 1)
        self.num[N] = dict(raw)
        den, typ = defaultdict(int), defaultdict(int)
        for (h, _), c in raw.items():
            den[h] += c
            typ[h] += 1
        self.den[N], self.types[N] = dict(den), dict(typ)
        for k in range(1, N):
            cont = defaultdict(int)
            for g in grams[k + 1]:
                cont[(g[1:-1], g[-1])] += 1
            den, typ = defaultdict(int), defaultdict(int)
            for (h, _), c in cont.items():
                den[h] += c
                typ[h] += 1
            self.num[k], self.den[k], self.types[k] = dict(cont), dict(den), dict(typ)

    def logp(self, hist: tuple, w: int) -> float:
        p = 1.0 / self.V
        for k in range(1, self.N + 1):
            h = hist[len(hist) - (k - 1):] if k > 1 else ()
            d = self.den[k].get(h, 0)
            if d == 0:
                continue
            c = self.num[k].get((h, w), 0)
            p = max(c - self.D, 0.0) / d + (self.D * self.types[k][h] / d) * p
        return math.log(p)

    def row_cross_entropy(self, rows: Iterable[Sequence[int]]) -> np.ndarray:
        """Mean -log p per token, for each row."""
        N, out = self.N, []
        for r in rows:
            seq = [BOS] * (N - 1) + [int(t) for t in r]
            tot = sum(-self.logp(tuple(seq[i - N + 1:i]), seq[i]) for i in range(N - 1, len(seq)))
            out.append(tot / max(len(seq) - (N - 1), 1))
        return np.array(out)

    def cross_entropy(self, rows: Iterable[Sequence[int]]) -> float:
        """Mean -log p over all tokens of all rows."""
        rows = [list(r) for r in rows]
        ce = self.row_cross_entropy(rows)
        w = np.array([len(r) for r in rows], dtype=np.float64)
        return float((ce * w).sum() / w.sum())


def reverse_ce(generated: List[Sequence[int]], real_fit: List[Sequence[int]], real_test: List[Sequence[int]],
               vocab_size: int, order: int = 4, discount: float = 0.75) -> Dict[str, object]:
    """``dict(reverse_ce_excess, ce_real, ce_generated, row_excess)``; fit sets should have equal size."""
    lm_g = KneserNeyLM(generated, vocab_size, order, discount)
    lm_r = KneserNeyLM(real_fit, vocab_size, order, discount)
    ce_g, ce_r = lm_g.row_cross_entropy(real_test), lm_r.row_cross_entropy(real_test)
    w = np.array([len(r) for r in real_test], dtype=np.float64)
    ex = ce_g - ce_r
    return dict(reverse_ce_excess=float((ex * w).sum() / w.sum()), ce_real=float((ce_r * w).sum() / w.sum()),
                ce_generated=float((ce_g * w).sum() / w.sum()), row_excess=ex)
