import os
import sys

import numpy as np
import pytest



class StubExtractor:
    """Deterministic 'features' of id rows: a fixed random projection of the unigram histogram (no model)."""
    pooling = "mean"

    def __init__(self, vocab=64, dim=16, seed=0):
        self.W = np.random.default_rng(seed).normal(size=(vocab, dim))
        self.calls = 0

    def __call__(self, samples):
        self.calls += 1
        out = []
        for r in samples.ids():
            h = np.bincount(r, minlength=self.W.shape[0]).astype(np.float64)
            out.append(h @ self.W / max(len(r), 1))
        return np.stack(out)


@pytest.fixture
def stub():
    return StubExtractor()


@pytest.fixture
def rng():
    return np.random.default_rng(0)
