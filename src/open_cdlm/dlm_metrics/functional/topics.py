"""Topic occupancy: k-means clusters of the REAL rows' features, and how the generated rows populate them.

``tv`` = total-variation distance between the cluster shares of the generated and the real rows; ``ratio`` = generated
share / real share per cluster (< 0.5 = an under-generated topic, > 2 = over-generated). Needs scikit-learn.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def topic_occupancy(X: np.ndarray, Y: np.ndarray, n_clusters: int = 16, seed: int = 0, n_init: int = 10) -> Dict[str, object]:
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("TopicOccupancy needs `pip install scikit-learn`") from e
    if len(X) < n_clusters:
        raise ValueError(f"TopicOccupancy needs at least n_clusters = {n_clusters} real rows, got {len(X)}")
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit(X)
    share_real = np.bincount(km.labels_, minlength=n_clusters) / len(X)
    share = np.bincount(km.predict(Y), minlength=n_clusters) / len(Y)
    ratio = share / np.maximum(share_real, 1e-12)
    return dict(tv=float(0.5 * np.abs(share - share_real).sum()), min_ratio=float(ratio.min()),
                n_under_half=int((ratio < 0.5).sum()), n_over_double=int((ratio > 2).sum()),
                share_real=share_real.tolist(), share=share.tolist(), ratio=ratio.tolist())
