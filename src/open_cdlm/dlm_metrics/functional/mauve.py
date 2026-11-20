"""MAUVE (Pillutla et al. 2021) from precomputed features, via the ``mauve-text`` package (``pip install mauve-text
faiss-cpu``; GPLv3, imported at run time only).

``num_buckets``: ``"auto"`` = ``round(min(n_p, n_q) / 10)`` (the package default), an int, or a float in (0, 1) read as a
fraction of ``min(n_p, n_q)``. The score depends on n and on the number of buckets (the real-vs-real value falls as k
grows): compare sets only at equal n and k. The reported value is the mean over k-means seeds, with their SD.
"""
from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np

DEFAULT_SEEDS = (25, 26, 27, 28, 29)


def _buckets(num_buckets: Union[str, int, float], n_p: int, n_q: int):
    if num_buckets == "auto" or isinstance(num_buckets, int):
        return num_buckets
    if isinstance(num_buckets, float) and 0 < num_buckets < 1:
        return max(2, int(round(min(n_p, n_q) * num_buckets)))
    raise ValueError("num_buckets must be 'auto', an int, or a fraction in (0, 1)")


def mauve_from_features(p_features: np.ndarray, q_features: np.ndarray, num_buckets: Union[str, int, float] = "auto",
                        seeds: Sequence[int] = DEFAULT_SEEDS, pca_max_data: int = -1, kmeans_explained_var: float = 0.9,
                        kmeans_num_redo: int = 5, kmeans_max_iter: int = 500,
                        divergence_curve_discretization_size: int = 25, mauve_scaling_factor: float = 5.0
                        ) -> Dict[str, object]:
    """MAUVE(P = real features, Q = generated features): ``dict(mauve, mauve_sd, per_seed, num_buckets)``."""
    try:
        from mauve.compute_mauve import compute_mauve
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("MAUVE needs `pip install mauve-text faiss-cpu`") from e
    p, q = np.asarray(p_features), np.asarray(q_features)
    k = _buckets(num_buckets, len(p), len(q))
    vals, used_k = [], None
    for s in seeds:
        out = compute_mauve(p_features=p, q_features=q, num_buckets=k, pca_max_data=pca_max_data,
                            kmeans_explained_var=kmeans_explained_var, kmeans_num_redo=kmeans_num_redo,
                            kmeans_max_iter=kmeans_max_iter,
                            divergence_curve_discretization_size=divergence_curve_discretization_size,
                            mauve_scaling_factor=mauve_scaling_factor, verbose=False, seed=s)
        vals.append(float(out.mauve))
        used_k = int(out.num_buckets)
    return dict(mauve=float(np.mean(vals)), mauve_sd=float(np.std(vals)), per_seed=vals, num_buckets=used_k)
