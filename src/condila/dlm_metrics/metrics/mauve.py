"""MAUVE as a stateful metric (needs ``pip install mauve-text faiss-cpu``)."""
from __future__ import annotations

from typing import Optional, Sequence, Union

from ..features import TextFeatureExtractor
from ..functional.mauve import DEFAULT_SEEDS, mauve_from_features
from .feature import FeatureMetric


class MAUVE(FeatureMetric):
    """MAUVE between real (P) and generated (Q) texts, mean +- SD over k-means seeds.

    Features: by default the ``mauve-text`` featurization (gpt2-large, last token, fp32, special tokens removed),
    computed by a ``TextFeatureExtractor(pooling="last")`` so the judge weights can be shared.

    Args:
        num_buckets: ``"auto"`` (= n/10, the package default), an int, or a fraction of n. MAUVE depends on n and k:
            compare only at equal n and k, and report the real-vs-real value at the same n, k.
        seeds: k-means seeds (default 25..29).
        pca_max_data, kmeans_explained_var, kmeans_num_redo, kmeans_max_iter, divergence_curve_discretization_size,
        mauve_scaling_factor: passed to ``mauve.compute_mauve`` (package defaults).
    """
    higher_is_better = True
    _pooling = "last"

    def __init__(self, num_buckets: Union[str, int, float] = "auto", seeds: Sequence[int] = DEFAULT_SEEDS,
                 pca_max_data: int = -1, kmeans_explained_var: float = 0.9, kmeans_num_redo: int = 5,
                 kmeans_max_iter: int = 500, divergence_curve_discretization_size: int = 25,
                 mauve_scaling_factor: float = 5.0, extractor: Optional[TextFeatureExtractor] = None, **kw):
        if extractor is not None and getattr(extractor, "pooling", "last") != "last":
            raise ValueError("MAUVE expects a last-token extractor (TextFeatureExtractor(pooling='last'))")
        super().__init__(extractor=extractor, normalize=False, **kw)
        self.kw = dict(num_buckets=num_buckets, seeds=tuple(seeds), pca_max_data=pca_max_data,
                       kmeans_explained_var=kmeans_explained_var, kmeans_num_redo=kmeans_num_redo,
                       kmeans_max_iter=kmeans_max_iter,
                       divergence_curve_discretization_size=divergence_curve_discretization_size,
                       mauve_scaling_factor=mauve_scaling_factor)
        self._hparams = dict(num_buckets=num_buckets, seeds=tuple(seeds))

    def _compute(self):
        import numpy as np

        P = np.concatenate(self.real_features) if self.real_features else None
        Q = np.concatenate(self.fake_features) if self.fake_features else None
        if P is None or Q is None:
            raise RuntimeError("MAUVE needs real and generated features")
        out = mauve_from_features(P, Q, **self.kw)
        out.update(n_real=len(P), n_fake=len(Q))
        return out
