"""Two-sample metrics in the judge's feature space: MMD (single- and multi-scale), KID, Frechet distance, k-NN
precision / recall / density / coverage, feature Vendi score, topic occupancy.

Usage (torchmetrics FID style)::

    fx = TextFeatureExtractor("gpt2-large", device="cuda")      # share it between metrics
    mmd = MMD(extractor=fx, tokenizer="gpt2")
    mmd.update(real_rows, real=True)
    mmd.update(generated_rows, real=False)
    mmd.compute()        # {'mmd': ..., 'mmd2': ..., 'bandwidth': ..., 'n_real': ..., 'n_fake': ...}

The real features survive ``reset()`` when ``reset_real_features=False``, so one reference serves many sample sets.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from ..base import Metric
from ..features import TextFeatureExtractor, resolve_extractor
from ..functional import kernel as K
from ..functional.topics import topic_occupancy
from ..inputs import Samples, TokenizerLike


class FeatureMetric(Metric):
    """Base class: collects real / generated features from a :class:`TextFeatureExtractor`.

    Args:
        extractor: a ``TextFeatureExtractor`` (shared), a judge name, or ``None`` (``gpt2-large``, mean pooling).
        device: device for a newly created extractor.
        normalize: L2-normalize the features before the statistic (default True).
        reset_real_features: clear the real features on ``reset()`` (default True).
        tokenizer: generator tokenizer (to decode token ids).
    """
    is_two_sample = True
    _pooling = "mean"

    def __init__(self, extractor: Union[TextFeatureExtractor, str, None] = None, device: Optional[str] = None,
                 normalize: bool = True, reset_real_features: bool = True, tokenizer: TokenizerLike = None, **kw):
        super().__init__(tokenizer=tokenizer, **kw)
        self.extractor = resolve_extractor(extractor, device, self._pooling)
        self.normalize = normalize
        self.add_state("real_features", keep_on_reset=not reset_real_features)
        self.add_state("fake_features")

    def update(self, samples: Any, real: bool = False) -> None:
        if not self.is_two_sample and real:
            return
        F = np.asarray(self.extractor(Samples.wrap(samples, self.tokenizer)))
        (self.real_features if real else self.fake_features).append(F)

    def update_features(self, features: np.ndarray, real: bool = False) -> None:
        """Add precomputed features ``[n, d]`` (from the same extractor configuration)."""
        (self.real_features if real else self.fake_features).append(np.asarray(features))

    def _stack(self, parts, name):
        if not parts:
            raise RuntimeError(f"{type(self).__name__}: no {name} features; call update(..., real={name == 'real'})")
        F = np.concatenate(parts).astype(np.float64)
        return K.l2_normalize(F) if self.normalize else F

    def _xy(self):
        return self._stack(self.real_features, "real"), self._stack(self.fake_features, "generated")


class MMD(FeatureMetric):
    """Maximum mean discrepancy with an RBF kernel; ``sign(MMD^2) sqrt|MMD^2|`` of the unbiased estimator.

    Args:
        bandwidth: ``"median"`` (median pairwise distance among the real rows; default) or a float.
        bandwidth_multiplier: scale of the bandwidth. ``1`` is the headline metric; ``0.25`` is the
            duplication / support guard (exact 2x duplication is ~130 null SDs at 0.25 vs ~2.5 subset SDs at 1).
        null_repeats: permutation re-splits for ``null_sd`` (the noise floor of this comparison at these n, m);
            0 = off. Costs one pooled kernel matrix, O((n + m)^2) memory.
        block_size: rows per kernel block.

    Returns ``mmd`` (0 = the reference distribution; can be slightly negative), ``mmd2``, ``bandwidth`` (the sigma
    actually used), ``n_real``, ``n_fake`` and, with ``null_repeats``, ``null_sd`` / ``null_mean``.
    """
    higher_is_better = False

    def __init__(self, bandwidth: Union[str, float] = "median", bandwidth_multiplier: float = 1.0,
                 null_repeats: int = 0, seed: int = 0, block_size: int = 2048, **kw):
        super().__init__(**kw)
        self.bandwidth, self.bandwidth_multiplier = bandwidth, bandwidth_multiplier
        self.null_repeats, self.seed, self.block_size = null_repeats, seed, block_size
        self._hparams = dict(bandwidth=bandwidth, bandwidth_multiplier=bandwidth_multiplier, null_repeats=null_repeats)

    def _sigma(self, X):
        return K.median_bandwidth(X) if self.bandwidth == "median" else float(self.bandwidth)

    def _compute(self) -> Dict[str, Any]:
        X, Y = self._xy()
        sigma = self._sigma(X)
        bw = self.bandwidth_multiplier * sigma
        v = float(K.mmd2_rbf(X, Y, [bw], self.block_size)[0])
        out = dict(mmd=K.signed_sqrt(v), mmd2=v, bandwidth=bw, n_real=len(X), n_fake=len(Y))
        if self.null_repeats:
            null = K.mmd_permutation_null(X, Y, sigma, self.null_repeats, self.seed, self.bandwidth_multiplier)
            out.update(null_sd=float(null.std()), null_mean=float(null.mean()))
        return out


class MultiScaleMMD(MMD):
    """MMD at bandwidths ``scale * sigma`` for each scale, and ``mmd_multiscale = signed_sqrt(mean_s MMD^2_s)``."""

    def __init__(self, scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0), **kw):
        super().__init__(**kw)
        self.scales = tuple(scales)
        self._hparams["scales"] = self.scales

    def _compute(self) -> Dict[str, Any]:
        X, Y = self._xy()
        sigma = self._sigma(X)
        v = K.mmd2_rbf(X, Y, [s * sigma for s in self.scales], self.block_size)
        out = {f"mmd_s{s:g}": K.signed_sqrt(float(x)) for s, x in zip(self.scales, v)}
        out.update(mmd_multiscale=K.signed_sqrt(float(np.mean(v))), bandwidth=sigma, n_real=len(X), n_fake=len(Y))
        return out


class KID(FeatureMetric):
    """Unbiased MMD^2 with the polynomial kernel ``(gamma x.y + coef0)^degree`` (KID). Default on UN-normalized
    features: on L2-normalized gpt2-large features ``x.y / d`` is ~1e-3 and the kernel is nearly constant."""
    higher_is_better = False

    def __init__(self, degree: int = 3, gamma: Optional[float] = None, coef0: float = 1.0, subsets: Optional[int] = None,
                 subset_size: int = 1000, seed: int = 0, normalize: bool = False, **kw):
        super().__init__(normalize=normalize, **kw)
        self.kw = dict(degree=degree, gamma=gamma, coef0=coef0, subsets=subsets, subset_size=subset_size, seed=seed)
        self._hparams = dict(self.kw, normalize=normalize)

    def _compute(self):
        X, Y = self._xy()
        out = K.kid(X, Y, **self.kw)
        out.update(n_real=len(X), n_fake=len(Y))
        return out


class FrechetDistance(FeatureMetric):
    """Frechet distance of Gaussian fits (the FID formula) in the feature space. Biased in n (compare at equal n) and
    blind to uniform duplication by construction."""
    higher_is_better = False

    def _compute(self):
        X, Y = self._xy()
        out = K.frechet_distance(X, Y)
        out.update(n_real=len(X), n_fake=len(Y))
        return out


class PRDC(FeatureMetric):
    """k-NN precision, recall, density, coverage (and the feature-space duplicate rate). n-dependent: compare at equal n."""

    def __init__(self, k: int = 5, duplicate_tol: float = 1e-6, **kw):
        super().__init__(**kw)
        self.k, self.duplicate_tol = k, duplicate_tol
        self._hparams = dict(k=k)

    def _compute(self):
        X, Y = self._xy()
        out = K.prdc(X, Y, self.k, self.duplicate_tol)
        out.update(n_real=len(X), n_fake=len(Y))
        return out


class FeatureVendi(FeatureMetric):
    """Vendi score of the generated rows' cosine kernel divided by their number (single-sample)."""
    is_two_sample = False
    higher_is_better = True

    def _compute(self):
        Y = self._stack(self.fake_features, "generated")
        return dict(vendi_per_n=K.feature_vendi(Y, normalize=False), n_fake=len(Y))


class TopicOccupancy(FeatureMetric):
    """k-means topics of the real rows and the generated rows' share of each (TV distance, worst ratio)."""

    def __init__(self, n_clusters: int = 16, seed: int = 0, n_init: int = 10, **kw):
        super().__init__(**kw)
        self.n_clusters, self.seed, self.n_init = n_clusters, seed, n_init
        self._hparams = dict(n_clusters=n_clusters, seed=seed)

    def _compute(self):
        X, Y = self._xy()
        out = topic_occupancy(X, Y, self.n_clusters, self.seed, self.n_init)
        out.update(n_real=len(X), n_fake=len(Y))
        return out
