"""Two-sample statistics in a feature space (rows = samples, float64).

Conventions (as in the reference evaluation scripts):

* ``X`` = real / reference features ``[n, d]``, ``Y`` = generated features ``[m, d]``;
* RBF kernel ``k(x, y) = exp(-||x - y||^2 / (2 sigma^2))`` with ``sigma`` = median pairwise distance among the REAL rows
  (so every generated set compared with the same reference shares one bandwidth), times ``bandwidth_multiplier``;
* unbiased MMD^2 (diagonals excluded), reported as ``sign(MMD^2) * sqrt(|MMD^2|)``: two samples of one distribution land
  near 0 and can be slightly negative.

All kernel sums are computed in row blocks, so memory is O(block * max(n, m)).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np


def l2_normalize(F: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    return F / np.linalg.norm(F, axis=1, keepdims=True).clip(eps)


def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances ``[len(A), len(B)]`` (not clipped)."""
    return (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T


def median_bandwidth(X: np.ndarray, max_rows: Optional[int] = None, seed: int = 0, *, lower: bool = False) -> float:
    """Pairwise bandwidth among the rows, optionally subsampled with ``max_rows``.

    Use the lower middle distance for even pair counts when ``lower=True``.
    """
    X = np.asarray(X, dtype=np.float64)
    if max_rows is not None and len(X) > max_rows:
        X = X[np.sort(np.random.default_rng(seed).choice(len(X), max_rows, replace=False))]
    d2 = sqdist(X, X)
    iu = np.triu_indices(len(X), 1)
    pairs = np.clip(d2[iu], 0, None)
    if lower:
        middle = (len(pairs) - 1) // 2
        pairs.partition(middle)
        return float(np.sqrt(pairs[middle]))
    return float(np.sqrt(np.median(pairs)))


def signed_sqrt(v: float) -> float:
    return float(np.sign(v) * math.sqrt(abs(v)))


def _rbf_sums(A: np.ndarray, B: np.ndarray, sigmas: Sequence[float], same: bool, block: int) -> np.ndarray:
    """sum_{i,j} k_sigma(a_i, b_j) for every sigma; ``same=True`` drops the diagonal entries (A is B)."""
    out = np.zeros(len(sigmas))
    nb = (B * B).sum(1)[None, :]
    for s in range(0, len(A), block):
        a = A[s:s + block]
        d2 = np.clip((a * a).sum(1)[:, None] + nb - 2.0 * a @ B.T, 0, None)
        for j, sig in enumerate(sigmas):
            K = np.exp(-d2 / (2.0 * sig ** 2))
            if same:
                r = np.arange(len(a))
                K[r, s + r] = 0.0
            out[j] += K.sum()
    return out


def mmd2_rbf(X: np.ndarray, Y: np.ndarray, sigmas: Sequence[float], block: int = 2048) -> np.ndarray:
    """Unbiased MMD^2 for each RBF bandwidth in ``sigmas`` (one pass over the distance blocks)."""
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        raise ValueError("MMD needs at least 2 real and 2 generated rows")
    kxx = _rbf_sums(X, X, sigmas, True, block) / (n * (n - 1))
    kyy = _rbf_sums(Y, Y, sigmas, True, block) / (m * (m - 1))
    kxy = _rbf_sums(X, Y, sigmas, False, block) / (n * m)
    return kxx + kyy - 2.0 * kxy


def mmd(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None, bandwidth_multiplier: float = 1.0) -> float:
    """``sign(MMD^2) sqrt|MMD^2|`` with the RBF kernel of bandwidth ``bandwidth_multiplier * sigma``
    (``sigma=None``: median distance among X)."""
    sigma = median_bandwidth(X) if sigma is None else sigma
    return signed_sqrt(float(mmd2_rbf(X, Y, [bandwidth_multiplier * sigma])[0]))


def multiscale_mmd(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None,
                   scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0)) -> Dict[str, float]:
    """MMD at several bandwidths ``scale * sigma`` and the multi-scale value ``signed_sqrt(mean_scale MMD^2)``."""
    sigma = median_bandwidth(X) if sigma is None else sigma
    v = mmd2_rbf(X, Y, [s * sigma for s in scales])
    out = {f"mmd_s{s:g}": signed_sqrt(float(x)) for s, x in zip(scales, v)}
    out["mmd_multiscale"] = signed_sqrt(float(np.mean(v)))
    return out


def mmd_permutation_null(X: np.ndarray, Y: np.ndarray, sigma: float, repeats: int = 12, seed: int = 0,
                         bandwidth_multiplier: float = 1.0) -> np.ndarray:
    """MMD values (signed sqrt) of random re-splits of the pooled rows into sizes (n, m): the spread under H0.
    Uses one pooled kernel matrix, O((n + m)^2) memory."""
    Z = np.vstack([np.asarray(X, np.float64), np.asarray(Y, np.float64)])
    n, N = len(X), len(X) + len(Y)
    sig = bandwidth_multiplier * sigma
    K = np.exp(-np.clip(sqdist(Z, Z), 0, None) / (2.0 * sig ** 2))
    g = np.random.default_rng(seed)
    out = []
    for _ in range(repeats):
        p = g.permutation(N)
        a, b = p[:n], p[n:]
        Kaa, Kbb, Kab = K[np.ix_(a, a)], K[np.ix_(b, b)], K[np.ix_(a, b)]
        na, nb = len(a), len(b)
        v = ((Kaa.sum() - np.trace(Kaa)) / (na * (na - 1)) + (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1))
             - 2.0 * Kab.mean())
        out.append(signed_sqrt(float(v)))
    return np.array(out)


def kid(X: np.ndarray, Y: np.ndarray, degree: int = 3, gamma: Optional[float] = None, coef0: float = 1.0,
        subsets: Optional[int] = None, subset_size: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Kernel Inception Distance-style unbiased MMD^2 with the polynomial kernel ``(gamma x.y + coef0)^degree``
    (``gamma = 1/d``). ``subsets``: average over random subsets of ``subset_size`` rows (Binkowski et al. 2018);
    ``None``: one estimate on all rows."""
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    gamma = 1.0 / X.shape[1] if gamma is None else gamma

    def one(A, B):
        n, m = len(A), len(B)
        Kaa, Kbb, Kab = (gamma * A @ A.T + coef0) ** degree, (gamma * B @ B.T + coef0) ** degree, (gamma * A @ B.T + coef0) ** degree
        return ((Kaa.sum() - np.trace(Kaa)) / (n * (n - 1)) + (Kbb.sum() - np.trace(Kbb)) / (m * (m - 1)) - 2.0 * Kab.mean())

    if not subsets:
        return dict(kid=float(one(X, Y)))
    g = np.random.default_rng(seed)
    s = min(subset_size, len(X), len(Y))
    vals = [one(X[g.choice(len(X), s, replace=False)], Y[g.choice(len(Y), s, replace=False)]) for _ in range(subsets)]
    return dict(kid=float(np.mean(vals)), kid_std=float(np.std(vals)))


def frechet_distance(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """Frechet distance of Gaussians fitted to X and Y (the FID formula). Biased in n and blind to uniform
    duplication (copies leave mean and covariance unchanged)."""
    from scipy import linalg

    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    mu1, mu2 = X.mean(0), Y.mean(0)
    C1, C2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    covmean = linalg.sqrtm(C1 @ C2)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    covmean = np.real(covmean)
    mean_part = float(((mu1 - mu2) ** 2).sum())
    return dict(fd=float(mean_part + np.trace(C1 + C2 - 2.0 * covmean)), fd_mean_part=mean_part)


def _kth_nn_radius(D: np.ndarray, k: int) -> np.ndarray:
    """k-th nearest-neighbour distance of each row of a square distance matrix (self excluded)."""
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    return np.partition(D, k - 1, axis=1)[:, k - 1]


def prdc(X: np.ndarray, Y: np.ndarray, k: int = 5, duplicate_tol: float = 1e-6) -> Dict[str, float]:
    """k-NN precision / recall (Kynkaanniemi et al. 2019), density / coverage (Naeem et al. 2020), plus the share of
    generated rows whose nearest other generated row is closer than ``duplicate_tol`` (feature-space duplicates).

    precision: generated rows inside the real manifold (union of real k-NN balls); recall: real rows inside the generated
    manifold; coverage: real rows with at least one generated row in their ball; density: mean number of real balls
    containing a generated row / k. All depend on n: compare at equal n."""
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    m = len(Y)
    rx = _kth_nn_radius(np.sqrt(np.clip(sqdist(X, X), 0, None)), k)
    Dyy = np.sqrt(np.clip(sqdist(Y, Y), 0, None))
    ry = _kth_nn_radius(Dyy, k)
    Dyy_off = Dyy.copy()
    np.fill_diagonal(Dyy_off, np.inf)
    nn_y = Dyy_off.min(1)
    Dxy = np.sqrt(np.clip(sqdist(X, Y), 0, None))
    inside_real = Dxy <= rx[:, None]
    return dict(precision=float(inside_real.any(0).mean()), recall=float((Dxy <= ry[None, :]).any(1).mean()),
                coverage=float(inside_real.any(1).mean()), density=float(inside_real.sum() / (k * m)),
                feature_dup_rate=float((nn_y < duplicate_tol).mean()))


def feature_vendi(Y: np.ndarray, normalize: bool = True) -> float:
    """Vendi score of the cosine kernel of the generated rows, divided by their number (1 = all distinct directions)."""
    Y = l2_normalize(Y) if normalize else np.asarray(Y, np.float64)
    K = np.clip(Y @ Y.T, -1, 1)
    lam = np.linalg.eigvalsh(K / len(Y))
    lam = lam[lam > 1e-12]
    return float(math.exp(-(lam * np.log(lam)).sum()) / len(Y))
