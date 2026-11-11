import math

import numpy as np

from condila.dlm_metrics.functional import (feature_vendi, frechet_distance, kid, l2_normalize, median_bandwidth, mmd, mmd2_rbf,
                                    mmd_permutation_null, multiscale_mmd, prdc)


def ref_mmd(X, Y, sigma):
    """Full-matrix numpy implementation of the original evaluation script."""
    def sq(A, B):
        return (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    k = lambda A, B: np.exp(-np.clip(sq(A, B), 0, None) / (2.0 * sigma ** 2))  # noqa: E731
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    v = Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2.0 * Kxy.mean()
    return float(np.sign(v) * math.sqrt(abs(v)))


def test_blockwise_equals_full(rng):
    X = l2_normalize(rng.normal(size=(300, 20)))
    Y = l2_normalize(rng.normal(size=(250, 20)) + 0.3)
    s = median_bandwidth(X)
    a = np.sign(mmd2_rbf(X, Y, [s], block=37)[0]) * math.sqrt(abs(mmd2_rbf(X, Y, [s], block=37)[0]))
    assert abs(a - ref_mmd(X, Y, s)) < 1e-12
    assert abs(mmd(X, Y) - ref_mmd(X, Y, s)) < 1e-12
    ms = multiscale_mmd(X, Y, s, scales=(0.5, 1.0))
    assert abs(ms["mmd_s1"] - ref_mmd(X, Y, s)) < 1e-12 and abs(ms["mmd_s0.5"] - ref_mmd(X, Y, 0.5 * s)) < 1e-12


def test_median_bandwidth(rng):
    X = rng.normal(size=(50, 5))
    d = [np.linalg.norm(X[i] - X[j]) for i in range(50) for j in range(i + 1, 50)]
    assert abs(median_bandwidth(X) - np.median(d)) < 1e-10


def test_duplication_identity(rng):
    """MMD^2(S x d) - MMD^2(S) = (d - 1)(1 - kbar_S) / (m - 1) for the unbiased estimator."""
    X = l2_normalize(rng.normal(size=(200, 8)))
    S = l2_normalize(rng.normal(size=(40, 8)))
    s = median_bandwidth(X)
    kss = np.exp(-np.clip(((S[:, None] - S[None]) ** 2).sum(-1), 0, None) / (2 * s ** 2))
    kbar = (kss.sum() - np.trace(kss)) / (40 * 39)
    for d in (2, 4):
        Yd = np.repeat(S, d, axis=0)
        diff = mmd2_rbf(X, Yd, [s])[0] - mmd2_rbf(X, S, [s])[0]
        assert abs(diff - (d - 1) * (1 - kbar) / (len(Yd) - 1)) < 1e-10


def test_null_is_centred(rng):
    X, Y = rng.normal(size=(200, 6)), rng.normal(size=(200, 6))
    null = mmd_permutation_null(X, Y, median_bandwidth(X), repeats=20)
    assert abs(null.mean()) < 3 * null.std() / math.sqrt(20) + 1e-3


def test_prdc_fd_vendi_kid(rng):
    X = rng.normal(size=(120, 4))
    p = prdc(X, X.copy(), k=5)
    assert p["precision"] == 1.0 and p["recall"] == 1.0 and p["coverage"] == 1.0 and p["feature_dup_rate"] == 0.0
    dup = prdc(X, np.repeat(X[:30], 4, axis=0), k=5)
    assert dup["feature_dup_rate"] == 1.0 and dup["recall"] < 1.0
    assert abs(frechet_distance(X, X)["fd"]) < 1e-6
    Y = X + np.array([1.0, 0, 0, 0])
    assert abs(frechet_distance(X, Y)["fd"] - 1.0) < 1e-6           # pure mean shift of 1
    assert abs(feature_vendi(np.eye(10)) - 1.0) < 1e-9 and abs(feature_vendi(np.ones((10, 3))) - 0.1) < 1e-9
    X2 = rng.normal(size=(120, 4))                                    # an independent sample of the same law
    assert abs(kid(X, X2)["kid"]) < 0.2 and kid(X, Y)["kid"] > 5 * abs(kid(X, X2)["kid"])
    assert set(kid(X, Y, subsets=5, subset_size=50)) == {"kid", "kid_std"}


def test_topic_occupancy_detects_missing_clusters(rng):
    import pytest
    pytest.importorskip("sklearn")
    from condila.dlm_metrics.functional import topic_occupancy
    centers = rng.normal(size=(4, 6)) * 10
    X = np.vstack([c + rng.normal(size=(100, 6)) for c in centers])
    Y = np.vstack([c + rng.normal(size=(200, 6)) for c in centers[:2]])       # two of the four topics only
    out = topic_occupancy(X, Y, n_clusters=4, seed=0)
    assert out["n_under_half"] == 2 and abs(out["tv"] - 0.5) < 0.02
    same = topic_occupancy(X, np.vstack([c + rng.normal(size=(100, 6)) for c in centers]), n_clusters=4)
    assert same["tv"] < 0.05
