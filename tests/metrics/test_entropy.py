import numpy as np
import scipy.sparse as sp

from open_cdlm.dlm_metrics import EntropyDecomposition, JSDRows, PooledEntropy, SampleEntropy
from open_cdlm.dlm_metrics.functional import entropy_decomposition, pooled_entropy, row_entropy, sample_entropy


def reference_entropies(rows):
    """Independent implementation (sparse count matrix), as in the original evaluation script."""
    V = max(int(r.max()) for r in rows) + 1
    r_, c_, v_ = [], [], []
    for i, x in enumerate(rows):
        u, k = np.unique(x, return_counts=True)
        r_.append(np.full(len(u), i)); c_.append(u); v_.append(k)
    C = sp.csr_matrix((np.concatenate(v_).astype(float), (np.concatenate(r_), np.concatenate(c_))), shape=(len(rows), V))
    lens = np.asarray(C.sum(1)).ravel()
    H = np.empty(C.shape[0])
    for i in range(C.shape[0]):
        c = C.data[C.indptr[i]:C.indptr[i + 1]]
        p = c / c.sum()
        H[i] = -(p * np.log(p)).sum()
    pooled = np.asarray(C.sum(0)).ravel()
    p = pooled[pooled > 0] / pooled.sum()
    H_pool = float(-(p * np.log(p)).sum())
    w = lens / lens.sum()
    return dict(H_row=float(H.mean()), H_pool=H_pool, JSD_rows=float(H_pool - (w * H).sum()))


def test_matches_reference_and_identity(rng):
    rows = [rng.integers(0, 50, size=40) for _ in range(30)]
    a, b = entropy_decomposition(rows), reference_entropies(rows)
    for k in b:
        assert abs(a[k] - b[k]) < 1e-12
    assert abs(a["H_pool"] - a["H_row"] - a["JSD_rows"]) < 1e-12          # equal lengths: H_pool = H_row + JSD


def test_ragged_rows_weighting(rng):
    rows = [rng.integers(0, 30, size=int(L)) for L in rng.integers(5, 60, size=25)]
    a, b = entropy_decomposition(rows), reference_entropies(rows)
    for k in b:
        assert abs(a[k] - b[k]) < 1e-12


def test_copies_have_zero_jsd(rng):
    row = rng.integers(0, 100, size=64)
    out = entropy_decomposition([row] * 10)
    assert abs(out["JSD_rows"]) < 1e-12 and abs(out["H_row"] - out["H_pool"]) < 1e-12
    assert abs(sample_entropy([row] * 10) - row_entropy([row])[0]) < 1e-12


def test_metric_batches_equal_one_shot(rng):
    ids = rng.integers(0, 80, size=(40, 32))
    m = EntropyDecomposition(split_half=4)
    for s in range(0, 40, 7):
        m.update(ids[s:s + 7])
    out = m.compute()
    ref = entropy_decomposition(list(ids))
    for k in ref:
        assert abs(out[k] - ref[k]) < 1e-12
    assert out["n_rows"] == 40 and all(f"{k}_sd_half" in out for k in ref)
    m.reset()
    assert m.counts == []


def test_single_value_classes_and_ignore(rng):
    ids = rng.integers(0, 20, size=(10, 50))
    assert set(SampleEntropy().__class__.__mro__) >= {EntropyDecomposition}
    s, p, j = SampleEntropy(), PooledEntropy(), JSDRows()
    for m in (s, p, j):
        m.update(ids)
    assert set(s.compute()) == {"H_row", "n_rows"} and set(p.compute()) == {"H_pool", "n_rows"}
    assert abs(p.compute()["H_pool"] - pooled_entropy(list(ids))) < 1e-12
    e = EntropyDecomposition(ignore_ids=[0, 1])
    e.update(ids)
    filtered = [r[(r != 0) & (r != 1)] for r in ids]
    assert abs(e.compute()["H_row"] - reference_entropies(filtered)["H_row"]) < 1e-12
