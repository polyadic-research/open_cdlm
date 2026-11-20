import numpy as np

from open_cdlm.dlm_metrics import DistinctN, ExactDuplicateRate, NGramDiversity, RepetitionRate
from open_cdlm.dlm_metrics.functional import distinct_n, exact_duplicate_rate, lzma_bits_per_token, ngram_diversity, rep_n


def naive_sets(rows, k):
    return [set(tuple(r[i:i + k]) for i in range(len(r) - k + 1)) for r in rows]


def test_rep_n_known_row():
    row = np.array([1, 2, 3, 4, 1, 2, 3, 4, 5])           # 6 4-grams, 5 distinct
    assert abs(rep_n([row], 4) - (1 - 5 / 6)) < 1e-12


def test_duplication_gives_one_over_k(rng):
    base = [rng.integers(0, 1000, size=64) for _ in range(8)]
    for k in (1, 2, 4):
        rows = base * k
        d = ngram_diversity(rows, 4)
        assert abs(d["VS"] - 1.0 / k) < 1e-6 and abs(d["xD"] - 1.0 / k) < 1e-12
        assert exact_duplicate_rate(rows) == (0.0 if k == 1 else 1.0)


def test_against_naive_python(rng):
    rows = [rng.integers(0, 6, size=30) for _ in range(12)]          # small vocabulary: many shared 3-grams
    sets = naive_sets(rows, 3)
    union = set().union(*sets)
    d = ngram_diversity(rows, 3)
    assert abs(d["xD"] - len(union) / sum(len(s) for s in sets)) < 1e-12
    J = np.array([[len(a & b) / len(a | b) for b in sets] for a in sets])
    off = J[~np.eye(len(rows), dtype=bool)]
    assert abs(d["MPJ"] - off.mean()) < 1e-6
    total = sum(len(r) - 2 for r in rows)
    allg = [tuple(r[i:i + 3]) for r in rows for i in range(len(r) - 2)]
    assert abs(distinct_n(rows, 3) - len(set(allg)) / total) < 1e-12


def test_metric_classes(rng):
    ids = rng.integers(0, 500, size=(20, 40))
    for cls, key in ((RepetitionRate, "rep_4"), (DistinctN, "distinct_4"), (ExactDuplicateRate, "exact_dup_rate")):
        m = cls()
        m.update(ids[:10]); m.update(ids[10:])
        assert key in m.compute()
    m = NGramDiversity(n=2)
    m.update(ids)
    assert {"VS_2", "xD_2", "NDR_2", "MPJ_2"} <= set(m.compute())
    assert lzma_bits_per_token(list(ids)) > 0
