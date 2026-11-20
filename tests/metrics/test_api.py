import json

import numpy as np
import pytest
import torch

from open_cdlm.dlm_metrics import (MMD, MTD, PRDC, EntropyDecomposition, ExactDuplicateRate, FeatureVendi,
                                  FrechetDistance, MetricCollection, MultiScaleMTD, Samples, load_samples)
from open_cdlm.dlm_metrics.functional import mmd as f_mmd, l2_normalize


def test_samples_wrap_and_subset(rng):
    a = rng.integers(0, 100, size=(5, 7))
    for x in (a, torch.as_tensor(a), [list(r) for r in a]):
        s = Samples.wrap(x)
        assert len(s) == 5 and s.has_ids and np.array_equal(s.ids()[2], a[2])
    t = Samples.wrap(["a b", "c"])
    assert t.from_texts and len(t) == 2
    assert Samples.wrap(a).digest() == Samples.wrap(torch.as_tensor(a)).digest()
    assert Samples.wrap(a).digest() != Samples.wrap(a[::-1].copy()).digest()
    sub = Samples.wrap(a).subset([0, 4])
    assert np.array_equal(sub.ids()[1], a[4])
    c = Samples.concat([Samples.wrap(a[:2]), Samples.wrap(a[2:])])
    assert len(c) == 5
    with pytest.raises(TypeError):
        Samples.wrap("single string")


def test_load_samples_formats(tmp_path, rng):
    a = rng.integers(0, 100, size=(4, 6))
    torch.save({"ids": torch.as_tensor(a), "meta": {"tokenizer": "gpt2"}}, tmp_path / "s.pt")
    np.save(tmp_path / "s.npy", a)
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps({"text": t}) for t in ["x", "y z"]))
    (tmp_path / "s.txt").write_text("one\ntwo\n")
    s = load_samples(str(tmp_path / "s.pt"))
    assert s.tokenizer == "gpt2" and np.array_equal(s.ids()[3], a[3])
    assert np.array_equal(load_samples(str(tmp_path / "s.npy")).ids()[0], a[0])
    assert load_samples(str(tmp_path / "s.jsonl")).texts() == ["x", "y z"]
    assert load_samples(str(tmp_path / "s.txt")).texts() == ["one", "two"]


def test_feature_metrics_lifecycle(stub, rng):
    real, fake = rng.integers(0, 64, size=(60, 32)), rng.integers(0, 40, size=(50, 32))
    m = MTD(extractor=stub, reset_real_features=False, null_repeats=5)
    m.update(real[:30], real=True); m.update(real[30:], real=True)
    m.update(fake)
    out = m.compute()
    X, Y = l2_normalize(stub(Samples.wrap(real))), l2_normalize(stub(Samples.wrap(fake)))
    assert abs(out["mtd"] - f_mmd(X, Y)) < 1e-12
    assert out["n_real"] == 60 and out["n_fake"] == 50 and "null_sd" in out
    legacy = MMD(extractor=stub, null_repeats=5)
    legacy.load_state_dict(m.state_dict())
    assert legacy.compute() == {k.replace("mtd", "mmd"): v for k, v in out.items()}
    m.reset()
    assert len(m.real_features) == 2 and m.fake_features == []        # real features kept
    m.update(real[:50])
    assert m.compute()["mtd"] < out["mtd"]
    q = MTD(extractor=stub, bandwidth_multiplier=0.25)
    q.update(real, real=True); q.update(fake)
    assert abs(q.compute()["bandwidth"] - 0.25 * out["bandwidth"]) < 1e-12
    with pytest.raises(RuntimeError):
        MTD(extractor=stub).compute()


def test_collection_shares_features(stub, rng):
    real, fake = rng.integers(0, 64, size=(40, 16)), rng.integers(0, 64, size=(40, 16))
    coll = MetricCollection({"mtd": MTD(extractor=stub), "ms": MultiScaleMTD(extractor=stub), "prdc": PRDC(extractor=stub),
                             "fd": FrechetDistance(extractor=stub), "vendi": FeatureVendi(extractor=stub),
                             "ent": EntropyDecomposition(), "dup": ExactDuplicateRate()})
    coll.update(real, real=True)
    coll.update(fake)
    out = coll.compute()
    assert set(out) == {"mtd", "ms", "prdc", "fd", "vendi", "ent", "dup"}
    assert out["ent"]["n_rows"] == 40                     # single-sample metrics never see the real rows
    assert out["vendi"]["n_fake"] == 40
    flat = coll.compute(flat=True)
    assert "mtd/mtd" in flat and "ms/mtd_multiscale" in flat and "prdc/recall" in flat


def test_state_dict_merge(rng):
    ids = rng.integers(0, 30, size=(20, 10))
    a, b = EntropyDecomposition(), EntropyDecomposition()
    a.update(ids[:10]); b.update(ids[10:])
    a.merge_state(b)
    c = EntropyDecomposition(); c.update(ids)
    assert abs(a.compute()["H_pool"] - c.compute()["H_pool"]) < 1e-12
    d = EntropyDecomposition(); d.load_state_dict(c.state_dict())
    assert d.compute() == c.compute()
