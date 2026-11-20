"""Likelihood estimators on a toy model with an exact denoiser and a known NLL."""
import torch

from open_cdlm.dlm_metrics.likelihood import GaussianMixtureTokens, NLLBound, bdmc_sandwich, ode_nll_bound, vdm_nelbo


def toy():
    return GaussianMixtureTokens(vocab_size=8, dim=4, gamma_min=-3.0, gamma_max=12.0, seed=0)


def test_ode_exact_draw_equals_true_nll():
    m = toy()
    ids = m.sample_ids(6, 5, seed=1)
    true = m.true_nll(ids).float()
    out = ode_nll_bound(m, ids, n_steps=128, generator=torch.Generator().manual_seed(0), divergence="exact")
    # exact model: the single-draw log-weight is the NLL for every encoder draw (solver error only, 2nd order)
    assert (out["nll"] - true).abs().max() < 0.04
    # the expectation form is unbiased but noisy per row
    assert (out["nll_expectation_form"] - true).abs().max() > (out["nll"] - true).abs().max()


def test_ode_hutchinson_unbiased():
    m = toy()
    ids = m.sample_ids(64, 5, seed=2)
    true = m.true_nll(ids).float()
    out = ode_nll_bound(m, ids, n_steps=48, generator=torch.Generator().manual_seed(1),
                        probe_generator=torch.Generator().manual_seed(2))
    d = out["nll"] - true
    assert abs(d.mean()) < 4 * d.std() / 8 + 0.02


def test_vdm_nelbo_unbiased():
    m = toy()
    ids = m.sample_ids(256, 5, seed=3)
    true = m.true_nll(ids).float()
    out = vdm_nelbo(m, ids, strata=16, generator=torch.Generator().manual_seed(0))
    d = out["nll"] - true
    assert abs(d.mean()) < 4 * d.std() / 16 + 0.01


def test_bdmc_sandwich_closes():
    m = toy()
    res = bdmc_sandwich(m, n_samples=8, length=5, dim=4, n_steps=128, batch_size=4, seed=0, device="cpu",
                        divergence="exact")
    true = m.true_nll(res["ids"]).float()
    # exact model: zero slack, both bounds equal the true NLL of the model's own samples (solver error only)
    assert res["width"].abs().max() < 0.06
    assert (res["lower"] - true).abs().max() < 0.06 and (res["upper"] - true).abs().max() < 0.06
    assert res["y1_recovery"].max() < 1e-3


def test_nll_bound_metric():
    m = toy()
    ids = m.sample_ids(64, 5, seed=4)
    metric = NLLBound(m, estimator="vdm", strata=16, device="cpu")
    metric.update(ids[:32]); metric.update(ids[32:])
    out = metric.compute()
    true = float(m.true_nll(ids).sum() / ids.numel())
    assert abs(out["nll_per_token"] - true) < 4 * out["se"] + 0.01 and out["n_rows"] == 64
