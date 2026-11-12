"""Reference correctness and probe coupling against a finite-endpoint exact flow."""
import math

import pytest
import torch

from condila.dlm_metrics.likelihood import ode_nll_bound


class AffineFlow:
    """dy/ds = M y+b; terminal N(0,I); decoder independent of z.

    This defines an exactly normalized discrete model at FINITE endpoints.
    No approximation of a finite-noise mixture marginal by N(0,I) is needed.
    """
    gamma_min, gamma_max = 0.0, 2.0

    def __init__(self, off_diagonal=False):
        self.M = torch.diag(torch.tensor([.4, -.3], dtype=torch.float64))
        if off_diagonal:
            self.M[0, 1] = .2
        self.b = torch.tensor([.2, -.1], dtype=torch.float64)

    def embed(self, ids):
        return torch.zeros(*ids.shape, 2, dtype=torch.float64)

    def denoise(self, z, gamma):
        return (z / torch.sigmoid(gamma).sqrt()) @ self.M.T.to(z) + self.b.to(z)

    def decoder_logits(self, z):
        return torch.tensor([.3, .7], dtype=z.dtype).log().expand(*z.shape[:-1], 2)

    def gamma_grid(self, n):
        return torch.linspace(self.gamma_min, self.gamma_max, n + 1, dtype=torch.float64)

    def exact_log_density(self, z):
        h = math.exp(-self.gamma_max / 2) - math.exp(-self.gamma_min / 2)
        E = torch.matrix_exp(self.M * h)
        shift = torch.linalg.solve(self.M, (E - torch.eye(2)) @ self.b)
        y1 = (z / math.sqrt(.5)) @ E.T + shift
        return (-.5 * y1.square().sum((1, 2)) - z.shape[1] * math.log(2 * math.pi)
                + z.shape[1] * h * torch.trace(self.M) - z.shape[1] * math.log(.5))

    def posterior_parameters(self):
        h = math.exp(-self.gamma_max / 2) - math.exp(-self.gamma_min / 2)
        scale = torch.exp(-self.M.diag() * h)
        mean = math.sqrt(.5) * (scale - 1) * self.b / self.M.diag()
        return mean, scale


def run(m, ids, **kw):
    return ode_nll_bound(m, ids, divergence='exact', state_dtype=torch.float64,
                         exact_endpoints=True, **kw)


def test_reference_log_weight_matches_exact_density_and_kl():
    m = AffineFlow()
    ids = torch.zeros(4, 1, dtype=torch.long)
    # Cubature with exact first/second moments, so all Gaussian KL averages are exact.
    eps = torch.cat([torch.eye(2), -torch.eye(2)])[:, None, :].double() * math.sqrt(2)
    mean, scale = m.posterior_parameters()
    for ref_mean, ref_scale in [(torch.zeros(2), torch.ones(2)), (mean, scale), (mean+.2, scale*.8)]:
        out = run(m, ids, eps=eps, ref_mean=ref_mean, ref_scale=ref_scale)
        z = ref_mean + math.sqrt(.5) * ref_scale * eps
        expected = out['log_ref'] - m.exact_log_density(z) - math.log(.3)
        assert torch.max(torch.abs(out['nll'] - expected)) < 2e-6
        kl = .5 * (((ref_scale/scale)**2 + (ref_mean-mean)**2/(.5*scale**2)
                    - 1 + 2*(scale/ref_scale).log()).sum())
        assert abs(out['nll'].mean() + math.log(.3) - kl) < 2e-6
        assert torch.max(torch.abs(out['nll'] - (out['log_ref']-out['log_flow']+out['reconst']))) < 1e-12


def test_supplied_z_scores_actual_shifted_scaled_reference():
    m = AffineFlow()
    ids = torch.zeros(3, 2, dtype=torch.long)
    eps = torch.randn(3, 2, 2, generator=torch.Generator().manual_seed(12), dtype=torch.float64)
    mean, scale = m.posterior_parameters()
    first = run(m, ids, eps=eps, ref_mean=mean, ref_scale=scale)
    second = run(m, ids, z0=mean+math.sqrt(.5)*scale*eps, ref_mean=mean, ref_scale=scale)
    for key in ('nll', 'eps_quad', 'log_ref'):
        torch.testing.assert_close(first[key], second[key], atol=1e-12, rtol=1e-12)


def test_fixed_probe_prefixes_share_identical_trajectory():
    m = AffineFlow(off_diagonal=True)
    ids = torch.zeros(2, 1, dtype=torch.long)
    eps = torch.ones(2, 1, 2, dtype=torch.float64)
    # Complete orthogonal basis: the average is exactly the trace.
    bank = (math.sqrt(2)*torch.eye(2, dtype=torch.float64))[:, None, None, :].expand(2,2,1,2)
    kw = dict(eps=eps, state_dtype=torch.float64, return_y1=True, return_probe_terms=True)
    one = ode_nll_bound(m, ids, probe_bank=bank, n_probes=1, **kw)
    both = ode_nll_bound(m, ids, probe_bank=bank, n_probes=2, **kw)
    exact = run(m, ids, eps=eps)
    torch.testing.assert_close(one['y1'], both['y1'], atol=0, rtol=0)
    torch.testing.assert_close(one['nll'], both['nll_per_probe'][0], atol=1e-12, rtol=0)
    torch.testing.assert_close(both['nll'], exact['nll'], atol=1e-12, rtol=0)


def test_step_convergence_with_exact_finite_terminal_prior():
    m = AffineFlow()
    ids = torch.zeros(1, 1, dtype=torch.long)
    eps = torch.tensor([[[1.5, -.7]]], dtype=torch.float64)
    errors = []
    for steps in [16, 32, 64, 128]:
        out = run(m, ids, eps=eps, n_steps=steps)
        errors.append(abs(float(out['log_flow'] - m.exact_log_density(math.sqrt(.5)*eps))))
    assert all(3.5 < a/b < 4.5 for a,b in zip(errors, errors[1:]))


@pytest.mark.parametrize('scale', [0., -1., float('nan'), float('inf')])
def test_invalid_scales_rejected(scale):
    with pytest.raises(ValueError, match='positive'):
        run(AffineFlow(), torch.zeros(1, 1, dtype=torch.long), ref_scale=scale)
