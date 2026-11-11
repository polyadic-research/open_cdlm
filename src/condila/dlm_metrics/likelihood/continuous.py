r"""Likelihood bounds for continuous (embedding-space, variance-preserving Gaussian) diffusion language models.

Convention (VDM / Plaid / LangFlow): ``gamma = log(sigma^2 / alpha^2) = -log SNR``, ``alpha^2 = sigmoid(-gamma)``,
``sigma^2 = sigmoid(gamma)``, ``z_gamma = alpha * x + sigma * eps`` with ``x = embed(ids)`` of shape ``[B, L, D]``.

A model is wrapped by an adapter with this duck-typed interface (see :class:`ContinuousDLM`)::

    gamma_min, gamma_max                       floats, the ends of the noise range
    embed(ids [B, L])          -> x [B, L, D]   clean embeddings
    denoise(z [B, L, D], gamma) -> x_hat        posterior-mean embedding E[x | z_gamma]; gamma is a 0-d tensor or [B].
                                                MUST be differentiable w.r.t. z end to end, INCLUDING any
                                                self-conditioning input computed from z: do not detach it.
    decoder_logits(z0 [B, L, D]) -> [B, L, V]   p(x | z_{gamma_min})
    gamma_grid(n_steps)        -> [n_steps + 1] gamma from gamma_min to gamma_max (integration grid)

Three estimators of the NLL of real rows (all are UPPER bounds on -log p(x) of a generative model defined by the
network; none is an exact likelihood, so a bound is comparable with an autoregressive exact NLL in one direction only:
bound < AR implies the model is better, bound > AR does not imply it is worse):

``vdm_nelbo``
    The continuous-time ELBO of the reverse SDE (Kingma et al. 2021):
    ``E[-log p(x | z_min)] + KL(q(z_max | x) || N(0, I)) + 1/2 int e^{-gamma} E||x - x_hat(z_gamma)||^2 dgamma``,
    the integral by stratified importance sampling of ``gamma ~ e^{-gamma}``. Function values only: no divergence,
    no ODE solver, no probe noise. About 10x cheaper than the ODE bound at 32-64 strata.
``ode_nll_bound``
    The probability-flow ODE model's likelihood through the instantaneous change of variables, integrated with Heun's
    method in ``(s, y) = (sqrt(SNR), z / sigma)`` coordinates where the flow is ``dy/ds = x_hat``. The divergence is
    a Hutchinson estimate ``u^T (d x_hat / d z) u`` by autograd through the WHOLE denoiser, so a self-conditioning
    path's chain-rule term is included automatically (estimators that detach the self-conditioning input drop it and
    under-estimate the NLL). Per row it returns a numerical estimate of the single-draw log-weight
    ``reconst + flow + 1/2||y_1||^2 - 1/2||eps||^2`` (``nll``) and the expectation form that replaces ``1/2||eps||^2``
    by ``N/2`` (``nll_expectation_form``; unbiased for the mean, but its per-row variance is large because the flow
    carries the chi-square fluctuation of ``||eps||^2`` into ``||y_1||^2``).
``bdmc_sandwich``
    Bidirectional Monte Carlo on the model's OWN samples (Grosse et al. 2015; Wu et al. 2017): ``z0* ~ p_ODE`` and
    ``x* ~ p(x | z0*)`` make ``z0*`` an exact posterior sample of ``x*``, so the same log-weight at ``z0*`` is a LOWER
    bound on the NLL of ``x*``; with the usual upper bound on the same ``x*`` the gap is ``KL(q||p) + KL(p||q)``, an
    upper bound on the slack of the ODE bound on model samples.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Protocol

import torch
import torch.nn.functional as F


class ContinuousDLM(Protocol):  # pragma: no cover - interface only
    gamma_min: float
    gamma_max: float

    def embed(self, ids: torch.Tensor) -> torch.Tensor: ...
    def denoise(self, z: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor: ...
    def decoder_logits(self, z0: torch.Tensor) -> torch.Tensor: ...
    def gamma_grid(self, n_steps: int) -> torch.Tensor: ...


def _alpha_sigma(gamma: torch.Tensor):
    return torch.sigmoid(-gamma).sqrt(), torch.sigmoid(gamma).sqrt()


def _randn(shape, like: torch.Tensor, generator: Optional[torch.Generator]):
    if generator is None:
        return torch.randn(shape, device=like.device, dtype=like.dtype)
    return torch.randn(shape, device=like.device, dtype=like.dtype, generator=generator)


def _rand(shape, like: torch.Tensor, generator: Optional[torch.Generator]):
    if generator is None:
        return torch.rand(shape, device=like.device, dtype=like.dtype)
    return torch.rand(shape, device=like.device, dtype=like.dtype, generator=generator)


def _reconst(model, z0: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model.decoder_logits(z0)
        if logits.dtype not in (torch.float32, torch.float64):
            logits = logits.float()
        return F.cross_entropy(logits.flatten(0, 1), ids.flatten(), reduction="none").view(ids.shape).double().sum(1)


# ------------------------------------------------------------------------------------------------ VDM NELBO

@torch.no_grad()
def vdm_nelbo(model: Any, ids: torch.Tensor, strata: int = 32, generator: Optional[torch.Generator] = None
              ) -> Dict[str, torch.Tensor]:
    """Per-row VDM NELBO terms (nats per row): ``reconst``, ``prior_kl``, ``diffusion``, ``nll`` (their sum),
    ``tokens``. Random numbers are drawn in the order: z_min noise, stratified u ``[B, strata]``, one noise per stratum."""
    B, L = ids.shape
    x = model.embed(ids).float()
    D = x.shape[2]
    gmin, gmax = float(model.gamma_min), float(model.gamma_max)
    g0 = torch.full((B,), gmin, device=x.device)
    a0, s0 = (t[:, None, None] for t in _alpha_sigma(g0))
    z0 = a0 * x + s0 * _randn(x.shape, x, generator)
    reconst = _reconst(model, z0, ids)
    a1sq, s1sq = 1.0 / (1.0 + math.exp(gmax)), 1.0 / (1.0 + math.exp(-gmax))
    prior_kl = 0.5 * (a1sq * (x ** 2).sum((1, 2)) + L * D * (s1sq - 1.0 - math.log(s1sq)))
    Z = math.exp(-gmin) - math.exp(-gmax)
    u = (torch.arange(strata, device=x.device)[None, :] + _rand((B, strata), x, generator)) / strata
    gam = -torch.log(math.exp(-gmin) - u * Z)
    mse = torch.zeros(B, device=x.device)
    for j in range(strata):
        g = gam[:, j]
        a, s = (t[:, None, None] for t in _alpha_sigma(g))
        z = a * x + s * _randn(x.shape, x, generator)
        mse += ((x - model.denoise(z, g).float()) ** 2).sum((1, 2))
    diffusion = 0.5 * Z * mse / strata
    return dict(reconst=reconst, prior_kl=prior_kl, diffusion=diffusion, nll=reconst + prior_kl + diffusion,
                tokens=torch.full((B,), L, device=x.device, dtype=torch.float32))


# ------------------------------------------------------------------------------------------------ ODE bound

def _probe(z: torch.Tensor, kind: str, generator: Optional[torch.Generator]) -> torch.Tensor:
    if kind == "gaussian":
        return torch.randn_like(z) if generator is None else torch.randn(z.shape, device=z.device, dtype=z.dtype,
                                                                         generator=generator)
    if kind == "rademacher":
        r = (torch.randint(0, 2, z.shape, device=z.device) if generator is None
             else torch.randint(0, 2, z.shape, device=z.device, generator=generator))
        return r.to(z.dtype) * 2 - 1
    raise ValueError("probe must be 'gaussian' or 'rademacher'")


def _velocity_and_div(model, y, s, shape, divergence, probe, probe_generator,
                      n_probes=1, probe_bank=None):
    """Return velocity and one divergence estimate per probe, in [P, B] order."""
    B, L, D = shape
    gamma = -2.0 * torch.log(s)
    sigma = torch.sigmoid(gamma).sqrt()
    with torch.enable_grad():
        z = (y.reshape(B, L, D) * sigma).detach().requires_grad_(True)
        xh = model.denoise(z, gamma)
        if divergence == "exact":
            flat = xh.reshape(B, -1)
            div = torch.zeros(B, device=z.device, dtype=torch.float64)
            for i in range(flat.shape[1]):
                gi = torch.autograd.grad(flat[:, i].sum(), z, retain_graph=i + 1 < flat.shape[1])[0]
                div += gi.reshape(B, -1)[:, i].double()
            div = div[None]
        else:
            values = []
            for j in range(n_probes):
                u = _probe(z, probe, probe_generator) if probe_bank is None else probe_bank[j]
                g = torch.autograd.grad((u * xh).sum(), z, retain_graph=j + 1 < n_probes)[0]
                values.append((u.double() * g.double()).sum((1, 2)))
            div = torch.stack(values)
    return xh.detach().reshape(B, -1), (sigma.double() * div).detach()


def ode_nll_bound(model: Any, ids: torch.Tensor, n_steps: int = 128, eps: Optional[torch.Tensor] = None,
                  z0: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None,
                  probe: str = "gaussian", probe_generator: Optional[torch.Generator] = None,
                  divergence: str = "hutchinson", return_y1: bool = False, *,
                  ref_mean: Optional[torch.Tensor] = None, ref_scale=1.0,
                  n_probes: int = 1, probe_bank: Optional[torch.Tensor] = None,
                  exact_endpoints: bool = False, state_dtype: torch.dtype = torch.float32,
                  return_probe_terms: bool = False) -> Dict[str, torch.Tensor]:
    """Estimate a Gaussian-reference ODE NLL upper bound, in nats per row.

    The reference is N(ref_mean, diag((sigma_min * ref_scale)**2)). The mean
    defaults to alpha_min * embed(ids); positive scales broadcast to [B,L,D],
    supporting scalar, channel-group and diagonal references. Mean/scale must
    be fixed independently of the evaluation eps/probes. They never change the
    model's velocity, decoder or terminal prior. Supplied z0 is scored under
    THIS reference, including its actual mean, scale and log determinant.

    probe_bank [P,B,L,D] is reused at every time point, so nested prefixes and
    different grids share identical vectors. Its vectors must have E[uu^T]=I.
    Without a bank, n_probes fresh vectors are drawn at every function call.
    return_probe_terms exposes [P,B] flow/nll for nested-probe comparisons.

    State arithmetic defaults to float32; scalar sums use float64. The returned
    log-weight is a numerical estimate: finite ODE steps introduce bias and
    Hutchinson probes introduce noise. A single draw is not itself a bound.
    Optional model.terminal_log_prob(y1) supplies the normalized terminal
    density in y=z/sigma coordinates (default: standard Gaussian).
    """
    if n_steps < 1 or n_probes < 1:
        raise ValueError("n_steps and n_probes must be positive")
    if divergence not in ("exact", "hutchinson"):
        raise ValueError("divergence must be 'exact' or 'hutchinson'")
    if z0 is not None and eps is not None:
        raise ValueError("provide eps or z0, not both")
    B, L = ids.shape
    x = model.embed(ids).to(dtype=state_dtype).detach()
    D = x.shape[2]
    N = L * D
    g0 = torch.tensor(float(model.gamma_min), device=x.device, dtype=x.dtype)
    a0, s0 = _alpha_sigma(g0)
    mean = a0 * x if ref_mean is None else torch.as_tensor(ref_mean, device=x.device, dtype=x.dtype)
    scale = torch.as_tensor(ref_scale, device=x.device, dtype=x.dtype)
    mean, scale = torch.broadcast_to(mean, x.shape), torch.broadcast_to(scale, x.shape)
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("reference mean must be finite; scales must be finite and positive")
    if z0 is None:
        eps = _randn(x.shape, x, generator) if eps is None else eps.to(x.device, x.dtype)
        if eps.shape != x.shape:
            raise ValueError("eps must match embedded ids shape")
        z0 = (mean + s0 * scale * eps).detach()
    else:
        z0 = z0.to(x.device, x.dtype).detach()
        if z0.shape != x.shape:
            raise ValueError("z0 must match embedded ids shape")
        eps = (z0 - mean) / (s0 * scale)
    if probe_bank is not None:
        if divergence != "hutchinson":
            raise ValueError("probe_bank requires Hutchinson divergence")
        if probe_bank.ndim != 4 or tuple(probe_bank.shape[1:]) != tuple(x.shape):
            raise ValueError("probe_bank must have shape [P,B,L,D]")
        if len(probe_bank) < n_probes:
            raise ValueError("probe_bank has fewer than n_probes vectors")
        probe_bank = probe_bank[:n_probes].to(x.device, x.dtype)
    reconst = _reconst(model, z0, ids).double()
    grid = model.gamma_grid(n_steps).to(x.device, x.dtype).clone()
    if grid.shape != (n_steps + 1,) or not torch.isfinite(grid).all():
        raise ValueError("gamma_grid must contain n_steps+1 finite values")
    if exact_endpoints:
        grid[0], grid[-1] = model.gamma_min, model.gamma_max
    if not (grid[1:] > grid[:-1]).all():
        raise ValueError("gamma_grid must be strictly increasing")
    s_grid = torch.exp(-0.5 * grid)
    y = (z0 / s0).reshape(B, -1)
    A = torch.zeros((1 if divergence == "exact" else n_probes, B), device=x.device, dtype=torch.float64)
    for a in range(n_steps):
        sa, sb = s_grid[a], s_grid[a + 1]
        h = sb - sa
        args = ((B, L, D), divergence, probe, probe_generator, n_probes, probe_bank)
        k1, d1 = _velocity_and_div(model, y, sa, *args)
        k2, d2 = _velocity_and_div(model, y + h * k1, sb, *args)
        y = y + h * (k1 * 0.5 + k2 * 0.5)
        A += h.double() * (d1 * 0.5 + d2 * 0.5)
    flow_per_probe = -A
    flow = flow_per_probe.mean(0)
    prior_quad = 0.5 * y.double().square().sum(1)
    eps_quad = 0.5 * eps.double().square().sum((1, 2))
    ref_logdet = scale.double().log().sum((1, 2))
    log_p_y = (-prior_quad - N / 2 * math.log(2 * math.pi))
    if hasattr(model, "terminal_log_prob"):
        log_p_y = model.terminal_log_prob(y.reshape(B, L, D)).double()
    prior_term = -log_p_y - N / 2 * math.log(2 * math.pi)
    base = reconst + flow + prior_term - ref_logdet
    log_ref = -eps_quad - ref_logdet - N * s0.double().log() - N / 2 * math.log(2 * math.pi)
    log_flow = log_p_y - flow - N * s0.double().log()
    out = dict(reconst=reconst, flow=flow, prior_quad=prior_quad, prior_term=prior_term,
               eps_quad=eps_quad, ref_logdet=ref_logdet, log_ref=log_ref, log_flow=log_flow,
               nll=base - eps_quad, nll_expectation_form=base - N / 2.0,
               tokens=torch.full((B,), L, device=x.device, dtype=torch.float64))
    if return_probe_terms:
        out["flow_per_probe"] = flow_per_probe
        out["nll_per_probe"] = reconst + flow_per_probe + prior_term - ref_logdet - eps_quad
    if return_y1:
        out["y1"] = y.reshape(B, L, D)
    return {k: v.detach() for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ sampling + BDMC

@torch.no_grad()
def sample_ode(model: Any, y1: torch.Tensor, n_steps: int = 128) -> torch.Tensor:
    """Integrate the probability-flow ODE from ``y1 ~ N(0, I)`` at gamma_max down to gamma_min (Heun, the same grid
    as the density); returns ``z0`` ``[B, L, D]``."""
    B, L, D = y1.shape
    s_grid = torch.exp(-0.5 * model.gamma_grid(n_steps).to(y1.device).float()).flip(0)
    y = y1.reshape(B, -1).float()

    def f(s, y):
        gamma = -2.0 * torch.log(s)
        return model.denoise(y.reshape(B, L, D) * torch.sigmoid(gamma).sqrt(), gamma).reshape(B, -1).float()

    for a in range(len(s_grid) - 1):
        sa, sb = s_grid[a], s_grid[a + 1]
        h = sb - sa
        k1 = f(sa, y)
        k2 = f(sb, y + h * k1)
        y = y + h * (k1 * 0.5 + k2 * 0.5)
    g0 = -2.0 * torch.log(s_grid[-1])
    return y.reshape(B, L, D) * torch.sigmoid(g0).sqrt()


@torch.no_grad()
def decode_sample(model: Any, z0: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """``x ~ p(x | z0)``: categorical sampling of the decoder (NOT argmax, which would break the joint sample)."""
    logits = model.decoder_logits(z0).float()
    dist = torch.distributions.Categorical(logits=logits)
    if generator is None:
        return dist.sample()
    p = dist.probs.reshape(-1, logits.shape[-1])
    return torch.multinomial(p, 1, True, generator=generator).view(logits.shape[:-1])


def bdmc_sandwich(model: Any, n_samples: int, length: int, dim: int, n_steps: int = 128, batch_size: int = 8,
                  seed: int = 0, device: Optional[str] = None, probe: str = "gaussian", divergence: str = "hutchinson",
                  verbose: bool = False) -> Dict[str, torch.Tensor]:
    """Upper and lower bounds on the NLL of the model's own samples and their gap (per row, nats per row).

    Returns ``upper``, ``lower``, ``width`` (= KL(q||p) + KL(p||q) per sample), ``tokens``, ``ids`` (the samples) and
    ``y1_recovery`` (mean squared error per dimension of the re-integrated ``y1``: a solver-consistency check).
    Per-sample values carry the Hutchinson probe noise of two ODE integrations; the MEAN width over samples is
    unbiased (the log-weight is linear in the divergence estimate). Average over enough samples, or use
    ``divergence="exact"`` for small models.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    acc: Dict[str, list] = {}
    for s in range(0, n_samples, batch_size):
        b = min(batch_size, n_samples - s)
        g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + s)
        y1 = torch.randn(b, length, dim, generator=g).to(device)
        z0 = sample_ode(model, y1, n_steps)
        gd = torch.Generator(device=device).manual_seed(seed * 1_000_003 + s + 1)
        x = decode_sample(model, z0, gd)
        gp = torch.Generator(device=device).manual_seed(seed * 1_000_003 + s + 2)
        rev = ode_nll_bound(model, x, n_steps, z0=z0, probe=probe, probe_generator=gp, divergence=divergence,
                            return_y1=True)
        ge = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + s + 3)
        eps = torch.randn(z0.shape, generator=ge).to(device)
        fwd = ode_nll_bound(model, x, n_steps, eps=eps, probe=probe, probe_generator=gp, divergence=divergence)
        out = dict(upper=fwd["nll"], lower=rev["nll"], width=fwd["nll"] - rev["nll"], tokens=fwd["tokens"],
                   ids=x, y1_recovery=((rev["y1"] - y1) ** 2).mean((1, 2)))
        for k, v in out.items():
            acc.setdefault(k, []).append(v.detach().cpu())
        if verbose:
            print(f"bdmc {s + b}/{n_samples}: width {(out['width'] / out['tokens']).mean():.4f} nats/token", flush=True)
    return {k: torch.cat(v) for k, v in acc.items()}
