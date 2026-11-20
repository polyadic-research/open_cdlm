"""Algorithm 1 and validation NELBO (paper Appendices C, D, and J)."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.nn import functional as F

from .config import ExperimentConfig
from .model import RePlaid, build_schedule_with_derivative, register_variance_minimizing_hooks


@dataclass
class BatchPlan:
    times: torch.Tensor
    reconstruction_mask: torch.Tensor
    self_condition_mask: torch.Tensor
    reconstruction_count: int
    diffusion_count: int

    def slice(self, start: int, count: int) -> BatchPlan:
        indices = slice(start, start + count)
        return BatchPlan(
            self.times[indices],
            self.reconstruction_mask[indices],
            self.self_condition_mask[indices],
            self.reconstruction_count,
            self.diffusion_count,
        )


def make_batch_plan(
    model: RePlaid,
    batch_size: int,
    *,
    generator: torch.Generator,
) -> BatchPlan:
    """Plan the entire per-rank optimizer batch BEFORE splitting into microbatches.

    A microbatch may contain zero reconstruction rows. Clamping the split inside
    each microbatch would bias the effective reconstruction/diffusion allocation.
    """
    device = model.embedding.weight.device
    if batch_size < 2:
        raise ValueError("the effective per-rank batch must contain at least two sequences")
    count = int(model.reconstruction_batch_size_tensor(batch_size))
    indices = torch.arange(batch_size, device=device)
    recon = indices < count
    times = (indices - count).clamp_min(0).double()
    times = times + torch.rand((), device=device, dtype=torch.float64, generator=generator)
    times = torch.where(recon, torch.zeros_like(times), times / (batch_size - count))
    sc = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if model.config.self_condition_probability:
        if batch_size % 4:
            raise ValueError(
                "exact 25% self-conditioning requires effective rank batch divisible by 4"
            )
        offset = torch.randint(4, (), device=device, generator=generator)
        sc[offset + 4 * torch.arange(batch_size // 4, device=device)] = True
    return BatchPlan(times, recon, sc, count, batch_size - count)


def nelbo_objective(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: ExperimentConfig,
    update_step: int,
    *,
    plan: BatchPlan,
    generator: torch.Generator,
    accumulation_steps: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Microbatch contribution normalized to Algorithm 1's full optimizer batch.

    The caller divides by accumulation_steps before backward. Statistics contain
    unnormalized sums so the moment EMAs can be updated once per optimizer step.
    """
    core = model.module if hasattr(model, "module") else model
    valid = attention_mask.float()
    valid_count = valid.sum(-1).clamp_min(1)
    recon, sc = plan.reconstruction_mask, plan.self_condition_mask
    selected = sc.nonzero(as_tuple=False)[:, 0]
    shape, shape_prime, gamma, gamma_prime = build_schedule_with_derivative(
        core,
        plan.times,
        recon,
    )
    holder = {}
    register_variance_minimizing_hooks(shape, shape_prime, recon, holder)
    gamma = torch.where(sc, gamma.detach(), gamma)
    gamma_prime = torch.where(sc, gamma_prime.detach(), gamma_prime)
    embedding = core.normalized_embedding()
    clean_trainable = core.embed_tokens(input_ids, normalized_embedding=embedding)
    clean = torch.where(sc[:, None, None], clean_trainable.detach(), clean_trainable)
    alpha, sigma = core.vp_parameters(gamma)
    state = core.sample_noisy_diffusion_state(
        clean,
        alpha.float()[:, None, None],
        sigma.float()[:, None, None],
        generator=generator,
    )
    scale = min(1.0, update_step / max(1, config.model.bias_warmup_steps))
    # Prepared blocks have no padding. Preserve support for masks in small tests.
    mask = None if bool(attention_mask.all()) else attention_mask
    self_condition = torch.zeros_like(clean)
    if selected.numel():
        with torch.no_grad():
            bootstrap = core(
                state.index_select(0, selected),
                gamma.index_select(0, selected),
                attention_mask=None if mask is None else mask.index_select(0, selected),
                output_prior_scale=scale,
                normalized_embedding=embedding,
            )
            predicted = core._self_condition_from_probabilities(
                bootstrap.float().softmax(-1),
                embedding,
            )
            self_condition.index_copy_(0, selected, predicted)
    logits = model(
        state,
        gamma,
        self_condition,
        mask,
        self_condition_mask=sc,
        self_condition_indices=selected,
        output_prior_scale=scale,
        normalized_embedding=embedding,
    )
    predicted, _ = core.probabilities_to_embedding(
        logits,
        sc,
        self_condition_indices=selected,
        normalized_embedding=embedding,
    )
    ce = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        input_ids.reshape(-1),
        reduction="none",
    ).reshape_as(input_ids)
    reconstruction = (ce * valid).sum(-1) / valid_count
    mse = ((predicted.float() - clean.float()).square().sum(-1) * valid).sum(-1) / valid_count
    diffusion = 0.5 * torch.exp(-gamma) * gamma_prime * mse.double()
    holder["diffusion"] = diffusion.detach()
    _, gamma_1 = core.gamma_bounds()
    alpha_1_sq, sigma_1_sq = torch.sigmoid(-gamma_1), torch.sigmoid(gamma_1)
    alpha_1_sq = torch.where(sc, alpha_1_sq.detach(), alpha_1_sq)
    sigma_1_sq = torch.where(sc, sigma_1_sq.detach(), sigma_1_sq)
    clean_norm = clean.float().square().sum(-1).double()
    prior = 0.5 * (
        alpha_1_sq[:, None] * clean_norm
        + core.diffusion_state_size()
        * (sigma_1_sq - 1 - sigma_1_sq.clamp_min(1e-16).log())[:, None]
    )
    prior = (prior * valid.double()).sum(-1) / valid_count.double()
    recon_sum = (reconstruction.double() * recon).sum()
    diff_sum = (diffusion * ~recon).sum()
    recon_loss = recon_sum * accumulation_steps / plan.reconstruction_count
    diff_loss = diff_sum * accumulation_steps / plan.diffusion_count
    loss = recon_loss + diff_loss + prior.mean()
    statistics = {
        "loss": loss.detach(),
        "reconstruction": recon_loss.detach(),
        "diffusion": diff_loss.detach(),
        "prior": prior.mean().detach(),
        "moments": torch.stack(
            [
                recon_sum.detach(),
                (reconstruction.detach().double().square() * recon).sum(),
                diff_sum.detach(),
                (diffusion.detach().square() * ~recon).sum(),
                recon.sum().double(),
                (~recon).sum().double(),
            ]
        ),
    }
    return loss, statistics


@torch.no_grad()
def update_loss_moments(model: RePlaid, totals: torch.Tensor) -> None:
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals)
    moments = torch.stack(
        [
            totals[0] / totals[4].clamp_min(1),
            totals[1] / totals[4].clamp_min(1),
            totals[2] / totals[5].clamp_min(1),
            totals[3] / totals[5].clamp_min(1),
        ]
    )
    model._apply_loss_moment_update(moments)


@torch.no_grad()
def validation_terms(
    model: RePlaid,
    input_ids: torch.Tensor,
    *,
    generator: torch.Generator,
    self_conditioning: bool | None = None,
) -> dict[str, torch.Tensor]:
    """Unbiased per-sequence estimates of ALL three per-token NELBO terms.

    Every sequence contributes to both reconstruction and diffusion. Validation
    uses a same-time bootstrap for 100% of examples when s.c. is enabled (D.1).
    It never changes the training loss-moment EMAs or applies training hooks.
    """
    if model.training:
        raise ValueError("validation_terms requires model.eval()")
    if self_conditioning is None:
        self_conditioning = model.config.self_condition_probability > 0
    batch = input_ids.shape[0]
    device = input_ids.device
    embedding = model.normalized_embedding().detach()
    clean = F.embedding(input_ids, embedding)
    times = (
        torch.arange(batch, device=device, dtype=torch.float64)
        + torch.rand((), device=device, dtype=torch.float64, generator=generator)
    ) / batch
    with torch.enable_grad():
        times.requires_grad_(True)
        gamma = model.gamma_at(times)
        gamma_prime = torch.autograd.grad(gamma.sum(), times)[0].detach()
        gamma = gamma.detach()

    def denoise(at_gamma):
        alpha, sigma = model.vp_parameters(at_gamma)
        state = model.sample_noisy_diffusion_state(
            clean,
            alpha.float()[:, None, None],
            sigma.float()[:, None, None],
            generator=generator,
        )
        sc = None
        if self_conditioning:
            first = model(state, at_gamma, normalized_embedding=embedding)
            sc = model._self_condition_from_probabilities(first.float().softmax(-1), embedding)
        logits = model(state, at_gamma, sc, normalized_embedding=embedding)
        return logits

    gamma_0, gamma_1 = model.gamma_bounds()
    recon_logits = denoise(gamma_0.expand(batch))
    reconstruction = (
        F.cross_entropy(
            recon_logits.reshape(-1, recon_logits.shape[-1]),
            input_ids.reshape(-1),
            reduction="none",
        )
        .reshape_as(input_ids)
        .mean(-1)
        .double()
    )
    logits = denoise(gamma)
    predicted = model._self_condition_from_probabilities(logits.float().softmax(-1), embedding)
    mse = (predicted - clean).square().sum(-1).mean(-1).double()
    diffusion = 0.5 * torch.exp(-gamma) * gamma_prime * mse
    alpha_sq, sigma_sq = torch.sigmoid(-gamma_1), torch.sigmoid(gamma_1)
    prior = 0.5 * (
        alpha_sq * clean.double().square().sum(-1).mean(-1)
        + model.diffusion_state_size() * (sigma_sq - 1 - sigma_sq.log())
    )
    return {
        "reconstruction": reconstruction,
        "diffusion": diffusion,
        "prior": prior,
        "nelbo": reconstruction + diffusion + prior,
    }
