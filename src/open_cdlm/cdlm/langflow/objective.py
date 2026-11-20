"""LangFlow's denoising and Gumbel schedule fitting losses."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def sample_gamma(model, batch_size, *, device, generator=None):
    cutoff = model.proposal.cutoff
    q = torch.rand(batch_size, device=device, generator=generator)
    return model.proposal(q * (1 - 2 * cutoff) + cutoff)


def training_objective(
    model,
    input_ids,
    gamma,
    *,
    attention_mask=None,
    self_condition_probability=0.25,
    scheduler_loss_weight=1.0,
    use_bias=None,
    generator=None,
):
    """Keep denoising gradients and proposal fitting gradients separate."""
    core = model.module if hasattr(model, "module") else model
    gamma = gamma.detach()
    clean = core._embed_tokens(input_ids)
    noisy = core._forward_diffusion(clean, gamma, generator=generator)
    kwargs = {"noisy_embeds": noisy, "timesteps": gamma, "return_dict": False, "use_bias": use_bias}
    self_condition = None
    if core.config.self_conditioning and self_condition_probability:
        draw = torch.rand((), device=input_ids.device, generator=generator)
        if draw < self_condition_probability:
            with torch.no_grad():
                logits = core(**kwargs)
                self_condition = core._embed_tokens(logits.float().softmax(dim=-1))
                del logits
    logits = model(**kwargs, x_self_cond=self_condition)
    token_ce = F.cross_entropy(
        logits.float().flatten(0, 1),
        input_ids.flatten(),
        reduction="none",
    ).reshape_as(input_ids)
    if attention_mask is None:
        denoising = token_ce.mean()
        observed = token_ce.detach().mean(dim=1)
    else:
        masked_ce = (token_ce * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1)
        denoising = masked_ce.sum() / counts.sum().clamp_min(1)
        observed = masked_ce.detach() / counts.clamp_min(1)

    proposal = core.proposal
    fitted = proposal.entropy * torch.exp(-torch.exp(-(gamma - proposal.loc) / proposal.scale))
    errors = (observed - fitted.float()).square()
    if attention_mask is None:
        schedule = errors.mean()
    else:
        valid = counts > 0
        schedule = (errors * valid).sum() / valid.sum().clamp_min(1)
    loss = denoising + scheduler_loss_weight * schedule
    return loss, {
        "loss": loss.detach(),
        "ce_loss": denoising.detach(),
        "scheduler_loss": schedule.detach(),
    }
