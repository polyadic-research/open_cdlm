"""Adapters that expose a model through the :class:`~open_cdlm.dlm_metrics.likelihood.continuous.ContinuousDLM` interface.

``LangFlowAdapter`` duck-types the official LangFlow model class (attributes ``_embed_tokens``, ``forward(noisy_embeds=,
timesteps=, x_self_cond=, return_dict=False)``, ``proposal`` with ``gamma_min`` / ``gamma_max``, ``config.self_conditioning``);
nothing from LangFlow is imported. Write an adapter of the same shape for other continuous DLMs.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class LangFlowAdapter:
    """LangFlow (and LangFlow-style self-conditioned embedding diffusion) as a ``ContinuousDLM``.

    Args:
        model: a loaded LangFlow model (``eval()`` mode; fp32 recommended for likelihoods).
        self_conditioning: use the two-pass self-conditioned denoiser ``x_hat(z, c(z))`` (the model's velocity field)
            when the model was trained with self-conditioning.
        sc_gradient: keep the gradient through the self-conditioning input (correct). ``False`` reproduces the
            official ``eval_ppl.py``, which detaches it and drops the chain-rule term of the divergence; that
            under-estimates the NLL (on the official OWT checkpoint: 3.18 instead of 3.48 nats/token).
    """

    def __init__(self, model: Any, self_conditioning: bool = True, sc_gradient: bool = True):
        self.model = model
        self.self_conditioning = bool(self_conditioning and getattr(model.config, "self_conditioning", False))
        self.sc_gradient = sc_gradient
        self.gamma_min = float(model.proposal.gamma_min)
        self.gamma_max = float(model.proposal.gamma_max)

    def _gb(self, z, gamma):
        gamma = torch.as_tensor(gamma, device=z.device, dtype=z.dtype)
        return gamma.expand(z.shape[0]) if gamma.dim() == 0 else gamma

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model._embed_tokens(ids).float()

    def denoise(self, z: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        gb = self._gb(z, gamma)
        sc = None
        if self.self_conditioning:
            sc = self.model._embed_tokens(F.softmax(self.model.forward(noisy_embeds=z, timesteps=gb, x_self_cond=None,
                                                                       return_dict=False), dim=-1))
            if not self.sc_gradient:
                sc = sc.detach()
        logits = self.model.forward(noisy_embeds=z, timesteps=gb, x_self_cond=sc, return_dict=False)
        return self.model._embed_tokens(F.softmax(logits, dim=-1))

    @torch.no_grad()
    def decoder_logits(self, z0: torch.Tensor) -> torch.Tensor:
        return self.decoder_logits_with_grad(z0)

    def decoder_logits_with_grad(self, z0: torch.Tensor) -> torch.Tensor:
        """The official reconstruction recipe: self-conditioning input from a no-SC pass at gamma_min."""
        gb = self._gb(z0, torch.tensor(self.gamma_min, device=z0.device))
        sc = None
        if self.self_conditioning:
            sc = self.model._embed_tokens(F.softmax(self.model.forward(noisy_embeds=z0, timesteps=gb, x_self_cond=None,
                                                                       return_dict=False), dim=-1))
        return self.model.forward(noisy_embeds=z0, timesteps=gb, x_self_cond=sc, return_dict=False)

    def gamma_grid(self, n_steps: int) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        t = torch.linspace(0.0, 1.0, n_steps + 1, device=dev).clamp(min=1e-5, max=1.0 - 1e-5)
        return self.model.proposal(t).float()
