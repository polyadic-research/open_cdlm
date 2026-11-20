"""A toy continuous DLM with an EXACT denoiser, for testing likelihood code (and your own adapters' conventions).

Tokens are i.i.d. over positions with prior ``pi`` over ``V`` embeddings ``e_v`` in ``R^D``. The posterior over the
token at a noisy position is closed-form, so ``denoise`` is the exact posterior mean, ``decoder_logits`` the exact
posterior, and the true NLL of a row is ``-sum_i log pi[x_i]``. With an exact denoiser the probability-flow ODE
preserves the marginals, so the exact single-draw ODE log-weight equals the true NLL for EVERY encoder draw (up to
solver / probe error), the VDM NELBO equals it in expectation, and the BDMC sandwich closes.
"""
from __future__ import annotations

import torch


class GaussianMixtureTokens:
    def __init__(self, vocab_size: int = 8, dim: int = 4, gamma_min: float = -3.0, gamma_max: float = 12.0,
                 seed: int = 0, device: str = "cpu", dtype: torch.dtype = torch.float64):
        g = torch.Generator().manual_seed(seed)
        self.E = torch.randn(vocab_size, dim, generator=g, dtype=dtype).to(device)
        logits = torch.randn(vocab_size, generator=g, dtype=dtype)
        self.log_pi = torch.log_softmax(logits, 0).to(device)
        self.gamma_min, self.gamma_max = float(gamma_min), float(gamma_max)
        self.V, self.D, self.device, self.dtype = vocab_size, dim, device, dtype

    def sample_ids(self, n: int, length: int, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.multinomial(self.log_pi.exp().cpu().float(), n * length, True, generator=g).view(n, length).to(self.device)

    def true_nll(self, ids: torch.Tensor) -> torch.Tensor:
        return -self.log_pi[ids].sum(1)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.E[ids]

    def _posterior_logits(self, z: torch.Tensor, gamma) -> torch.Tensor:
        gamma = torch.as_tensor(gamma, dtype=z.dtype, device=z.device)
        if gamma.dim() == 0:
            gamma = gamma.expand(z.shape[0])
        a = torch.sigmoid(-gamma).sqrt().view(-1, 1, 1, 1)
        s2 = torch.sigmoid(gamma).view(-1, 1, 1, 1)
        d2 = ((z.unsqueeze(2) - a * self.E.to(z.dtype)) ** 2).sum(-1, keepdim=True)   # [B, L, V, 1]
        return (self.log_pi.to(z.dtype).view(1, 1, -1) - d2.squeeze(-1) / (2.0 * s2.view(-1, 1, 1)))

    def denoise(self, z: torch.Tensor, gamma) -> torch.Tensor:
        w = torch.softmax(self._posterior_logits(z, gamma), dim=-1)
        return w @ self.E.to(z.dtype)

    def decoder_logits(self, z0: torch.Tensor) -> torch.Tensor:
        return self._posterior_logits(z0, torch.tensor(self.gamma_min, dtype=z0.dtype, device=z0.device))

    def gamma_grid(self, n_steps: int) -> torch.Tensor:
        return torch.linspace(self.gamma_min, self.gamma_max, n_steps + 1, dtype=torch.float64, device=self.device)

    def __repr__(self) -> str:
        return f"GaussianMixtureTokens(V={self.V}, D={self.D}, gamma=[{self.gamma_min}, {self.gamma_max}])"

