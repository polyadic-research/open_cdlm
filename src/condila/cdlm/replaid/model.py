"""RePlaid likelihood and samplers, adapted from Open CDLM's core implementation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .layers import (
    FinalLayer,
    GQADDiTBlock,
    RotaryEmbedding,
    SampleOutput,
    TimestepEmbedder,
)


class SphericalEmbedding(nn.Module):
    """Learned token codebook constrained row-wise to the unit sphere."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        value = F.normalize(torch.randn(vocab_size, embedding_dim), p=2.0, dim=-1)
        self.weight = nn.Parameter(value)

    def forward(self) -> torch.Tensor:
        return F.normalize(self.weight.float(), p=2.0, dim=-1, eps=1e-8)


class MonotoneNoiseSchedule(nn.Module):
    """RePlaID's normalized monotone MLP schedule, evaluated in float64."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(hidden_size, 1))
        self.b1 = nn.Parameter(torch.randn(hidden_size))
        self.w2 = nn.Parameter(torch.randn(1, hidden_size))

    def _raw(self, time: torch.Tensor) -> torch.Tensor:
        time64 = time.to(torch.float64)
        w1 = F.softplus(self.w1.to(torch.float64))
        w2 = 0.01 * F.softplus(self.w2.to(torch.float64))
        hidden = (time64[:, None] - 0.5) @ w1.mT + self.b1.to(torch.float64)[None]
        return (torch.tanh(hidden) @ w2.mT)[:, 0]

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(1, device=time.device, dtype=torch.float64)
        one = torch.ones(1, device=time.device, dtype=torch.float64)
        raw_zero, raw_one, raw_time = self._raw(zero), self._raw(one), self._raw(time)
        return (raw_time - raw_zero) / (raw_one - raw_zero).clamp_min(1e-12)


class RePlaid(nn.Module):
    """16-D spherical VP diffusion with the paper's bidirectional DDiT backbone.

    LayerNorm, ordinary MHA, GELU(tanh), biased MLPs, block/final adaLN-Zero,
    additive gamma features, and un-gated probability-weighted self-conditioning.
    Precision boundaries follow paper Appendix K, Table 6.
    """

    STATE_SPACE = "replaid_spherical_embedding_16d"
    SAMPLER = "replaid_ancestral_ddpm"
    BLOCK_CLASS: type[GQADDiTBlock] = GQADDiTBlock
    FINAL_LAYER_CLASS: type[FinalLayer] = FinalLayer

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.vocab_size is None:
            raise ValueError("vocab_size must be resolved before constructing RePlaid")
        self.config = config
        embedding_dim = int(config.replaid_embedding_dim)
        self.embedding = SphericalEmbedding(int(config.vocab_size), embedding_dim)
        self.input_projection = nn.Linear(embedding_dim, config.hidden_size, bias=False)
        self.self_condition_projection = nn.Linear(embedding_dim, config.hidden_size, bias=False)
        self.gamma_projection = nn.Linear(64, config.hidden_size, bias=False)
        self.time = TimestepEmbedder(config.cond_dim)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_length)
        self.blocks = nn.ModuleList(
            self.BLOCK_CLASS(config) for _ in range(config.num_hidden_layers)
        )
        self.output = self.FINAL_LAYER_CLASS(config)
        nn.init.zeros_(self.self_condition_projection.weight)
        nn.init.zeros_(self.gamma_projection.weight)

        self.noise_schedule = MonotoneNoiseSchedule(int(config.replaid_schedule_hidden_size))
        self.gamma_0 = nn.Parameter(torch.tensor(float(config.replaid_gamma_0_init)))
        self.gamma_1 = nn.Parameter(torch.tensor(float(config.replaid_gamma_1_init)))

        self.register_buffer("loss_ema_bias", torch.tensor(1e-8, dtype=torch.float64))
        self.register_buffer("recon_ema", torch.tensor(1e-8, dtype=torch.float64))
        self.register_buffer("recon_sqr_ema", torch.tensor(1e-8, dtype=torch.float64))
        self.register_buffer("diff_ema", torch.tensor(1e-8, dtype=torch.float64))
        self.register_buffer("diff_sqr_ema", torch.tensor(1e-8, dtype=torch.float64))
        self._pending_loss_moment_update = None
        self.gradient_checkpointing = False
        self.use_bfloat16 = True
        self._seed = 42
        self._generators: dict[str, torch.Generator] = {}

    def set_diffusion_seed(self, seed: int) -> None:
        self._seed = int(seed)
        self._generators.clear()

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._seed)
            self._generators[key] = generator
        return self._generators[key]

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def normalized_embedding(self) -> torch.Tensor:
        return self.embedding()

    def embed_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        detach: bool = False,
        normalized_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        table = (
            self.normalized_embedding() if normalized_embedding is None else normalized_embedding
        )
        return F.embedding(input_ids, table.detach() if detach else table)

    def diffusion_state_size(self) -> int:
        return int(self.config.replaid_embedding_dim)

    def gamma_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gamma_0.to(torch.float64), self.gamma_1.to(torch.float64)

    def schedule_shape(self, time: torch.Tensor) -> torch.Tensor:
        return self.noise_schedule(time)

    def gamma_at(self, time: torch.Tensor) -> torch.Tensor:
        shape = self.schedule_shape(time)
        gamma_0, gamma_1 = self.gamma_bounds()
        return gamma_0 + (gamma_1 - gamma_0) * shape

    @staticmethod
    def vp_parameters(gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(-gamma).sqrt(), torch.sigmoid(gamma).sqrt()

    @staticmethod
    def _gamma_features(gamma: torch.Tensor) -> torch.Tensor:
        frequencies = torch.linspace(-5.0, 5.0, 32, device=gamma.device, dtype=torch.float32).exp()
        phase = gamma.float()[:, None] * frequencies[None]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    def normalize_model_input(self, model_input: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        alpha_squared = torch.sigmoid(-gamma.float())[:, None, None]
        sigma_squared = torch.sigmoid(gamma.float())[:, None, None]
        variance = alpha_squared / float(self.diffusion_state_size()) + sigma_squared
        return model_input.float() / variance.sqrt().clamp_min(1e-6)

    def _prepare_backbone_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply an optional experiment-specific transform before the first block."""
        return hidden

    def _refresh_backbone_hidden(
        self,
        hidden: torch.Tensor,
        *,
        block_number: int,
        normalized_state: torch.Tensor,
        self_condition: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Apply an optional experiment-specific refresh after one block."""
        del block_number, normalized_state, self_condition, condition
        return hidden

    def _finalize_backbone_hidden(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply an optional experiment-specific transform before the output head."""
        del condition
        return hidden

    def sample_noisy_diffusion_state(
        self,
        clean_embedding: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        noise = torch.randn(
            clean_embedding.shape,
            device=clean_embedding.device,
            dtype=clean_embedding.dtype,
            generator=generator,
        )
        return alpha * clean_embedding + sigma * noise

    def _backbone_hidden(
        self,
        state: torch.Tensor,
        gamma: torch.Tensor,
        self_condition: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_state = self.normalize_model_input(state, gamma)
        # These three projections and their resulting residual stream are FP32,
        # even inside an outer autocast context (paper Table 6).
        with torch.autocast(device_type=state.device.type, enabled=False):
            normalized_state = self.normalize_model_input(state, gamma)
            gamma_hidden = self.gamma_projection(self._gamma_features(gamma))
            hidden = self.input_projection(normalized_state)
            hidden = hidden + self.self_condition_projection(
                self_condition.float() * math.sqrt(float(self.diffusion_state_size()))
            )
            hidden = hidden + gamma_hidden[:, None]
        context = torch.autocast(
            device_type=state.device.type,
            dtype=torch.bfloat16,
            enabled=state.device.type == "cuda" and self.use_bfloat16,
            cache_enabled=False,
        )
        with context:
            hidden = self._prepare_backbone_hidden(hidden)
            condition = F.silu(self.time(gamma.float()))
            for block_number, block in enumerate(self.blocks, start=1):
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():

                    def custom_forward(
                        states: torch.Tensor,
                        conditioning: torch.Tensor,
                        module: GQADDiTBlock = block,
                    ) -> torch.Tensor:
                        return module(states, conditioning, self.rope, attention_mask)

                    hidden = checkpoint(custom_forward, hidden, condition, use_reentrant=False)
                else:
                    hidden = block(hidden, condition, self.rope, attention_mask)
                hidden = self._refresh_backbone_hidden(
                    hidden,
                    block_number=block_number,
                    normalized_state=normalized_state,
                    self_condition=self_condition,
                    condition=condition,
                )
            logits = self.output(
                self._finalize_backbone_hidden(hidden, condition),
                condition,
            )
        return logits, normalized_state

    @staticmethod
    def output_prior_logits(
        model_input: torch.Tensor,
        gamma: torch.Tensor,
        normalized_embedding: torch.Tensor,
        output_prior_scale: float,
    ) -> torch.Tensor:
        alpha = torch.sigmoid(-gamma.float()).sqrt()[:, None, None]
        sigma_squared = torch.sigmoid(gamma.float())[:, None, None]
        scaled_input = (
            float(output_prior_scale) * alpha / sigma_squared.clamp_min(1e-8) * model_input.float()
        )
        with torch.autocast(device_type=model_input.device.type, enabled=False):
            return torch.matmul(scaled_input.float(), normalized_embedding.float().t())

    def forward(
        self,
        state: torch.Tensor,
        gamma: torch.Tensor,
        self_condition: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        self_condition_mask: torch.Tensor | None = None,
        self_condition_indices: torch.Tensor | None = None,
        output_prior_scale: float = 1.0,
        normalized_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, length, state_size = state.shape
        if state_size != self.diffusion_state_size():
            raise ValueError(f"RePlaID state must end in dimension {self.diffusion_state_size()}")
        if length > self.config.max_seq_length:
            raise ValueError("sequence length exceeds model.max_seq_length")
        if self_condition is None:
            self_condition = torch.zeros_like(state)
        embedding = (
            self.normalized_embedding() if normalized_embedding is None else normalized_embedding
        )
        transformer_logits, _ = self._backbone_hidden(state, gamma, self_condition, attention_mask)
        prior = self.output_prior_logits(state, gamma, embedding, output_prior_scale)
        if self_condition_mask is None or not torch.is_grad_enabled():
            return transformer_logits.float() + prior
        if self_condition_indices is None:
            raise ValueError(
                "self_condition_indices are required with a training self_condition_mask"
            )
        if self_condition_indices.numel() == 0:
            return transformer_logits.float() + prior
        selected = self_condition_indices
        detached_prior = self.output_prior_logits(
            state.index_select(0, selected),
            gamma.index_select(0, selected),
            embedding.detach(),
            output_prior_scale,
        )
        return transformer_logits.float() + prior.index_copy(0, selected, detached_prior)

    def _self_condition_from_probabilities(
        self,
        probabilities: torch.Tensor,
        normalized_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Convert probabilities to the RePlaID state used for self-conditioning."""
        with torch.autocast(device_type=probabilities.device.type, enabled=False):
            return torch.matmul(probabilities.float(), normalized_embedding.float())

    def _sampling_self_condition_from_probabilities(
        self,
        probabilities: torch.Tensor,
        normalized_embedding: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Build sampling SC, optionally bypassing an experiment-specific gate."""
        if mode == "standard":
            return torch.matmul(probabilities.float(), normalized_embedding.float())
        if mode == "model_default":
            return self._self_condition_from_probabilities(
                probabilities,
                normalized_embedding,
            )
        raise ValueError("self_conditioning_mode must be 'model_default' or 'standard'")

    def probabilities_to_embedding(
        self,
        logits: torch.Tensor,
        self_condition_mask: torch.Tensor | None = None,
        *,
        self_condition_indices: torch.Tensor | None = None,
        normalized_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probabilities = F.softmax(logits.float(), dim=-1)
        embedding = (
            self.normalized_embedding() if normalized_embedding is None else normalized_embedding
        )
        predicted = self._self_condition_from_probabilities(probabilities, embedding)
        if self_condition_mask is not None and torch.is_grad_enabled():
            if self_condition_indices is None:
                raise ValueError(
                    "self_condition_indices are required with a training self_condition_mask"
                )
            if self_condition_indices.numel() > 0:
                selected_probabilities = probabilities.index_select(0, self_condition_indices)
                predicted_detached = self._self_condition_from_probabilities(
                    selected_probabilities, embedding.detach()
                )
                predicted = predicted.index_copy(0, self_condition_indices, predicted_detached)
        return predicted, probabilities

    def reconstruction_batch_size_tensor(self, batch_size: int) -> torch.Tensor:
        if batch_size < 2:
            raise ValueError("RePlaID adaptive batch splitting requires batch_size >= 2")
        self.finish_pending_loss_moment_update()
        correction = self.loss_ema_bias.clamp_min(1e-12).reciprocal()
        reconstruction_variance = (
            correction * self.recon_sqr_ema - (correction * self.recon_ema).square()
        ).clamp_min(0.0)
        diffusion_variance = (
            correction * self.diff_sqr_ema - (correction * self.diff_ema).square()
        ).clamp_min(0.0)
        reconstruction_std = reconstruction_variance.sqrt()
        diffusion_std = diffusion_variance.sqrt()
        fraction = reconstruction_std / (reconstruction_std + diffusion_std + 1e-8)
        return torch.round(fraction * batch_size).clamp(1, batch_size - 1).to(torch.long)

    @torch.no_grad()
    def _apply_loss_moment_update(self, moments: torch.Tensor) -> None:
        rate = 1.0 - float(self.config.replaid_reconstruction_ema_decay)
        self.loss_ema_bias.lerp_(torch.ones_like(self.loss_ema_bias), rate)
        self.recon_ema.lerp_(moments[0], rate)
        self.recon_sqr_ema.lerp_(moments[1], rate)
        self.diff_ema.lerp_(moments[2], rate)
        self.diff_sqr_ema.lerp_(moments[3], rate)

    @torch.no_grad()
    def finish_pending_loss_moment_update(self) -> None:
        pending = self._pending_loss_moment_update
        if pending is None:
            return
        work, moments, world_size = pending
        work.wait()
        moments.div_(world_size)
        self._apply_loss_moment_update(moments)
        self._pending_loss_moment_update = None

    @torch.no_grad()
    def update_loss_moments(
        self,
        reconstruction_losses: torch.Tensor,
        diffusion_losses: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> None:
        self.finish_pending_loss_moment_update()
        reconstruction = reconstruction_losses.detach().to(torch.float64)
        diffusion = diffusion_losses.detach().to(torch.float64)
        reconstruction_weight = reconstruction_mask.to(reconstruction)
        diffusion_weight = (~reconstruction_mask).to(diffusion)
        reconstruction_count = reconstruction_weight.sum().clamp_min(1.0)
        diffusion_count = diffusion_weight.sum().clamp_min(1.0)
        moments = torch.stack(
            (
                (reconstruction * reconstruction_weight).sum() / reconstruction_count,
                (reconstruction.square() * reconstruction_weight).sum() / reconstruction_count,
                (diffusion * diffusion_weight).sum() / diffusion_count,
                (diffusion.square() * diffusion_weight).sum() / diffusion_count,
            )
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            work = torch.distributed.all_reduce(
                moments, op=torch.distributed.ReduceOp.SUM, async_op=True
            )
            self._pending_loss_moment_update = (work, moments, world_size)
        else:
            self._apply_loss_moment_update(moments)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        length: int,
        *,
        steps: int = 128,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
        sample_seeds: tuple[int, ...] | None = None,
        return_logits: bool = False,
        sampler: str = "ddpm",
        self_conditioning_mode: str = "standard",
        self_conditioning: bool | None = None,
    ) -> SampleOutput:
        """Sample with learned-schedule DDPM, DDIM, or deterministic Heun."""
        if min(batch_size, length, steps) <= 0:
            raise ValueError("batch_size, length, and steps must be positive")
        if length > self.config.max_seq_length:
            raise ValueError("length exceeds max_seq_length")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if sampler not in {"ddpm", "ddim", "heun"}:
            raise ValueError("sampler must be 'ddpm', 'ddim', or 'heun'")
        if self_conditioning_mode not in {"model_default", "standard"}:
            raise ValueError("self_conditioning_mode must be 'model_default' or 'standard'")
        device = self.embedding.weight.device
        if self.training:
            raise ValueError("call model.eval() before sampling")
        if self_conditioning is None:
            self_conditioning = self.config.self_condition_probability > 0
        expected = (batch_size, length, self.diffusion_state_size())
        sample_generators = None
        if sample_seeds is not None:
            if generator is not None or initial_noise is not None:
                raise ValueError("sample_seeds cannot be combined with generator or initial_noise")
            if len(sample_seeds) != batch_size:
                raise ValueError("sample_seeds must contain one seed per batch row")
            sample_generators = [
                torch.Generator(device=device).manual_seed(int(seed)) for seed in sample_seeds
            ]
        elif generator is None:
            generator = self._generator(device)

        def draw_noise(dtype: torch.dtype) -> torch.Tensor:
            if sample_generators is None:
                return torch.randn(
                    expected,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            return torch.cat(
                [
                    torch.randn(
                        1,
                        length,
                        self.diffusion_state_size(),
                        device=device,
                        dtype=dtype,
                        generator=row_generator,
                    )
                    for row_generator in sample_generators
                ],
                dim=0,
            )

        if initial_noise is None:
            state = draw_noise(torch.float32)
        else:
            if tuple(initial_noise.shape) != expected:
                raise ValueError(f"initial_noise must have shape {expected}")
            state = initial_noise.to(device=device, dtype=torch.float32)

        edges = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float64)
        gamma = self.gamma_at(edges)
        embedding = self.normalized_embedding()
        self_condition = torch.zeros_like(state)
        self_condition_mask = torch.ones(batch_size, device=device, dtype=torch.bool)

        for index in range(steps):
            gamma_t = gamma[index]
            gamma_s = gamma[index + 1]
            logits = self(
                state,
                gamma_t.expand(batch_size),
                self_condition,
                self_condition_mask=self_condition_mask,
                output_prior_scale=1.0,
                normalized_embedding=embedding,
            )
            probabilities = F.softmax(logits.float() / float(temperature), dim=-1)
            predicted_clean = self._self_condition_from_probabilities(probabilities, embedding)
            self_condition = self._sampling_self_condition_from_probabilities(
                probabilities,
                embedding,
                self_conditioning_mode,
            )
            if not self_conditioning:
                self_condition.zero_()
            alpha_t, sigma_t = self.vp_parameters(gamma_t)
            alpha_s, sigma_s = self.vp_parameters(gamma_s)
            predicted_noise = (
                state.double() - alpha_t * predicted_clean.double()
            ) / sigma_t.clamp_min(1e-12)
            if sampler == "heun":
                # Integrate the VP probability-flow ODE in the natural
                # noise-to-signal coordinate tau=sigma/alpha.  In the scaled
                # state y=x/alpha, dy/dtau is the model-predicted noise.  The
                # DDIM update is the Euler predictor; a second denoiser call at
                # the predicted endpoint supplies the Heun corrector slope.
                tau_t = sigma_t / alpha_t.clamp_min(1e-12)
                tau_s = sigma_s / alpha_s.clamp_min(1e-12)
                step_size = tau_s - tau_t
                scaled_state = state.double() / alpha_t.clamp_min(1e-12)
                predicted_scaled_state = scaled_state + step_size * predicted_noise
                predicted_state = (alpha_s * predicted_scaled_state).float()
                endpoint_logits = self(
                    predicted_state,
                    gamma_s.expand(batch_size),
                    self_condition,
                    self_condition_mask=self_condition_mask,
                    output_prior_scale=1.0,
                    normalized_embedding=embedding,
                )
                endpoint_probabilities = F.softmax(
                    endpoint_logits.float() / float(temperature), dim=-1
                )
                endpoint_clean = self._self_condition_from_probabilities(
                    endpoint_probabilities, embedding
                )
                self_condition = self._sampling_self_condition_from_probabilities(
                    endpoint_probabilities,
                    embedding,
                    self_conditioning_mode,
                )
                if not self_conditioning:
                    self_condition.zero_()
                endpoint_noise = (
                    predicted_scaled_state - endpoint_clean.double()
                ) / tau_s.clamp_min(1e-12)
                corrected_scaled_state = scaled_state + 0.5 * step_size * (
                    predicted_noise + endpoint_noise
                )
                state = (alpha_s * corrected_scaled_state).float()
            elif sampler == "ddim":
                state = (alpha_s * predicted_clean.double() + sigma_s * predicted_noise).float()
            else:
                coefficient = -torch.expm1(gamma_s - gamma_t)
                mean = (1.0 - coefficient) * (alpha_s / alpha_t) * state.double()
                mean = mean + coefficient * alpha_s * predicted_clean.double()
                standard_deviation = (coefficient * (1.0 - alpha_s.square())).clamp_min(0.0).sqrt()
                reverse_noise = draw_noise(torch.float64)
                # Keep the finite-gamma(0) final posterior stochastic, matching
                # the referenced RePlaID ancestral sampler.
                state = (mean + standard_deviation * reverse_noise).float()

        final_logits = self(
            state,
            gamma[-1].expand(batch_size),
            self_condition,
            self_condition_mask=self_condition_mask,
            output_prior_scale=1.0,
            normalized_embedding=embedding,
        ) / float(temperature)
        final_probabilities = F.softmax(final_logits.float(), dim=-1)
        entropy = -(final_probabilities * final_probabilities.clamp_min(1e-10).log()).sum(-1)
        token_ids = final_logits.argmax(-1)
        return SampleOutput(
            token_ids=token_ids,
            sampled_token_ids=token_ids,
            token_entropy=entropy,
            logits=final_logits if return_logits else None,
        )


def build_schedule_with_derivative(
    model: RePlaid,
    times: torch.Tensor,
    reconstruction_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build gamma(t) and dgamma/dt with RePlaID's exact gradient graph."""
    with torch.enable_grad():
        time = times.detach().to(torch.float64).requires_grad_(True)
        raw_shape = model.schedule_shape(time)
        shape = torch.where(reconstruction_mask, raw_shape.detach(), raw_shape)
        shape_prime = torch.autograd.grad(shape.sum(), time, create_graph=True)[0]
    gamma_0, gamma_1 = model.gamma_bounds()
    gamma = gamma_0 + (gamma_1 - gamma_0) * shape
    gamma_prime = (gamma_1 - gamma_0) * shape_prime
    return shape, shape_prime, gamma, gamma_prime


def register_variance_minimizing_hooks(
    shape: torch.Tensor,
    shape_prime: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    loss_holder: dict[str, torch.Tensor],
) -> None:
    """Make schedule shape minimize E[L_t^2], as in the reference trainer."""

    def hook(gradient: torch.Tensor) -> torch.Tensor:
        losses = loss_holder.get("diffusion")
        if losses is None:
            raise RuntimeError("diffusion loss must be populated before backward")
        multiplier = torch.where(
            reconstruction_mask,
            torch.ones_like(gradient),
            2.0 * losses.detach().to(gradient),
        )
        return gradient * multiplier

    if shape.requires_grad:
        shape.register_hook(hook)
    if shape_prime.requires_grad:
        shape_prime.register_hook(hook)
