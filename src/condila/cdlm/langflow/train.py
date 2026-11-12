"""PyTorch Lightning training module for LangFlow.

Implements the training procedure from Algorithm 1 of the paper:
  LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling
  (https://arxiv.org/abs/2604.11748)

Training follows the Bregman-divergence Flow Matching formulation:
  1. Embed tokens → clean embeddings z
  2. Sample γ from learnable Gumbel distribution
  3. Add noise: z_γ = α_γ * z + σ_γ * ε
  4. Predict token probabilities x̂_θ(z_γ, γ) via cross-entropy loss
  5. Jointly learn the noise schedule via scheduler fitting loss
"""

import math
import os
import typing
import warnings

import lightning as L
import torch
import torch.nn.functional as F

from .model import LangFlow
from .config import LangFlowConfig


def _remap_legacy_proposal_keys(state_dict: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
    """Convert legacy ``proposal.scale`` parameters to ``proposal.raw_scale``.

    Older checkpoints stored the Gumbel scale as a raw (unconstrained) parameter
    named ``scale``. The current ``GumbelProposal`` stores ``raw_scale`` and
    exposes ``scale = softplus(raw_scale)``. To load a legacy checkpoint we must
    invert the softplus: ``raw_scale = log(exp(scale) - 1) = log(expm1(scale))``.

    Keys that already use ``raw_scale`` (current local checkpoints) pass
    through untouched.
    """
    if not any(k.endswith("proposal.scale") for k in state_dict):
        return state_dict

    remapped: typing.Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.endswith("proposal.scale"):
            new_key = k[: -len("scale")] + "raw_scale"
            scale = v.float().clamp(min=1e-6)
            remapped[new_key] = (scale + torch.log(-torch.expm1(-scale))).to(v.dtype)
        else:
            remapped[k] = v
    return remapped


class LangFlowLightningModule(L.LightningModule):
    """PyTorch Lightning module for training LangFlow.

    Implements Algorithm 1 (Training) from the LangFlow paper with:
      - Cross-entropy loss on predicted token probabilities
      - Learnable Gumbel noise schedule (information-uniform principle)
      - Self-conditioning (randomly enabled during training)
      - Preconditioning bias skip connection with warmup
      - Gradient clipping and learning rate warmup
    """

    def __init__(
        self,
        # Model config
        config: typing.Optional[LangFlowConfig] = None,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        cond_dim: int = 128,
        n_blocks: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        model_length: int = 1024,
        # Training hyperparameters
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        warmup_steps: int = 5000,
        total_steps: int = 1_000_000,
        # Self-conditioning
        self_conditioning: bool = True,
        self_cond_prob: float = 0.25,
        # Bias warmup
        bias_warmup_steps: int = 5000,
        # Gradient clipping
        max_grad_norm: float = 1.0,
        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        # Scheduler loss weight
        scheduler_loss_weight: float = 1.0,
        # Gumbel initial parameters
        gumbel_loc: float = 4.723,
        gumbel_scale: float = 0.852,
        gumbel_cutoff: float = 1e-5,
        gumbel_entropy: float = 7.02,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])

        # Build or use provided config
        if config is not None:
            self.model_config = config
        else:
            self.model_config = LangFlowConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                cond_dim=cond_dim,
                n_blocks=n_blocks,
                n_heads=n_heads,
                dropout=dropout,
                model_length=model_length,
                self_conditioning=self_conditioning,
                use_bias=True,
                gumbel_loc=gumbel_loc,
                gumbel_scale=gumbel_scale,
                gumbel_cutoff=gumbel_cutoff,
                gumbel_entropy=gumbel_entropy,
            )

        self.model = LangFlow(self.model_config)

        # EMA model (for evaluation)
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        if use_ema:
            self.ema_model = LangFlow(self.model_config)
            self.ema_model.load_state_dict(self.model.state_dict())
            for param in self.ema_model.parameters():
                param.requires_grad = False
        else:
            self.ema_model = None

        # For manual global step tracking (needed for schedulers)
        self._global_step = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_global_step(self) -> int:
        """Return the current global step.

        Handles three cases:
        1. Attached to a Trainer → use trainer.global_step
        2. Not attached but has trainer property that raises → fall back
        3. No trainer at all → use internal counter
        """
        try:
            if self.trainer is not None:
                return self.trainer.global_step
        except RuntimeError:
            pass
        return self._global_step

    def _sample_gamma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample γ from the learnable Gumbel distribution.

        Follows Algorithm 1 lines 3-4:
          q ~ clip(Uniform(0,1), 1e-5, 1-1e-5)
          γ = P_μ - P_β * log(-log q)

        Args:
            batch_size: Number of samples.
            device: Target device.

        Returns:
            gamma: [B] tensor of log-SNR values.
        """
        cutoff = self.model.proposal.cutoff
        # Low-discrepancy sampling in [cutoff, 1 - cutoff]
        q = torch.rand(batch_size, device=device) * (1.0 - 2.0 * cutoff) + cutoff
        gamma = self.model.proposal(q)
        return gamma

    def _forward_diffusion(
        self,
        x_embed: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to clean embeddings (forward diffusion).

        z_γ = α_γ * z + σ_γ * ε,  ε ~ N(0, I)

        Args:
            x_embed: Clean token embeddings [B, L, D].
            gamma: Log-SNR values [B].

        Returns:
            z_gamma: Noisy embeddings [B, L, D].
        """
        gamma_f = gamma.float()
        alpha = torch.sigmoid(-gamma_f).sqrt()[:, None, None]
        sigma = torch.sigmoid(gamma_f).sqrt()[:, None, None]
        noise = torch.randn_like(x_embed)
        return (x_embed * alpha + noise * sigma).to(x_embed.dtype)

    # ------------------------------------------------------------------
    # Core training step (Algorithm 1)
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: typing.Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Execute one training step (Algorithm 1 from the paper).

        Args:
            batch: Dict with 'input_ids' [B, L] and optional 'attention_mask' [B, L].
            batch_idx: Index of the batch.

        Returns:
            Total loss (scalar).
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        B, L = input_ids.shape
        device = input_ids.device

        # === Line 2: Embed tokens ===
        x_embed = self.model._embed_tokens(input_ids)  # [B, L, D]

        # === Lines 3-4: Sample γ from Gumbel distribution ===
        # `gamma` is differentiable w.r.t. the proposal params (loc/scale). It is
        # used ONLY for the scheduler fitting loss below. The denoiser / CE path
        # must NOT backprop into the schedule, otherwise CE could shift the
        # schedule toward easy (low-noise) γ and collapse the proposal. So we
        # feed a detached copy into the forward diffusion and the model.
        gamma = self._sample_gamma(B, device)
        gamma_model = gamma.detach()

        # === Lines 5-6: Forward diffusion (add noise) ===
        z_gamma = self._forward_diffusion(x_embed, gamma_model)

        # === Lines 7-13: Self-conditioning (with probability p_SC) ===
        x_self_cond = None
        if self.hparams.self_conditioning and torch.rand(1, device=device).item() < self.hparams.self_cond_prob:
            # First pass without self-conditioning
            with torch.no_grad():
                logits_first = self.model.forward(
                    noisy_embeds=z_gamma,
                    timesteps=gamma_model,
                    x_self_cond=None,
                    return_dict=False,
                )
                probs_first = F.softmax(logits_first.float(), dim=-1)
                x_self_cond = self.model._embed_tokens(probs_first).detach()

        # === Line 14: Model forward (with optional self-conditioning) ===
        logits = self.model.forward(
            noisy_embeds=z_gamma,
            timesteps=gamma_model,
            x_self_cond=x_self_cond,
            return_dict=False,
        )  # [B, L, V]

        # === Line 15: Cross-entropy loss ===
        # Compute per-token CE, respecting attention mask if provided
        ce_loss_per_token = F.cross_entropy(
            logits.flatten(0, 1),
            input_ids.flatten(),
            reduction="none",
        ).reshape(B, L)

        if attention_mask is not None:
            valid_tokens = attention_mask.sum()
            ce_loss = (ce_loss_per_token * attention_mask).sum() / valid_tokens.clamp(min=1)
        else:
            ce_loss = ce_loss_per_token.mean()

        # === Lines 16-17: Scheduler fitting loss ===
        # H_γ = H_∞ * exp(-exp(-(γ - P_μ) / P_β))
        # L_Scheduler = (stopgrad(L_CE(γ)) - H_γ)^2
        # This is a curve fit of H(γ) onto the observed CE: γ is the independent
        # variable (a fixed data coordinate, hence detached) and loc/scale/entropy
        # are the only parameters that should receive gradient here.
        with torch.no_grad():
            if attention_mask is None:
                ce_per_sample = ce_loss_per_token.mean(dim=1)  # [B]
                valid_samples = torch.ones(B, device=device, dtype=torch.bool)
            else:
                token_counts = attention_mask.sum(dim=1)
                ce_per_sample = (ce_loss_per_token * attention_mask).sum(dim=1) / token_counts.clamp(min=1)
                valid_samples = token_counts > 0
        h_gamma = self.model.proposal.entropy * torch.exp(
            -torch.exp(-(gamma.detach() - self.model.proposal.loc) / self.model.proposal.scale)
        )
        scheduler_errors = (ce_per_sample.float() - h_gamma.float()).square()
        scheduler_loss = (scheduler_errors * valid_samples).sum() / valid_samples.sum().clamp(min=1)

        # === Line 18: Total loss ===
        total_loss = ce_loss + self.hparams.scheduler_loss_weight * scheduler_loss

        # === Logging ===
        self.log("train/ce_loss", ce_loss, on_step=True, on_epoch=False,
                 prog_bar=True, sync_dist=False)
        self.log("train/scheduler_loss", scheduler_loss, on_step=True, on_epoch=False,
                 prog_bar=True, sync_dist=False)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=False,
                 prog_bar=True, sync_dist=False)

        # Log gamma statistics
        with torch.no_grad():
            gamma_mean = gamma.mean()
            gamma_std = gamma.std(unbiased=False)
            self.log("train/gamma_mean", gamma_mean, on_step=True, on_epoch=False)
            self.log("train/gamma_std", gamma_std, on_step=True, on_epoch=False)

            # Log Gumbel distribution parameters
            self.log("train/gumbel_loc", self.model.proposal.loc, on_step=True, on_epoch=False)
            self.log("train/gumbel_scale", self.model.proposal.scale, on_step=True, on_epoch=False)
            self.log("train/gumbel_h_inf", self.model.proposal.entropy, on_step=True, on_epoch=False)

            # Log SNR stats
            alpha = torch.sigmoid(-gamma).sqrt()
            self.log("train/alpha_mean", alpha.mean(), on_step=True, on_epoch=False)

        return total_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self,
        batch: typing.Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> typing.Dict[str, torch.Tensor]:
        """Compute validation cross-entropy at a fixed set of γ levels.

        Evaluates the model across the noise spectrum to track per-noise-level
        performance. Uses the EMA model if available.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        B, L = input_ids.shape
        device = input_ids.device

        model = self.ema_model if (self.use_ema and self.ema_model is not None) else self.model

        # Evaluate at multiple γ levels spanning the noise range
        gamma_levels = torch.linspace(-2.0, 8.0, 11, device=device)  # -2 to 8 in 11 steps
        ce_losses = []

        with torch.no_grad():
            x_embed = model._embed_tokens(input_ids)

            for gamma_val in gamma_levels:
                gamma_batch = gamma_val.expand(B)
                z_gamma = self._forward_diffusion(x_embed, gamma_batch)

                # One forward pass without self-conditioning for clean evaluation
                logits = model.forward(
                    noisy_embeds=z_gamma,
                    timesteps=gamma_batch,
                    x_self_cond=None,
                    return_dict=False,
                )

                ce_per_token = F.cross_entropy(
                    logits.flatten(0, 1),
                    input_ids.flatten(),
                    reduction="none",
                ).reshape(B, L)

                if attention_mask is not None:
                    valid_tokens = attention_mask.sum()
                    ce = (ce_per_token * attention_mask).sum() / valid_tokens.clamp(min=1)
                else:
                    ce = ce_per_token.mean()
                ce_losses.append(ce)

        # Average across gamma levels
        val_ce = torch.stack(ce_losses).mean()
        val_ppl = torch.exp(val_ce)

        self.log("val/ce_loss", val_ce, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("val/ppl", val_ppl, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        return {"val_ce": val_ce, "val_ppl": val_ppl}

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Configure AdamW optimizer with cosine decay and linear warmup.

        Uses separate parameter groups:
          - No weight decay for bias, layernorm, and embedding parameters.
          - Weight decay for all other parameters.

        Learning rate schedule:
          - Linear warmup for warmup_steps
          - Cosine decay to 0 over the remaining steps
        """
        # Collect proposal parameter IDs first so we can exclude them
        # from the main model groups (they get their own group with lower LR)
        proposal_param_ids = {id(p) for p in self.model.proposal.parameters()}

        # Separate parameters that should not have weight decay
        decay_params = []
        no_decay_params = []
        no_decay_keywords = ["bias", "norm", "embedding", "ln_", "LayerNorm"]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in proposal_param_ids:
                continue  # Handled separately below
            if any(kw in name for kw in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Gumbel proposal parameters (learnable noise schedule)
        proposal_params = list(self.model.proposal.parameters())

        lr = float(self.hparams.learning_rate)
        opt_groups = [
            {
                "params": decay_params,
                "lr": lr,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": no_decay_params,
                "lr": lr,
                "weight_decay": 0.0,
            },
            {
                "params": proposal_params,
                "lr": lr * 0.1,  # Lower LR for scheduler params
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )

        # Cosine decay scheduler with linear warmup.
        # NOTE: We use a linear warmup followed by cosine decay.
        # The scheduler applies a multiplier to the base LR stored in each param group.
        warmup = self.hparams.warmup_steps
        total = self.hparams.total_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            progress = float(current_step - warmup) / float(max(1, total - warmup))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    # ------------------------------------------------------------------
    # EMA management
    # ------------------------------------------------------------------

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """Update EMA once per optimizer step, including with accumulation."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.use_ema and self.ema_model is not None:
            # Update EMA
            with torch.no_grad():
                for ema_param, model_param in zip(
                    self.ema_model.parameters(), self.model.parameters()
                ):
                    ema_param.data.mul_(self.ema_decay).add_(
                        model_param.data, alpha=1.0 - self.ema_decay
                    )

    # ------------------------------------------------------------------
    # Bias warmup (preconditioning skip connection)
    # ------------------------------------------------------------------

    def on_train_batch_start(self, batch, batch_idx):
        """Handle bias warmup: enable bias after bias_warmup_steps."""
        global_step = self._get_global_step()
        if global_step < self.hparams.bias_warmup_steps:
            # Disable bias during warmup
            self.model.config.use_bias = False
        else:
            self.model.config.use_bias = True

    # ------------------------------------------------------------------
    # Gradient clipping
    # ------------------------------------------------------------------

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """Apply gradient clipping."""
        if self.hparams.max_grad_norm is not None and self.hparams.max_grad_norm > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.hparams.max_grad_norm,
                gradient_clip_algorithm="norm",
            )

    # ------------------------------------------------------------------
    # State dict helpers (save EMA weights)
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        """Save EMA model state alongside the main model."""
        checkpoint["condila_model_config"] = self.model_config.to_dict()
        if self.use_ema and self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()
        checkpoint["gumbel_loc"] = self.model.proposal.loc
        checkpoint["gumbel_scale"] = self.model.proposal.scale
        checkpoint["gumbel_entropy"] = self.model.proposal.entropy

    def on_load_checkpoint(self, checkpoint):
        """Load EMA model state."""
        if self.use_ema and self.ema_model is not None and "ema_state_dict" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_state_dict"])

    # ------------------------------------------------------------------
    # Warm-start from pretrained weights (does NOT restore optimizer/step)
    # ------------------------------------------------------------------

    def load_pretrained_weights(
        self,
        path: str,
        load_ema: bool = True,
        strict: bool = False,
    ) -> None:
        """Warm-start the model from a pretrained checkpoint (weights only).

        Use this to *continue training on top of the official checkpoint*. It
        loads weights into ``self.model`` (and ``self.ema_model``) but does NOT
        restore the optimizer, LR scheduler, or global step — training starts a
        fresh schedule from step 0. For exact full-state resumption of a run,
        pass ``ckpt_path=`` to ``Trainer.fit`` instead.

        Supported formats:
          - ``*.safetensors`` : HuggingFace weights (inner-model keys, e.g. the
            official ``model.safetensors``). These were exported from the EMA
            model, so they seed both ``model`` and ``ema_model``.
          - ``*.ckpt``        : Lightning checkpoint (``model.*`` / ``ema_model.*``
            prefixed keys, plus an optional ``ema_state_dict``).

        Legacy ``proposal.scale`` parameters are auto-remapped to
        ``proposal.raw_scale``.

        Args:
            path: Path to the checkpoint or safetensors file.
            load_ema: Also (re)initialize the EMA model from the loaded weights.
            strict: Forwarded to ``load_state_dict``. Defaults to False so that
                missing buffers (e.g. rotary caches) don't raise.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pretrained weights not found: {path}")

        ema_state: typing.Optional[typing.Dict[str, torch.Tensor]] = None

        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            model_state = load_file(path)  # inner-model keys: backbone.*, proposal.*
            if any(k.startswith("model.") for k in model_state):
                # Defensive: a safetensors that kept the LightningModule prefix.
                model_state = {
                    k[len("model."):]: v
                    for k, v in model_state.items()
                    if k.startswith("model.")
                }
        else:
            ckpt = torch.load(path, map_location="cpu")
            full_state = ckpt.get("state_dict", ckpt)
            model_state = {
                k[len("model."):]: v
                for k, v in full_state.items()
                if k.startswith("model.")
            }
            if not model_state:
                # Raw inner-model state dict (no LightningModule prefix).
                model_state = full_state
            if "ema_state_dict" in ckpt:
                ema_state = ckpt["ema_state_dict"]
            else:
                ema_prefixed = {
                    k[len("ema_model."):]: v
                    for k, v in full_state.items()
                    if k.startswith("ema_model.")
                }
                if ema_prefixed:
                    ema_state = ema_prefixed

        model_state = _remap_legacy_proposal_keys(model_state)
        missing, unexpected = self.model.load_state_dict(model_state, strict=strict)
        if missing:
            warnings.warn(f"[load_pretrained_weights] missing keys in model: {missing}")
        if unexpected:
            warnings.warn(f"[load_pretrained_weights] unexpected keys in model: {unexpected}")

        if load_ema and self.use_ema and self.ema_model is not None:
            # Fall back to the main weights when no dedicated EMA state exists
            # (e.g. the official safetensors export).
            ema_to_load = _remap_legacy_proposal_keys(ema_state) if ema_state is not None else model_state
            self.ema_model.load_state_dict(ema_to_load, strict=strict)

        print(f"[load_pretrained_weights] warm-started from {path} "
              f"(ema={'yes' if (load_ema and self.use_ema) else 'no'}, "
              f"loaded {len(model_state)} tensors)")
