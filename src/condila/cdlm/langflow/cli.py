"""Training entry point for LangFlow.

Launches PyTorch Lightning training with Hydra configuration.

Usage:
    # Train on OpenWebText (GPT-2 tokenizer, 1024 context)
    condila-langflow-train \
        data=openwebtext \
        training.total_steps=1000000

    # Train on LM1B (BERT tokenizer, 128 context)
    condila-langflow-train \
        data=lm1b \
        training.total_steps=1000000

    # Multi-GPU training
    torchrun --nproc_per_node=8 -m condila.cdlm.langflow.cli \
        data=openwebtext \
        training.total_steps=1000000

    # Resume from checkpoint
    condila-langflow-train \
        data=openwebtext \
        training.total_steps=1000000 \
        resume=path/to/checkpoint.ckpt
"""

import os
import typing

import hydra
import lightning as L
import omegaconf
import torch
import datetime
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from .train import LangFlowLightningModule
from .data import LangFlowDataModule

# Register OmegaConf resolvers
for _name, _resolver in {
    "cwd": os.getcwd,
    "device_count": lambda: torch.cuda.device_count(),
    "div_up": lambda x, y: (x + y - 1) // y,
}.items():
    if not omegaconf.OmegaConf.has_resolver(_name):
        omegaconf.OmegaConf.register_new_resolver(_name, _resolver)


def build_callbacks(config) -> typing.List[L.Callback]:
    """Build Lightning callbacks from config."""
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.training.checkpoint_dir,
        filename="langflow-{step:07d}-{val/ppl:.2f}",
        auto_insert_metric_name=False,
        save_top_k=config.training.save_top_k,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        monitor="val/ppl",
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    # Step-based checkpoints cannot monitor a validation value that has not yet
    # been produced. Keep the latest periodic state separately from best-val states.
    if config.training.save_every_steps > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=config.training.checkpoint_dir,
            filename="step-{step:07d}",
            auto_insert_metric_name=False,
            every_n_train_steps=config.training.save_every_steps,
            save_on_train_epoch_end=False,
            save_top_k=1,
            save_last=True,
        ))

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    return callbacks


def build_loggers(config) -> typing.List[L.pytorch.loggers.Logger]:
    """Build loggers from config."""
    loggers = []

    # CSV logger for local logging
    csv_logger = CSVLogger(
        save_dir=config.training.log_dir,
        name="langflow",
    )
    loggers.append(csv_logger)

    # Wandb logger (optional)
    if config.training.get("use_wandb", False):
        from lightning.pytorch.loggers import WandbLogger

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_base_name = config.training.get("wandb_run_name", None)
        wandb_name = f"{wandb_base_name}_{timestamp}" if wandb_base_name else None
        wandb_logger = WandbLogger(
            project=config.training.get("wandb_project", "langflow-ori"),
            entity=config.training.get("wandb_entity", os.environ.get("WANDB_ENTITY")),
            name=wandb_name,
            save_dir=config.training.log_dir,
        )
        loggers.append(wandb_logger)

    return loggers


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="train",
)
def main(config):
    """Main training entry point."""
    torch.set_float32_matmul_precision("high")
    # Set seed for reproducibility
    L.seed_everything(config.training.seed, workers=True)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = LangFlowLightningModule(
        # Architecture
        vocab_size=config.model.vocab_size,
        hidden_size=config.model.hidden_size,
        cond_dim=config.model.cond_dim,
        n_blocks=config.model.n_blocks,
        n_heads=config.model.n_heads,
        dropout=config.model.dropout,
        model_length=config.model.length,
        # Training
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        beta1=config.training.beta1,
        beta2=config.training.beta2,
        warmup_steps=config.training.warmup_steps,
        total_steps=config.training.total_steps,
        # Self-conditioning
        self_conditioning=config.model.get("self_conditioning", True),
        self_cond_prob=config.training.self_cond_prob,
        # Bias warmup
        bias_warmup_steps=config.training.bias_warmup_steps,
        # Gradient clipping
        max_grad_norm=config.training.max_grad_norm,
        # EMA
        use_ema=config.training.get("use_ema", True),
        ema_decay=config.training.get("ema_decay", 0.9999),
        # Scheduler loss
        scheduler_loss_weight=config.training.get("scheduler_loss_weight", 1.0),
        # Gumbel initial parameters
        gumbel_loc=config.model.get("gumbel_loc", 4.723),
        gumbel_scale=config.model.get("gumbel_scale", 0.852),
        gumbel_cutoff=config.model.get("gumbel_cutoff", 1e-5),
        gumbel_entropy=config.model.get("gumbel_entropy", 7.02),
    )

    # Print parameter count
    n_params = sum(p.numel() for p in model.model.parameters())
    n_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"\nLangFlow model: {n_params / 1e6:.1f}M total, "
          f"{n_trainable / 1e6:.1f}M trainable parameters\n")

    # ------------------------------------------------------------------
    # Warm-start from pretrained weights (e.g. the official safetensors).
    # This loads weights only — a fresh optimizer / LR schedule / step counter
    # is used. For exact full-state resumption use `resume=` instead (below).
    # ------------------------------------------------------------------
    init_from = config.get("init_from", None)
    if init_from:
        model.load_pretrained_weights(init_from, load_ema=model.use_ema)
        if config.training.get("bias_warmup_steps", 0) > 0:
            print(
                "[warn] init_from is set but training.bias_warmup_steps="
                f"{config.training.bias_warmup_steps} > 0: the preconditioning "
                "bias will be disabled for the first "
                f"{config.training.bias_warmup_steps} steps even though the "
                "pretrained model was trained WITH it. When continuing from a "
                "trained checkpoint, set training.bias_warmup_steps=0."
            )

    # ------------------------------------------------------------------
    # Build data module
    # ------------------------------------------------------------------
    data_module = LangFlowDataModule(
        train_dataset=config.data.get("train", "openwebtext"),
        valid_dataset=config.data.get("valid", "wikitext103"),
        tokenizer_name_or_path=config.data.get("tokenizer_name_or_path", "gpt2"),
        cache_dir=config.data.get("cache_dir", "/tmp/data"),
        wrap=config.data.get("wrap", True),
        streaming=config.data.get("streaming", False),
        insert_train_eos=config.data.get("insert_train_eos", True),
        insert_valid_eos=config.data.get("insert_valid_eos", True),
        batch_size=config.loader.batch_size,
        eval_batch_size=config.loader.get("eval_batch_size", config.loader.batch_size),
        num_workers=config.loader.get("num_workers", 16),
        pin_memory=config.loader.get("pin_memory", True),
        model_length=config.model.length,
    )

    # ------------------------------------------------------------------
    # Build trainer
    # ------------------------------------------------------------------
    callbacks = build_callbacks(config)
    loggers = build_loggers(config)

    # Compute effective global batch size
    num_gpus = max(torch.cuda.device_count(), 1)
    num_nodes = config.trainer.get("num_nodes", 1)
    accumulate = config.trainer.get("accumulate_grad_batches", 1)
    effective_batch = config.loader.batch_size * num_gpus * num_nodes * accumulate
    print(f"Effective global batch size: {effective_batch} "
          f"({config.loader.batch_size} per GPU × {num_gpus} GPUs "
          f"× {num_nodes} nodes × {accumulate} accumulate)\n")

    trainer = L.Trainer(
        # Hardware
        devices=config.trainer.get("devices", "auto"),
        num_nodes=num_nodes,
        accelerator=config.trainer.get("accelerator", "auto"),
        # Precision
        precision=config.trainer.get("precision", "bf16-mixed"),
        # Training steps
        max_steps=config.training.total_steps,
        val_check_interval=config.training.get("val_check_interval", 2000),
        # Gradient accumulation
        accumulate_grad_batches=accumulate,
        # Gradient clipping is handled in LightningModule
        # Logging
        log_every_n_steps=config.training.get("log_every_n_steps", 10),
        # Callbacks and loggers
        callbacks=callbacks,
        logger=loggers,
        # Distributed
        strategy=config.trainer.get("strategy", "ddp_find_unused_parameters_true"),
        # Misc
        default_root_dir=config.training.log_dir,
        enable_progress_bar=True,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    # Full-state resume: restores weights + optimizer + LR scheduler + global
    # step + EMA + callback state from a Lightning `.ckpt`. Note this respects
    # the original `total_steps`, so to train *further* than the saved step you
    # must also raise `training.total_steps`. Use `init_from=` (weights-only
    # warm-start) for the official safetensors, which has no optimizer state.
    resume = config.get("resume", None)
    if resume:
        print(f"[resume] restoring full training state from: {resume}")
    trainer.fit(model, datamodule=data_module, ckpt_path=resume)

    # Save final checkpoint
    final_path = os.path.join(config.training.checkpoint_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete! Final checkpoint saved to: {final_path}")

    # Also save as HuggingFace-compatible safetensors
    if config.training.get("save_safetensors", True):
        from safetensors.torch import save_file
        safetensors_path = os.path.join(
            config.training.checkpoint_dir, "model.safetensors"
        )
        if model.use_ema and model.ema_model is not None:
            save_file(model.ema_model.state_dict(), safetensors_path)
        else:
            save_file(model.model.state_dict(), safetensors_path)

        # Save config.json alongside
        model.model_config.save_pretrained(config.training.checkpoint_dir)
        print(f"HuggingFace-format model saved to: {config.training.checkpoint_dir}")


if __name__ == "__main__":
    main()
