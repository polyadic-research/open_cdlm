from pathlib import Path

import lightning as L
from omegaconf import OmegaConf
import torch

from condila.cdlm import from_pretrained
from condila.cdlm.langflow import LangFlowLightningModule
from condila.cdlm.langflow.cli import build_callbacks, build_loggers
from tests.test_cdlm_api import tiny_config


def test_periodic_and_best_checkpoints_are_saved_after_the_correct_events(tmp_path):
    config = OmegaConf.create({"training": {
        "checkpoint_dir": str(tmp_path / "checkpoints"), "save_top_k": 2,
        "save_every_steps": 1, "log_dir": str(tmp_path / "logs"), "use_wandb": False,
    }})
    callbacks = build_callbacks(config)
    module = LangFlowLightningModule(
        config=tiny_config("langflow"), warmup_steps=0, total_steps=2,
        bias_warmup_steps=0, self_conditioning=False,
    )
    rows = [{"input_ids": torch.randint(18, (8,))} for _ in range(4)]
    data = torch.utils.data.DataLoader(rows, batch_size=2)
    trainer = L.Trainer(
        accelerator="cpu", devices=1, max_steps=2, val_check_interval=1,
        num_sanity_val_steps=0, logger=build_loggers(config), callbacks=callbacks,
        enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=1,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=data, val_dataloaders=data)
    folder = Path(config.training.checkpoint_dir)
    assert 1 <= len(list(folder.glob("langflow-*.ckpt"))) <= 2
    assert len(list(folder.glob("step-*.ckpt"))) == 1
    assert (folder / "last.ckpt").is_file()
    checkpoint = torch.load(folder / "last.ckpt", map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 2
    assert from_pretrained("langflow", folder / "last.ckpt").config.hidden_size == 16
