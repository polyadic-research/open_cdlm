import pytest
import torch
import torch.nn.functional as F
import lightning as L

from condila.cdlm.langflow import LangFlow, LangFlowConfig
from condila.cdlm.langflow.model import GumbelProposal
from condila.cdlm.langflow.train import LangFlowLightningModule


def small_config(**kwargs):
    options = dict(vocab_size=17, hidden_size=16, cond_dim=8, n_blocks=1,
                   n_heads=2, dropout=0.0, model_length=8)
    options.update(kwargs)
    return LangFlowConfig(**options)


@pytest.mark.parametrize("scale", [0.001, 0.852, 1000.0])
def test_gumbel_scale_initialization_and_positivity(scale):
    proposal = GumbelProposal(scale=scale)
    torch.testing.assert_close(proposal.scale, torch.tensor(scale))
    with torch.no_grad():
        proposal.raw_scale.fill_(-10)
    assert proposal.scale.item() > 0
    gamma = proposal(torch.tensor([0.1, 0.5, 0.9]))
    assert torch.isfinite(gamma).all()
    gamma.sum().backward()
    assert torch.isfinite(proposal.raw_scale.grad)


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_invalid_gumbel_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        GumbelProposal(scale=scale)


def test_legacy_proposal_keys_load_through_model_and_training(tmp_path):
    model = LangFlow(small_config())
    state = dict(model.state_dict())
    state.pop("proposal.raw_scale")
    state["proposal.scale"] = torch.tensor(1.25)
    model.load_state_dict(state, strict=True)
    torch.testing.assert_close(model.proposal.scale, torch.tensor(1.25))
    module = LangFlowLightningModule(config=small_config())
    path = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in state.items()}}, path)
    module.load_pretrained_weights(str(path), strict=True)
    torch.testing.assert_close(module.model.proposal.scale, torch.tensor(1.25))
    torch.testing.assert_close(module.ema_model.proposal.scale, torch.tensor(1.25))
    model.save_pretrained(tmp_path / "hf")
    restored = LangFlow.from_pretrained(tmp_path / "hf")
    torch.testing.assert_close(restored.proposal.scale, model.proposal.scale)


def test_scheduler_ignores_padding_and_empty_sequences(monkeypatch):
    module = LangFlowLightningModule(config=small_config(), use_ema=False,
                                    self_conditioning=False)
    logged = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kw: logged.update({name: value}))
    gamma = torch.tensor([3.0, 4.0, 5.0])
    monkeypatch.setattr(module, "_sample_gamma", lambda size, device: gamma.to(device))
    logits = torch.randn(3, 4, 17, requires_grad=True)
    monkeypatch.setattr(module.model, "forward", lambda **kw: logits)
    ids = torch.zeros(3, 4, dtype=torch.long)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    ce = F.cross_entropy(logits.flatten(0, 1), ids.flatten(), reduction="none").view(3, 4)
    targets = torch.stack((ce[0, :2].mean(), ce[1, 0]))
    fitted = module.model.proposal.entropy * module.model.proposal._get_distribution().cdf(gamma)
    expected = F.mse_loss(fitted[:2], targets.detach())
    loss = module.training_step({"input_ids": ids, "attention_mask": mask}, 0)
    torch.testing.assert_close(logged["train/scheduler_loss"], expected)
    loss.backward()
    assert module.model.proposal.raw_scale.grad is not None
    assert torch.isfinite(module.model.proposal.raw_scale.grad)


def test_ema_updates_once_per_optimizer_step_with_gradient_accumulation(tmp_path):
    module = LangFlowLightningModule(config=small_config(), learning_rate=1e-3,
                                    warmup_steps=0, total_steps=2,
                                    bias_warmup_steps=0, self_conditioning=False,
                                    ema_decay=0.5)

    class CheckEMA(L.Callback):
        def __init__(self):
            self.expected = {k: v.detach().clone() for k, v in module.model.named_parameters()}
            self.last_step = 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step != self.last_step:
                for name, param in pl_module.model.named_parameters():
                    self.expected[name].lerp_(param.detach(), 0.5)
                self.last_step = trainer.global_step
            for name, param in pl_module.ema_model.named_parameters():
                torch.testing.assert_close(param, self.expected[name])

    callback = CheckEMA()
    trainer = L.Trainer(accelerator="cpu", devices=1, max_steps=2,
                        accumulate_grad_batches=2, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, limit_val_batches=0,
                        callbacks=[callback], default_root_dir=tmp_path)
    data = [{"input_ids": torch.randint(17, (8,))} for _ in range(4)]
    trainer.fit(module, train_dataloaders=torch.utils.data.DataLoader(data, batch_size=1))
    assert callback.last_step == 2
