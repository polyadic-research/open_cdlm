from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from condila.cdlm import (
    LangFlowConfig,
    RePlaidConfig,
    create_model,
    generate,
    get_model_spec,
    register_model,
)


def tiny_config(name):
    if name == "langflow":
        return LangFlowConfig(vocab_size=18, hidden_size=16, cond_dim=8, n_blocks=1,
                              n_heads=2, dropout=0.0, model_length=8)
    return RePlaidConfig(
        vocab_size=18, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, max_seq_length=8, cond_dim=8, dropout=0.0,
        replaid_embedding_dim=4, replaid_schedule_hidden_size=16,
    )


@pytest.mark.parametrize("name,sampler,nfe", [
    ("langflow", "euler_edm", 3), ("replaid", "ddpm", 3),
    ("replaid", "ddim", 3), ("replaid", "heun", 5),
])
def test_common_sampling_delegates_correct_steps_and_preserves_rng_and_modes(name, sampler, nfe):
    torch.manual_seed(30)
    model = create_model(name, tiny_config(name)).train()
    # A mixed train/eval state must also be restored.
    next(iter(model.children())).eval()
    modes = [module.training for module in model.modules()]
    state = torch.get_rng_state().clone()
    with patch.object(model, "forward", wraps=model.forward) as forward:
        ids = generate(model, num_samples=2, length=8, steps=2, seed=7, sampler=sampler)
    assert forward.call_count == nfe
    assert ids.shape == (2, 8) and ids.dtype == torch.long
    assert 0 <= ids.min() <= ids.max() < 18
    assert [module.training for module in model.modules()] == modes
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    torch.testing.assert_close(
        ids, generate(model, num_samples=2, length=8, steps=2, seed=7, sampler=sampler),
        rtol=0, atol=0,
    )
    separated = torch.cat([
        generate(model, num_samples=1, length=8, steps=2, seed=7 + i, sampler=sampler)
        for i in range(2)
    ])
    torch.testing.assert_close(ids, separated, rtol=0, atol=0)


def test_creation_validation_and_unknown_model():
    config = tiny_config("langflow")
    assert type(create_model("LANGFLOW", config.to_dict()).config) is LangFlowConfig
    with pytest.raises(ValueError, match="unknown model"):
        create_model("missing")
    with pytest.raises(TypeError, match="requires"):
        create_model("langflow", tiny_config("replaid"))
    with pytest.raises(TypeError, match="overrides"):
        create_model("langflow", config, n_blocks=2)
    with pytest.raises(ValueError, match="head"):
        create_model("langflow", hidden_size=15, n_heads=2)


@pytest.mark.parametrize("options", [
    {"steps": 0}, {"steps": 1.5}, {"num_samples": 0}, {"length": 0},
    {"seed": -1}, {"temperature": float("nan")}, {"sampler": "ddpm"},
    {"temperature": 0.8}, {"length": 9}, {"unknown_option": True},
])
def test_invalid_options_raise_and_restore_training_state(options):
    model = create_model("langflow", tiny_config("langflow")).train()
    with pytest.raises((ValueError, TypeError)):
        generate(model, **options)
    assert model.training


def test_native_noise_options_and_seed_conflicts():
    model = create_model("langflow", tiny_config("langflow"))
    noise = torch.randn(1, 8, 16)
    with pytest.raises(ValueError, match="seed=None"):
        generate(model, initial_noise=noise)
    first = generate(model, steps=1, seed=None, initial_noise=noise)
    second = generate(model, steps=1, seed=None, initial_noise=noise)
    torch.testing.assert_close(first, second)


def test_external_model_registration(monkeypatch):
    from condila.cdlm import registry

    monkeypatch.setattr(registry, "_MODELS", registry._MODELS.copy())

    @dataclass
    class Config:
        width: int = 3

    class Model(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    def sample(model, *, num_samples, length, **kwargs):
        assert not model.training
        return torch.ones(num_samples, length or model.config.width, dtype=torch.long)

    register_model("external", Model, Config, sampler=sample, samplers=("native",))
    model = create_model("external", {"width": 4})
    assert generate(model, num_samples=2).shape == (2, 4)
    assert get_model_spec("EXTERNAL").name == "external"
    with pytest.raises(ValueError, match="already registered"):
        register_model("external", Model, Config)


def test_langflow_likelihood_adapter_uses_namespaced_native_model():
    from condila.dlm_metrics.likelihood import LangFlowAdapter

    model = create_model("langflow", tiny_config("langflow")).eval()
    adapter = LangFlowAdapter(model)
    ids = torch.randint(18, (2, 8))
    state = adapter.embed(ids).detach().requires_grad_(True)
    predicted = adapter.denoise(state, torch.tensor(3.0))
    assert predicted.shape == state.shape
    assert torch.isfinite(torch.autograd.grad(predicted.square().sum(), state)[0]).all()
    assert adapter.decoder_logits(state).shape == (2, 8, 18)


def test_subclass_registration_uses_its_own_sampler(monkeypatch):
    from condila.cdlm import LangFlow, registry

    monkeypatch.setattr(registry, "_MODELS", registry._MODELS.copy())

    class CustomLangFlow(LangFlow):
        pass

    def custom_sample(model, *, num_samples, length, **kwargs):
        return torch.zeros(num_samples, length or 8, dtype=torch.long)

    register_model("custom_langflow", CustomLangFlow, LangFlowConfig, sampler=custom_sample)
    model = create_model("custom_langflow", tiny_config("langflow"))
    assert torch.equal(generate(model, steps=1), torch.zeros(1, 8, dtype=torch.long))
