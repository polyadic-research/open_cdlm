from dataclasses import replace
from pathlib import Path

import pytest
import torch

from open_cdlm.cdlm.replaid.config import load_config
from open_cdlm.cdlm.replaid.layers import FP32LayerNorm
from open_cdlm.cdlm.replaid.model import (
    RePlaid,
    build_schedule_with_derivative,
    register_variance_minimizing_hooks,
)
from open_cdlm.cdlm.replaid.objective import (
    make_batch_plan,
    nelbo_objective,
    update_loss_moments,
    validation_terms,
)
from open_cdlm.cdlm.replaid.optim import build_optimizer, build_scheduler


def activate(model):
    with torch.no_grad():
        model.output.linear.weight.normal_(std=0.1)
        for block in model.blocks:
            block.adaln.weight.normal_(std=0.1)
            block.adaln.bias.normal_(std=0.1)


def test_paper_recipe_and_architecture():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "src/open_cdlm/cdlm/replaid/configs/owt_sc.yaml")
    assert config.model.vocab_size == 50257
    assert config.model.cond_dim == 128
    assert config.model.replaid_embedding_dim == 16
    assert config.train.ema_decay == 0.9999
    assert config.train.total_steps == 1_000_000
    assert config.train.global_batch_size == 512
    with torch.device("meta"):
        model = RePlaid(config.model)
    assert sum(p.numel() for p in model.parameters()) == 131_909_219
    assert isinstance(model.blocks[0].norm1, FP32LayerNorm)
    assert isinstance(model.output.norm, FP32LayerNorm)
    assert model.blocks[0].num_heads == model.blocks[0].num_key_value_heads == 12
    assert hasattr(model, "gamma_projection") and hasattr(model.output, "adaln")
    assert not hasattr(model.blocks[0], "q_norm")
    assert model.blocks[0].mlp[0].bias is not None
    assert model.blocks[0].mlp[1].approximate == "tanh"


def test_schedule_sphere_output_prior_and_ungated_sc(tiny_config):
    torch.manual_seed(2)
    model = RePlaid(tiny_config.model).eval()
    embedding = model.normalized_embedding()
    torch.testing.assert_close(embedding.norm(dim=-1), torch.ones(18))
    times = torch.linspace(0, 1, 100, dtype=torch.float64)
    gamma = model.gamma_at(times)
    assert gamma.dtype == torch.float64
    assert bool((gamma[1:] > gamma[:-1]).all())
    torch.testing.assert_close(gamma[[0, -1]], torch.tensor([-3.0, 6.0], dtype=torch.float64))
    state = torch.randn(2, 8, 4)
    values = torch.tensor([-2.0, 3.0], dtype=torch.float64)
    with torch.no_grad():
        logits = model(state, values)
        expected = torch.sigmoid(-values.float()).sqrt() / torch.sigmoid(values.float())
        expected = (expected[:, None, None] * state) @ embedding.t()
        torch.testing.assert_close(logits, expected)
    probabilities = torch.full((2, 8, 18), 1 / 18)
    torch.testing.assert_close(
        model._self_condition_from_probabilities(probabilities, embedding),
        probabilities @ embedding,
    )
    assert model._self_condition_from_probabilities(probabilities, embedding).abs().sum() > 0


def test_attention_is_bidirectional(tiny_config):
    torch.manual_seed(3)
    model = RePlaid(tiny_config.model).eval()
    activate(model)
    state = torch.randn(1, 8, 4)
    changed = state.clone()
    changed[:, -1] += 3
    with torch.no_grad():
        first = model(state, torch.ones(1), output_prior_scale=0)
        second = model(changed, torch.ones(1), output_prior_scale=0)
    assert not torch.allclose(first[:, 0], second[:, 0])


def test_schedule_shape_optimizes_second_moment_but_endpoints_optimize_mean(tiny_config):
    torch.manual_seed(4)
    hooked = RePlaid(tiny_config.model)
    variance = RePlaid(tiny_config.model)
    ordinary = RePlaid(tiny_config.model)
    variance.load_state_dict(hooked.state_dict())
    ordinary.load_state_dict(hooked.state_dict())
    times = torch.tensor([0.15, 0.35, 0.65, 0.85], dtype=torch.float64)
    mask = torch.zeros(4, dtype=torch.bool)

    def losses(model, hook=False):
        shape, prime, gamma, gamma_prime = build_schedule_with_derivative(model, times, mask)
        loss = torch.exp(-gamma) * gamma_prime * (2 + gamma.square())
        if hook:
            register_variance_minimizing_hooks(shape, prime, mask, {"diffusion": loss.detach()})
        return loss

    losses(hooked, hook=True).mean().backward()
    losses(variance).square().mean().backward()
    losses(ordinary).mean().backward()
    for name, parameter in hooked.noise_schedule.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(variance.noise_schedule.named_parameters())[name].grad
        )
    torch.testing.assert_close(hooked.gamma_0.grad, ordinary.gamma_0.grad)
    torch.testing.assert_close(hooked.gamma_1.grad, ordinary.gamma_1.grad)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_nelbo_backward_and_geometry_detach_on_sc(tiny_config, checkpointing):
    torch.manual_seed(5)
    model = RePlaid(tiny_config.model).train()
    activate(model)
    if checkpointing:
        model.gradient_checkpointing_enable()
    generator = torch.Generator().manual_seed(6)
    rows = torch.randint(18, (8, 8))
    plan = make_batch_plan(model, 8, generator=generator)
    assert int(plan.self_condition_mask.sum()) == 2
    loss, statistics = nelbo_objective(
        model, rows, torch.ones_like(rows), tiny_config, 5000, plan=plan, generator=generator
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters())
    update_loss_moments(model, statistics["moments"])
    assert model.loss_ema_bias > 1e-8
    model.zero_grad(set_to_none=True)
    plan.self_condition_mask.fill_(True)
    loss, _ = nelbo_objective(
        model, rows, torch.ones_like(rows), tiny_config, 5000, plan=plan, generator=generator
    )
    loss.backward()
    assert model.output.linear.weight.grad.abs().sum() > 0
    for parameter in [
        model.embedding.weight,
        model.gamma_0,
        model.gamma_1,
        *model.noise_schedule.parameters(),
    ]:
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_accumulated_gradient_matches_full_rank_batch(tiny_config):
    torch.manual_seed(7)
    tiny_config.model = replace(tiny_config.model, self_condition_probability=0.0)
    full = RePlaid(tiny_config.model)
    activate(full)
    micro = RePlaid(tiny_config.model)
    micro.load_state_dict(full.state_dict())
    rows = torch.randint(18, (8, 8))
    plan = make_batch_plan(full, 8, generator=torch.Generator().manual_seed(8))
    # The second microbatch contains no reconstruction samples, a case that the
    # original local experiment's per-microbatch clamping could not represent.
    assert not bool(plan.reconstruction_mask[4:].any())
    g1, g2 = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
    loss, _ = nelbo_objective(
        full, rows, torch.ones_like(rows), tiny_config, 5000, plan=plan, generator=g1
    )
    loss.backward()
    for start in (0, 4):
        batch = rows[start : start + 4]
        loss, _ = nelbo_objective(
            micro,
            batch,
            torch.ones_like(batch),
            tiny_config,
            5000,
            plan=plan.slice(start, 4),
            generator=g2,
            accumulation_steps=2,
        )
        (loss / 2).backward()
    for name, parameter in full.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(micro.named_parameters())[name].grad,
            rtol=3e-5,
            atol=3e-6,
            msg=name,
        )


@pytest.mark.parametrize("sampler", ["ddpm", "ddim", "heun"])
@pytest.mark.parametrize("sc", [False, True])
def test_sampling_and_validation_modes(tiny_config, sampler, sc):
    config = replace(tiny_config.model, self_condition_probability=0.25 if sc else 0.0)
    model = RePlaid(config).eval()
    observed = []
    hook = model.register_forward_pre_hook(lambda module, args: observed.append(args[2].clone()))
    samples = model.sample(
        2, 8, steps=2, sample_seeds=(11, 12), sampler=sampler, return_logits=True
    )
    hook.remove()
    assert samples.token_ids.shape == (2, 8) and samples.logits.isfinite().all()
    assert torch.count_nonzero(observed[0]) == 0
    assert (observed[1].abs().sum() > 0) if sc else (torch.count_nonzero(observed[1]) == 0)
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(1))
    terms = validation_terms(
        model, torch.randint(18, (2, 8)), generator=torch.Generator().manual_seed(13)
    )
    handle.remove()
    assert len(calls) == (4 if sc else 2)
    torch.testing.assert_close(
        terms["nelbo"], terms["reconstruction"] + terms["diffusion"] + terms["prior"]
    )
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)


def test_optimizer_groups_and_constant_after_warmup(tiny_config):
    model = RePlaid(tiny_config.model)
    tiny_config.train.warmup_steps = 2
    optimizer = build_optimizer(model, tiny_config)
    schedule = build_scheduler(optimizer, tiny_config)
    for _ in range(4):
        optimizer.step()
        schedule.step()
    assert [g["lr"] for g in optimizer.param_groups] == [3e-4, 1e-2, 1e-2, 1e-2]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0, 0, 0, 0.1]
    assert optimizer.defaults["betas"] == (0.9, 0.999)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_precision_boundaries_and_backward(tiny_config):
    model = RePlaid(tiny_config.model).cuda().train()
    generator = torch.Generator(device="cuda").manual_seed(20)
    plan = make_batch_plan(model, 8, generator=generator)
    rows = torch.randint(18, (8, 8), device="cuda")
    dtypes = {}
    handles = [
        model.input_projection.register_forward_hook(
            lambda module, args, output: dtypes.update(input_projection=output.dtype)
        ),
        model.blocks[0].register_forward_pre_hook(
            lambda module, args: dtypes.update(residual=args[0].dtype)
        ),
        model.blocks[0].qkv.register_forward_hook(
            lambda module, args, output: dtypes.update(attention=output.dtype)
        ),
        model.output.linear.register_forward_hook(
            lambda module, args, output: dtypes.update(logit_head=output.dtype)
        ),
    ]
    loss, _ = nelbo_objective(
        model, rows, torch.ones_like(rows), tiny_config, 5000, plan=plan, generator=generator
    )
    loss.backward()
    for handle in handles:
        handle.remove()
    assert loss.dtype == torch.float64 and loss.isfinite()
    assert dtypes == dict(
        input_projection=torch.float32,
        residual=torch.float32,
        attention=torch.bfloat16,
        logit_head=torch.bfloat16,
    )
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
