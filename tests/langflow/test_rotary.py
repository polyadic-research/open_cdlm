"""RoPE regression tests; no downloaded model or dataset is needed."""

import pytest
import torch

from open_cdlm.cdlm.langflow.model import Rotary, split_and_apply_rotary_pos_emb


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def reference_tables(rotary, length, device):
    with torch.autocast(device_type=device.type, enabled=False):
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, rotary.inv_freq.float())
        angles = torch.cat((angles, angles), dim=-1)
        cos = angles.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        sin = angles.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        cos[:, :, 2].fill_(1)
        sin[:, :, 2].zero_()
    return cos, sin


def assert_tables(actual, expected):
    for got, want in zip(actual, expected):
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("first_amp", [False, True])
def test_autocast_order_does_not_change_positions(device, dtype, first_amp):
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("CPU float16 autocast is unsupported by older PyTorch")
    rotary = Rotary(64).to(device)
    x = torch.zeros(1, 1024, 64, device=device)
    expected = reference_tables(rotary, 1024, x.device)
    for amp in (first_amp, not first_amp, first_amp):
        with torch.autocast(device, dtype=dtype, enabled=amp):
            actual = rotary(x)
        assert_tables(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_low_precision_input_and_sequence_length_changes(device):
    rotary = Rotary(64).to(device)
    for length in (128, 1024, 128):
        x = torch.zeros(1, length, 64, device=device, dtype=torch.bfloat16)
        with torch.autocast(device, dtype=torch.bfloat16):
            actual = rotary(x)
        assert_tables(actual, reference_tables(rotary, length, x.device))
        reused = rotary(x)
        assert reused[0] is actual[0]
        assert reused[1] is actual[1]


@pytest.mark.parametrize("cache_name", ["cos_cached", "sin_cached"])
@pytest.mark.parametrize("damage", ["missing", "low_precision"])
def test_invalid_cache_is_rebuilt(cache_name, damage):
    rotary = Rotary(64)
    x = torch.zeros(1, 128, 64)
    rotary(x)
    old = getattr(rotary, cache_name)
    setattr(rotary, cache_name, None if damage == "missing" else old.bfloat16())
    assert_tables(rotary(x), reference_tables(rotary, 128, x.device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_device_move_with_same_sequence_length():
    rotary = Rotary(64)
    rotary(torch.zeros(1, 128, 64))
    rotary.cuda()
    x = torch.zeros(1, 128, 64, device="cuda")
    assert_tables(rotary(x), reference_tables(rotary, 128, x.device))


def test_rotation_matches_complex_multiplication_and_preserves_values():
    torch.manual_seed(0)
    qkv = torch.randn(2, 16, 3, 2, 8, requires_grad=True)
    rotary = Rotary(8)
    q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary(qkv))
    angles = torch.outer(torch.arange(16).float(), rotary.inv_freq)
    phase = torch.polar(torch.ones_like(angles), angles)[None, :, None, :]
    for result, original in ((q, qkv[:, :, 0]), (k, qkv[:, :, 1])):
        rotated = torch.complex(original[..., :4], original[..., 4:]) * phase
        expected = torch.cat((rotated.real, rotated.imag), dim=-1)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result.norm(dim=-1), original.norm(dim=-1))
    torch.testing.assert_close(v, qkv[:, :, 2], rtol=0, atol=0)
    (q.sum() + k.sum() + v.sum()).backward()
    assert qkv.grad is not None and torch.isfinite(qkv.grad).all()


def test_checkpoint_keys_stay_compatible():
    rotary = Rotary(64)
    x = torch.zeros(1, 128, 64)
    expected = rotary(x)
    assert set(rotary.state_dict()) == {"inv_freq"}
    restored = Rotary(64)
    restored.load_state_dict(rotary.state_dict(), strict=True)
    assert_tables(restored(x), expected)
