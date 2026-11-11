from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None]) + shift[:, None]


class FP32LayerNorm(nn.Module):
    """Bias-free LayerNorm that preserves the source model's FP32 boundary."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = (
            torch.autocast(device_type="cuda", enabled=False)
            if x.device.type == "cuda"
            else nullcontext()
        )
        with context:
            return F.layer_norm(x.float(), (self.dim,)) * self.weight.float()


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=gamma.device, dtype=torch.float32)
            / half
        )
        angles = gamma.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if self.frequency_dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return self.mlp(embedding)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_length: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if base <= 0.0:
            raise ValueError("RoPE base must be positive")
        self.base = float(base)
        inverse = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        angles = torch.outer(torch.arange(max_length, dtype=torch.float32), inverse)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def apply(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        length = x.shape[1]
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        cosine_slice = self.cos[offset : offset + length]
        sine_slice = self.sin[offset : offset + length]
        cosine = torch.cat((cosine_slice, cosine_slice), dim=-1)[None, :, None]
        sine = torch.cat((sine_slice, sine_slice), dim=-1)[None, :, None]
        return x * cosine.to(x) + rotated * sine.to(x)


class GQADDiTBlock(nn.Module):
    """Non-causal adaLN-Zero DDiT block with grouped-query attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.query_dim = self.num_heads * self.head_dim
        self.key_value_dim = self.num_key_value_heads * self.head_dim
        dim = config.hidden_size

        self.norm1 = FP32LayerNorm(dim)
        self.qkv = nn.Linear(dim, self.query_dim + 2 * self.key_value_dim, bias=False)
        # Keep names aligned with the extracted experiment so its safetensors
        # weights can be compared or converted without a key-remapping table.
        self.attn_out = nn.Linear(self.query_dim, dim, bias=False)
        self.norm2 = FP32LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, config.intermediate_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.intermediate_size, dim, bias=True),
        )
        self.adaln = nn.Linear(config.cond_dim, 6 * dim, bias=True)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)
        self.dropout = config.dropout

    def _project_qkv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.split(
            self.qkv(hidden),
            (self.query_dim, self.key_value_dim, self.key_value_dim),
            dim=-1,
        )

    def _normalize_query_key(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return query, key

    def _condition_query_key(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional conditioning after QK normalization and before position encoding."""
        return query, key

    def _apply_position_encoding(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rope: RotaryEmbedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the experiment-specific Q/K positional encoding."""
        return rope.apply(query), rope.apply(key)

    def _gate_attention_output(
        self,
        attention_output: torch.Tensor,
        attention_input: torch.Tensor,
    ) -> torch.Tensor:
        """Optionally gate per-head SDPA outputs before concatenation."""
        del attention_input
        return attention_output

    def _condition_attention_output(
        self,
        attention_output: torch.Tensor,
        attention_input: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Optionally condition per-head SDPA outputs before concatenation."""
        del condition, attention_mask
        return self._gate_attention_output(attention_output, attention_input)

    def _normalize_attention_branch(self, branch: torch.Tensor) -> torch.Tensor:
        """Optionally normalize the projected attention branch before its residual."""
        return branch

    def _normalize_mlp_branch(self, branch: torch.Tensor) -> torch.Tensor:
        """Optionally normalize the MLP branch before its residual."""
        return branch

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        rope: RotaryEmbedding,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaln(
            condition
        ).chunk(6, dim=-1)
        attention_input = modulate(self.norm1(x), shift_attn, scale_attn)
        batch_size, length, _ = attention_input.shape
        query, key, value = self._project_qkv(attention_input)
        query = query.view(batch_size, length, self.num_heads, self.head_dim)
        key = key.view(batch_size, length, self.num_key_value_heads, self.head_dim)
        value = value.view(batch_size, length, self.num_key_value_heads, self.head_dim)
        query, key = self._normalize_query_key(query, key)
        query, key = self._condition_query_key(query, key, condition)
        query, key = self._apply_position_encoding(query, key, rope)

        sdpa_mask = None
        if attention_mask is not None:
            sdpa_mask = attention_mask[:, None, None, :].bool()
        hidden = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden = hidden.transpose(1, 2)
        hidden = self._condition_attention_output(
            hidden, attention_input, condition, attention_mask
        )
        hidden = hidden.reshape(batch_size, length, self.query_dim)
        attention_branch = self._normalize_attention_branch(self.attn_out(hidden))
        x = x + gate_attn[:, None] * F.dropout(
            attention_branch,
            self.dropout,
            self.training,
        )
        hidden = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        mlp_branch = self._normalize_mlp_branch(hidden)
        return x + gate_mlp[:, None] * F.dropout(
            mlp_branch,
            self.dropout,
            self.training,
        )


class FinalLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = FP32LayerNorm(config.hidden_size)
        self.adaln = nn.Linear(config.cond_dim, 2 * config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, int(config.vocab_size))
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


@dataclass
class SampleOutput:
    token_ids: torch.Tensor
    sampled_token_ids: torch.Tensor
    token_entropy: torch.Tensor
    logits: torch.Tensor | None = None
