"""Load LangFlow weights using the bundled implementation and native state keys."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .config import LangFlowConfig
from .model import LangFlow


def _weights(path):
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _directory_state(directory):
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = directory / name
        if path.is_file():
            return _weights(path)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / name
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            state = {}
            for shard in sorted(set(mapping.values())):
                values = _weights(directory / shard)
                overlap = state.keys() & values.keys()
                if overlap:
                    raise ValueError(f"duplicate checkpoint tensors across shards: {sorted(overlap)}")
                state.update(values)
            if state.keys() != mapping.keys():
                raise ValueError("checkpoint shards do not match the index weight map")
            return state
    raise FileNotFoundError(f"no model weights found in {directory}")


def from_pretrained(
    checkpoint,
    *,
    config=None,
    device="cpu",
    dtype=None,
    weights="ema",
    revision=None,
    cache_dir=None,
    local_files_only=False,
    token=None,
):
    """Load an HF directory/model ID, safetensors file, or Lightning ``.ckpt``.

    ``config`` can be a LangFlowConfig, dict, or config directory. File checkpoints
    need a sibling config.json or an explicit config. Condila Lightning checkpoints
    also carry their architecture. Legacy ``proposal.scale`` keys are handled by
    the native model. EMA weights are preferred when a Lightning checkpoint has them.
    """
    if weights not in {"raw", "ema"}:
        raise ValueError("weights must be raw or ema")
    source = Path(checkpoint).expanduser()
    if not source.exists():
        name = str(checkpoint)
        if source.is_absolute() or name.startswith(("./", "../")) or source.suffix in {
            ".safetensors", ".bin", ".ckpt", ".pt",
        }:
            raise FileNotFoundError(source)
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(
            repo_id=name, revision=revision, cache_dir=cache_dir,
            local_files_only=local_files_only, token=token,
            allow_patterns=["config.json", "*.safetensors", "*.bin", "*.index.json"],
        ))
    payload = None
    if source.is_dir():
        state = _directory_state(source)
        config_dir = source
    elif source.suffix in {".ckpt", ".pt"}:
        payload = torch.load(source, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        ema = payload.get("ema_state_dict")
        if weights == "ema" and ema is not None:
            state = ema
        elif weights == "ema" and any(key.startswith("ema_model.") for key in state):
            state = {key[len("ema_model."):]: value for key, value in state.items()
                     if key.startswith("ema_model.")}
        elif any(key.startswith("model.") for key in state):
            state = {key[len("model."):]: value for key, value in state.items()
                     if key.startswith("model.")}
        config_dir = source.parent
    else:
        state = _weights(source)
        config_dir = source.parent
    if config is None:
        if (config_dir / "config.json").is_file():
            config = LangFlowConfig.from_pretrained(config_dir, local_files_only=True)
        elif payload is not None and "condila_model_config" in payload:
            config = LangFlowConfig(**payload["condila_model_config"])
        else:
            raise ValueError("checkpoint architecture is missing; supply config=LangFlowConfig(...)")
    elif isinstance(config, dict):
        config = LangFlowConfig(**config)
    elif not isinstance(config, LangFlowConfig):
        config = LangFlowConfig.from_pretrained(config)
    config.validate()
    model = LangFlow(config)
    model.load_state_dict(state, strict=True)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device).eval()


@torch.no_grad()
def sample_tokens(
    model, *, num_samples, length, steps, seed, sampler, temperature, **kwargs,
):
    if temperature != 1.0:
        raise ValueError("the native LangFlow Euler-EDM sampler supports temperature=1 only")
    if sampler != "euler_edm":
        raise ValueError("LangFlow supports sampler='euler_edm'")
    if set(kwargs) - {"generator", "initial_noise"}:
        raise TypeError(f"unsupported LangFlow sampling options: {', '.join(sorted(kwargs))}")
    length = model.config.model_length if length is None else length
    if length > model.config.model_length:
        raise ValueError("length exceeds model_length")
    device = next(model.parameters()).device
    if seed is not None:
        if kwargs:
            raise ValueError("set seed=None when passing native generator or initial_noise")
        kwargs["initial_noise"] = torch.cat([
            torch.randn(1, length, model.config.hidden_size, device=device,
                        generator=torch.Generator(device=device).manual_seed(seed + index))
            for index in range(num_samples)
        ])
    return model.generate_samples(
        num_samples=num_samples, seq_length=length, num_steps=steps + 1, device=device, **kwargs,
    )

