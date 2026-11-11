"""Model discovery and common creation, loading and token-generation APIs.

Native model APIs remain available; sampling delegates to their original algorithms.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_class: type | str
    config_class: type | str
    loader: Callable | str | None = None
    sampler: Callable | str | None = None
    samplers: tuple[str, ...] = ()


def _resolve(value):
    if isinstance(value, str):
        module, attribute = value.split(":", 1)
        return getattr(import_module(module), attribute)
    return value


_MODELS: dict[str, ModelSpec] = {}


def register_model(
    name: str,
    model_class: type | str,
    config_class: type | str,
    *,
    loader: Callable | str | None = None,
    sampler: Callable | str | None = None,
    samplers=(),
    overwrite: bool = False,
) -> ModelSpec:
    """Register a model; strings use ``'module:attribute'`` for lazy imports.

    A loader receives ``(checkpoint, device=..., **kwargs)`` and returns a model.
    A sampler receives the keyword arguments of :func:`generate` and returns token IDs.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model name must be a nonempty string")
    key = name.strip().lower()
    if key in _MODELS and not overwrite:
        raise ValueError(f"model {key!r} is already registered")
    spec = ModelSpec(key, model_class, config_class, loader, sampler, tuple(samplers))
    _MODELS[key] = spec
    return spec


def list_models() -> tuple[str, ...]:
    """Return registered model names without importing torch or model implementations."""
    return tuple(sorted(_MODELS))


def get_model_spec(name: str) -> ModelSpec:
    if not isinstance(name, str):
        raise TypeError("model name must be a string")
    try:
        return _MODELS[name.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(list_models())}") from None


def create_model(name: str, config: Any = None, **config_kwargs):
    """Construct native weights from a native config, config dict, or config keywords.

    This creates randomly initialized weights. Use :func:`from_pretrained` to load weights.
    """
    spec = get_model_spec(name)
    config_class = _resolve(spec.config_class)
    if config is None:
        config = config_class(**config_kwargs)
    elif isinstance(config, Mapping):
        config = config_class(**{**config, **config_kwargs})
    elif not isinstance(config, config_class):
        raise TypeError(f"{spec.name} requires {config_class.__name__} or a config mapping")
    elif config_kwargs:
        raise TypeError("pass overrides in a config mapping or construct the config explicitly")
    if hasattr(config, "validate"):
        config.validate()
    return _resolve(spec.model_class)(config)


def from_pretrained(name: str, checkpoint, *, device="cpu", **kwargs):
    """Load native pretrained weights; tokenizer loading remains explicit.

    LangFlow accepts HF directories, HF model IDs, safetensors files and Lightning checkpoints.
    RePlaid accepts a local reproduction checkpoint directory (EMA by default).
    """
    spec = get_model_spec(name)
    if spec.loader is None:
        raise NotImplementedError(f"no checkpoint loader registered for {spec.name}")
    return _resolve(spec.loader)(checkpoint, device=device, **kwargs)


def generate(
    model,
    *,
    num_samples: int = 1,
    length: int | None = None,
    steps: int = 128,
    seed: int | None = 0,
    sampler: str | None = None,
    temperature: float = 1.0,
    **kwargs,
):
    """Generate an integer ``[num_samples, length]`` tensor with a native sampler.

    ``steps`` counts solver intervals: LangFlow calls its native API with
    ``num_steps=steps+1`` (grid points), while RePlaid uses ``steps`` directly.
    Model training/evaluation flags are restored even if generation fails.
    """
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if length is not None and (isinstance(length, bool) or not isinstance(length, int) or length < 1):
        raise ValueError("length must be a positive integer")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    # Prefer the most specific registered class, including third-party subclasses.
    spec = None
    for cls in type(model).__mro__:
        reference = f"{cls.__module__}:{cls.__qualname__}"
        spec = next((entry for entry in _MODELS.values()
                     if entry.model_class is cls or entry.model_class == reference), None)
        if spec is not None:
            break
    else:
        # A string registration can refer to an alias of a native class.
        spec = next((entry for entry in _MODELS.values()
                     if isinstance(model, _resolve(entry.model_class))), None)
    if spec is None:
        raise TypeError("model is not an instance of a registered model class")
    if spec.sampler is None:
        raise NotImplementedError(f"no common sampler registered for {spec.name}")
    chosen = sampler or (spec.samplers[0] if spec.samplers else None)
    if spec.samplers and chosen not in spec.samplers:
        raise ValueError(f"{spec.name} sampler must be one of {spec.samplers}")
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        return _resolve(spec.sampler)(
            model, num_samples=num_samples, length=length, steps=steps,
            seed=seed, sampler=chosen, temperature=temperature, **kwargs,
        )
    finally:
        for module, training in modes:
            module.training = training


register_model(
    "langflow", "condila.cdlm.langflow.model:LangFlow", "condila.cdlm.langflow.config:LangFlowConfig",
    loader="condila.cdlm.langflow.loading:from_pretrained",
    sampler="condila.cdlm.langflow.loading:sample_tokens", samplers=("euler_edm",),
)
register_model(
    "replaid", "condila.cdlm.replaid.model:RePlaid", "condila.cdlm.replaid.config:ModelConfig",
    loader="condila.cdlm.replaid.api:from_pretrained",
    sampler="condila.cdlm.replaid.api:sample_tokens", samplers=("ddpm", "ddim", "heun"),
)

