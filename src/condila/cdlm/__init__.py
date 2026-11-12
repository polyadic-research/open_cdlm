"""Continuous diffusion language models with lazy native-model exports."""
from importlib import import_module

from .registry import (
    ModelSpec,
    create_model,
    from_pretrained,
    generate,
    get_model_spec,
    list_models,
    register_model,
)

_EXPORTS = {
    "LangFlow": (".langflow", "LangFlow"),
    "LangFlowConfig": (".langflow", "LangFlowConfig"),
    "RePlaid": (".replaid", "RePlaid"),
    "RePlaidConfig": (".replaid", "RePlaidConfig"),
    "RePlaidExperimentConfig": (".replaid", "ExperimentConfig"),
}
__all__ = [
    "LangFlow",
    "LangFlowConfig",
    "ModelSpec",
    "RePlaid",
    "RePlaidConfig",
    "RePlaidExperimentConfig",
    "create_model",
    "from_pretrained",
    "generate",
    "get_model_spec",
    "langflow",
    "list_models",
    "register_model",
    "replaid",
]


def __getattr__(name):
    if name in {"langflow", "replaid"}:
        value = import_module(f"{__name__}.{name}")
    elif name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module, __name__), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

