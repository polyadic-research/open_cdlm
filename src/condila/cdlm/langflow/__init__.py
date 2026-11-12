"""LangFlow models, checkpoint loading, and optional Lightning training."""
from importlib import import_module

_EXPORTS = {
    "LangFlow": (".model", "LangFlow"),
    "LangFlowConfig": (".config", "LangFlowConfig"),
    "from_pretrained": (".loading", "from_pretrained"),
    "LangFlowLightningModule": (".train", "LangFlowLightningModule"),
    "LangFlowDataModule": (".data", "LangFlowDataModule"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    try:
        value = getattr(import_module(module, __name__), attribute)
    except ModuleNotFoundError as exc:
        if exc.name in {"lightning", "datasets", "hydra", "omegaconf"}:
            raise ImportError("LangFlow training requires: pip install 'condila[train]'") from exc
        raise
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

