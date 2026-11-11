"""Condila: continuous diffusion language models and evaluation.

Model and metric dependencies are imported only when their namespace is used.
"""
from importlib import import_module

__version__ = "0.1.0"
__all__ = ["__version__", "cdlm", "dlm_metrics"]


def __getattr__(name):
    if name in {"cdlm", "dlm_metrics"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

