"""Compatibility spelling for the requested ``condia.dlm_metrics`` namespace.

The distribution and primary package are named ``condila``.
"""
from importlib import import_module

__all__ = ["dlm_metrics"]


def __getattr__(name):
    if name == "dlm_metrics":
        value = import_module(f"{__name__}.dlm_metrics")
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

