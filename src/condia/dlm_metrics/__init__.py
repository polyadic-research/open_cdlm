"""Alias the complete metrics namespace without duplicating classes or state."""
import pkgutil
import sys
from importlib import import_module

_canonical = import_module("condila.dlm_metrics")
for _info in pkgutil.walk_packages(_canonical.__path__, prefix=_canonical.__name__ + "."):
    _module = import_module(_info.name)
    sys.modules[__name__ + _info.name[len(_canonical.__name__):]] = _module
sys.modules[__name__] = _canonical
