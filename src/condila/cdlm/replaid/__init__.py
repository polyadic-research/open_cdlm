"""RePlaid reproduction: model, native NELBO objective, training and sampling."""
from importlib import import_module

_EXPORTS = {
    "RePlaid": (".model", "RePlaid"),
    "RePlaidConfig": (".config", "ModelConfig"),
    "ModelConfig": (".config", "ModelConfig"),
    "ExperimentConfig": (".config", "ExperimentConfig"),
    "DataConfig": (".config", "DataConfig"),
    "TrainConfig": (".config", "TrainConfig"),
    "load_config": (".config", "load_config"),
    "from_pretrained": (".api", "from_pretrained"),
    "load_checkpoint": (".checkpoint", "load_model"),
    "validation_terms": (".objective", "validation_terms"),
    "nelbo_objective": (".objective", "nelbo_objective"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

