"""torchmetrics-style metric base class and a metric collection.

Every metric follows the same life cycle::

    metric = SomeMetric(**hyperparameters)
    for batch in batches:
        metric.update(batch)                 # two-sample metrics: metric.update(batch, real=False / True)
    result = metric.compute()                # a dict of floats
    metric.reset()

Unlike torchmetrics there is no batch-level ``forward()``: distribution-level quantities (KTD, MTD, pooled entropy, MAUVE,
generative perplexity) are not meaningful on a single batch. States are lists that ``compute()`` concatenates; under
``torch.distributed`` they are all-gathered first (``sync_on_compute=True``), so every rank returns the global value.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Mapping, Optional, Union

from .inputs import Samples, TokenizerLike


class Metric:
    #: True for metrics that compare generated samples with real ones (``update(x, real=True/False)``)
    is_two_sample: bool = False
    #: direction of the headline value, for reports
    higher_is_better: Optional[bool] = None

    def __init__(self, tokenizer: TokenizerLike = None, sync_on_compute: bool = True):
        self.tokenizer = tokenizer
        self.sync_on_compute = sync_on_compute
        self._defaults: Dict[str, Any] = {}
        self._keep_on_reset: set = set()

    # ------------------------------------------------------------------ states
    def add_state(self, name: str, default: Any = None, keep_on_reset: bool = False) -> None:
        """Register a list state (concatenated across updates and ranks)."""
        default = [] if default is None else default
        if not isinstance(default, list):
            raise TypeError("states are lists; aggregate them in compute()")
        self._defaults[name] = copy.deepcopy(default)
        if keep_on_reset:
            self._keep_on_reset.add(name)
        setattr(self, name, copy.deepcopy(default))

    def reset(self) -> None:
        for name, default in self._defaults.items():
            if name not in self._keep_on_reset:
                setattr(self, name, copy.deepcopy(default))

    def state_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self._defaults}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for name in self._defaults:
            if name in state:
                setattr(self, name, list(state[name]))

    def merge_state(self, other: "Metric") -> None:
        """Append another instance's states (e.g. from a different process) to this one."""
        for name in self._defaults:
            getattr(self, name).extend(getattr(other, name))

    @contextmanager
    def _synced(self):
        """Temporarily replace local states with the states gathered from all ranks."""
        import torch.distributed as dist

        if not (self.sync_on_compute and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            yield
            return
        local = {name: getattr(self, name) for name in self._defaults}
        world = dist.get_world_size()
        for name, value in local.items():
            gathered = [None] * world
            dist.all_gather_object(gathered, value)
            setattr(self, name, [x for part in gathered for x in part])
        try:
            yield
        finally:
            for name, value in local.items():
                setattr(self, name, value)

    # ------------------------------------------------------------------ interface
    def _wrap(self, x: Any) -> Samples:
        return Samples.wrap(x, self.tokenizer)

    def update(self, *args, **kwargs) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _compute(self) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def compute(self) -> Dict[str, Any]:
        with self._synced():
            return self._compute()

    def __repr__(self) -> str:
        hp = getattr(self, "_hparams", {})
        return f"{type(self).__name__}(" + ", ".join(f"{k}={v!r}" for k, v in hp.items()) + ")"


class MetricCollection:
    """Run several metrics on the same batches. Two-sample metrics receive real batches too; single-sample metrics
    only see generated batches. Metrics that share a :class:`~open_cdlm.dlm_metrics.features.TextFeatureExtractor` reuse its
    features, so each batch is encoded once.

    >>> coll = MetricCollection({"ent": EntropyDecomposition(), "mtd": MTD(extractor=fx)}, tokenizer="gpt2")
    >>> coll.update(reference_ids, real=True); coll.update(sample_ids)
    >>> coll.compute()          # {"ent": {...}, "mtd": {...}}
    """

    def __init__(self, metrics: Union[Mapping[str, Metric], Iterable[Metric]], tokenizer: TokenizerLike = None):
        if isinstance(metrics, Mapping):
            self.metrics = dict(metrics)
        else:
            self.metrics = {type(m).__name__: m for m in metrics}
        self.tokenizer = tokenizer

    def update(self, samples: Any, real: bool = False) -> None:
        s = Samples.wrap(samples, self.tokenizer)
        for m in self.metrics.values():
            if m.is_two_sample:
                m.update(s, real=real)
            elif not real:
                m.update(s)

    def compute(self, flat: bool = False) -> Dict[str, Any]:
        out = {name: m.compute() for name, m in self.metrics.items()}
        if not flat:
            return out
        return {f"{name}/{k}": v for name, res in out.items() for k, v in res.items()}

    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()

    def __getitem__(self, name: str) -> Metric:
        return self.metrics[name]

    def __repr__(self) -> str:
        return "MetricCollection(\n" + "".join(f"  {k}: {v!r}\n" for k, v in self.metrics.items()) + ")"
