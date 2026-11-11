"""``NLLBound``: a stateful wrapper of the per-row bound estimators."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..base import Metric
from .continuous import ode_nll_bound, vdm_nelbo


class NLLBound(Metric):
    """Aggregate a likelihood bound of a continuous DLM over real rows: ``nll_per_token`` (sum of row bounds / sum of
    tokens), ``ppl = exp(nll_per_token)``, and ``se`` (SE of the mean per-row nats/token, which contains the Monte Carlo
    noise of the estimator as well as the spread between rows).

    Args:
        model: a :class:`~condila.dlm_metrics.likelihood.continuous.ContinuousDLM` adapter.
        estimator: ``"vdm"`` (default; cheap, no probes) or ``"ode"`` (probability-flow ODE).
        strata: VDM strata per row.
        n_steps, probe: ODE options.
        seed: seed of a per-metric generator (``None``: global RNG).

    The value is an UPPER bound on the NLL of the model's SDE (``vdm``) or ODE (``ode``) generative model; it is
    comparable with an exact autoregressive NLL in one direction only.
    """
    higher_is_better = False

    def __init__(self, model: Any, estimator: str = "vdm", strata: int = 32, n_steps: int = 128,
                 probe: str = "gaussian", seed: Optional[int] = 0, device: Optional[str] = None, **kw):
        super().__init__(**kw)
        if estimator not in ("vdm", "ode"):
            raise ValueError("estimator must be 'vdm' or 'ode'")
        self.model, self.estimator, self.strata, self.n_steps, self.probe = model, estimator, strata, n_steps, probe
        self.seed, self.device = seed, device
        self._gen = None
        self._hparams = dict(estimator=estimator, strata=strata, n_steps=n_steps)
        self.add_state("row_nll")
        self.add_state("row_tokens")
        self.add_state("row_nll_expectation_form")

    def _generator(self, device):
        if self.seed is None:
            return None
        if self._gen is None:
            self._gen = torch.Generator(device=device).manual_seed(self.seed)
        return self._gen

    def update(self, ids: Any) -> None:
        ids = torch.as_tensor(np.asarray(ids) if not torch.is_tensor(ids) else ids).long()
        dev = self.device or (ids.device if ids.is_cuda else ("cuda" if torch.cuda.is_available() else "cpu"))
        ids = ids.to(dev)
        g = self._generator(ids.device)
        if self.estimator == "vdm":
            out = vdm_nelbo(self.model, ids, self.strata, g)
        else:
            out = ode_nll_bound(self.model, ids, self.n_steps, generator=g, probe=self.probe, probe_generator=g)
            self.row_nll_expectation_form.extend(out["nll_expectation_form"].double().cpu().tolist())
        self.row_nll.extend(out["nll"].double().cpu().tolist())
        self.row_tokens.extend(out["tokens"].double().cpu().tolist())

    def reset(self) -> None:
        super().reset()
        self._gen = None

    def _compute(self) -> Dict[str, Any]:
        nll, tok = np.asarray(self.row_nll), np.asarray(self.row_tokens)
        per = nll / tok
        val = float(nll.sum() / tok.sum())
        out = dict(nll_per_token=val, ppl=math.exp(val), se=float(per.std(ddof=1) / math.sqrt(len(per))) if len(per) > 1
                   else float("nan"), n_rows=len(per), estimator=self.estimator)
        if self.row_nll_expectation_form:
            e = np.asarray(self.row_nll_expectation_form)
            out["nll_per_token_expectation_form"] = float(e.sum() / tok.sum())
        return out
