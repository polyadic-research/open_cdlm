"""One-call evaluation with the recommended protocol.

MTD uses mean-pooled features and reports its permutation null SD. KTD uses projected token statistics and a
separate real calibration corpus. The suite also reports quarter-bandwidth MTD, neighborhood coverage, duplicate
rates, repetition, attributes, Gen PPL, entropy decomposition, and MAUVE.

Every metric is also computed on a second random sample of real rows of the same size: the floor a perfect generator
of that size reaches. A difference is readable only when it exceeds the metric's noise at that n.
"""
from __future__ import annotations

import json
import math
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .base import Metric
from .features import TextFeatureExtractor
from .inputs import PRESETS, Samples, load_samples
from .metrics import (KTD, MAUVE, MMD, MTD, PRDC, AttributePanel, CompressionRate, DistinctN,
                      EntropyDecomposition, ExactDuplicateRate, FeatureVendi, FrechetDistance,
                      GenerativePerplexity, MultiScaleMTD, NGramDiversity, RepetitionRate, ReverseNGramCE,
                      TopicOccupancy)
from .models import get_judge
from .stats import real_floor

BASIC = ("gen_ppl", "entropy")
RECOMMENDED = ("gen_ppl", "entropy", "mtd", "mtd_quarter", "prdc", "dup", "rep", "mauve", "attributes")
ALL = RECOMMENDED + ("multiscale", "fd", "vendi", "ngram", "distinct", "compression", "topics", "reverse_ce")
GROUPS = {"basic": BASIC, "recommended": RECOMMENDED, "all": ALL}

#: guard thresholds relative to the real floor (starting points calibrated on constructed OWT failure sets)
DEFAULT_GUARDS = dict(mtd_quarter_null_sd=3.0, jsd_sd=3.0, recall_drop=0.05, coverage_drop=0.05, exact_dup_rate=0.01,
                      rep_rise=0.05, attribute_retention_low=0.5, attribute_retention_high=2.0)


def _mauve_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("mauve") is not None


def build_metrics(names: Sequence[str], tokenizer, judge, fx_mean: TextFeatureExtractor,
                  fx_last: Optional[TextFeatureExtractor], protocol: str = "langflow", split_half: int = 8,
                  mtd_null_repeats: int = 12, gen_ppl_batch_size: Optional[int] = None, *,
                  fx_ktd: Optional[TextFeatureExtractor] = None) -> "OrderedDict[str, Metric]":
    """Fresh metric objects for a suite (real features kept across ``reset()``)."""
    kw = dict(tokenizer=tokenizer)
    fkw = dict(extractor=fx_mean, reset_real_features=False, tokenizer=tokenizer)
    make = {
        "gen_ppl": lambda: GenerativePerplexity(judge=judge, protocol=protocol, split_half=split_half,
                                                batch_size=gen_ppl_batch_size, **kw),
        "entropy": lambda: EntropyDecomposition(split_half=split_half, **kw),
        "mtd": lambda: MTD(null_repeats=mtd_null_repeats, **fkw),
        "mtd_quarter": lambda: MTD(bandwidth_multiplier=0.25, null_repeats=mtd_null_repeats, **fkw),
        "ktd": lambda: KTD(extractor=fx_ktd, reset_real_features=False, **kw),
        "mmd": lambda: MMD(null_repeats=mtd_null_repeats, **fkw),
        "mmd_quarter": lambda: MMD(bandwidth_multiplier=0.25, null_repeats=mtd_null_repeats, **fkw),
        "prdc": lambda: PRDC(**fkw),
        "dup": lambda: ExactDuplicateRate(**kw),
        "rep": lambda: RepetitionRate(n=4, **kw),
        "mauve": lambda: MAUVE(extractor=fx_last, reset_real_features=False, tokenizer=tokenizer),
        "attributes": lambda: AttributePanel(reset_real_features=False, **kw),
        "multiscale": lambda: MultiScaleMTD(**fkw),
        "fd": lambda: FrechetDistance(**fkw),
        "vendi": lambda: FeatureVendi(**fkw),
        "ngram": lambda: NGramDiversity(n=4, **kw),
        "distinct": lambda: DistinctN(n=4, **kw),
        "compression": lambda: CompressionRate(**kw),
        "topics": lambda: TopicOccupancy(**fkw),
        "reverse_ce": lambda: ReverseNGramCE(reset_real_features=False, **kw),
    }
    out = OrderedDict()
    for name in names:
        if name not in make:
            raise ValueError(f"unknown metric {name!r}; choose from {sorted(make)}")
        out[name] = make[name]()
    return out


def _as_samples(x: Any, tokenizer, preset) -> Samples:
    if isinstance(x, str):
        return load_samples(x, tokenizer, preset)
    s = Samples.wrap(x, tokenizer)
    if s.tokenizer is None:
        s.tokenizer = tokenizer
    return s


def evaluate(samples: Union[Any, Mapping[str, Any]], reference: Any = None, *, calibration: Any = None, tokenizer=None,
             preset: Optional[str] = None, n: Optional[int] = None, metrics: Union[str, Sequence[str]] = "recommended",
             device: Optional[str] = None, judge: Any = "gpt2-large", protocol: str = "langflow", ref_seed: int = 0,
             split_half: int = 8, mtd_null_repeats: Optional[int] = None, feature_batch_size: int = 8,
             gen_ppl_batch_size: Optional[int] = None, cache_dir: Optional[str] = None, verbose: bool = True,
             mmd_null_repeats: Optional[int] = None) -> "Report":
    """Evaluate one or several sample sets against random real rows.

    Args:
        samples: a sample set (ids / texts / file path) or ``{name: set}``.
        reference: real rows of the evaluation split (ids / texts / file path), at least ``2 n`` rows: ``n`` random rows
            are the reference, ``n`` others are scored like a generator (the floor). ``None``: only the single-sample
            metrics (gen ppl, entropies, repetition, duplicates) are computed, without a floor.
        calibration: independent real samples for KTD, in the same formats as reference.
            Supplying calibration enables KTD in the recommended and all metric groups.
        tokenizer / preset: the generator tokenizer (``preset`` in ``owt``, ``owt-sub``, ``lm1b``).
        n: evaluate the first ``n`` rows of each set (default: all rows; sets of different size get separate floors).
        metrics: a group name, a metric name, or a list of metric names. Defaults to ``"recommended"``.
        protocol: generative-perplexity protocol (``"langflow"`` or ``"flm"``).
        split_half: half-splits for the +- of gen ppl and the entropies (0 = off).
        mtd_null_repeats: permutation re-splits for the MTD null SD. Defaults to 12, with 0 turning this off.
        mmd_null_repeats: compatibility spelling for ``mtd_null_repeats``.
        cache_dir: on-disk feature cache (reference features are reused across runs).
    """
    tokenizer = tokenizer or (PRESETS[preset][0] if preset else None)
    sets = OrderedDict()
    if isinstance(samples, Mapping):
        items = list(samples.items())
    else:
        items = [(samples if isinstance(samples, str) else "samples", samples)]
    for name, x in items:
        s = _as_samples(x, tokenizer, preset)
        if n is not None:
            if len(s) < n:
                raise ValueError(f"{name}: {len(s)} rows < n = {n}")
            s = s.subset(range(n))
        sets[str(name)] = s
    tokenizer = tokenizer or next(iter(sets.values())).tokenizer
    names = list(GROUPS.get(metrics, (metrics,))) if isinstance(metrics, str) else list(metrics)
    if isinstance(metrics, str) and metrics in ("recommended", "all") and calibration is not None:
        names.append("ktd")
    if "ktd" in names and (reference is None or calibration is None):
        raise ValueError("KTD requires a scoring reference and a separate real calibration corpus")
    ref_all = _as_samples(reference, tokenizer, preset) if reference is not None else None
    cal_rows = _as_samples(calibration, tokenizer, preset) if "ktd" in names else None
    if cal_rows is not None and cal_rows.digest() == ref_all.digest():
        raise ValueError("KTD calibration samples must be separate from the scoring reference")
    if ref_all is None:
        names = [m for m in names if m in ("gen_ppl", "entropy", "dup", "rep", "ngram", "distinct", "compression",
                                           "vendi")]
    if "mauve" in names and not _mauve_available():
        warnings.warn("mauve-text is not installed: MAUVE skipped (pip install mauve-text faiss-cpu)")
        names = [m for m in names if m != "mauve"]
    J = get_judge(judge, device)
    fx_mean = TextFeatureExtractor(J, pooling="mean", batch_size=feature_batch_size, cache_dir=cache_dir)
    fx_last = TextFeatureExtractor(J, pooling="last", batch_size=feature_batch_size, cache_dir=cache_dir)
    if mtd_null_repeats is None:
        mtd_null_repeats = 12 if mmd_null_repeats is None else mmd_null_repeats
    settings = dict(tokenizer=str(tokenizer), judge=J.name, protocol=protocol, metrics=names,
                    ref_seed=ref_seed, split_half=split_half, mtd_null_repeats=mtd_null_repeats)
    fx_ktd, ktd_state = None, None
    if cal_rows is not None:
        fx_ktd = TextFeatureExtractor(J, pooling="ktd", batch_size=feature_batch_size, cache_dir=cache_dir)
        ktd_state = KTD(extractor=fx_ktd).fit(cal_rows).state_dict()
        settings["ktd_calibration"] = {**ktd_state["calibration"], "sample_digest": cal_rows.digest()}
    results: Dict[str, Dict[str, Any]] = OrderedDict()
    floors: Dict[int, Dict[str, Any]] = {}
    for size in sorted({len(s) for s in sets.values()}):
        ms = build_metrics(names, tokenizer, J, fx_mean, fx_last, protocol, split_half, mtd_null_repeats,
                           gen_ppl_batch_size, fx_ktd=fx_ktd)
        if ktd_state is not None:
            ms["ktd"].load_state_dict(ktd_state)
        group = [k for k, s in sets.items() if len(s) == size]
        todo = []
        if ref_all is not None:
            ref_rows, real_rows = real_floor(ref_all, size, ref_seed)
            ref_rows.tokenizer = real_rows.tokenizer = tokenizer
            for m in ms.values():
                if m.is_two_sample:
                    m.update(ref_rows, real=True)
            todo.append(("__real__", real_rows))
        todo += [(k, sets[k]) for k in group]
        for key, s in todo:
            res = OrderedDict()
            for name, m in ms.items():
                m.reset()
                if m.is_two_sample:
                    m.update(s, real=False)
                else:
                    m.update(s)
                res[name] = m.compute()
            if key == "__real__":
                floors[size] = res
            else:
                results[key] = res
            if verbose:
                print(_one_line("real rows" if key == "__real__" else key, res), flush=True)
    return Report(results, floors, settings)


# ------------------------------------------------------------------------------------------------ report

def _get(res, metric, key, default=float("nan")):
    if metric in ("mtd", "mtd_quarter") and metric not in res:
        metric = metric.replace("mtd", "mmd", 1)
        key = key.replace("mtd", "mmd", 1)
    return res.get(metric, {}).get(key, default)


def _one_line(label, res) -> str:
    parts = [f"[{label}]"]
    for metric, key, fmt in (("gen_ppl", "gen_ppl", "gen_ppl {:.2f}"), ("entropy", "H_row", "H_row {:.3f}"),
                             ("entropy", "JSD_rows", "JSD_rows {:.3f}"), ("mtd", "mtd", "MTD {:.4f}"),
                             ("ktd", "ktd", "KTD {:.6g}"), ("mtd_quarter", "mtd", "MTD_s/4 {:.4f}"),
                             ("mauve", "mauve", "MAUVE {:.3f}")):
        v = _get(res, metric, key, None)
        if v is not None:
            parts.append(fmt.format(v))
    return "  ".join(parts)


def _fmt(v, f="{:.3f}", sd=None):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    s = f.format(v)
    if sd is not None and not (isinstance(sd, float) and math.isnan(sd)):
        s += " ± " + f.format(sd)
    return s


class Report:
    """Results of :func:`evaluate`: ``results[set][metric] -> dict``, ``floors[n][metric] -> dict``."""

    def __init__(self, results, floors, settings):
        self.results, self.floors, self.settings = results, floors, settings

    def floor_for(self, name: str) -> Optional[Dict[str, Any]]:
        r = self.results[name]
        n = (_get(r, "entropy", "n_rows", None) or _get(r, "gen_ppl", "n_rows", None)
             or _get(r, "mtd", "n_fake", None) or _get(r, "ktd", "n_fake", None)
             or _get(r, "mtd_quarter", "n_fake", None))
        return self.floors.get(n) if n is not None else None

    def _rows(self):
        out = []
        for size, f in sorted(self.floors.items()):
            out.append((f"real rows (n = {size})", f))
        out += [(k, v) for k, v in self.results.items()]
        return out

    def table(self) -> str:
        """Markdown: a headline table and a guard table, each with the real-row floor first."""
        rows = self._rows()
        sd = lambda r, m, k: _get(r, m, k + "_sd_half", None)  # noqa: E731
        has_ktd = any("ktd" in result for _, result in rows)
        head = ["set", "MTD"]
        if has_ktd:
            head.append("KTD")
        head += ["gen ppl", "H_row", "H_pool", "JSD_rows", "MAUVE"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for label, r in rows:
            cells = [label, _fmt(_get(r, "mtd", "mtd", None), "{:.4f}", _get(r, "mtd", "null_sd", None))]
            if has_ktd:
                cells.append(_fmt(_get(r, "ktd", "ktd", None), "{:.6g}"))
            cells += [
                _fmt(_get(r, "gen_ppl", "gen_ppl", None), "{:.2f}", sd(r, "gen_ppl", "gen_ppl")),
                _fmt(_get(r, "entropy", "H_row", None), "{:.3f}", sd(r, "entropy", "H_row")),
                _fmt(_get(r, "entropy", "H_pool", None), "{:.3f}", sd(r, "entropy", "H_pool")),
                _fmt(_get(r, "entropy", "JSD_rows", None), "{:.3f}", sd(r, "entropy", "JSD_rows")),
                _fmt(_get(r, "mauve", "mauve", None), "{:.3f}", _get(r, "mauve", "mauve_sd", None))]
            lines.append("| " + " | ".join(cells) + " |")
        ghead = ["set", "MTD σ/4", "recall", "coverage", "exact dup", "feature dup", "rep-4", "min attr. retention"]
        glines = ["| " + " | ".join(ghead) + " |", "|" + "---|" * len(ghead)]
        for label, r in rows:
            att = r.get("attributes", {})
            glines.append("| " + " | ".join([
                label, _fmt(_get(r, "mtd_quarter", "mtd", None), "{:.4f}", _get(r, "mtd_quarter", "null_sd", None)),
                _fmt(_get(r, "prdc", "recall", None)), _fmt(_get(r, "prdc", "coverage", None)),
                _fmt(_get(r, "dup", "exact_dup_rate", None)), _fmt(_get(r, "prdc", "feature_dup_rate", None)),
                _fmt(_get(r, "rep", "rep_4", None)),
                (_fmt(att.get("min_retention")) + (f" ({att.get('min_retention_attribute')})" if att else ""))
                if att else "–"]) + " |")
        notes = (f"\nMTD ± = permutation null SD. Gen PPL and entropy ± = SD over {self.settings['split_half']} random "
                 f"half-splits (n/2 rows). MAUVE ± = SD over k-means seeds. Judge {self.settings['judge']}, Gen PPL "
                 f"protocol {self.settings['protocol']}.\n")
        flags = self.flags()
        ftxt = "".join(f"- {k}: " + "; ".join(v) + "\n" for k, v in flags.items() if v)
        return ("**Headline**\n\n" + "\n".join(lines) + "\n\n**Guards**\n\n" + "\n".join(glines) + "\n" + notes
                + (("\n**Guard flags (deviations from the real floor beyond the thresholds)**\n\n" + ftxt) if ftxt else ""))

    def flags(self, thresholds: Optional[Mapping[str, float]] = None) -> Dict[str, List[str]]:
        """Guards that deviate from the real-rows floor of the same n beyond ``thresholds`` (default
        :data:`DEFAULT_GUARDS`): MTD at a quarter bandwidth above the floor by more than k null SDs, ``JSD_rows`` off the
        floor (either direction) by more than k SDs, recall / coverage below the floor, rep-4 above it, any exact
        duplicates beyond a rate, and attributes whose retention leaves [low, high]."""
        thresholds = thresholds or {}
        t = dict(DEFAULT_GUARDS, **thresholds)
        quarter_limit = thresholds.get("mtd_quarter_null_sd", t.get("mmd_quarter_null_sd", t["mtd_quarter_null_sd"]))
        out = {}
        for name, r in self.results.items():
            f = self.floor_for(name)
            fl = []
            if f is not None:
                q = _get(r, "mtd_quarter", "mtd", None)
                qf = _get(f, "mtd_quarter", "mtd", None)
                qsd = _get(r, "mtd_quarter", "null_sd", None)
                if None not in (q, qf, qsd) and q > qf + quarter_limit * qsd:
                    fl.append(f"MTD σ/4 {q:.4f} vs floor {qf:.4f} (> {quarter_limit:g} null SD)")
                j, jf, jsd = _get(r, "entropy", "JSD_rows", None), _get(f, "entropy", "JSD_rows", None), _get(r, "entropy", "JSD_rows_sd_half", None)
                if None not in (j, jf, jsd) and abs(j - jf) > t["jsd_sd"] * jsd / math.sqrt(2):
                    fl.append(f"JSD_rows {j:.3f} vs floor {jf:.3f}")
                for key, lim in (("recall", "recall_drop"), ("coverage", "coverage_drop")):
                    v, vf = _get(r, "prdc", key, None), _get(f, "prdc", key, None)
                    if None not in (v, vf) and v < vf - t[lim]:
                        fl.append(f"{key} {v:.3f} vs floor {vf:.3f}")
                rp, rpf = _get(r, "rep", "rep_4", None), _get(f, "rep", "rep_4", None)
                if None not in (rp, rpf) and rp > rpf + t["rep_rise"]:
                    fl.append(f"rep-4 {rp:.3f} vs floor {rpf:.3f}")
            d = _get(r, "dup", "exact_dup_rate", None)
            if d is not None and d > t["exact_dup_rate"]:
                fl.append(f"exact duplicates {d:.3f}")
            att = r.get("attributes", {}).get("retention", {})
            for k, v in att.items():
                if np.isfinite(v) and (v < t["attribute_retention_low"] or v > t["attribute_retention_high"]):
                    fl.append(f"attribute {k} retention {v:.2f}")
            out[name] = fl
        return out

    def to_dict(self) -> Dict[str, Any]:
        def clean(x):
            if isinstance(x, dict):
                return {str(k): clean(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [clean(v) for v in x]
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
            return x
        return clean(dict(settings=self.settings, results=self.results,
                          floors={str(k): v for k, v in self.floors.items()}, flags=self.flags()))

    def save(self, path: str) -> None:
        json.dump(self.to_dict(), open(path, "w"), indent=1)

    def __repr__(self) -> str:
        return self.table()

