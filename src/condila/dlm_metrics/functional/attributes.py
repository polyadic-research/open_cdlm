"""Row-level text attributes and their retention: the statistics that see SPARSE failures (a few percent of tokens or
rows) which every global distance (MMD, MAUVE, FD, entropies) averages away.

For each attribute, a "mode" is the set of rows above the real rows' ``percentile``-th percentile. ``retention`` =
share of generated rows above that threshold / share of real rows above it (1 = kept, 0 = lost, > 1 = over-produced);
``mean_ratio`` = mean attribute value of the generated rows / of the real rows.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np

_CODE = set("{}[]();=<>|\\/_#$%^&*")
_WORD = re.compile(r"[A-Za-z]+")


def _digits(s: str) -> float:
    return sum(ch.isdigit() for ch in s) / max(len(s), 1)


def _non_ascii(s: str) -> float:
    return sum(ord(ch) > 127 for ch in s) / max(len(s), 1)


def _code_chars(s: str) -> float:
    return sum(ch in _CODE for ch in s) / max(len(s), 1)


def _url(s: str) -> float:
    return float(("http" in s) or ("www." in s) or (".com" in s))


def _quotes(s: str) -> float:
    return (s.count('"') + s.count("“") + s.count("”")) / max(len(s), 1) * 1000


def _newlines(s: str) -> float:
    return s.count("\n") / max(len(s), 1) * 1000


def _caps_words(s: str) -> float:
    words = _WORD.findall(s)
    return sum(1 for w in words if len(w) >= 2 and w.isupper()) / max(len(words), 1)


def _word_len(s: str) -> float:
    words = _WORD.findall(s)
    return float(np.mean([len(w) for w in words])) if words else 0.0


#: name -> f(text without special tokens) -> float
DEFAULT_ATTRIBUTES: Dict[str, Callable[[str], float]] = {
    "digits": _digits, "non_ascii": _non_ascii, "code_chars": _code_chars, "url": _url, "quotes": _quotes,
    "newlines": _newlines, "caps_words": _caps_words, "word_len": _word_len,
}
#: binary attributes use a fixed threshold instead of a percentile
BINARY_THRESHOLDS = {"url": 0.5}


def attribute_table(texts: List[str], attributes: Optional[Mapping[str, Callable[[str], float]]] = None,
                    docs_per_row: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
    """``{attribute: per-row values}``; ``docs_per_row`` (e.g. EOS counts) is added when given."""
    attributes = DEFAULT_ATTRIBUTES if attributes is None else attributes
    out = {k: np.array([f(t) for t in texts], dtype=np.float64) for k, f in attributes.items()}
    if docs_per_row is not None:
        out["docs_per_row"] = np.asarray(docs_per_row, dtype=np.float64)
    return out


def attribute_retention(real: Mapping[str, np.ndarray], generated: Mapping[str, np.ndarray], percentile: float = 90.0,
                        binary_thresholds: Mapping[str, float] = BINARY_THRESHOLDS) -> Dict[str, Dict[str, float]]:
    """Per attribute: ``dict(threshold, share_real, share, retention, mean_ratio)``."""
    out = {}
    for k in real:
        if k not in generated:
            continue
        thr = binary_thresholds.get(k, float(np.percentile(real[k], percentile)))
        base = float((real[k] > thr).mean())
        share = float((generated[k] > thr).mean())
        out[k] = dict(threshold=thr, share_real=base, share=share,
                      retention=share / base if base > 0 else float("nan"),
                      mean_ratio=float(generated[k].mean() / real[k].mean()) if real[k].mean() != 0 else float("nan"))
    return out
