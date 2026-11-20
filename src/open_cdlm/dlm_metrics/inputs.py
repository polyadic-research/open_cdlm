"""Sample containers, tokenizers and file loaders.

Metrics accept samples as

* token ids: a ``torch.Tensor`` / ``np.ndarray`` of shape ``[n, L]``, or a list of 1-D integer sequences (ragged rows), or
* texts: a ``list[str]``.

Token-level metrics (entropies, n-gram statistics) work on ids; model-based metrics (generative perplexity, feature
metrics, MAUVE) work on texts. :class:`Samples` converts between the two with the *generator* tokenizer, i.e. the
tokenizer the evaluated model was trained with (``gpt2`` for OpenWebText models, ``bert-base-uncased`` for LM1B models).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from typing import Any, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

#: name -> (generator tokenizer, row length) of the common DLM benchmarks
PRESETS = {
    "owt": ("gpt2", 1024),
    "owt-sub": ("gpt2", 128),
    "lm1b": ("bert-base-uncased", 128),
}
GPT2_FAMILY = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")

TokenizerLike = Union[str, Any]


def is_gpt2_family(name: TokenizerLike) -> bool:
    return isinstance(name, str) and name.split("/")[-1] in GPT2_FAMILY


@functools.lru_cache(maxsize=None)
def _auto_tokenizer(name: str):
    """AutoTokenizer as LangFlow's gen_ppl.py decodes; ``[PAD]`` is added when missing (only affects ids >= vocab)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


def get_tokenizer(tokenizer: TokenizerLike):
    """A tokenizer object from a Hugging Face name or an existing tokenizer object."""
    if tokenizer is None:
        raise ValueError("a generator tokenizer is needed to convert between token ids and texts")
    return _auto_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer


@functools.lru_cache(maxsize=None)
def flm_tokenizer(name: str):
    """The decoding tokenizer of the MDLM / DUO / FLM code bases (``dataloader.get_tokenizer``)."""
    import transformers

    if name == "bert-base-uncased":
        tok = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        tok = transformers.AutoTokenizer.from_pretrained(name)
    gpt2_classes = tuple(c for c in (getattr(transformers, "GPT2TokenizerFast", None),
                                     getattr(transformers, "GPT2Tokenizer", None)) if c is not None)
    if gpt2_classes and isinstance(tok, gpt2_classes) and hasattr(tok, "_tokenizer"):
        import tokenizers

        tok._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tok.bos_token, tok.bos_token_id), (tok.eos_token, tok.eos_token_id))
    if tok.bos_token is None:
        tok.bos_token = tok.cls_token
    if tok.eos_token is None:
        tok.eos_token = tok.sep_token
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


def _tokenizer_key(tokenizer: TokenizerLike) -> str:
    if tokenizer is None:
        return "none"
    if isinstance(tokenizer, str):
        return tokenizer
    return getattr(tokenizer, "name_or_path", type(tokenizer).__name__)


class Samples:
    """A set of generated (or real) rows: token ids and/or texts plus the generator tokenizer.

    Build it with :meth:`Samples.wrap`; metrics call it for you, so you rarely need it directly.
    """

    def __init__(self, ids: Optional[List[np.ndarray]] = None, texts: Optional[List[str]] = None,
                 tokenizer: TokenizerLike = None):
        if ids is None and texts is None:
            raise ValueError("Samples needs ids or texts")
        if ids is not None and texts is not None and len(ids) != len(texts):
            raise ValueError("ids and texts have different lengths")
        self._ids, self._texts, self.tokenizer = ids, texts, tokenizer
        self._cache = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def wrap(cls, x: Any, tokenizer: TokenizerLike = None) -> "Samples":
        """Accept ``Samples``, a list of strings, an id tensor/array ``[n, L]`` or a list of id sequences."""
        if isinstance(x, Samples):
            if tokenizer is not None and x.tokenizer is None:
                x.tokenizer = tokenizer
            return x
        if isinstance(x, str):
            raise TypeError("pass a list of texts, not a single string")
        if torch.is_tensor(x) or isinstance(x, np.ndarray):
            a = np.asarray(x.detach().cpu() if torch.is_tensor(x) else x)
            if a.ndim == 1:
                a = a[None]
            if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer):
                raise ValueError(f"expected an integer array [n, L] of token ids, got {a.dtype} {a.shape}")
            return cls(ids=[r.astype(np.int64) for r in a], tokenizer=tokenizer)
        x = list(x)
        if not x:
            raise ValueError("empty sample set")
        if all(isinstance(t, str) for t in x):
            return cls(texts=x, tokenizer=tokenizer)
        return cls(ids=[np.asarray(r, dtype=np.int64).reshape(-1) for r in x], tokenizer=tokenizer)

    # ------------------------------------------------------------------ views
    def __len__(self) -> int:
        return len(self._ids) if self._ids is not None else len(self._texts)

    @property
    def from_texts(self) -> bool:
        return self._ids is None

    @property
    def has_ids(self) -> bool:
        return self._ids is not None

    def ids(self) -> List[np.ndarray]:
        """Token-id rows. Texts are encoded with the generator tokenizer (no special tokens added)."""
        if self._ids is None:
            if "ids" not in self._cache:
                tok = get_tokenizer(self.tokenizer)
                self._cache["ids"] = [np.asarray(tok(t, add_special_tokens=False)["input_ids"], dtype=np.int64)
                                      for t in self._texts]
            return self._cache["ids"]
        return self._ids

    def texts(self, skip_special_tokens: bool = False) -> List[str]:
        """Decoded texts. ``skip_special_tokens=False`` keeps e.g. ``<|endoftext|>`` / ``[CLS]`` as text (the
        LangFlow / MDLM gen-ppl convention); ``True`` drops them (the MAUVE convention)."""
        key = ("texts", skip_special_tokens)
        if key not in self._cache:
            if self._texts is not None:
                out = list(self._texts)
                if skip_special_tokens:
                    specials = [s for s in getattr(get_tokenizer(self.tokenizer), "all_special_tokens", []) if s]
                    for s in sorted(specials, key=len, reverse=True):
                        out = [t.replace(s, "") for t in out]
            else:
                tok = get_tokenizer(self.tokenizer)
                out = tok.batch_decode([r.tolist() for r in self._ids], skip_special_tokens=skip_special_tokens)
            self._cache[key] = out
        return self._cache[key]

    def subset(self, idx: Iterable[int]) -> "Samples":
        idx = list(idx)
        return Samples(ids=[self._ids[i] for i in idx] if self._ids is not None else None,
                       texts=[self._texts[i] for i in idx] if self._texts is not None else None,
                       tokenizer=self.tokenizer)

    def lengths(self) -> List[int]:
        return [len(r) for r in self.ids()]

    def digest(self) -> str:
        """Content hash (ids if available, else texts) used for feature caching."""
        if "digest" not in self._cache:
            h = hashlib.sha1()
            if self._ids is not None:
                for r in self._ids:
                    h.update(np.asarray(r, dtype=np.int64).tobytes())
                    h.update(b"|")
            else:
                for t in self._texts:
                    h.update(t.encode("utf-8"))
                    h.update(b"\x00")
            h.update(_tokenizer_key(self.tokenizer).encode())
            self._cache["digest"] = h.hexdigest()
        return self._cache["digest"]

    @staticmethod
    def concat(parts: Sequence["Samples"]) -> "Samples":
        parts = list(parts)
        if not parts:
            raise ValueError("nothing to concatenate")
        if all(p.has_ids for p in parts):
            return Samples(ids=[r for p in parts for r in p.ids()], tokenizer=parts[0].tokenizer)
        return Samples(texts=[t for p in parts for t in p.texts(False)], tokenizer=parts[0].tokenizer)


# ---------------------------------------------------------------------------------------------- file loading

def _read_texts(path: str) -> List[str]:
    if path.endswith(".txt"):
        return [line.rstrip("\n") for line in open(path, encoding="utf-8") if line.strip()]
    if path.endswith(".jsonl"):
        out = []
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                j = json.loads(line)
                out.append(j if isinstance(j, str) else j["text"])
        return out
    j = json.load(open(path, encoding="utf-8"))
    if isinstance(j, dict):
        j = j.get("texts") or j.get("samples") or j.get("text")
    if not (isinstance(j, list) and all(isinstance(t, str) for t in j)):
        raise ValueError(f"{path}: expected a list of strings or {{'texts': [...]}}")
    return j


def _unpack(obj: Any, path: str):
    """(ids tensor or None, texts or None, meta) from a loaded torch / numpy object."""
    if isinstance(obj, np.lib.npyio.NpzFile):
        key = "ids" if "ids" in obj.files else obj.files[0]
        return torch.as_tensor(obj[key]), None, {}
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return torch.as_tensor(obj), None, {}
    if isinstance(obj, list) and obj and all(isinstance(t, str) for t in obj):
        return None, obj, {}
    if isinstance(obj, dict):
        meta = dict(obj["meta"]) if isinstance(obj.get("meta"), dict) else {}
        if isinstance(obj.get("tokenizer"), str):
            meta["tokenizer"] = obj["tokenizer"]
        for k in ("ids", "samples", "tokens", "input_ids"):
            v = obj.get(k)
            if torch.is_tensor(v) or isinstance(v, np.ndarray):
                return torch.as_tensor(v), None, meta
        for k in ("texts", "text", "samples"):
            v = obj.get(k)
            if isinstance(v, list) and v and all(isinstance(t, str) for t in v):
                return None, v, meta
    raise ValueError(f"{path}: expected a tensor [n, L], a dict with 'ids' / 'samples' / 'tokens', or texts")


def load_samples(path: str, tokenizer: TokenizerLike = None, preset: Optional[str] = None) -> Samples:
    """Load a sample file.

    Formats: torch ``.pt`` (tensor ``[n, L]`` or dict with ``ids`` / ``samples`` / ``tokens`` / ``input_ids``, optional
    ``meta={'tokenizer': ...}``), numpy ``.npy`` / ``.npz``, or texts (``.jsonl`` with a ``text`` field, ``.json`` list
    of strings or ``{"texts": [...]}``, ``.txt`` one sample per line). The generator tokenizer is ``tokenizer``, else
    the file's meta, else the preset's.
    """
    if path.endswith((".jsonl", ".json", ".txt")):
        ids, texts, meta = None, _read_texts(path), {}
    elif path.endswith((".npy", ".npz")):
        ids, texts, meta = _unpack(np.load(path, allow_pickle=False), path)
    else:
        ids, texts, meta = _unpack(torch.load(path, map_location="cpu", weights_only=False), path)
    tok = tokenizer or meta.get("tokenizer") or (PRESETS[preset][0] if preset else None)
    if texts is not None:
        return Samples(texts=texts, tokenizer=tok)
    if ids.dim() != 2:
        raise ValueError(f"{path}: expected a 2-D tensor of token ids, got shape {tuple(ids.shape)}")
    return Samples(ids=[r for r in ids.long().numpy()], tokenizer=tok)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def env_threads(n: int = 16) -> None:
    """Cap BLAS threads: the CPU-side scoring thrashes when several evaluation jobs share a machine."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, str(n))
