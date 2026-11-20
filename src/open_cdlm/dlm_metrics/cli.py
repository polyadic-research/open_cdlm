"""Command line: ``open_cdlm-metrics eval`` and ``open_cdlm-metrics make-ref``.

    open_cdlm-metrics make-ref --from-disk /path/openwebtext-valid_validation_bs1024_wrapped.dat --rows 4096 \
        --preset owt --out owt_ref.pt
    open_cdlm-metrics eval samples.pt --reference owt_ref.pt --preset owt --n 1024 --device cuda:0 --out results.json
    open_cdlm-metrics eval a.pt b.pt --reference owt_ref.pt --preset owt --metrics basic
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def cmd_make_ref(argv) -> None:
    import torch

    from .inputs import PRESETS, _unpack

    p = argparse.ArgumentParser(prog="open_cdlm-metrics make-ref", description="Draw RANDOM real rows into a reference file.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-disk", help="datasets.save_to_disk directory with an 'input_ids' column")
    src.add_argument("--from-ids", help="a .pt / .npy tensor [N, L] of real rows")
    p.add_argument("--split", default=None, help="split name when --from-disk holds a DatasetDict")
    p.add_argument("--rows", type=int, default=4096, help="rows to draw (>= 2 x the largest n you will evaluate)")
    p.add_argument("--length", type=int, default=None, help="truncate rows to this many tokens")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--preset", choices=list(PRESETS), default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.from_disk:
        import datasets

        ds = datasets.load_from_disk(a.from_disk)
        if isinstance(ds, datasets.DatasetDict):
            ds = ds[a.split or next(iter(ds))]
        N = len(ds)
        idx = np.sort(np.random.default_rng(a.seed).choice(N, size=min(a.rows, N), replace=False))
        ds = ds.with_format("numpy")
        rows = [np.asarray(ds[int(i)]["input_ids"]) for i in idx]
        source = os.path.basename(os.path.normpath(a.from_disk)) + (f" [{a.split}]" if a.split else "")
    else:
        x, _, _ = _unpack(np.load(a.from_ids) if a.from_ids.endswith((".npy", ".npz"))
                          else torch.load(a.from_ids, map_location="cpu", weights_only=False), a.from_ids)
        N = len(x)
        idx = np.sort(np.random.default_rng(a.seed).choice(N, size=min(a.rows, N), replace=False))
        rows = [x[int(i)].numpy() for i in idx]
        source = os.path.basename(a.from_ids)
    L = a.length or min(len(r) for r in rows)
    ids = torch.as_tensor(np.stack([r[:L] for r in rows]).astype(np.int64)).to(torch.int32)
    tok = a.tokenizer or (PRESETS[a.preset][0] if a.preset else None)
    torch.save(dict(ids=ids, idx=torch.as_tensor(idx), source=source, split_rows=N, seed=a.seed,
                    meta={"tokenizer": tok} if tok else {}, note="uniform random rows of an evaluation split"), a.out)
    print(f"wrote {a.out}: {tuple(ids.shape)} rows drawn from {N} ({source}), tokenizer {tok}")


def cmd_eval(argv) -> None:
    from .inputs import PRESETS, env_threads

    env_threads()
    from .suite import GROUPS, evaluate

    p = argparse.ArgumentParser(prog="open_cdlm-metrics eval", description="Evaluate sample files against random real rows.")
    p.add_argument("samples", nargs="+", help=".pt / .npy / .npz ids, or .jsonl / .json / .txt texts")
    p.add_argument("--reference", default=None, help="reference file of real rows (make-ref)")
    p.add_argument("--calibration", default=None, help="separate real sample file for KTD calibration")
    p.add_argument("--preset", choices=list(PRESETS), default=None)
    p.add_argument("--tokenizer", default=None, help="generator tokenizer (overrides preset / file meta)")
    p.add_argument("--n", type=int, default=None, help="evaluate the first n rows of each file")
    p.add_argument("--metrics", default="recommended", help=f"{' | '.join(GROUPS)} or a comma-separated list")
    p.add_argument("--protocol", default="langflow", choices=["langflow", "flm"])
    p.add_argument("--judge", default="gpt2-large")
    p.add_argument("--ref-seed", type=int, default=0)
    p.add_argument("--split-half", type=int, default=8)
    p.add_argument("--mtd-null-repeats", "--mmd-null-repeats", type=int, default=12)
    p.add_argument("--feature-batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/open_cdlm.dlm_metrics"))
    p.add_argument("--out", default=None, help="JSON output")
    p.add_argument("--table", default=None, help="append the markdown tables to this file")
    a = p.parse_args(argv)
    metrics = a.metrics if a.metrics in GROUPS else [m.strip() for m in a.metrics.split(",") if m.strip()]
    sets = {os.path.splitext(os.path.basename(f))[0]: f for f in a.samples}
    rep = evaluate(sets, a.reference, calibration=a.calibration, tokenizer=a.tokenizer,
                   preset=a.preset, n=a.n, metrics=metrics, device=a.device,
                   judge=a.judge, protocol=a.protocol, ref_seed=a.ref_seed, split_half=a.split_half,
                   mtd_null_repeats=a.mtd_null_repeats, feature_batch_size=a.feature_batch_size,
                   cache_dir=a.cache_dir or None)
    text = rep.table()
    print("\n" + text)
    if a.out:
        rep.save(a.out)
        print("wrote", a.out)
    if a.table:
        with open(a.table, "a") as fh:
            fh.write(text + "\n")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "make-ref":
        cmd_make_ref(rest)
    elif cmd == "eval":
        cmd_eval(rest)
    else:
        raise SystemExit(f"unknown command {cmd!r}: use 'eval' or 'make-ref'")


if __name__ == "__main__":
    main()
