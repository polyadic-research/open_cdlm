"""Export a fixed held-out text reference for generation-quality evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .data import PreparedDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    dataset = PreparedDataset(
        args.data_dir,
        "validation",
        sequence_length=manifest["sequence_length"],
        vocab_size=manifest["vocab_size"],
    )
    if args.num_samples <= 0 or args.num_samples > len(dataset):
        raise ValueError("reference size must be positive and no larger than the held-out corpus")
    tokenizer = AutoTokenizer.from_pretrained(args.data_dir / "tokenizer", use_fast=True)
    indices = np.random.default_rng(args.seed).choice(len(dataset), args.num_samples, replace=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for sample_id, index in enumerate(indices):
            ids = dataset[int(index)].tolist()
            stream.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "source_row": int(index),
                        "token_ids": ids,
                        "text": tokenizer.decode(ids, skip_special_tokens=False),
                        "clean_text": tokenizer.decode(ids, skip_special_tokens=True),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "num_samples": args.num_samples,
                "split": "validation",
                "split_sha256": manifest["splits"]["validation"]["sha256"],
                "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
