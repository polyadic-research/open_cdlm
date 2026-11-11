"""Generate LangFlow token files or JSONL records suitable for metric evaluation."""
import argparse
import json
from pathlib import Path

import torch

from .loading import from_pretrained


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True, help=".pt token IDs or .jsonl texts")
    parser.add_argument("--tokenizer", help="required for JSONL output")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length", type=int)
    parser.add_argument("--steps", type=int, default=128, help="Euler-EDM solver intervals")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if min(args.num_samples, args.batch_size, args.steps) < 1:
        parser.error("num-samples, batch-size and steps must be positive")
    if args.output.suffix not in {".pt", ".jsonl"}:
        parser.error("output must end in .pt or .jsonl")
    if args.output.suffix == ".jsonl" and not args.tokenizer:
        parser.error("JSONL output requires --tokenizer")
    from ..registry import generate

    model = from_pretrained(args.checkpoint, device=args.device)
    # Seed per output row so batching reuses the same initial noise.
    rows = []
    for start in range(0, args.num_samples, args.batch_size):
        count = min(args.batch_size, args.num_samples - start)
        rows.append(generate(
            model, num_samples=count, length=args.length, steps=args.steps, seed=args.seed + start,
        ).cpu())
    ids = torch.cat(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "langflow", "steps": args.steps, "seed": args.seed,
        "num_samples": len(ids), "length": ids.shape[1], "tokenizer": args.tokenizer,
    }
    if args.output.suffix == ".pt":
        torch.save({"ids": ids, "meta": metadata}, args.output)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        texts = tokenizer.batch_decode(ids, skip_special_tokens=False)
        clean = tokenizer.batch_decode(ids, skip_special_tokens=True)
        with args.output.open("w", encoding="utf-8") as stream:
            for index, (text, clean_text, tokens) in enumerate(zip(texts, clean, ids.tolist())):
                record = {
                    "sample_id": index, "text": text, "clean_text": clean_text,
                    "token_ids": tokens,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
