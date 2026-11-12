"""Sample a pretrained CDLM, save token IDs, and evaluate against held-out real rows."""
import argparse
from pathlib import Path

import torch

from condila.cdlm import from_pretrained, generate
from condila.dlm_metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("langflow", "replaid"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length", type=int)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--sampler")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    if min(args.samples, args.batch_size) < 1:
        parser.error("samples and batch-size must be positive")
    model = from_pretrained(args.model, args.checkpoint, device=args.device)
    outputs = []
    for start in range(0, args.samples, args.batch_size):
        outputs.append(generate(
            model, num_samples=min(args.batch_size, args.samples - start), length=args.length,
            steps=args.steps, sampler=args.sampler, seed=args.seed + start,
        ).cpu())
    ids = torch.cat(outputs)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "meta": {"tokenizer": args.tokenizer}}, args.output / "samples.pt")
    report = evaluate(
        {args.model: ids}, reference=args.reference, tokenizer=args.tokenizer,
        n=args.samples, device=args.device,
    )
    print(report.table())
    report.save(str(args.output / "metrics.json"))


if __name__ == "__main__":
    main()
