"""GenPPL/entropy and optional ModernBERT MAUVE on generated decoded texts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .metrics import ARScorer, token_sequence_metrics


@torch.no_grad()
def modernbert_features(texts, *, model_name, device, batch_size, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "reference_compile"):
        # Eager reference math also works with the existing PyTorch 2.3 runtime;
        # optional Triton compilation is unnecessary for extracting fixed features.
        config.reference_compile = False
    model = (
        AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    features = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,
            return_tensors="pt",
        ).to(device)
        output = model(**encoded).last_hidden_state
        positions = encoded.attention_mask.sum(-1) - 1
        vectors = output[torch.arange(output.shape[0], device=device), positions]
        features.append(vectors.float().cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scorer-model", default="openai-community/gpt2-large")
    parser.add_argument("--skip-gen-ppl", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--text-field", choices=("text", "clean_text"), default="text")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--mauve", action="store_true")
    parser.add_argument("--feature-model", default="answerdotai/ModernBERT-large")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.batch_size, args.max_length) <= 0:
        raise ValueError("batch-size and max-length must be positive")
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    records = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("sample file is empty")
    metrics = token_sequence_metrics([r["token_ids"] for r in records])
    metrics["num_generated_samples"] = len(records)
    if not args.skip_gen_ppl:
        scorer = ARScorer(args.scorer_model, device=device)
        try:
            metrics.update(
                scorer.score(
                    records,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    text_field=args.text_field,
                )
            )
        finally:
            scorer.close()
    if args.mauve:
        if args.reference is None:
            raise ValueError("--mauve requires a fixed --reference JSONL")
        import mauve

        reference = [
            json.loads(line) for line in args.reference.read_text().splitlines() if line.strip()
        ]
        if len(reference) != len(records):
            raise ValueError("MAUVE reference and sample counts must match (paper: 5120 each)")
        kwargs = dict(
            model_name=args.feature_model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        p = modernbert_features([r[args.text_field] for r in reference], **kwargs)
        q = modernbert_features([r[args.text_field] for r in records], **kwargs)
        result = mauve.compute_mauve(p_features=p, q_features=q, seed=args.seed, verbose=False)
        metrics["mauve"] = float(result.mauve)
        metrics["mauve_feature_model"] = args.feature_model
        metrics["mauve_pooling"] = "last_valid_token"
        metrics["num_reference_samples"] = len(reference)
    metrics.update(
        text_field=args.text_field,
        max_length=args.max_length,
        scorer_model=None if args.skip_gen_ppl else args.scorer_model,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
