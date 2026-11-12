"""torchmetrics-style usage: stream batches into metrics, share one feature extractor, keep reference features.

    python examples/modular_metrics.py --reference owt_ref.pt --samples samples.pt
"""
import argparse

import torch

from condila.dlm_metrics import (MAUVE, MMD, PRDC, EntropyDecomposition, GenerativePerplexity, MetricCollection,
                         TextFeatureExtractor, load_samples)

p = argparse.ArgumentParser()
p.add_argument("--reference", required=True)
p.add_argument("--samples", required=True)
p.add_argument("--n", type=int, default=512)
a = p.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

ref = load_samples(a.reference, tokenizer="gpt2").subset(range(a.n))
gen = load_samples(a.samples, tokenizer="gpt2").subset(range(a.n))

# one judge, one extractor per pooling: every metric below shares the gpt2-large weights and the features
fx = TextFeatureExtractor("gpt2-large", device=device, pooling="mean", batch_size=8)
fx_last = TextFeatureExtractor(fx.judge, pooling="last", batch_size=8)
metrics = MetricCollection({
    "gen_ppl": GenerativePerplexity(judge=fx.judge, split_half=8),
    "entropy": EntropyDecomposition(split_half=8),
    "mmd": MMD(extractor=fx, null_repeats=12, reset_real_features=False),
    "mmd_quarter": MMD(extractor=fx, bandwidth_multiplier=0.25, reset_real_features=False),
    "prdc": PRDC(extractor=fx, k=5, reset_real_features=False),
    "mauve": MAUVE(extractor=fx_last, num_buckets="auto", reset_real_features=False),
}, tokenizer="gpt2")

metrics.update(ref, real=True)                      # reference rows: two-sample metrics only
for s in range(0, len(gen), 64):                    # stream the generated rows in batches
    metrics.update(gen.subset(range(s, min(s + 64, len(gen)))))
for name, res in metrics.compute().items():
    print(name, {k: v for k, v in res.items() if not isinstance(v, (list, dict))})

metrics.reset()                                     # generated states cleared, reference features kept
