"""Apply the existing metric library's likelihood estimator to bundled LangFlow."""
import argparse

import numpy as np
import torch

from open_cdlm.cdlm import from_pretrained
from open_cdlm.dlm_metrics import load_samples
from open_cdlm.dlm_metrics.likelihood import LangFlowAdapter, NLLBound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.rows, args.batch_size) < 1:
        parser.error("rows and batch-size must be positive")
    model = from_pretrained("langflow", args.checkpoint, device=args.device)
    samples = load_samples(args.reference)
    if len(samples) < args.rows:
        parser.error("reference contains fewer rows than requested")
    ids = torch.as_tensor(np.stack(samples.subset(range(args.rows)).ids())).to(args.device)
    metric = NLLBound(LangFlowAdapter(model), estimator="ode", n_steps=args.steps)
    for start in range(0, args.rows, args.batch_size):
        metric.update(ids[start:start + args.batch_size])
    print(metric.compute())


if __name__ == "__main__":
    main()
