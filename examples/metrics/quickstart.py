"""Evaluate sample files with the recommended protocol (headline MMD + guards + legacy gen ppl / entropies + MAUVE).

    python examples/quickstart.py samples.pt --reference owt_ref.pt --preset owt --n 1024

``samples.pt``: a tensor [n, L] of generated token ids (or a dict with 'ids'); ``owt_ref.pt``: >= 2n random real rows
of the evaluation split, e.g. from ``dlm-metrics make-ref``.
"""
import argparse

from condila.dlm_metrics import evaluate

p = argparse.ArgumentParser()
p.add_argument("samples", nargs="+")
p.add_argument("--reference", required=True)
p.add_argument("--preset", default="owt")
p.add_argument("--n", type=int, default=None)
p.add_argument("--device", default=None)
a = p.parse_args()

report = evaluate({f: f for f in a.samples}, a.reference, preset=a.preset, n=a.n, device=a.device)
print(report.table())
report.save("results.json")
