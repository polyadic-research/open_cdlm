"""Evaluate sample files with MTD, optionally KTD, and the recommended diagnostics.

    python examples/metrics/quickstart.py samples.pt --reference owt_ref.pt --preset owt --n 1024

``samples.pt``: a tensor [n, L] of generated token ids (or a dict with 'ids'); ``owt_ref.pt``: >= 2n random real rows
of the evaluation split, e.g. from ``open_cdlm-metrics make-ref``. Pass a separate real sample file via
``--calibration`` to include KTD.
"""
import argparse

from open_cdlm.dlm_metrics import evaluate

p = argparse.ArgumentParser()
p.add_argument("samples", nargs="+")
p.add_argument("--reference", required=True)
p.add_argument("--calibration")
p.add_argument("--preset", default="owt")
p.add_argument("--n", type=int, default=None)
p.add_argument("--device", default=None)
a = p.parse_args()

report = evaluate({f: f for f in a.samples}, a.reference, calibration=a.calibration, preset=a.preset,
                  n=a.n, device=a.device)
print(report.table())
report.save("results.json")
