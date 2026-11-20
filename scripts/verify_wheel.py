"""Install a built wheel into a temporary target and exercise it outside the source tree.

The current interpreter must already have Open CDLM's base and training dependencies.
No model weights or datasets are downloaded.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        parser.error(f"wheel not found: {wheel}")
    with tempfile.TemporaryDirectory(prefix="open_cdlm-wheel-") as directory:
        target = Path(directory) / "installed"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-index",
             "--no-deps", "--target", str(target), str(wheel)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        environment = {
            **os.environ, "PYTHONPATH": str(target), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        }
        code = r'''
import sys
from pathlib import Path
from importlib.metadata import distribution
from importlib.resources import files
import open_cdlm
target = Path(sys.argv[1])
assert Path(open_cdlm.__file__).is_relative_to(target)
metadata = distribution('open_cdlm')
assert metadata.metadata['Name'] == 'open_cdlm'
assert Path(metadata.locate_file('')).resolve() == target.resolve()
assert all(
    ep.name == 'open_cdlm' or ep.name.startswith('open_cdlm-')
    for ep in metadata.entry_points if ep.group == 'console_scripts'
)
assert not any(target.joinpath(name).exists() for name in ('condila', 'condia'))
from open_cdlm.cdlm import list_models
assert list_models() == ('langflow', 'replaid')
assert 'torch' not in sys.modules
from open_cdlm.dlm_metrics import KTD, MTD
assert MTD.__module__ == 'open_cdlm.dlm_metrics.metrics.feature'
assert KTD.__module__ == 'open_cdlm.dlm_metrics.metrics.ktd'
assert files('open_cdlm.cdlm.langflow.configs').joinpath('train.yaml').is_file()
assert files('open_cdlm.cdlm.langflow.configs').joinpath('model/langflow-owt.yaml').is_file()
assert files('open_cdlm.cdlm.langflow.configs').joinpath('owt_pytorch.yaml').is_file()
assert files('open_cdlm.cdlm.replaid.configs').joinpath('owt_sc.yaml').is_file()
from open_cdlm.cdlm import create_model, generate
settings = {
 'langflow': dict(vocab_size=18, hidden_size=16, cond_dim=8, n_blocks=1,
                 n_heads=2, dropout=0., model_length=8),
 'replaid': dict(vocab_size=18, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, max_seq_length=8,
                cond_dim=8, dropout=0., replaid_embedding_dim=4, replaid_schedule_hidden_size=16),
}
for name, config in settings.items():
    assert generate(create_model(name, config), num_samples=2, steps=2).shape == (2, 8)
assert not {'langflow', 'duo', 'replaid_original', 'dlm_metrics'} & sys.modules.keys()
'''
        commands = [
            [sys.executable, "-c", code, str(target)],
            [str(target / "bin/open_cdlm"), "--version"],
            [str(target / "bin/open_cdlm"), "models"],
            [str(target / "bin/open_cdlm-metrics"), "--help"],
            [str(target / "bin/open_cdlm-langflow-train"), "--dry-run", "--device", "cpu"],
            [str(target / "bin/open_cdlm-replaid-train"), "--dry-run", "--device", "cpu"],
        ]
        for command in commands:
            result = subprocess.run(
                command, check=False, cwd=directory, env=environment,
                text=True, capture_output=True, timeout=90,
            )
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        print(json.dumps({"wheel": wheel.name, "checks": len(commands), "status": "passed"}))


if __name__ == "__main__":
    main()
