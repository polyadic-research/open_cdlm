# <img src="docs/assets/condila-logo.svg" alt="" width="58" height="36" align="absmiddle"> Condila

Condila is a Python library for continuous diffusion language models. It provides
model configuration, training, checkpoint management, sampling, and generation
evaluation, with support for LangFlow and RePlaid.

```python
from condila.dlm_metrics import MMD, GenerativePerplexity, evaluate
from condila.cdlm import LangFlow, LangFlowConfig, RePlaid, RePlaidConfig
```

## Installation

Requires Python 3.10+ and PyTorch 2.3+. For GPU support, install a PyTorch build
compatible with your CUDA version. From the project root, run:

```bash
python -m pip install -e .
```

Install optional dependencies for the features you need:

| Extra | Features |
|---|---|
| `train` | Lightning / Hydra training and data loading |
| `data` | Dataset processing and Parquet support |
| `mauve` | MAUVE evaluation |
| `topics` | Topic coverage statistics |
| `tracking` | W&B experiment tracking |
| `test` | Testing, linting, and build tools |
| `all` | All optional runtime features |

```bash
python -m pip install -e '.[train,data]'
python -m pip install -e '.[mauve,topics,tracking]'
```

## Namespaces

| Namespace | Contents |
|---|---|
| `condila.dlm_metrics` | `Metric`, `MetricCollection`, MMD, MAUVE, PRDC, GenPPL, entropy, repetition, and coverage diagnostics |
| `condila.dlm_metrics.functional` | Functional metric interfaces |
| `condila.dlm_metrics.likelihood` | NELBO / ODE likelihood estimation and model adapters |
| `condila.cdlm` | Model discovery, registration, construction, loading, and unified token sampling |
| `condila.cdlm.langflow` | LangFlow configuration, Hugging Face model, sampling, LightningModule, and DataModule |
| `condila.cdlm.replaid` | RePlaid configuration, model, objectives, data processing, training, evaluation, and sampling |

`condia.dlm_metrics` is a compatibility namespace that shares the same modules
and classes as `condila.dlm_metrics`.

## Models and Unified Sampling

```python
from condila.cdlm import create_model, list_models, from_pretrained, generate

print(list_models())  # ('langflow', 'replaid')

# Create a randomly initialized model from a configuration
model = create_model(
    'langflow', vocab_size=32, hidden_size=16, cond_dim=8,
    n_blocks=1, n_heads=2, dropout=0.0, model_length=8,
)
ids = generate(model, num_samples=2, length=8, steps=2, seed=42)
assert ids.shape == (2, 8)

# Load a LangFlow checkpoint
model = from_pretrained('langflow', '/path/to/langflow-checkpoint', device='cuda')
ids = generate(model, num_samples=8, length=1024, steps=128, seed=42)

# Load a RePlaid checkpoint with EMA weights by default
model = from_pretrained('replaid', '/path/to/checkpoint-1000000', device='cuda')
ids = generate(model, num_samples=8, steps=128, sampler='ddpm', seed=42)
```

`generate` returns a token-ID tensor of shape `[num_samples, length]`. It runs
the model in evaluation mode during sampling and restores each module's training
state afterward. When `seed` is specified, sample i uses the independent seed
`seed+i`, without modifying the global random number generator state.

The unified `steps` parameter counts solver intervals. It maps to
`num_steps=steps+1` in the native LangFlow interface and `steps=steps` in the
native RePlaid interface.

| Model | Sampler | Denoiser calls per interval |
|---|---|---|
| LangFlow | `euler_edm`, `temperature=1` | 1 |
| RePlaid | `ddpm`, `ddim` | 1 |
| RePlaid | `heun` | 2 |

All samplers make one additional decoder call at the end.

Native model interfaces are also available:

```python
from condila.cdlm.langflow import LangFlow, LangFlowConfig
from condila.cdlm.replaid import RePlaid, RePlaidConfig, load_checkpoint

langflow = LangFlow(LangFlowConfig())
replaid = RePlaid(RePlaidConfig())
model, tokenizer, experiment = load_checkpoint('/path/to/checkpoint', device='cpu')
```

LangFlow supports Hugging Face model IDs, local model directories, safetensors
files, sharded weights, and Lightning checkpoints. Standalone weight files require
a sibling `config.json` or an explicit `config=` argument. Lightning checkpoints
can use their embedded architecture configuration.

A RePlaid checkpoint directory contains `experiment_config.json`,
`model.safetensors`, `ema.safetensors`, and `tokenizer/`. Use `load_checkpoint`
to retrieve the model, tokenizer, and experiment configuration together. Set
`weights='raw'` to load non-EMA weights.

## Evaluation and Likelihood Estimation

```python
import numpy as np
from condila.dlm_metrics import MMD

# Evaluate distributional discrepancy using precomputed features
rng = np.random.default_rng(42)
metric = MMD(bandwidth=1.0)
metric.update_features(rng.normal(size=(64, 16)), real=True)
metric.update_features(rng.normal(size=(64, 16)) + 0.3)
print(metric.compute())
```

```python
from condila.dlm_metrics import evaluate

report = evaluate(
    {'langflow': '/path/to/samples.pt'}, reference='/path/to/owt_reference.pt',
    tokenizer='gpt2', n=512, device='cuda',
)
print(report.table())
report.save('quality.json')
```

For an evaluation with `n` generated samples, provide at least `n` generated rows
and `2*n` reference rows. The reference rows are split between the scoring
reference and a real-versus-real control. `evaluate` uses GPT-2-large as its
default evaluation model and downloads the weights on first use if they are not
cached. The tokenizer must match the token-ID encoding of the generated samples.

MMD returns both U-MMD squared and its signed square root for display. The
`null_sd` field reports the standard deviation of the permutation null
distribution; it is not a confidence interval for model comparisons. See the
[metrics reference](docs/metrics.md) for parameters and statistical conventions.

Use the likelihood module with LangFlow:

```python
from condila.cdlm import from_pretrained
from condila.dlm_metrics.likelihood import LangFlowAdapter, NLLBound

langflow_model = from_pretrained('langflow', '/path/to/langflow-checkpoint', device='cuda')
bound = NLLBound(LangFlowAdapter(langflow_model), estimator='ode', n_steps=128)
# real_ids contains held-out data encoded with the same tokenizer
bound.update(real_ids)
print(bound.compute())
```

For RePlaid held-out NELBO evaluation, use
`condila.cdlm.replaid.objective.validation_terms` or `condila-replaid-evaluate`.

## Training and Command-Line Tools

```bash
condila models
condila metrics --help
condila-metrics eval samples.pt --reference owt_ref.pt --preset owt

# LangFlow
condila-langflow-train --cfg job
condila-langflow-train model=langflow-owt data=openwebtext
condila-langflow-sample --checkpoint /path/to/langflow --output samples.pt \
  --num-samples 512 --batch-size 8 --steps 128

# RePlaid
condila-replaid-train --dry-run --device cpu
condila-replaid-prepare --help
condila-replaid-train --data-dir /path/to/owt_gpt2 --output-dir outputs/replaid
condila-replaid-evaluate --checkpoint /path/to/checkpoint --output nelbo.json
condila-replaid-sample --checkpoint /path/to/checkpoint --output samples.jsonl
condila-replaid-score --samples samples.jsonl --output quality.json
condila-replaid-reference --help
```

Model commands also support `python -m condila.cdlm.<model>.<module>`. The LangFlow
training entry point is `condila.cdlm.langflow.cli`, and its sampling entry point
is `condila.cdlm.langflow.inference`. Launch distributed RePlaid training with
`torchrun -m condila.cdlm.replaid.train ...`.

LangFlow uses Hydra configuration with parameter overrides, `resume` for restoring
training state, and `init_from` for initializing weights. It retains the best
validation checkpoints and periodically saves the latest training state and
`last.ckpt`. At the end of training, it exports `final_model.ckpt` and weights
in Hugging Face format.

Experiment logs are written to local CSV files by default. Enable W&B tracking
with the `tracking` extra, environment variables, and training configuration.

## Development

```bash
python -m pip install -e '.[train,data,test]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m build
```

The default test suite excludes `slow` tests that download model weights. Run
`python -m pytest -m slow` to include those integration tests.

## Documentation

- [Metrics reference](docs/metrics.md)
- [Extending models and metrics](docs/extending.md)
- [Examples](examples)

## License

[MIT License](LICENSE)
