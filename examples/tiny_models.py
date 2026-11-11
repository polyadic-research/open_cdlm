"""Run both native CDLMs and token metrics without downloading any weights."""
import torch

from condila.cdlm import create_model, generate
from condila.dlm_metrics import EntropyDecomposition, MetricCollection, RepetitionRate


def main():
    torch.set_num_threads(1)
    options = {
        "langflow": {
            "vocab_size": 32, "hidden_size": 16, "cond_dim": 8,
            "n_blocks": 1, "n_heads": 2, "dropout": 0.0, "model_length": 8,
        },
        "replaid": {
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 1, "num_attention_heads": 2, "max_seq_length": 8,
            "cond_dim": 8, "dropout": 0.0, "replaid_embedding_dim": 4,
            "replaid_schedule_hidden_size": 16,
        },
    }
    for name, config in options.items():
        model = create_model(name, config)
        ids = generate(model, num_samples=4, steps=2, seed=42)
        metrics = MetricCollection({"entropy": EntropyDecomposition(), "repeat": RepetitionRate()})
        metrics.update(ids)
        print(name, tuple(ids.shape), metrics.compute())


if __name__ == "__main__":
    main()
