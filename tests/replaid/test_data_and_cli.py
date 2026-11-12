from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from safetensors.torch import load_file
import torch

from condila.cdlm.replaid.checkpoint import (
    initialize_ema,
    latest_checkpoint,
    load_model,
    save_checkpoint,
)
from condila.cdlm.replaid.data import PreparedDataset, RandomBatchSampler
from condila.cdlm.replaid.prepare import pack_documents, prepare
from condila.cdlm.replaid.model import RePlaid
from condila.cdlm.replaid.optim import build_optimizer, build_scheduler


ROOT = Path(__file__).resolve().parents[2]


def command(*args, timeout=90):
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false"
    )
    result = subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_mdlm_document_eos_and_block_boundaries(local_tokenizer):
    rows = list(pack_documents(["word0 word1", "word2 word3"], local_tokenizer, 8))
    assert rows == [[0, 2, 3, 0, 4, 5, 0, 0]]


def test_holdout_split_precedes_tokenization(tmp_path, local_tokenizer):
    source = tmp_path / "raw.jsonl"
    source.write_text(
        "".join(
            json.dumps({"text": " ".join([word] * 24)}) + "\n"
            for word in ["word0"] * 3 + ["word15"] * 3
        )
    )
    output = tmp_path / "prepared"
    manifest = prepare([source], output, local_tokenizer, validation_documents=3, length=8)
    train = PreparedDataset(output, "train", sequence_length=8, vocab_size=18)
    valid = PreparedDataset(output, "validation", sequence_length=8, vocab_size=18)
    assert set(torch.cat([train[i] for i in range(len(train))]).tolist()) == {0, 2}
    assert set(torch.cat([valid[i] for i in range(len(valid))]).tolist()) == {0, 17}
    assert manifest["train_documents"] == manifest["validation_documents"] == 3
    assert manifest["splits"]["train"]["sha256"] != manifest["splits"]["validation"]["sha256"]
    with pytest.raises(FileExistsError):
        prepare([source], output, local_tokenizer, validation_documents=3, length=8)


def test_parquet_input_and_incomplete_data_detection(tmp_path, local_tokenizer):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "raw.parquet"
    pq.write_table(pa.table({"text": ["word0 " * 30] * 10}), source)
    output = tmp_path / "prepared"
    prepare([source], output, local_tokenizer, validation_documents=2, length=8)
    with pytest.raises(ValueError, match="sequence length"):
        PreparedDataset(output, "train", sequence_length=16, vocab_size=18)
    with (output / "train.bin").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="incomplete"):
        PreparedDataset(output, "train", sequence_length=8, vocab_size=18)


def test_sampler_resume_does_not_skip_or_repeat_prefix():
    def batches(start, rank=0):
        return list(RandomBatchSampler(100, 4, seed=1, rank=rank, start_batch=start, stop_batch=10))

    assert batches(5) == batches(0)[5:]
    assert batches(0, rank=1) != batches(0, rank=0)


def test_checkpoint_retention_preserves_milestones_and_refuses_overwrite(
    tmp_path,
    local_tokenizer,
    tiny_config,
):
    model = RePlaid(tiny_config.model)
    optimizer = build_optimizer(model, tiny_config)
    scheduler = build_scheduler(optimizer, tiny_config)
    tiny_config.train.save_total_limit = 2
    tiny_config.train.checkpoint_keep_steps = [1]
    ema = initialize_ema(model)
    generator = torch.Generator().manual_seed(1)
    for step in range(1, 5):
        save_checkpoint(
            tmp_path,
            model,
            local_tokenizer,
            tiny_config,
            optimizer,
            scheduler,
            ema,
            step,
            generator,
            {},
            {},
        )
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == [
        "checkpoint-1",
        "checkpoint-3",
        "checkpoint-4",
    ]
    assert latest_checkpoint(tmp_path).name == "checkpoint-4"
    with pytest.raises(RuntimeError, match="already exists"):
        save_checkpoint(
            tmp_path,
            model,
            local_tokenizer,
            tiny_config,
            optimizer,
            scheduler,
            ema,
            4,
            generator,
            {},
            {},
        )


def test_full_pipeline_exact_resume_and_evaluation(tmp_path, prepared_data, tiny_config):
    # Dropout exposes lost/restored RNG state, instead of checking only weights
    # in a deterministic forward. The nonzero eval interval also exercises EMA.
    tiny_config.model = replace(tiny_config.model, dropout=0.1)
    tiny_config.data.directory = str(prepared_data)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(tiny_config)))
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    common = ["-m", "condila.cdlm.replaid.train", "--config", config_path, "--device", "cpu"]
    command(*common, "--output-dir", full)
    command(*common, "--output-dir", resumed, "--stop-after-steps", "2")
    command(*common, "--output-dir", resumed, "--resume", "latest")
    checkpoint = resumed / "checkpoints/checkpoint-4"
    for file in ("model.safetensors", "ema.safetensors"):
        reference = load_file(str(full / "checkpoints/checkpoint-4" / file))
        restored = load_file(str(checkpoint / file))
        for name in reference:
            torch.testing.assert_close(reference[name], restored[name], rtol=0, atol=0, msg=name)
    raw, _, _ = load_model(checkpoint, weights="raw")
    ema, _, _ = load_model(checkpoint, weights="ema")
    assert not torch.equal(raw.output.linear.weight, ema.output.linear.weight)
    torch.testing.assert_close(raw.loss_ema_bias, ema.loss_ema_bias, rtol=0, atol=0)
    evaluation = tmp_path / "nelbo.json"
    command(
        "-m",
        "condila.cdlm.replaid.evaluate",
        "--checkpoint",
        checkpoint,
        "--output",
        evaluation,
        "--device",
        "cpu",
        "--max-batches",
        2,
        "--batch-size",
        2,
        "--mc-samples",
        2,
    )
    metrics = json.loads(evaluation.read_text())
    assert metrics["num_sequences"] == 4 and metrics["num_estimates"] == 8
    assert np.isfinite(metrics["ppl_upper_bound"])
    assert metrics["self_conditioning"] is True
    samples = tmp_path / "samples.jsonl"
    command(
        "-m",
        "condila.cdlm.replaid.sample",
        "--checkpoint",
        checkpoint,
        "--output",
        samples,
        "--device",
        "cpu",
        "--num-samples",
        4,
        "--batch-size",
        2,
        "--length",
        8,
        "--steps",
        2,
    )
    records = [json.loads(line) for line in samples.read_text().splitlines()]
    assert [row["sample_id"] for row in records] == list(range(4))
    assert all(len(row["token_ids"]) == 8 for row in records)
    reference = tmp_path / "reference.jsonl"
    command(
        "-m",
        "condila.cdlm.replaid.reference",
        "--data-dir",
        prepared_data,
        "--output",
        reference,
        "--num-samples",
        4,
    )
    scores = tmp_path / "quality.json"
    command(
        "-m", "condila.cdlm.replaid.score", "--samples", samples, "--skip-gen-ppl", "--output", scores
    )
    assert json.loads(scores.read_text())["num_generated_samples"] == 4


def test_torchrun_two_rank_training_and_sampling(tmp_path, prepared_data, tiny_config):
    tiny_config.data.directory = str(prepared_data)
    tiny_config.train.output_dir = str(tmp_path / "distributed")
    tiny_config.train.total_steps = 2
    tiny_config.train.global_batch_size = 8
    config_path = tmp_path / "config.json"
    tiny_config.save(config_path)
    command(
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        "-m",
        "condila.cdlm.replaid.train",
        "--config",
        config_path,
        "--device",
        "cpu",
    )
    checkpoint = Path(tiny_config.train.output_dir) / "checkpoints/checkpoint-2"
    assert (checkpoint / "training.pt").exists()
    output = tmp_path / "ddp_samples.jsonl"
    command(
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        "-m",
        "condila.cdlm.replaid.sample",
        "--checkpoint",
        checkpoint,
        "--output",
        output,
        "--num-samples",
        5,
        "--batch-size",
        2,
        "--length",
        8,
        "--steps",
        2,
        "--device",
        "cpu",
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_id"] for row in records] == list(range(5))
