from dataclasses import replace
import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from open_cdlm.cdlm.replaid.config import ExperimentConfig
from open_cdlm.cdlm.replaid.prepare import prepare


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_config():
    config = ExperimentConfig()
    config.model = replace(
        config.model,
        vocab_size=18,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_seq_length=8,
        cond_dim=8,
        dropout=0.0,
        replaid_embedding_dim=4,
        replaid_schedule_hidden_size=16,
    )
    config.train = replace(
        config.train,
        total_steps=4,
        batch_size=2,
        global_batch_size=8,
        warmup_steps=0,
        num_workers=0,
        log_every=1,
        save_every=2,
        eval_every=2,
        eval_max_batches=2,
        eval_batch_size=2,
        mixed_precision="no",
    )
    return config


@pytest.fixture
def local_tokenizer():
    vocabulary = {"<|endoftext|>": 0, "[UNK]": 1}
    vocabulary.update({f"word{i}": i + 2 for i in range(16)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="[UNK]",
    )


@pytest.fixture
def prepared_data(tmp_path, local_tokenizer):
    source = tmp_path / "documents.jsonl"
    source.write_text(
        "".join(
            json.dumps({"text": " ".join(f"word{(row + column) % 16}" for column in range(24))})
            + "\n"
            for row in range(40)
        )
    )
    destination = tmp_path / "prepared"
    prepare(
        [source],
        destination,
        local_tokenizer,
        validation_documents=8,
        length=8,
        document_batch_size=4,
    )
    return destination
