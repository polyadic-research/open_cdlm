from argparse import Namespace

import datasets
import pytest
import torch
from safetensors.torch import save_file
from transformers import BertTokenizer

from open_cdlm.cdlm.langflow import LangFlow, LangFlowConfig




def test_wrapped_bert_data_uses_sep_for_eos(tmp_path, monkeypatch):
    from open_cdlm.cdlm.langflow._data import dataloader

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
    tokenizer = BertTokenizer(vocab_file=str(vocab))
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    raw = datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"text": ["hello world hello world"]})
    })
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: raw)
    data = dataloader.get_dataset("lm1b", tokenizer, wrap=True, mode="train",
                                  cache_dir=str(tmp_path), block_size=6, num_proc=1)
    ids = data[0]["input_ids"]
    assert ids[0] == tokenizer.cls_token_id
    assert ids[-1] == tokenizer.sep_token_id


def test_data_cache_distinguishes_tokenizers_and_revisions(tmp_path, monkeypatch):
    from open_cdlm.cdlm.langflow._data import dataloader

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
    raw = datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"text": ["HELLO world HELLO world"]})
    })
    loads = []

    def load_dataset(*args, **kwargs):
        loads.append(kwargs.get("revision"))
        return raw

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    def load(lower_case, revision):
        tokenizer = BertTokenizer(vocab_file=str(vocab), do_lower_case=lower_case)
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
        return dataloader.get_dataset("lm1b", tokenizer, wrap=True, mode="train",
                                      cache_dir=str(tmp_path), block_size=6,
                                      num_proc=1, revision=revision)

    lower = load(True, "v1")[0]["input_ids"]
    upper = load(False, "v1")[0]["input_ids"]
    assert not torch.equal(lower, upper)
    load(True, "v2")
    load(True, "v1")
    assert loads == ["v1", "v1", "v2"]






def test_model_can_train_after_generation_on_a_different_device():
    config = LangFlowConfig(vocab_size=17, hidden_size=16, cond_dim=8,
                           n_blocks=1, n_heads=2, dropout=0.0, model_length=8)
    model = LangFlow(config)
    with torch.no_grad():
        model.generate_samples(num_samples=1, num_steps=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    logits = model(input_ids=torch.randint(17, (2, 8), device=device),
                   timesteps=torch.ones(2, device=device), return_dict=False)
    logits.float().mean().backward()
    assert torch.isfinite(logits).all()
    assert model.backbone.output_layer.linear.weight.grad is not None
