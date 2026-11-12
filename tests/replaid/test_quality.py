import math

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, ModernBertConfig, ModernBertModel

from condila.cdlm.replaid.metrics import ARScorer
from condila.cdlm.replaid.score import modernbert_features


def test_generation_ppl_api_with_local_random_gpt2(tmp_path, local_tokenizer):
    local_tokenizer.pad_token = local_tokenizer.eos_token
    local_tokenizer.save_pretrained(tmp_path)
    torch.manual_seed(30)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=18,
            n_embd=16,
            n_layer=2,
            n_head=2,
            n_positions=16,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
        )
    )
    model.save_pretrained(tmp_path)
    scorer = ARScorer(str(tmp_path), device="cpu")
    records = [
        {"text": "word1 word2 word3", "clean_text": "word1 word2 word3"},
        {"text": "word4 word5", "clean_text": "word4 word5"},
    ]
    try:
        scores = scorer.score(records, batch_size=2, max_length=8, text_field="text")
    finally:
        scorer.close()
    assert scores["ar_scored_tokens"] == 3
    assert math.isfinite(scores["gpt2_gen_ppl"]) and scores["gpt2_gen_ppl"] > 1


def test_modernbert_feature_extraction_api(tmp_path, local_tokenizer):
    local_tokenizer.pad_token = local_tokenizer.eos_token
    local_tokenizer.save_pretrained(tmp_path)
    config = ModernBertConfig(
        vocab_size=18,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        local_attention=8,
        global_attn_every_n_layers=2,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        reference_compile=False,
    )
    ModernBertModel(config).save_pretrained(tmp_path)
    features = modernbert_features(
        ["word1 word2 word3", "word4 word5"],
        model_name=str(tmp_path),
        device="cpu",
        batch_size=2,
        max_length=8,
    )
    assert features.shape == (2, 16)
    assert np.isfinite(features).all()
