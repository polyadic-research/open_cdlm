"""Integration tests with a small Hugging Face model (downloads gpt2 once). Run with: pytest -m slow"""
import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow

TEXTS = ["The quick brown fox jumps over the lazy dog.",
         "In 2019, the city council approved a new budget of 12 million dollars for public transport.",
         "def f(x):\n    return x ** 2  # square",
         "She said: “We will be back next week.”<|endoftext|>Another document starts here."] * 3


@pytest.fixture(scope="module")
def judge():
    from open_cdlm.dlm_metrics import JudgeLM
    return JudgeLM("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")


def test_gen_ppl_matches_manual(judge):
    from open_cdlm.dlm_metrics import GenerativePerplexity
    m = GenerativePerplexity(judge=judge, tokenizer="gpt2", batch_size=4)
    m.update(TEXTS[:5]); m.update(TEXTS[5:])
    out = m.compute()
    nll, tok = 0.0, 0
    with torch.no_grad():
        for t in TEXTS:
            x = judge.tok(t, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(judge.device)
            logits = judge.lm(x).logits[0, :-1].float()
            nll += float(torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum"))
            tok += x.shape[1] - 1
    assert out["n_tokens"] == tok
    assert abs(out["nll_per_token"] - nll / tok) < 1e-4
    assert abs(out["gen_ppl"] - math.exp(nll / tok)) / out["gen_ppl"] < 1e-4


def test_ids_and_texts_agree(judge):
    from open_cdlm.dlm_metrics import GenerativePerplexity, Samples
    ids = [np.asarray(judge.tok(t)["input_ids"]) for t in TEXTS]
    a = GenerativePerplexity(judge=judge, tokenizer="gpt2"); a.update(Samples.wrap(ids, "gpt2"))
    b = GenerativePerplexity(judge=judge, tokenizer="gpt2"); b.update(TEXTS)
    assert abs(a.compute()["gen_ppl"] - b.compute()["gen_ppl"]) < 1e-3


def test_last_token_features_match_mauve_text(judge):
    mauve = pytest.importorskip("mauve.compute_mauve")
    from open_cdlm.dlm_metrics import TextFeatureExtractor
    fx = TextFeatureExtractor(judge, pooling="last", batch_size=4)
    ours = fx(TEXTS, tokenizer="gpt2")
    dev = judge.device.index if judge.device.type == "cuda" else -1
    clean = [t.replace("<|endoftext|>", "") for t in TEXTS]
    ref = np.asarray(mauve.get_features_from_input(None, None, clean, "gpt2", 1024, dev, "q", batch_size=4, verbose=False))
    assert ours.shape == ref.shape
    assert np.abs(ours - ref).max() < 1e-3 * np.abs(ref).max()


def test_mean_features_and_suite(judge, tmp_path):
    from open_cdlm.dlm_metrics import MTD, TextFeatureExtractor, evaluate
    fx = TextFeatureExtractor(judge, pooling="mean", batch_size=4, cache_dir=str(tmp_path))
    F1 = fx(TEXTS, tokenizer="gpt2")
    F2 = TextFeatureExtractor(judge, pooling="mean", batch_size=4, cache_dir=str(tmp_path))(TEXTS, tokenizer="gpt2")
    assert np.array_equal(F1, F2) and F1.shape == (len(TEXTS), judge.lm.config.n_embd)
    m = MTD(extractor=fx, tokenizer="gpt2", reset_real_features=False)
    m.update(TEXTS, real=True); m.update(TEXTS[::-1])
    same = m.compute()["mtd"]          # identical multisets: the unbiased MMD^2 is <= 0 (cross term has self-pairs)
    m.reset(); m.update(["Completely different text about cooking pasta with tomatoes."] * 4 + TEXTS[:4])
    assert same <= 0 < m.compute()["mtd"]
    ref = TEXTS * 2
    from open_cdlm.dlm_metrics.suite import ALL
    names = [m for m in ALL if m != "topics"                      # 16 clusters need >= 16 reference rows
             and (m != "mauve" or __import__("importlib").util.find_spec("mauve") is not None)]
    rep = evaluate({"a": TEXTS[:6]}, ref, tokenizer="gpt2", judge=judge, metrics=names, split_half=2,
                   mtd_null_repeats=3, verbose=False)
    assert "a" in rep.results and 6 in rep.floors
    assert set(rep.results["a"]) == set(names) and set(rep.floors[6]) == set(names)
    assert "Headline" in rep.table() and "Guards" in rep.table()
    rep.save(str(tmp_path / "r.json"))
