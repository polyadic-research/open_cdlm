from __future__ import annotations

import gc
import math
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def token_sequence_metrics(
    sequences: list[list[int]], *, max_ngram_size: int = 4
) -> dict[str, float]:
    """Entropy and diversity metrics for tokenized generated text."""
    if max_ngram_size <= 0:
        raise ValueError("max_ngram_size must be positive")

    nonempty_sequences = [sequence for sequence in sequences if sequence]
    pooled_tokens = [token for sequence in nonempty_sequences for token in sequence]
    pooled_counts = Counter(pooled_tokens)

    def entropy(sequence: list[int]) -> float:
        counts = Counter(sequence)
        total = len(sequence)
        return -sum(
            (count / total) * math.log(max(count / total, 1e-12)) for count in counts.values()
        )

    sample_entropies = [entropy(sequence) for sequence in nonempty_sequences]
    corpus_entropy = entropy(pooled_tokens) if pooled_tokens else float("nan")
    metrics = {
        "num_scored_samples": float(len(nonempty_sequences)),
        "num_scorer_tokens": float(len(pooled_tokens)),
        "sample_unigram_entropy": (
            sum(sample_entropies) / len(sample_entropies) if sample_entropies else float("nan")
        ),
        "corpus_unigram_entropy": corpus_entropy,
        "unique_token_count": float(len(pooled_counts)),
        "unique_token_ratio": len(pooled_counts) / max(1, len(pooled_tokens)),
        "mean_max_token_frac": (
            sum(max(Counter(sequence).values()) / len(sequence) for sequence in nonempty_sequences)
            / max(1, len(nonempty_sequences))
        ),
    }
    metrics["unigram_sample_entropy"] = metrics["sample_unigram_entropy"]

    for width in range(1, max_ngram_size + 1):
        all_ngrams: list[tuple[int, ...]] = []
        per_sample_repetition = []
        for sequence in nonempty_sequences:
            sample_ngrams = list(zip(*(sequence[offset:] for offset in range(width))))
            all_ngrams.extend(sample_ngrams)
            if sample_ngrams:
                per_sample_repetition.append(1.0 - len(set(sample_ngrams)) / len(sample_ngrams))
        metrics[f"distinct_{width}"] = len(set(all_ngrams)) / max(1, len(all_ngrams))
        metrics[f"repetition_{width}"] = sum(per_sample_repetition) / max(
            1, len(per_sample_repetition)
        )
    return metrics


def repetition_metrics(records: list[dict[str, Any]], *, ngram_size: int = 4) -> dict[str, float]:
    """Corpus duplicates plus source-token repetition diagnostics."""
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    normalized_texts = [" ".join(record["clean_text"].split()) for record in records]
    sequences = [record["token_ids"] for record in records]

    total_tokens = 0
    repeated_tokens = 0
    total_ngrams = 0
    repeated_ngrams = 0
    for sequence in sequences:
        token_counts = Counter(sequence)
        total_tokens += len(sequence)
        repeated_tokens += sum(count - 1 for count in token_counts.values())
        ngrams = list(zip(*(sequence[offset:] for offset in range(ngram_size))))
        ngram_counts = Counter(ngrams)
        total_ngrams += len(ngrams)
        repeated_ngrams += sum(count - 1 for count in ngram_counts.values())

    count = max(1, len(normalized_texts))
    unique_text_fraction = len(set(normalized_texts)) / count
    return {
        "unique_text_fraction": unique_text_fraction,
        "duplicate_text_fraction": 1.0 - unique_text_fraction,
        "source_token_repetition_fraction": repeated_tokens / max(1, total_tokens),
        f"source_repeated_{ngram_size}gram_fraction": repeated_ngrams / max(1, total_ngrams),
    }


def tensorboard_sample_markdown(records: list[dict[str, Any]], *, max_samples: int) -> str:
    lines = ["| sample | generated text |", "|---:|:---|"]
    for record in records[:max_samples]:
        text = " ".join(record["clean_text"].split()).replace("|", "\\|")
        if len(text) > 2_000:
            text = text[:2_000] + " …"
        lines.append(f"| {record['sample_id']} | {text or '*empty*'} |")
    return "\n".join(lines)


class ARScorer:
    """GPT-2 generation perplexity scorer, loaded only when explicitly requested."""

    def __init__(self, name_or_path: str, *, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("AR scorer tokenizer must define EOS or PAD")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                name_or_path,
                torch_dtype=dtype,
            )
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def score(
        self,
        records: list[dict[str, Any]],
        *,
        batch_size: int,
        max_length: int,
        max_ngram_size: int = 4,
        text_field: str | None = None,
    ) -> dict[str, float]:
        if text_field not in {None, "text", "clean_text"}:
            raise ValueError("text_field must be text, clean_text, or None")
        texts = [
            record[text_field]
            if text_field is not None
            else (record["clean_text"] or record["text"])
            for record in records
        ]
        total_nll = torch.zeros((), device=self.device, dtype=torch.float64)
        total_entropy = torch.zeros((), device=self.device, dtype=torch.float64)
        total_tokens = 0
        tokenized_samples: list[list[int]] = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                texts[start : start + batch_size],
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            tokenized_samples.extend(
                ids[mask.bool()].tolist()
                for ids, mask in zip(
                    encoded.input_ids,
                    encoded.attention_mask,
                    strict=True,
                )
            )
            if encoded.input_ids.shape[1] < 2:
                continue
            logits = self.model(**encoded).logits[:, :-1].float()
            labels = encoded.input_ids[:, 1:]
            mask = encoded.attention_mask[:, 1:].bool()
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape_as(labels)
            probabilities = F.softmax(logits, dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-10).log()).sum(dim=-1)
            total_nll += nll[mask].double().sum()
            total_entropy += entropy[mask].double().sum()
            total_tokens += int(mask.sum())

        metrics = token_sequence_metrics(
            tokenized_samples,
            max_ngram_size=max_ngram_size,
        )
        if total_tokens == 0:
            metrics.update(
                {
                    "ar_generation_ppl": float("nan"),
                    "gpt2_gen_ppl": float("nan"),
                    "ar_prediction_entropy": float("nan"),
                    "ar_scored_tokens": 0.0,
                }
            )
            return metrics
        mean_nll = float(total_nll / total_tokens)
        ppl = math.exp(min(mean_nll, 80.0))
        metrics.update(
            {
                "ar_generation_ppl": ppl,
                "gpt2_gen_ppl": ppl,
                "ar_prediction_entropy": float(total_entropy / total_tokens),
                "ar_scored_tokens": float(total_tokens),
            }
        )
        return metrics

    def close(self) -> None:
        model = self.model
        self.model = None
        del model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
