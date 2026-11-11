"""MDLM-style OWT packing: document EOS, then block BOS + (L-2) tokens + EOS."""

from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .data import tokenizer_fingerprint


def source_files(patterns: list[str]) -> list[Path]:
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if not matches:
            raise FileNotFoundError(f"no input files matched {pattern}")
        files.extend(Path(p).resolve() for p in matches)
    if len(set(files)) != len(files):
        raise ValueError("input patterns select duplicate files")
    return files


def documents(paths: list[Path], text_field: str):
    for path in paths:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(path).iter_batches(columns=[text_field], batch_size=4096):
                for text in batch.column(0).to_pylist():
                    if not isinstance(text, str):
                        raise ValueError(f"non-string {text_field} in {path}")
                    yield text
        elif path.suffix in (".jsonl", ".txt"):
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if path.suffix == ".jsonl":
                        if not line.strip():
                            continue
                        text = json.loads(line)[text_field]
                    else:
                        text = line.rstrip("\n")
                    if not isinstance(text, str):
                        raise ValueError(f"non-string {text_field} in {path}")
                    yield text
        else:
            raise ValueError(f"unsupported input format: {path}; use parquet, jsonl, or txt")


def count_documents(paths: list[Path], text_field: str) -> int:
    count = 0
    for path in paths:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            count += pq.ParquetFile(path).metadata.num_rows
        else:
            count += sum(1 for _ in documents([path], text_field))
    return count


def pack_documents(records, tokenizer, length: int, document_batch_size: int = 1000):
    if length < 3 or document_batch_size < 1:
        raise ValueError("sequence length must be >=3 and document batch size positive")
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    if bos is None or eos is None:
        raise ValueError("tokenizer must define BOS and EOS")
    iterator = iter(records)
    while batch := list(itertools.islice(iterator, document_batch_size)):
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        tokens = list(itertools.chain.from_iterable(ids + [eos] for ids in encoded))
        usable = len(tokens) // (length - 2) * (length - 2)
        for start in range(0, usable, length - 2):
            yield [bos] + tokens[start : start + length - 2] + [eos]


def write_split(path: Path, rows, *, vocab_size: int) -> dict:
    count = 0
    checksum = hashlib.sha256()
    with path.open("xb") as stream:
        for row in rows:
            if min(row) < 0 or max(row) >= vocab_size:
                raise ValueError("token ID outside the configured vocabulary")
            content = np.asarray(row, dtype="<u2").tobytes()
            stream.write(content)
            checksum.update(content)
            count += 1
    if not count:
        raise ValueError(f"{path.stem} produces no full blocks; provide more documents")
    return {"sequences": count, "sha256": checksum.hexdigest()}


def prepare(
    paths,
    output: Path,
    tokenizer,
    *,
    validation_documents: int,
    length: int,
    text_field: str = "text",
    document_batch_size: int = 1000,
) -> dict:
    total = count_documents(paths, text_field)
    if validation_documents < 1 or total <= validation_documents:
        raise ValueError(f"need more than {validation_documents} documents; found {total}")
    if len(tokenizer) > 65536:
        raise ValueError("this GPT-2 pipeline uses uint16; vocabulary must be <=65536")
    output.mkdir(parents=True, exist_ok=False)
    records = iter(documents(paths, text_field))
    training_count = total - validation_documents
    training_records = itertools.islice(records, training_count)
    splits = {}
    splits["train"] = write_split(
        output / "train.bin",
        pack_documents(training_records, tokenizer, length, document_batch_size),
        vocab_size=len(tokenizer),
    )
    # Exhausting train's packing iterator consumed exactly training_count raw
    # documents; validation is the LAST documents, never a sample of training.
    splits["validation"] = write_split(
        output / "validation.bin",
        pack_documents(records, tokenizer, length, document_batch_size),
        vocab_size=len(tokenizer),
    )
    tokenizer.save_pretrained(output / "tokenizer")
    manifest = {
        "format_version": 1,
        "sequence_length": length,
        "dtype": "<u2",
        "vocab_size": len(tokenizer),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "packing": "document EOS; concatenate in batches; block BOS + L-2 tokens + EOS",
        "document_batch_size": document_batch_size,
        "documents": total,
        "train_documents": training_count,
        "validation_documents": validation_documents,
        "split_policy": "last documents in input order held out before tokenization",
        "sources": [{"path": str(p), "bytes": p.stat().st_size} for p in paths],
        "splits": splits,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="+", required=True, help="ordered parquet/jsonl/txt patterns"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", default="openai-community/gpt2")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--validation-documents", type=int, default=100_000)
    parser.add_argument("--document-batch-size", type=int, default=1000)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    manifest = prepare(
        source_files(args.input),
        args.output,
        tokenizer,
        validation_documents=args.validation_documents,
        length=args.sequence_length,
        text_field=args.text_field,
        document_batch_size=args.document_batch_size,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
