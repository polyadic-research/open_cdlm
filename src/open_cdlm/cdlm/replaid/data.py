from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


def tokenizer_fingerprint(tokenizer) -> str:
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop("padding", None)
    backend.pop("truncation", None)
    return hashlib.sha256(json.dumps(backend, sort_keys=True).encode()).hexdigest()


class PreparedDataset(Dataset):
    """Fixed-length, unpadded GPT-2 rows, memory mapped rather than loaded into RAM."""

    def __init__(self, directory: str | Path, split: str, *, sequence_length: int, vocab_size: int):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["sequence_length"] != sequence_length:
            raise ValueError("prepared sequence length differs from the model context length")
        if self.manifest["vocab_size"] != vocab_size:
            raise ValueError("prepared vocabulary differs from model.vocab_size")
        self.split = split
        self.sequence_length = sequence_length
        self.rows = int(self.manifest["splits"][split]["sequences"])
        self.dtype = np.dtype(self.manifest["dtype"])
        self.path = self.directory / f"{split}.bin"
        expected = self.rows * sequence_length * self.dtype.itemsize
        if self.rows <= 0 or self.path.stat().st_size != expected:
            raise ValueError(f"empty or incomplete prepared split: {self.path}")
        self._array = None

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        if self._array is None:
            self._array = np.memmap(
                self.path,
                dtype=self.dtype,
                mode="r",
                shape=(self.rows, self.sequence_length),
            )
        # An owned int64 copy prevents writes to the read-only memory map.
        return torch.from_numpy(np.array(self._array[index], dtype=np.int64))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        return state


class RandomBatchSampler(Sampler):
    """Uniform block sampling with replacement, addressed by global microbatch index.

    This ordering has no hidden worker RNG state and resumes without scanning or
    skipping the corpus. Different ranks receive independently seeded batches.
    """

    def __init__(
        self, size: int, batch_size: int, *, seed: int, rank: int, start_batch: int, stop_batch: int
    ):
        self.size, self.batch_size = size, batch_size
        self.seed, self.rank = seed, rank
        self.start_batch, self.stop_batch = start_batch, stop_batch

    def __iter__(self):
        for index in range(self.start_batch, self.stop_batch):
            seed = np.random.SeedSequence([self.seed, self.rank, index])
            yield np.random.default_rng(seed).integers(self.size, size=self.batch_size).tolist()

    def __len__(self):
        return max(0, self.stop_batch - self.start_batch)


def validation_loader(dataset, *, batch_size, max_batches, rank, world):
    """Shard the same fixed validation budget across all ranks."""
    size = len(dataset) if max_batches is None else min(len(dataset), max_batches * batch_size)
    subset = Subset(dataset, range(rank, size, world))
    return DataLoader(subset, batch_size=batch_size, shuffle=False), size
