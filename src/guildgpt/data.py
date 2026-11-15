from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_NAME = "andrewelawrence/writingPrompts-merged"
SPLIT_METHOD = "sha256(normalized_prompt), first 8 bytes modulo 100, validation below 5"


def prompt_split(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", (prompt or "").strip().casefold())
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return "validation" if int.from_bytes(digest[:8], "big") % 100 < 5 else "train"


def tokenize_stories(records, tokenizer, train_path: Path, val_path: Path) -> dict:
    counts = {
        "train_stories": 0,
        "validation_stories": 0,
        "train_tokens": 0,
        "validation_tokens": 0,
    }
    eos = tokenizer.eos_token_id
    if eos is None or not 0 <= eos <= 65535:
        raise ValueError("Tokenizer EOS must fit uint16")
    with open(train_path, "wb") as train_file, open(val_path, "wb") as val_file:
        for record in records:
            story = record.get("story")
            if not isinstance(story, str) or not story.strip():
                continue
            ids = tokenizer.encode(story, add_special_tokens=False)
            if not ids:
                continue
            ids.append(eos)
            if min(ids) < 0 or max(ids) > 65535:
                raise ValueError("Token ID exceeds uint16 range")
            split = prompt_split(record.get("prompt", ""))
            np.asarray(ids, dtype=np.uint16).tofile(
                val_file if split == "validation" else train_file
            )
            counts[f"{split}_stories"] += 1
            counts[f"{split}_tokens"] += len(ids)
    return counts


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_data(output: str | Path, *, revision: str | None = None, force: bool = False) -> dict:
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    if not revision:
        raise ValueError("Pin --revision to a dataset commit for reproducible preprocessing")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("dataset") != DATASET_NAME
            or manifest.get("revision_requested") != revision
        ):
            raise ValueError("Existing preprocessing inputs differ; use --force to replace")
        for split in ("train", "validation"):
            path = output / f"{split}.bin"
            if not path.exists() or file_sha256(path) != manifest[f"{split}_sha256"]:
                raise ValueError("Existing token file is missing or changed; use --force")
        return manifest
    if any((output / f"{s}.bin").exists() for s in ("train", "validation")) and not force:
        raise ValueError("Token files exist without a matching manifest; use --force")
    dataset = load_dataset(DATASET_NAME, split="train", revision=revision)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_tmp, val_tmp = output / "train.bin.tmp", output / "validation.bin.tmp"
    counts = tokenize_stories(dataset, tokenizer, train_tmp, val_tmp)
    if min(counts["train_tokens"], counts["validation_tokens"]) < 2:
        raise ValueError("Both splits need at least two tokens")
    manifest = {
        "dataset": DATASET_NAME,
        "revision_requested": revision,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "tokenizer": "gpt2",
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "split_method": SPLIT_METHOD,
        "eos_token_id": tokenizer.eos_token_id,
        "dtype": "uint16",
        **counts,
        "train_sha256": file_sha256(train_tmp),
        "validation_sha256": file_sha256(val_tmp),
    }
    os.replace(train_tmp, output / "train.bin")
    os.replace(val_tmp, output / "validation.bin")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_tokens(data_dir: str | Path, split: str):
    if split not in {"train", "validation"}:
        raise ValueError("Invalid split")
    path = Path(data_dir) / f"{split}.bin"
    return np.memmap(path, dtype=np.uint16, mode="r")


def prepare_synthetic(output, vocab_size: int, seed: int = 42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = {
        "dataset": "synthetic",
        "seed": seed,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "tokenizer": None,
        "eos_token_id": None,
    }
    prior = output / "manifest.json"
    if prior.exists():
        old = json.loads(prior.read_text())
        if any(old.get(key) != value for key, value in manifest.items()):
            raise ValueError("Existing synthetic data configuration differs")
        for split in ("train", "validation"):
            path = output / f"{split}.bin"
            if not path.exists() or file_sha256(path) != old[f"{split}_sha256"]:
                raise ValueError("Existing synthetic data changed")
        return old
    if any((output / f"{split}.bin").exists() for split in ("train", "validation")):
        raise ValueError("Synthetic token files exist without manifest")
    for split, count in (("train", 75), ("validation", 29)):
        path = output / f"{split}.bin"
        rng.integers(0, vocab_size, size=count, dtype=np.uint16).tofile(path)
        manifest[f"{split}_tokens"] = path.stat().st_size // 2
        manifest[f"{split}_sha256"] = file_sha256(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@dataclass
class Batch:
    tokens: np.ndarray  # [micro_batch, seq_len + 1]
    mask: np.ndarray  # [micro_batch, seq_len], valid target positions


class WindowLoader:
    def __init__(self, tokens, seq_len: int, batch_size: int, seed: int = 42, shuffle: bool = True):
        if len(tokens) < 2 or seq_len < 1 or batch_size < 1:
            raise ValueError("Invalid token array, sequence length, or batch size")
        self.tokens = tokens
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.num_windows = math.ceil((len(tokens) - 1) / seq_len)
        self.num_batches = math.ceil(self.num_windows / batch_size)
        self._cached_epoch = None
        self._cached_order = None

    def order(self, epoch: int):
        if self._cached_epoch != epoch:
            windows = np.arange(self.num_windows)
            self._cached_order = (
                np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(
                    windows
                )
                if self.shuffle
                else windows
            )
            self._cached_epoch = epoch
        return self._cached_order

    def get_batch(self, epoch: int, batch_index: int) -> Batch:
        if not 0 <= batch_index < self.num_batches:
            raise IndexError(batch_index)
        selected = self.order(epoch)[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        ids = np.zeros((self.batch_size, self.seq_len + 1), dtype=np.int32)
        mask = np.zeros((self.batch_size, self.seq_len), dtype=np.float32)
        for row, window in enumerate(selected):
            start = int(window) * self.seq_len
            count = min(self.seq_len, len(self.tokens) - 1 - start)
            ids[row, : count + 1] = self.tokens[start : start + count + 1]
            mask[row, :count] = 1
        return Batch(ids, mask)
