"""Streaming data pipeline for moe-lab.

Streams HuggingFaceFW/fineweb-edu, tokenizes on the fly with a local HF
`tokenizers` BPE, and packs tokens into fixed-length sequences. Tokenized
shards are cached on disk as uint16 .npy files so restarts do not re-tokenize.
A manifest (manifest.json) tracks how many stream documents and shards have
been consumed so a fresh process can skip ahead without redoing work.

~0.1% of the token stream is reserved as a held-out: cached shard 0 is used for
evaluation/pruning calibration (kind="val") and the training stream (kind="train")
starts at shard 1, so train never sees held-out documents.

Also provides synthetic_batches(): fixed random-token batches for smoke tests
(no HF download; memorizable, so the loss provably decreases).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch


def load_tokenizer(path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def _stream_texts(cfg):
    from datasets import load_dataset

    try:
        ds = load_dataset(cfg.dataset, name=cfg.subset, split="train", streaming=True)
    except Exception:
        ds = load_dataset(cfg.dataset, split="train", streaming=True)
    for rec in ds:
        yield rec["text"]


def _manifest_path(cfg) -> Path:
    return Path(cfg.cache_dir) / "manifest.json"


def _load_manifest(cfg) -> dict:
    p = _manifest_path(cfg)
    if p.exists():
        return json.loads(p.read_text())
    return {"docs_seen": 0, "train_shards": 0, "val_shards": 0}


def _save_manifest(cfg, m: dict) -> None:
    p = _manifest_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m))


def ensure_shard(cfg, kind: str, idx: int) -> np.ndarray:
    """Return token shard `idx` ("train"|"val") as a uint16 array.

    Loads from the on-disk cache when present; otherwise resumes the stream
    after the last consumed document, tokenizes, caches every complete shard
    on the way, and returns the requested one.
    """
    assert kind in ("train", "val")
    path = Path(cfg.cache_dir) / f"{kind}_shard_{idx:05d}.npy"
    if path.exists():
        return np.load(path)
    manifest = _load_manifest(cfg)
    counts = {"train": manifest["train_shards"], "val": manifest["val_shards"]}
    if counts[kind] > idx:
        raise RuntimeError(f"manifest says {kind} shard {idx} exists but file is missing: {path}")
    tok = load_tokenizer(cfg.tokenizer_path)
    eos = tok.token_to_id("[EOS]")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer: list = []
    stream = _stream_texts(cfg)
    if manifest["docs_seen"]:
        stream = itertools.islice(stream, manifest["docs_seen"], None)
    docs_iter = iter(stream)
    train_shards = counts["train"]
    while True:
        # Chunked encode_batch: the tokenizers Rust core fans out across cores
        # (single-doc encode is one core and caps the pipeline at ~4k tok/s).
        docs = list(itertools.islice(docs_iter, 128))
        if not docs:
            break
        manifest["docs_seen"] += len(docs)
        for enc in tok.encode_batch(docs, add_special_tokens=False):
            buffer.extend(enc.ids)
            buffer.append(eos)
            while len(buffer) >= cfg.shard_tokens:
                arr = np.asarray(buffer[: cfg.shard_tokens], dtype=np.uint16)
                del buffer[: cfg.shard_tokens]
                np.save(Path(cfg.cache_dir) / f"train_shard_{train_shards:05d}.npy", arr)
                train_shards += 1
                manifest["train_shards"] = train_shards
                _save_manifest(cfg, manifest)
                if train_shards - 1 == idx:
                    return arr
    raise RuntimeError("dataset stream exhausted before producing the requested shard")


def batch_stream(cfg, kind: str, device, batch_size: int):
    """Infinite generator of (input, target) batches, each (batch_size, seq_len).

    Packs non-overlapping seq_len+1 windows from consecutive cached shards.
    Cached shard 0 is reserved as the held-out: kind="val" reads only shard 0,
    kind="train" iterates shards 1..N (never shard 0).
    """
    idx = 0 if kind == "val" else 1
    while True:
        arr = torch.from_numpy(ensure_shard(cfg, "train", idx).astype(np.int64))
        idx += 1
        n_win = (arr.numel() - 1) // cfg.seq_len
        usable = n_win - (n_win % batch_size)
        for i in range(0, usable, batch_size):
            seqs = [arr[(i + j) * cfg.seq_len : (i + j) * cfg.seq_len + cfg.seq_len + 1] for j in range(batch_size)]
            chunk = torch.stack(seqs)
            yield chunk[:, :-1].to(device), chunk[:, 1:].to(device)


def synthetic_batches(vocab_size: int, batch_size: int, seq_len: int, n_batches: int, seed: int, device):
    """Fixed random-token batches (no HF download). Deterministic given `seed`.

    The same small set of batches is meant to be cycled during smoke tests, so
    the model can memorize them and the loss provably decreases.
    """
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        x = torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=g)
        batches.append((x[:, :-1].to(device), x[:, 1:].to(device)))
    return batches
