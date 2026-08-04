#!/usr/bin/env python
"""Train a 16k BPE tokenizer on ~1GB of the fineweb-edu stream.

Usage: python scripts/train_tokenizer.py [--out tokenizer] [--bytes 1000000000]
Saves <out>/tokenizer.json. Standalone: only needs `datasets` + `tokenizers`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def text_iterator(dataset: str, subset: str, byte_budget: int):
    from datasets import load_dataset

    try:
        ds = load_dataset(dataset, name=subset, split="train", streaming=True)
    except Exception:
        ds = load_dataset(dataset, split="train", streaming=True)
    n = 0
    for rec in ds:
        text = rec["text"]
        n += len(text.encode("utf-8"))
        yield text
        if n >= byte_budget:
            print(f"[tokenizer] reached byte budget: {n / 1e9:.2f} GB")
            return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--bytes", type=int, default=1_000_000_000, dest="byte_budget")
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--out", default="tokenizer")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        show_progress=True,
    )
    tok.train_from_iterator(
        text_iterator(args.dataset, args.subset, args.byte_budget), trainer=trainer
    )
    path = out / "tokenizer.json"
    tok.save(str(path))
    print(f"[tokenizer] saved {path} vocab={tok.get_vocab_size()} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
