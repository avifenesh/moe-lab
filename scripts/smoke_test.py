#!/usr/bin/env python
"""Smoke test: 30 training steps in each of the 3 balancing modes.

Tiny model (d_model=128, n_layer=2, 4 experts, top-2, seq 128) on fixed
random-token synthetic batches — no HF download, tiny tensors, runs in well
under 2 minutes. Asserts:
  - loss decreases (first-5-step mean vs last-5-step mean)
  - router telemetry fields exist in telemetry.jsonl (train + eval records)
  - prune_eval.prunability_curve runs and returns a finite curve
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from moe_lab.config import Config, DataConfig, ModelConfig, TrainConfig
from moe_lab.data import synthetic_batches
from moe_lab.prune_eval import prunability_curve
from train import train

VOCAB = 256
BATCH = 8
SEQ = 128
STEPS = 30


def make_cfg(mode: str) -> Config:
    return Config(
        model=ModelConfig(
            vocab_size=VOCAB, d_model=128, n_layer=2, n_head=4, max_seq_len=SEQ,
            n_experts=4, top_k=2, expert_hidden=64,
            balancing=mode, aux_loss_coeff=0.01, decorr_coeff=0.01, bias_update_rate=1e-3,
        ),
        data=DataConfig(seq_len=SEQ),
        train=TrainConfig(
            run_name=f"smoke_{mode}", seed=0, batch_size=BATCH, grad_accum=1,
            total_tokens=STEPS * BATCH * SEQ, lr=3e-3, min_lr_ratio=0.1, warmup_steps=5,
            weight_decay=0.0, grad_clip=1.0, bf16=torch.cuda.is_available(),
            eval_interval=15, eval_batches=2, ckpt_interval=10 ** 9,
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
    )


def cycler(batches):
    while True:
        for b in batches:
            yield b


def run_mode(mode: str) -> dict:
    cfg = make_cfg(mode)
    device = cfg.train.device
    train_batches = synthetic_batches(VOCAB, BATCH, SEQ, n_batches=4, seed=1, device=device)
    val_batches = synthetic_batches(VOCAB, BATCH, SEQ, n_batches=2, seed=2, device=device)
    out = train(cfg, batch_source=cycler(train_batches), val_batches=val_batches)

    losses = out["losses"]
    assert len(losses) == STEPS, f"{mode}: expected {STEPS} steps, got {len(losses)}"
    head = sum(losses[:5]) / 5
    tail = sum(losses[-5:]) / 5
    assert tail < head, f"{mode}: loss did not decrease (first5={head:.4f} last5={tail:.4f})"

    with open(out["run_dir"] / "telemetry.jsonl") as f:
        recs = [json.loads(line) for line in f]
    tr = [r for r in recs if r["type"] == "train"]
    ev = [r for r in recs if r["type"] == "eval"]
    assert tr, f"{mode}: no train telemetry records"
    last = tr[-1]
    for field in ("loss", "aux_loss", "reg_loss", "router_entropy", "expert_counts", "lr"):
        assert field in last, f"{mode}: train telemetry missing '{field}'"
    assert len(last["router_entropy"]) == 2, f"{mode}: router_entropy not per-layer"
    assert len(last["expert_counts"]) == 2 and len(last["expert_counts"][0]) == 4, (
        f"{mode}: expert_counts shape wrong")
    assert ev, f"{mode}: no eval telemetry records"
    for field in ("val_loss", "expert_act_norms", "expert_weight_cosim"):
        assert field in ev[-1], f"{mode}: eval telemetry missing '{field}'"

    curve = prunability_curve(out["model"], val_batches, device, ks=(1, 2), n_random=3)
    assert curve["curves"], f"{mode}: prune_eval returned an empty curve"
    for k, c in curve["curves"].items():
        assert math.isfinite(c["reap_delta"]), f"{mode}: non-finite reap delta at k={k}"
        assert math.isfinite(c["random_delta_mean"]), f"{mode}: non-finite random delta at k={k}"
    return {"first5": head, "last5": tail, "curves": curve["curves"]}


def main():
    t0 = time.time()
    results = {}
    for mode in ("aux", "aux_free", "aux_decorr"):
        results[mode] = run_mode(mode)
        r = results[mode]
        c1 = r["curves"][1]
        print(f"[smoke] {mode:11s} loss first5={r['first5']:.4f} -> last5={r['last5']:.4f} | "
              f"prune k=1: reap_delta={c1['reap_delta']:+.4f} random_delta={c1['random_delta_mean']:+.4f}")
    print(f"[smoke] ALL MODES GREEN in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
