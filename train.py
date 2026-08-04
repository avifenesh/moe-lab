#!/usr/bin/env python
"""Single-GPU training loop for moe-lab.

Usage: python train.py --config configs/base.yaml [--name RUN_NAME]

bf16 autocast, AdamW, cosine schedule with warmup, grad clipping, optional
grad checkpointing / grad accumulation. JSONL telemetry per step (train loss,
aux/reg losses, per-layer router entropy, per-expert token counts) and per
eval (val loss, per-expert activation norms, pairwise expert weight cosine
similarity — all on a fixed held-out calibration slice). Checkpoints and
telemetry go to runs/<run_name>/.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch

from moe_lab import data as data_mod
from moe_lab.config import Config
from moe_lab.model import MoETransformer
from moe_lab.prune_eval import expert_weight_cosim


def lr_at(step: int, cfg: Config) -> float:
    t = cfg.train
    total_steps = max(1, t.total_tokens // cfg.tokens_per_step)
    if step < t.warmup_steps:
        return t.lr * (step + 1) / max(1, t.warmup_steps)
    p = min(1.0, (step - t.warmup_steps) / max(1, total_steps - t.warmup_steps))
    return t.lr * (t.min_lr_ratio + (1 - t.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def build_optimizer(model, cfg: Config, device: torch.device):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.train.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.train.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=(device.type == "cuda"),
    )


@torch.no_grad()
def evaluate(model, val_batches, device: torch.device, bf16: bool) -> dict:
    was_training = model.training
    model.eval()
    moes = [blk.moe for blk in model.blocks]
    for m in moes:
        m.capture = True
    ce, n = 0.0, 0
    act = [torch.zeros(model.cfg.n_experts) for _ in moes]
    for x, y in val_batches:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            out = model(x, y)
        ce += out["ce_loss"].item()
        n += 1
        for a, m in zip(act, moes):
            a += m.last_act_norms.cpu()
    for m in moes:
        m.capture = False
    if was_training:
        model.train()
    return {
        "val_loss": round(ce / max(1, n), 5),
        "expert_act_norms": [[round(v, 4) for v in (a / max(1, n)).tolist()] for a in act],
        "expert_weight_cosim": [round(expert_weight_cosim(m), 6) for m in moes],
        "expert_bias": [[round(v, 5) for v in m.router.expert_bias.tolist()] for m in moes],
    }


def save_ckpt(model, opt, step: int, tokens: int, cfg: Config, path: Path) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "tokens": tokens,
            "config": dataclasses.asdict(cfg),
        },
        path,
    )


def train(cfg: Config, batch_source=None, val_batches=None) -> dict:
    """Run the training loop. Returns {"model", "run_dir", "losses"}.

    batch_source/val_batches are injectable so the smoke test can run on
    synthetic data without touching HF; by default they come from the
    fineweb-edu stream (moe_lab.data).
    """
    t = cfg.train
    device = torch.device(t.device if (t.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(t.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(t.seed)

    run_dir = Path(t.out_root) / t.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")

    model = MoETransformer(cfg.model).to(device)
    if t.grad_checkpoint:
        model.grad_checkpoint = True
    n_total = sum(p.numel() for p in model.parameters())
    expert_params = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
    n_active = n_total - (cfg.model.n_experts - cfg.model.top_k) * expert_params * cfg.model.n_layer
    print(f"[train] run={t.run_name} balancing={cfg.model.balancing} "
          f"params total={n_total / 1e6:.1f}M active~={n_active / 1e6:.1f}M device={device}")

    if batch_source is None:
        batch_source = data_mod.batch_stream(cfg.data, "train", device, t.batch_size)
    if val_batches is None:
        val_stream = data_mod.batch_stream(cfg.data, "val", device, t.batch_size)
        val_batches = [next(val_stream) for _ in range(t.eval_batches)]

    opt = build_optimizer(model, cfg, device)
    log_f = open(run_dir / "telemetry.jsonl", "a")

    total_steps = max(1, t.total_tokens // cfg.tokens_per_step)
    losses, tokens, step = [], 0, 0
    t0 = time.time()
    while tokens < t.total_tokens:
        model.train()
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        ce = aux = reg = 0.0
        counts = [torch.zeros(cfg.model.n_experts, dtype=torch.long) for _ in model.blocks]
        for _ in range(t.grad_accum):
            x, y = next(batch_source)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=t.bf16):
                out = model(x, y)
                loss = out["loss"] / t.grad_accum
            loss.backward()
            ce += out["ce_loss"].item() / t.grad_accum
            aux += out["aux_loss"].item() / t.grad_accum
            reg += out["reg_loss"].item() / t.grad_accum
            for li, blk in enumerate(model.blocks):
                counts[li] += blk.moe.last_counts.cpu()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if cfg.model.balancing == "aux_free":
            for li, blk in enumerate(model.blocks):
                blk.moe.router.balance_update(counts[li])
        step += 1
        tokens += cfg.tokens_per_step
        losses.append(ce)
        if t.telemetry:
            rec = {
                "type": "train", "step": step, "tokens": tokens,
                "loss": round(ce, 5), "aux_loss": round(aux, 5), "reg_loss": round(reg, 5),
                "lr": lr,
                "router_entropy": [round(blk.moe.last_entropy, 5) for blk in model.blocks],
                "expert_counts": [c.tolist() for c in counts],
                "tok_per_s": int(tokens / max(1e-9, time.time() - t0)),
            }
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
        if step % t.eval_interval == 0 or step == total_steps:
            rec = evaluate(model, val_batches, device, t.bf16)
            rec.update({"type": "eval", "step": step, "tokens": tokens})
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(f"[train] step {step}/{total_steps} loss={ce:.4f} val={rec['val_loss']:.4f}")
        if step % t.ckpt_interval == 0:
            save_ckpt(model, opt, step, tokens, cfg, run_dir / f"ckpt_step{step}.pt")
    save_ckpt(model, opt, step, tokens, cfg, run_dir / "last.pt")
    log_f.close()
    return {"model": model, "run_dir": run_dir, "losses": losses}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to a run YAML (see configs/base.yaml)")
    p.add_argument("--name", default=None, help="override train.run_name")
    args = p.parse_args()
    cfg = Config.from_yaml(args.config)
    if args.name:
        cfg.train.run_name = args.name
    train(cfg)


if __name__ == "__main__":
    main()
