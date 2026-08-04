"""REAP-ish expert pruning evaluation.

Saliency per expert = sum over calibration tokens of (router gate weight) x
(L2 norm of the expert's output) — the router-weighted activation norm from
REAP. Experts are ranked per MoE layer; pruning masks the lowest-saliency
experts out of the router and the gates re-normalize over the kept experts
(handled by Router.keep_mask / set_keep_mask).

prunability_curve() reports val-loss delta vs baseline for k in ks, plus a
random-prune control (mean over n_random random masks). k is clamped so at
least top_k experts remain routable per layer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _moe_layers(model):
    return [blk.moe for blk in model.blocks]


def _dev(device):
    return device if isinstance(device, torch.device) else torch.device(device)


@torch.no_grad()
def mean_val_loss(model, batches, device, bf16: bool = True) -> float:
    device = _dev(device)
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for x, y in batches:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            out = model(x, y)
        total += out["ce_loss"].item()
        n += 1
    if was_training:
        model.train()
    return total / max(1, n)


@torch.no_grad()
def expert_saliency(model, batches, device, bf16: bool = True):
    """Per MoE layer, per expert: accumulated router-weighted activation norm."""
    device = _dev(device)
    was_training = model.training
    model.eval()
    moes = _moe_layers(model)
    for m in moes:
        m.capture = True
    acc = None
    for x, _ in batches:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            model(x)
        if acc is None:
            acc = [torch.zeros_like(m.last_act_norms.cpu()) for m in moes]
        for a, m in zip(acc, moes):
            a += m.last_act_norms.cpu()
    for m in moes:
        m.capture = False
    if was_training:
        model.train()
    return acc


def expert_weight_cosim(moe) -> float:
    """Mean off-diagonal cosine similarity between flattened expert weights."""
    ws = [torch.cat([p.detach().float().flatten() for p in e.parameters()]) for e in moe.experts]
    W = F.normalize(torch.stack(ws), dim=-1)
    S = W @ W.T
    n = S.shape[0]
    if n < 2:
        return 0.0
    return float((S.sum() - S.diagonal().sum()) / (n * (n - 1)))


def set_prune_masks(model, masks) -> None:
    """masks: one bool tensor (n_experts,) per MoE layer, True = keep routable."""
    for moe, keep in zip(_moe_layers(model), masks):
        moe.router.set_keep_mask(keep)


def clear_prune_masks(model) -> None:
    for moe in _moe_layers(model):
        moe.router.set_keep_mask(torch.ones_like(moe.router.keep_mask))


def prunability_curve(model, batches, device, ks=(1, 2, 4), n_random: int = 3, seed: int = 0, bf16: bool = True) -> dict:
    """Val-loss delta after pruning k lowest-saliency experts per layer, vs random control."""
    device = _dev(device)
    moes = _moe_layers(model)
    n_experts = model.cfg.n_experts
    top_k = model.cfg.top_k
    baseline = mean_val_loss(model, batches, device, bf16)
    saliency = expert_saliency(model, batches, device, bf16)
    gen = torch.Generator().manual_seed(seed)
    result = {
        "baseline_val_loss": baseline,
        "saliency": [s.tolist() for s in saliency],
        "curves": {},
    }
    for k in ks:
        k_eff = min(k, n_experts - top_k)  # always keep at least top_k experts routable
        if k_eff < 1:
            continue
        masks = []
        for s in saliency:
            order = torch.argsort(s)  # ascending saliency: prune the lowest first
            keep = torch.ones(n_experts, dtype=torch.bool)
            keep[order[:k_eff]] = False
            masks.append(keep)
        set_prune_masks(model, masks)
        loss_reap = mean_val_loss(model, batches, device, bf16)
        rand_deltas = []
        for _ in range(n_random):
            rmasks = []
            for _layer in moes:
                perm = torch.randperm(n_experts, generator=gen)
                keep = torch.ones(n_experts, dtype=torch.bool)
                keep[perm[:k_eff]] = False
                rmasks.append(keep)
            set_prune_masks(model, rmasks)
            rand_deltas.append(mean_val_loss(model, batches, device, bf16) - baseline)
        result["curves"][k] = {
            "k_eff": k_eff,
            "reap_val_loss": loss_reap,
            "reap_delta": loss_reap - baseline,
            "random_delta_mean": sum(rand_deltas) / len(rand_deltas),
            "random_deltas": rand_deltas,
        }
    clear_prune_masks(model)
    return result
