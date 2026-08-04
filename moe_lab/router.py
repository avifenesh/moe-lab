"""Routing logic and load-balancing mechanisms for moe-lab MoE layers.

Kept separate from the model so each piece is unit-testable on its own.

Mechanisms:
  R0 "aux"        : Switch-style load-balancing loss  n_experts * sum_i(f_i * P_i),
                    f_i = fraction of selections to expert i (no grad),
                    P_i = mean softmax probability of expert i (grad).
  R1 "aux_free"   : per-expert bias added to the selection logits only (gate
                    values come from unbiased softmax). Between optimizer steps
                    the trainer calls Router.balance_update(counts), which moves
                    the bias by update_rate * sign(mean_load - load_i). No grad.
  R2 "aux_decorr" : differentiable decorrelation penalty on expert *outputs*
                    (decorrelation_penalty below), applied per MoE layer.
"""
from __future__ import annotations

from typing import List, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouteOutput(NamedTuple):
    gates: torch.Tensor    # (N, top_k) gate weights, renormalized over selected experts
    indices: torch.Tensor  # (N, top_k) selected expert ids
    probs: torch.Tensor    # (N, n_experts) full softmax distribution (fp32)
    counts: torch.Tensor   # (n_experts,) selection counts this forward (long)
    aux_loss: torch.Tensor # scalar Switch-style load-balancing loss (fp32)
    entropy: torch.Tensor  # scalar mean router entropy in nats (fp32)


def load_balancing_loss(probs: torch.Tensor, counts: torch.Tensor, top_k: int) -> torch.Tensor:
    """Switch-style aux loss. probs: (N, E) softmax probs, counts: (E,) selections."""
    n_experts = probs.shape[1]
    f = counts.float() / (probs.shape[0] * top_k)
    return n_experts * (f * probs.mean(dim=0)).sum()


def decorrelation_penalty(
    expert_outputs: List[torch.Tensor],
    expert_gates: List[torch.Tensor],
) -> torch.Tensor:
    """Mean off-diagonal cosine similarity between experts' gate-weighted mean outputs.

    expert_outputs[i]: (n_i, d_model) outputs of one expert on the tokens routed to it.
    expert_gates[i]:   (n_i,) gate weights for those tokens.
    Differentiable through both outputs and gates. Experts with zero tokens are
    simply absent from the lists; with <2 experts present the penalty is 0.
    """
    mus = []
    for o, g in zip(expert_outputs, expert_gates):
        w = g.float() / g.float().sum().clamp_min(1e-9)
        mus.append((o.float() * w[:, None]).sum(dim=0))
    if len(mus) < 2:
        return torch.zeros((), device=expert_gates[0].device if expert_gates else "cpu")
    M = F.normalize(torch.stack(mus), dim=-1)
    S = M @ M.T
    n = S.shape[0]
    return (S.sum() - S.diagonal().sum()) / (n * (n - 1))


class Router(nn.Module):
    """Softmax router: top-k selection, gates renormalized over the selected experts.

    Selection uses logits + expert_bias (bias is all zeros unless the R1/aux-free
    updater moves it). keep_mask (all True unless a pruning eval sets it) removes
    experts from selection; gates then renormalize over the kept experts.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int, bias_update_rate: float = 1e-3):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.register_buffer("keep_mask", torch.ones(n_experts, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> RouteOutput:
        logits = self.gate(x)  # (N, E)
        probs = F.softmax(logits.float(), dim=-1)
        sel = logits.float() + self.expert_bias
        sel = sel.masked_fill(~self.keep_mask, float("-inf"))
        _, idx = sel.topk(self.top_k, dim=-1)
        gates = probs.gather(1, idx)
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        counts = torch.bincount(idx.reshape(-1), minlength=self.n_experts)
        aux = load_balancing_loss(probs, counts, self.top_k)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
        return RouteOutput(gates.to(x.dtype), idx, probs, counts, aux, entropy)

    @torch.no_grad()
    def balance_update(self, counts: torch.Tensor) -> None:
        """R1 aux-free update: nudge the per-expert bias by the sign of the load error."""
        total = counts.sum()
        if total == 0:
            return
        load = counts.float().to(self.expert_bias.device) / total
        self.expert_bias.add_(self.bias_update_rate * torch.sign(load.mean() - load))

    def set_keep_mask(self, keep: torch.Tensor) -> None:
        """Pruning hook: keep=True means the expert stays routable."""
        self.keep_mask.copy_(keep.to(self.keep_mask.device, dtype=torch.bool))
