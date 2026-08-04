"""Decoder-only MoE transformer. Single-file readable, torch only.

Pre-norm RMSNorm, RoPE, causal attention via F.scaled_dot_product_attention,
SwiGLU experts, tied input/output embeddings. Every FFN is an MoE layer:
n_experts experts, top-k routing with softmax gates (see router.py).

Balancing modes come from ModelConfig.balancing:
  "aux"        R0: aux load-balancing loss added to the training loss
  "aux_free"   R1: no aux loss; trainer updates router bias between steps
  "aux_decorr" R2: aux loss + differentiable expert-output decorrelation reg

Expert-output capture hook points (used by the R2 regularizer, eval telemetry
and prune_eval saliency):
  MoELayer.capture = True   -> per-forward per-expert gate-weighted activation
                               norms are stashed in MoELayer.last_act_norms
  every forward             -> MoELayer.last_counts / MoELayer.last_entropy
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import ModelConfig
from .router import Router, decorrelation_penalty


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rope_cache(seq_len: int, head_dim: int, base: float, device):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()  # (seq_len, head_dim/2)


def apply_rope(x, cos, sin):
    # x: (B, T, H, Dh); cos/sin: (T, Dh/2); rotation applied to interleaved pairs
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = apply_rope(q.view(B, T, self.n_head, self.head_dim), cos, sin).transpose(1, 2)
        k = apply_rope(k.view(B, T, self.n_head, self.head_dim), cos, sin).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class Expert(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.router = Router(cfg.d_model, cfg.n_experts, cfg.top_k, cfg.bias_update_rate)
        self.experts = nn.ModuleList([Expert(cfg.d_model, cfg.expert_hidden) for _ in range(cfg.n_experts)])
        self.use_decorr = cfg.balancing == "aux_decorr" and cfg.decorr_coeff > 0
        self.capture = False  # telemetry hook: stash per-expert activation norms
        self.last_counts = torch.zeros(cfg.n_experts, dtype=torch.long)
        self.last_entropy = 0.0
        self.last_act_norms = torch.zeros(cfg.n_experts)

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        r = self.router(flat)
        out = torch.zeros_like(flat)
        want_outputs = self.capture or (self.training and self.use_decorr)
        outs, gates = [], []
        act_norms = torch.zeros(self.cfg.n_experts, device=x.device, dtype=torch.float32) if self.capture else None
        for e, expert in enumerate(self.experts):
            tok, slot = (r.indices == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            o = expert(flat[tok])
            g = r.gates[tok, slot]
            out.index_add_(0, tok, (o * g.unsqueeze(-1).to(o.dtype)).to(out.dtype))
            if want_outputs:
                outs.append(o)
                gates.append(g)
            if self.capture:
                act_norms[e] += (g.float() * o.detach().float().norm(dim=-1)).sum()
        if self.training and self.use_decorr and len(outs) >= 2:
            reg = decorrelation_penalty(outs, gates)
        else:
            reg = torch.zeros((), device=x.device, dtype=torch.float32)
        self.last_counts = r.counts.detach()
        self.last_entropy = float(r.entropy.detach())
        if self.capture:
            self.last_act_norms = act_norms
        return out.view(B, T, C), r.aux_loss, reg


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.moe = MoELayer(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        m, aux, reg = self.moe(self.ln2(x))
        return x + m, aux, reg


class MoETransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied input/output embeddings
        cos, sin = rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_head, cfg.rope_base, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.grad_checkpoint = False
        self.aux_w = cfg.aux_loss_coeff if cfg.balancing in ("aux", "aux_decorr") else 0.0
        self.reg_w = cfg.decorr_coeff if cfg.balancing == "aux_decorr" else 0.0
        self.apply(self._init_weights)
        std = 0.02 / math.sqrt(2 * cfg.n_layer)  # scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "w2.weight")):
                nn.init.normal_(p, mean=0.0, std=std)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        h = self.embed(idx)
        aux_total = torch.zeros((), device=idx.device)
        reg_total = torch.zeros((), device=idx.device)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                h, a, r = torch.utils.checkpoint.checkpoint(blk, h, cos, sin, use_reentrant=False)
            else:
                h, a, r = blk(h, cos, sin)
            aux_total = aux_total + a
            reg_total = reg_total + r
        logits = self.lm_head(self.ln_f(h))
        out = {"logits": logits, "aux_loss": aux_total, "reg_loss": reg_total}
        if targets is not None:
            ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
            out["ce_loss"] = ce
            out["loss"] = ce + self.aux_w * aux_total + self.reg_w * reg_total
        return out
