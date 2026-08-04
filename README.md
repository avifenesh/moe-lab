# moe-lab

Tiny from-scratch MoE language models for a training-dynamics research program.

**Research question:** how should experts *form* during training so that they are
prunable later? We train identical tiny MoE LMs under three router-balancing
regimes and compare post-hoc prunability (REAP-style saliency pruning) against a
random-prune control.

## Model

Decoder-only transformer, torch only, single-file readable (`moe_lab/model.py`):
pre-norm RMSNorm, RoPE, SwiGLU experts, tied input/output embeddings. Every FFN
is an MoE layer: 8 experts, top-2 routing with softmax gates. Default size
(`configs/base.yaml`, d_model=512, n_layer=8, expert_hidden=320, vocab 16k):
~48M total params / ~24M active (incl. the shared 8.2M embedding matrix).

## Training variants (run matrix)

| Run | `model.balancing` | Balancing mechanism |
|-----|-------------------|---------------------|
| R0  | `aux`             | Switch-style auxiliary load-balancing loss (`aux_loss_coeff`) |
| R1  | `aux_free`        | Aux-loss-free (DeepSeek-V3 style): per-expert router bias updated between steps by `update_rate * sign(mean_load − load_i)`, no grad; bias affects selection only, not gate values |
| R2  | `aux_decorr`      | Aux loss + differentiable expert-output decorrelation regularizer (`decorr_coeff`): mean off-diagonal cosine similarity of gate-weighted mean expert outputs per MoE layer, on the training batch |

Matrix: R0/R1/R2 × seeds {0, 1} at 0.5–1B tokens. Stretch: R3 granularity
(16 experts, top-4, expert_hidden halved, R0 balancing, 1 seed).

Note: R1 keeps Switch-style softmax gates for comparability across variants;
only the balancing mechanism differs, per the design above.

## Metrics (all in `runs/<name>/telemetry.jsonl`)

Per step: train loss, aux loss, reg (decorrelation) loss, lr, per-layer router
entropy (nats), per-expert token counts, tok/s.
Per eval (fixed held-out calibration batches, crc32-diverted ~0.1% of the
stream): val loss, per-expert gate-weighted activation norms, per-layer pairwise
expert-weight cosine similarity, router bias.

**Prunability curve** (`moe_lab/prune_eval.py`): on N calibration batches,
saliency per expert = Σ gate × ‖expert output‖ (router-weighted activation
norm, REAP-ish). Mask the k lowest-saliency experts per layer out of the router
(gates re-normalize over kept experts), measure val-loss delta vs baseline for
k ∈ {1, 2, 4}, plus a random-prune control (mean over 3 random masks). k is
clamped so at least top-k experts remain.

## Kill criteria

- K1: a variant >2% above the best variant's val loss at 50% of the token
  budget → stop that variant.
- K2: routing collapse (any expert >50% of selections, or mean router entropy
  < 0.5·ln(8), at 2 consecutive evals) → stop; log as failure mode.
- K3: at the final checkpoint, |reap_delta − random_delta| at k=2 within the
  spread of the 3 random masks → variant not prunable; skip its second seed.

## Layout

- `moe_lab/model.py` — transformer + MoE layer (+ expert-output capture hooks)
- `moe_lab/router.py` — routing, both balancing mechanisms, decorrelation penalty
- `moe_lab/config.py` — dataclass config, YAML load/save
- `moe_lab/data.py` — fineweb-edu streaming → BPE tokens → packed 1024-seq,
  uint16 shard cache on disk (restarts don't re-tokenize), val divert
- `moe_lab/prune_eval.py` — REAP-ish saliency + prunability curve
- `train.py` — single-GPU loop (bf16 autocast, AdamW, cosine+warmup, grad clip,
  optional grad checkpoint/accum, JSONL telemetry, checkpoints to `runs/<name>/`)
- `scripts/train_tokenizer.py` — train 16k BPE on ~1GB of the stream
- `scripts/smoke_test.py` — 30 steps × 3 modes on synthetic data + prune_eval
- `scripts/jetson_deploy.sh` — rsync to the Jetson + print the launch command
- `configs/base.yaml` — R0 seed-0 config

## Usage

```bash
python scripts/train_tokenizer.py                 # once, ~1GB stream
python train.py --config configs/base.yaml        # R0 seed 0
python scripts/smoke_test.py                      # desktop smoke test
scripts/jetson_deploy.sh r0-seed0 configs/base.yaml   # deploy + print launch cmd
```

No wandb / external services: telemetry is JSONL only.
