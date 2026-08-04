"""Configuration for moe-lab runs: dataclasses + YAML load/save.

Balancing modes (ModelConfig.balancing):
  "aux"        R0: Switch-style auxiliary load-balancing loss
  "aux_free"   R1: aux-loss-free (DeepSeek-V3 style router bias updates)
  "aux_decorr" R2: aux loss + differentiable expert-output decorrelation reg
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

BALANCING_MODES = ("aux", "aux_free", "aux_decorr")


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    max_seq_len: int = 1024
    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 320
    balancing: str = "aux"  # one of BALANCING_MODES
    aux_loss_coeff: float = 0.01
    decorr_coeff: float = 0.01
    bias_update_rate: float = 0.001
    rope_base: float = 10000.0

    def __post_init__(self):
        if self.balancing not in BALANCING_MODES:
            raise ValueError(f"balancing must be one of {BALANCING_MODES}, got {self.balancing!r}")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.top_k > self.n_experts:
            raise ValueError("top_k must be <= n_experts")


@dataclass
class DataConfig:
    dataset: str = "HuggingFaceFW/fineweb-edu"
    subset: str = "sample-10BT"
    tokenizer_path: str = "tokenizer/tokenizer.json"
    cache_dir: str = "data_cache"
    seq_len: int = 1024
    shard_tokens: int = 1_000_000  # tokens per cached on-disk shard


@dataclass
class TrainConfig:
    run_name: str = "r0-seed0"
    seed: int = 0
    batch_size: int = 8
    grad_accum: int = 4
    total_tokens: int = 500_000_000
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    bf16: bool = True
    grad_checkpoint: bool = False
    eval_interval: int = 500
    eval_batches: int = 8
    ckpt_interval: int = 2500
    telemetry: bool = True  # per-step JSONL records (eval records are always written)
    device: str = "cuda"
    out_root: str = "runs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def tokens_per_step(self) -> int:
        return self.train.batch_size * self.data.seq_len * self.train.grad_accum

    @classmethod
    def from_yaml(cls, path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}

        def build(dc, d):
            known = {f.name for f in dataclasses.fields(dc)}
            return dc(**{k: v for k, v in (d or {}).items() if k in known})

        return cls(
            model=build(ModelConfig, raw.get("model")),
            data=build(DataConfig, raw.get("data")),
            train=build(TrainConfig, raw.get("train")),
        )

    def to_yaml(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(dataclasses.asdict(self), sort_keys=False))
