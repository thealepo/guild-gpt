from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GuildGPTConfig:
    max_seq_len: int = 1024
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    hidden_size: int = 768
    mlp_hidden_size: int | None = None
    attention_backend: str = "reference"
    compute_dtype: str = "float32"

    def __post_init__(self):
        if self.mlp_hidden_size is None:
            object.__setattr__(self, "mlp_hidden_size", 4 * self.hidden_size)
        if (
            min(
                self.max_seq_len,
                self.vocab_size,
                self.n_layers,
                self.n_heads,
                self.hidden_size,
                self.mlp_hidden_size,
            )
            <= 0
        ):
            raise ValueError("Model dimensions must be positive")
        if self.hidden_size % self.n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        if self.attention_backend not in {"reference", "sdpa"}:
            raise ValueError("attention_backend must be reference or sdpa")
        if self.compute_dtype not in {"float32", "bfloat16"}:
            raise ValueError("compute_dtype must be float32 or bfloat16")

    def parameter_count(self) -> int:
        h, m = self.hidden_size, self.mlp_hidden_size
        embeddings = (self.vocab_size + self.max_seq_len) * h
        per_layer = 4 * (h * h + h) + (h * m + m) + (m * h + h) + 4 * h
        return embeddings + self.n_layers * per_layer + 2 * h


@dataclass(frozen=True)
class RunConfig:
    model: GuildGPTConfig = field(default_factory=GuildGPTConfig)
    data_dir: str = "data/processed"
    checkpoint_dir: str = "outputs/checkpoints"
    seed: int = 42
    epochs: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    init_lr: float = 0.0
    peak_lr: float = 3e-4
    final_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 1000
    checkpoint_every: int = 2000
    checkpoint_minutes: int = 30
    keep_checkpoints: int = 3
    log_every: int = 20
    sample_every: int = 1000
    wandb_project: str | None = "GuildGPT"
    wandb_name: str | None = None
    tokenizer: str = "gpt2"
    synthetic: bool = False

    def __post_init__(self):
        if (
            min(
                self.epochs,
                self.micro_batch_size,
                self.gradient_accumulation_steps,
                self.eval_every,
                self.checkpoint_every,
                self.log_every,
                self.sample_every,
                self.keep_checkpoints,
            )
            <= 0
        ):
            raise ValueError("Counts and intervals must be positive")
        if self.warmup_steps < 0 or self.checkpoint_minutes < 0:
            raise ValueError("Warmup and checkpoint minutes cannot be negative")
        if not 0 <= self.init_lr <= self.peak_lr or not 0 <= self.final_lr <= self.peak_lr:
            raise ValueError("Invalid learning rates")
        if self.tokenizer != "gpt2":
            raise ValueError("GuildGPT requires the original GPT-2 tokenizer")

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> RunConfig:
    with open(path, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("Configuration must be a mapping")
    model = GuildGPTConfig(**raw.pop("model"))
    return RunConfig(model=model, **raw)
