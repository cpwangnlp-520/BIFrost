"""Configuration dataclasses for BIF."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SGLDConfig:
    """Configuration for Localized SGLD sampler."""

    lr: float = 5e-6
    gamma: float = 1e-3
    beta: float = 1.0
    noise_scale: float = 1.0
    num_chains: int = 4
    draws_per_chain: int = 60
    burn_in: int = 0
    thinning: int = 1
    seed: int = 42
    grad_clip: float | None = None
    weight_decay: float = 0.0

    @property
    def total_sampling_steps(self) -> int:
        return self.burn_in + self.draws_per_chain * self.thinning


@dataclass
class ReplayTrainConfig:
    """Configuration for replay-aware CPT training."""

    schedule: str = "mixed"
    replay_mode: str = "selected"
    replay_ratio: float = 0.2
    learning_rate: float = 5e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_length: int = 256
    weight_decay: float = 0.01
    warmup_ratio: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 0
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    deepspeed: str | None = None
    fsdp: str = ""
    fsdp_transformer_layer_cls_to_wrap: str | None = None
    seed: int = 42
