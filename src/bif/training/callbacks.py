"""Shared HF Trainer callbacks, base trainers, and training helpers for BIF CPT."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import torch
from transformers import Trainer, TrainerCallback

from bif.constants import IGNORE_INDEX


class _SwanLogCallback(TrainerCallback):
    """Minimal SwanLab logger: only loss/eval_loss/grad_norm/lr at integer steps.

    If ``metric_prefix`` is set (e.g. ``"seq_sel"``), logged keys become
    ``seq_sel/loss``, ``seq_sel/eval_loss`` etc. — allowing multiple
    training runs to share one SwanLab experiment.
    """

    _KEEP = {"loss", "eval_loss", "grad_norm", "learning_rate"}

    def __init__(self, metric_prefix: str = ""):
        self._prefix = f"{metric_prefix}/" if metric_prefix else ""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        try:
            import swanlab
            if swanlab.get_run() is None:
                return
        except Exception:
            return
        metrics = {}
        for k, v in logs.items():
            if k in self._KEEP:
                metrics[f"{self._prefix}{k}"] = v
        if metrics:
            swanlab.log(metrics, step=int(state.global_step))


class _GradNormCallback(TrainerCallback):
    """Buffer gradient norms on pre_optimizer_step, flush on log."""

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        if not hasattr(self, "_buffered_norms"):
            self._buffered_norms = []
        self._buffered_norms.append(total_norm)

    def on_log(self, args, state, control, logs=None, **kwargs):
        norms = getattr(self, "_buffered_norms", [])
        if not norms:
            return
        avg_norm = sum(norms) / len(norms)
        if logs is not None:
            logs["grad_norm"] = round(avg_norm, 4)
        self._buffered_norms = []


class _LossLogCallback(TrainerCallback):
    """Record training and evaluation loss history to JSON files."""

    def __init__(self) -> None:
        self.train_logs: list[dict[str, Any]] = []
        self.eval_logs: list[dict[str, Any]] = []
        self._seen_steps: set[int] = set()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record: dict[str, Any] = {"step": int(state.global_step)}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                record[k] = float(v)
        if "loss" in record and "eval_loss" not in record:
            self.train_logs.append(record)
            self._seen_steps.add(state.global_step)
        if "eval_loss" in record:
            self.eval_logs.append(record)

    def on_train_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer") or kwargs.get("model")
        gl = getattr(trainer, "_pending_group_losses", None) if trainer else None
        if gl and state.global_step not in self._seen_steps:
            record: dict[str, Any] = {"step": int(state.global_step)}
            record.update(gl)
            self.train_logs.append(record)

    def save(self, out_dir: str) -> None:
        from bif.io import ensure_dir
        logs_dir = f"{out_dir}/logs"
        ensure_dir(logs_dir)
        for name, data in [
            ("train_log_history.json", self.train_logs),
            ("eval_log_history.json", self.eval_logs),
        ]:
            with open(f"{logs_dir}/{name}", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


import random

from torch.utils.data import Sampler


class _ProportionalBatchSampler(Sampler):
    """Ensures each batch has the exact ratio of replay to target samples.

    Dataset layout: target samples at indices [0, target_size),
    replay samples at indices [target_size, target_size + replay_size).
    """

    def __init__(
        self,
        target_size: int,
        replay_size: int,
        batch_size: int,
        replay_ratio: float,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
    ):
        self.target_size = target_size
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.n_target = max(1, round(batch_size * (1 - replay_ratio)))
        self.n_replay = batch_size - self.n_target
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        per_rank_targets = target_size // world_size
        self.num_batches = max(1, per_rank_targets // self.n_target)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        target_indices = list(range(self.target_size))
        rng.shuffle(target_indices)

        per_rank = self.target_size // self.world_size
        start = self.rank * per_rank
        end = start + per_rank if self.rank < self.world_size - 1 else self.target_size
        target_indices = target_indices[start:end]

        replay_indices = list(range(self.target_size, self.target_size + self.replay_size))

        for batch_idx in range(self.num_batches):
            t_start = batch_idx * self.n_target
            t_end = min(t_start + self.n_target, len(target_indices))
            if t_end <= t_start:
                break
            batch_target = target_indices[t_start:t_end]
            batch_replay = rng.choices(replay_indices, k=self.n_replay)
            yield batch_target + batch_replay

    def __len__(self):
        return self.num_batches


class CPTTrainer(Trainer):
    """Base Trainer for all BIF CPT training.

    Key fix: sets ``model_accepts_loss_kwargs = False`` so that
    ``training_step`` correctly normalises loss by
    ``gradient_accumulation_steps``.  Without this, GPTNeoX's ``**kwargs``
    in ``forward`` causes HF 5.x to infer ``model_accepts_loss_kwargs=True``
    which skips the GA normalisation and produces inflated loss values.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class ReplayTrainer(CPTTrainer):
    """CPT Trainer with replay-aware loss tracking and schedule support.

    Schedules:
      sequential    — replay first, then target (no shuffle)
      mixed         — shuffle replay and target together (random per-batch ratio)
      proportional  — each batch has exact replay:target ratio (DDP-safe)
    """

    def __init__(self, schedule: str = "mixed", replay_ratio: float = 0.0, **kwargs):
        self._schedule = schedule
        self._replay_ratio = replay_ratio
        self._target_len = 0
        self._replay_len = 0
        super().__init__(**kwargs)

    def _setup_proportional_indices(self, target_len: int, replay_len: int) -> None:
        self._target_len = target_len
        self._replay_len = replay_len

    def get_train_dataloader(self):
        if self._schedule == "sequential":
            from torch.utils.data import DataLoader, DistributedSampler
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=False,
                drop_last=self.args.dataloader_drop_last,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        if self._schedule == "proportional" and self._target_len > 0 and self._replay_len > 0:
            from torch.utils.data import DataLoader
            batch_sampler = _ProportionalBatchSampler(
                target_size=self._target_len,
                replay_size=self._replay_len,
                batch_size=self.args.train_batch_size,
                replay_ratio=self._replay_ratio,
                world_size=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        return super().get_train_dataloader()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        import numpy as np
        from bif.training.loss import per_example_causal_lm_loss

        groups = inputs.pop("groups", None)
        outputs = model(**inputs)
        loss = outputs.loss

        if groups:
            with torch.no_grad():
                labels = inputs.get("labels")
                if labels is not None:
                    per_ex = per_example_causal_lm_loss(labels, outputs.logits)
                    target_mask = np.array([g == "target" for g in groups])
                    replay_mask = np.array([g == "replay" for g in groups])
                    per_ex_np = per_ex.detach().cpu().float().numpy()
                    extra: dict[str, float] = {}
                    if target_mask.any():
                        extra["train_loss_target"] = float(per_ex_np[target_mask].mean())
                    if replay_mask.any():
                        extra["train_loss_replay"] = float(per_ex_np[replay_mask].mean())
                    n_total = len(groups)
                    if n_total > 0:
                        extra["data_mixture_replay_frac"] = float(replay_mask.sum()) / n_total
                        extra["data_mixture_target_frac"] = float(target_mask.sum()) / n_total
                    if extra:
                        self._pending_group_losses = extra

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        gl = getattr(self, "_pending_group_losses", None)
        if gl:
            logs.update(gl)
            del self._pending_group_losses
        super().log(logs, start_time=start_time)


# ─── Shared training helpers ─────────────────────────────────────────────────


def estimate_total_steps(
    num_examples: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int,
    num_epochs: float,
) -> int:
    effective_batch = (
        per_device_batch_size * max(1, num_gpus) * gradient_accumulation_steps
    )
    steps_per_epoch = math.ceil(num_examples / effective_batch)
    return max(1, math.ceil(steps_per_epoch * num_epochs))


def infer_interval(total_steps: int, target_checkpoints: int, min_interval: int) -> int:
    return max(1, max(min_interval, total_steps // max(1, target_checkpoints)))


def can_load_best_model(deepspeed_config: str | None, fsdp: str) -> bool:
    if fsdp and "full_shard" in fsdp:
        return False
    if deepspeed_config:
        try:
            with open(deepspeed_config, encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("zero_optimization", {}).get("stage", 0) >= 3:
                return False
        except Exception:
            pass
    return True


def log_eval_step0(trainer, metric_prefix: str = "") -> None:
    """Evaluate and log step=0 baseline on all ranks (DDP-safe)."""
    metrics0 = trainer.evaluate()
    if trainer.args.local_process_index == 0:
        prefix = f"{metric_prefix}/" if metric_prefix else ""
        try:
            import swanlab
            if swanlab.get_run() is not None:
                swanlab.log({f"{prefix}eval_loss": metrics0["eval_loss"]}, step=0)
        except Exception:
            pass
