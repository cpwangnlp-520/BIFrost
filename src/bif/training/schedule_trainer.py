"""Replay-aware CPT training with HF Trainer.

Schedules:
  sequential — replay data first, then target data (no shuffle)
  mixed      — interleave replay and target data (shuffle)
  (replay_mode=none → target-only baseline, schedule is irrelevant)

Replay selection:
  selected    — top-K BIF-scored samples from PT pool
  random      — random samples from PT pool (ablation)
  top_random  — top-K samples, randomly ordered
  none        — no replay (baseline)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from bif.data.dataset import DataCollatorForLM, LMTextDataset
from bif.io import ensure_dir, read_jsonl, save_json
from bif.training.callbacks import (
    ReplayTrainer,
    _GradNormCallback,
    _LossLogCallback,
    _SwanLogCallback,
    can_load_best_model,
    estimate_total_steps,
    infer_interval,
    log_eval_step0,
)
from bif.utils.tracker import finish, init_run


def _build_replay_rows(
    pool_rows: list[dict[str, Any]],
    replay_n: int,
    mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    if replay_n > len(pool_rows):
        replay_n = len(pool_rows)
    if mode == "selected":
        return pool_rows[:replay_n]
    if mode == "top_random":
        top = list(pool_rows[:replay_n])
        random.Random(seed).shuffle(top)
        return top
    rng = random.Random(seed)
    idx = list(range(len(pool_rows)))
    rng.shuffle(idx)
    return [pool_rows[i] for i in idx[:replay_n]]


def _build_training_data(
    schedule: str,
    target_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[str] | None]:
    n_target = len(target_rows)
    n_replay = len(replay_rows)
    if n_replay == 0:
        return list(target_rows), None
    if schedule == "sequential":
        rows = list(replay_rows) + list(target_rows)
        labels = ["replay"] * n_replay + ["target"] * n_target
    elif schedule == "proportional":
        rows = list(target_rows) + list(replay_rows)
        labels = ["target"] * n_target + ["replay"] * n_replay
    else:
        rows = list(target_rows) + list(replay_rows)
        labels = ["target"] * n_target + ["replay"] * n_replay
        paired = list(zip(rows, labels))
        random.Random(seed).shuffle(paired)
        rows, labels = [p[0] for p in paired], [p[1] for p in paired]
    return rows, labels


def train_schedule_compare(
    base_model_path: str,
    tokenizer_path: str,
    target_train_jsonl: str,
    target_val_jsonl: str,
    replay_pool_jsonl: str,
    output_dir: str,
    run_name: str,
    schedule: str,
    replay_mode: str,
    replay_ratio: float,
    text_key: str = "text",
    max_length: int = 256,
    seed: int = 42,
    random_seed_for_replay: int = 1,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.01,
    lr_scheduler_type: str = "cosine",
    logging_steps: int = 10,
    eval_steps: int = 0,
    save_steps: int = 0,
    save_total_limit: int = 3,
    target_num_checkpoints: int = 6,
    bf16: bool = False,
    fp16: bool = False,
    gradient_checkpointing: bool = False,
    deepspeed: str | None = None,
    fsdp: str = "",
    fsdp_transformer_layer_cls_to_wrap: str | None = None,
    experiment_name: str = "replay_train",
    manage_tracking: bool = True,
    swanlab_run_id: str | None = None,
    metric_prefix: str = "",
) -> dict[str, Any]:
    set_seed(seed)
    ensure_dir(output_dir)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_rows = read_jsonl(target_train_jsonl)
    val_rows = read_jsonl(target_val_jsonl)

    if replay_mode == "none":
        replay_rows: list[dict[str, Any]] = []
    else:
        pool_rows = read_jsonl(replay_pool_jsonl)
        replay_n = int(round(len(target_rows) * replay_ratio))
        if replay_n > len(pool_rows):
            print(
                f"[replay] replay_n={replay_n} > pool_size={len(pool_rows)}, clamping"
            )
            replay_n = len(pool_rows)
        replay_rows = _build_replay_rows(
            pool_rows, replay_n, replay_mode, random_seed_for_replay
        )

    train_rows, group_labels = _build_training_data(
        schedule, target_rows, replay_rows, seed
    )
    n_target = len(target_rows)
    n_replay = len(replay_rows)

    collator = DataCollatorForLM(tokenizer, pad_to_multiple_of=8)
    train_ds = LMTextDataset(
        train_rows, tokenizer, max_length, text_key, group_labels=group_labels
    )
    val_ds = LMTextDataset(val_rows, tokenizer, max_length, text_key)

    num_gpus = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    total_steps = estimate_total_steps(
        len(train_ds),
        per_device_train_batch_size,
        gradient_accumulation_steps,
        num_gpus,
        num_train_epochs,
    )
    actual_eval = (
        eval_steps
        if eval_steps > 0
        else infer_interval(total_steps, target_num_checkpoints, 20)
    )
    actual_save = save_steps if save_steps > 0 else actual_eval

    cuda_available = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=(
            torch.bfloat16 if bf16 else (torch.float16 if fp16 else None)
        ),
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if cuda_available and not deepspeed and not fsdp:
        model = model.to(f"cuda:{local_rank}")

    history_cb = _LossLogCallback()
    callbacks = [history_cb, _GradNormCallback()]
    if local_rank == 0 and manage_tracking:
        if swanlab_run_id:
            import swanlab
            swanlab.init(
                project=os.environ.get("SWANLAB_PROJECT", "bif"),
                experiment_name=experiment_name,
                id=swanlab_run_id,
                resume="allow",
            )
        else:
            init_run(
                experiment_name=experiment_name,
                run_name=run_name,
                config={
                    "base_model": base_model_path,
                    "schedule": schedule,
                    "replay_mode": replay_mode,
                    "replay_ratio": replay_ratio,
                    "learning_rate": learning_rate,
                    "n_target": n_target,
                    "n_replay": n_replay,
                },
            )
        callbacks.append(_SwanLogCallback(metric_prefix=metric_prefix))

    load_best = can_load_best_model(deepspeed, fsdp)
    warmup_steps = int(total_steps * warmup_ratio)

    fsdp_config: dict[str, Any] = {}
    if fsdp:
        fsdp_config["fsdp_min_num_params"] = 1e8
        if fsdp_transformer_layer_cls_to_wrap:
            fsdp_config["fsdp_transformer_layer_cls_to_wrap"] = (
                fsdp_transformer_layer_cls_to_wrap
            )
        if gradient_checkpointing:
            fsdp_config["fsdp_activation_checkpointing"] = True

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=actual_eval,
        save_steps=actual_save,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16 and cuda_available,
        fp16=fp16 and cuda_available,
        use_cpu=not cuda_available,
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        deepspeed=deepspeed,
        fsdp=fsdp,
        fsdp_config=fsdp_config if fsdp_config else None,
    )

    trainer = ReplayTrainer(
        schedule=schedule,
        replay_ratio=replay_ratio,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    if schedule == "proportional" and n_replay > 0:
        trainer._setup_proportional_indices(n_target, n_replay)

    log_eval_step0(trainer, metric_prefix=metric_prefix)

    trainer.train()

    if not load_best and trainer.state.best_model_checkpoint:
        trainer._load_best_model()

    final_model_dir = f"{output_dir}/final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    metrics = trainer.evaluate()
    history_cb.save(output_dir)

    save_json(
        f"{output_dir}/run_config.json",
        {
            "base_model_path": base_model_path,
            "schedule": schedule,
            "replay_mode": replay_mode,
            "replay_ratio": replay_ratio,
            "learning_rate": learning_rate,
        },
    )
    save_json(
        f"{output_dir}/dataset_summary.json",
        {
            "target_train_examples": n_target,
            "target_val_examples": len(val_rows),
            "replay_examples_used": n_replay,
            "train_examples_total": len(train_rows),
            "schedule": schedule,
            "replay_mode": replay_mode,
            "replay_ratio": replay_ratio,
        },
    )
    save_json(f"{output_dir}/final_eval_metrics.json", metrics)
    save_json(
        f"{output_dir}/run_summary.json",
        {
            "run_name": run_name,
            "best_eval_loss": getattr(trainer.state, "best_metric", None),
            "global_step": int(trainer.state.global_step),
            "total_steps": total_steps,
            "schedule": schedule,
            "replay_mode": replay_mode,
            "replay_ratio": replay_ratio,
        },
    )

    if local_rank == 0 and manage_tracking:
        if swanlab_run_id:
            import swanlab
            if swanlab.get_run() is not None:
                swanlab.finish()
        else:
            finish()

    return {
        "run_name": run_name,
        "best_eval_loss": getattr(trainer.state, "best_metric", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay-aware CPT training.")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--target_train_jsonl", required=True)
    parser.add_argument("--target_val_jsonl", required=True)
    parser.add_argument("--replay_pool_jsonl", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument(
        "--schedule", required=True, choices=["sequential", "mixed"]
    )
    parser.add_argument(
        "--replay_mode",
        required=True,
        choices=["selected", "random", "top_random", "none"],
    )
    parser.add_argument("--replay_ratio", type=float, default=0.0)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_seed_for_replay", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--fsdp", default="")
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", default=None)
    parser.add_argument("--experiment_name", default="replay_train")
    parser.add_argument("--swanlab_run_id", default=None)
    parser.add_argument("--metric_prefix", default="")
    args = parser.parse_args()

    result = train_schedule_compare(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
