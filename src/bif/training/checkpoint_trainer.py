"""Stage-2 LM fine-tuning with checkpoint management."""

from __future__ import annotations

import argparse
import json
import os
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
    CPTTrainer,
    _GradNormCallback,
    _LossLogCallback,
    _SwanLogCallback,
    can_load_best_model,
    estimate_total_steps,
    infer_interval,
    log_eval_step0,
)
from bif.utils.tracker import finish, init_run


def train_with_checkpoints(
    base_model_path: str,
    tokenizer_path: str,
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str,
    text_key: str = "text",
    max_length: int = 512,
    seed: int = 42,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    logging_steps: int = 10,
    target_num_checkpoints: int = 6,
    min_save_steps: int = 20,
    min_eval_steps: int = 20,
    save_steps: int = 0,
    eval_steps: int = 0,
    save_total_limit: int = 6,
    gradient_checkpointing: bool = False,
    bf16: bool = False,
    fp16: bool = False,
    deepspeed: str | None = None,
    fsdp: str = "",
    fsdp_transformer_layer_cls_to_wrap: str | None = None,
    experiment_name: str = "stage2_train",
    run_name: str | None = None,
    manage_tracking: bool = True,
) -> dict[str, Any]:
    set_seed(seed)
    ensure_dir(output_dir)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    train_rows = read_jsonl(train_jsonl)
    val_rows = read_jsonl(val_jsonl)
    train_ds = LMTextDataset(train_rows, tokenizer, max_length, text_key)
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
        else infer_interval(total_steps, target_num_checkpoints, min_eval_steps)
    )
    actual_save = save_steps if save_steps > 0 else actual_eval
    load_best = can_load_best_model(deepspeed, fsdp)

    cuda_available = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=(torch.bfloat16 if bf16 else (torch.float16 if fp16 else None)),
    )
    if cuda_available and not deepspeed and not fsdp:
        model = model.to(f"cuda:{local_rank}")

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    collator = DataCollatorForLM(tokenizer, pad_to_multiple_of=8)
    history_cb = _LossLogCallback()
    callbacks: list = [history_cb, _GradNormCallback()]
    if local_rank == 0 and manage_tracking:
        init_run(
            experiment_name=experiment_name,
            run_name=run_name,
            config={
                "base_model": base_model_path,
                "learning_rate": learning_rate,
                "epochs": num_train_epochs,
                "batch_size": per_device_train_batch_size,
                "gradient_accumulation": gradient_accumulation_steps,
                "max_length": max_length,
                "bf16": bf16,
                "fp16": fp16,
            },
        )
        callbacks.append(_SwanLogCallback())

    use_bf16 = bf16 and cuda_available
    use_fp16 = fp16 and cuda_available
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
        bf16=use_bf16,
        fp16=use_fp16,
        use_cpu=not cuda_available,
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        deepspeed=deepspeed,
        fsdp=fsdp,
        fsdp_config=fsdp_config if fsdp_config else None,
    )

    trainer = CPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    log_eval_step0(trainer)

    trainer.train()

    if not load_best and trainer.state.best_model_checkpoint:
        trainer._load_best_model()

    final_model_dir = f"{output_dir}/final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    metrics = trainer.evaluate()
    history_cb.save(output_dir)

    summary = {
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "best_metric": getattr(trainer.state, "best_metric", None),
        "global_step": int(trainer.state.global_step),
        "total_steps": total_steps,
        "save_steps": actual_save,
        "eval_steps": actual_eval,
        "deepspeed": deepspeed,
        "fsdp": fsdp,
    }
    save_json(f"{output_dir}/logs/run_summary.json", summary)
    save_json(f"{output_dir}/logs/final_eval_metrics.json", metrics)

    if local_rank == 0 and manage_tracking:
        finish()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train stage-2 LM with checkpoint saves."
    )
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--target_num_checkpoints", type=int, default=6)
    parser.add_argument("--min_save_steps", type=int, default=20)
    parser.add_argument("--min_eval_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=6)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--fsdp", default="")
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", default=None)
    parser.add_argument("--experiment_name", default="stage2_train")
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    summary = train_with_checkpoints(**vars(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
