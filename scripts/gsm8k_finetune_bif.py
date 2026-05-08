#!/usr/bin/env python3
"""Finetune Pythia-70M on GSM8K, run BIF on checkpoints, analyze source shift.

Uses BIFrost framework's `train` command with --loss_mode response_only.
All steps log to SwanLab project: BIFrost-gsm8k-finetune

Step 1: Finetune on GSM8K, 16 epochs, save every 2 epochs → 8 ckpts
Step 2: Run BIF on all 8 checkpoints using best sweep config
Step 3: Analyze BIF results — source shift across finetune checkpoints

Usage:
  export SWANLAB_API_KEY=...
  python scripts/gsm8k_finetune_bif.py              # all steps
  python scripts/gsm8k_finetune_bif.py finetune      # step 1 only
  python scripts/gsm8k_finetune_bif.py run-bif       # step 2 only
  python scripts/gsm8k_finetune_bif.py analyze       # step 3 only
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = "/workspace/pku_percy/models/pythia-70m-step1000"
GSM8K_TRAIN = "/workspace/pku_percy/data/gsm8k/main/train-00000-of-00001.parquet"
GSM8K_TEST = "/workspace/pku_percy/data/gsm8k/main/test-00000-of-00001.parquet"
OUT_BASE = ROOT / "runs" / "gsm8k_finetune"
POOL_JSONL = "data/pool_800.jsonl"
QUERY_JSONL = "data/query_gsm8k_20_answer_only.jsonl"
SWANLAB_PROJECT = "BIFrost-gsm8k-finetune"
SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", "RGvhmcyaE940jILdlzCGg")


def finetune():
    print("=" * 60)
    print("Step 1: Finetuning Pythia-70M on GSM8K (response_only)")
    print("=" * 60)

    output_dir = str(OUT_BASE / "finetune")
    os.makedirs(output_dir, exist_ok=True)

    cmd = (
        f"python -m bif.cli train"
        f"  --base_model_path {MODEL_PATH}"
        f"  --tokenizer_path {MODEL_PATH}"
        f"  --train_jsonl {GSM8K_TRAIN}"
        f"  --val_jsonl {GSM8K_TEST}"
        f"  --output_dir {output_dir}"
        f"  --loss_mode response_only"
        f"  --prompt_key question"
        f"  --response_key answer"
        f"  --num_train_epochs 16"
        f"  --learning_rate 3e-5"
        f"  --per_device_train_batch_size 16"
        f"  --per_device_eval_batch_size 16"
        f"  --gradient_accumulation_steps 2"
        f"  --max_length 512"
        f"  --target_num_checkpoints 8"
        f"  --bf16"
        f"  --gradient_checkpointing"
        f"  --warmup_ratio 0.05"
        f"  --logging_steps 20"
        f"  --experiment_name gsm8k-finetune-70m"
    )

    env = os.environ.copy()
    env["SWANLAB_API_KEY"] = SWANLAB_API_KEY
    env["SWANLAB_PROJECT"] = SWANLAB_PROJECT

    ret = os.system(cmd)
    if ret != 0:
        print("[WARN] Finetune failed")
    else:
        print(f"[done] Checkpoints: {sorted(os.listdir(output_dir))}")


def run_bif():
    print("\n" + "=" * 60)
    print("Step 2: Running BIF on all finetune checkpoints")
    print("=" * 60)

    finetune_dir = str(OUT_BASE / "finetune")
    ckpt_dirs = sorted([
        d for d in os.listdir(finetune_dir)
        if d.startswith("checkpoint-") or d == "final_model"
    ])
    print(f"Found {len(ckpt_dirs)} checkpoints: {ckpt_dirs}")

    bif_out = str(OUT_BASE / "bif")
    os.makedirs(bif_out, exist_ok=True)

    for ckpt_name in ckpt_dirs:
        ckpt_path = f"{finetune_dir}/{ckpt_name}"
        out_dir = f"{bif_out}/{ckpt_name}"
        exp_name = f"gsm8k-bif-{ckpt_name}"

        if os.path.isfile(f"{out_dir}/chain_000/observable_loss_trace.npz"):
            print(f"  SKIP: {ckpt_name}")
            continue

        print(f"\n--- BIF on {ckpt_name} ---")
        cmd = (
            f"python -m bif.cli run-bif"
            f"  --model_name_or_path {ckpt_path}"
            f"  --pool_jsonl {POOL_JSONL}"
            f"  --query_jsonl {QUERY_JSONL}"
            f"  --out_dir {out_dir}"
            f"  --sampler_type rmsprop_sgld"
            f"  --num_chains 3 --draws_per_chain 500"
            f"  --num_burnin_steps 100 --num_steps_bw_draws 1"
            f"  --train_batch_size 32 --eval_batch_size 256"
            f"  --batches_per_draw 3 --gradient_accumulation_steps 1"
            f"  --lr 1e-6 --gamma 100 --nbeta 10 --nbeta_mode devinterp"
            f"  --beta 0.125 --rmsprop_eps 1e-4"
            f"  --dtype bfloat16 --max_length 512"
            f"  --experiment_name {exp_name}"
        )

        env = os.environ.copy()
        env["SWANLAB_API_KEY"] = SWANLAB_API_KEY
        env["SWANLAB_PROJECT"] = SWANLAB_PROJECT

        ret = os.system(cmd)
        if ret != 0:
            print(f"  [WARN] run-bif failed for {ckpt_name}")
            continue

        analyze_one(ckpt_name, out_dir)


def analyze_one(ckpt_name, bif_dir):
    out_dir = f"{bif_dir}/analysis"
    exp_name = f"gsm8k-analyze-{ckpt_name}"

    print(f"  Analyzing {ckpt_name}...")
    cmd = (
        f"CUDA_VISIBLE_DEVICES='' python -m bif.cli analyze-bif"
        f"  --bif_root {bif_dir}"
        f"  --out_dir {out_dir}"
        f"  --score_col cross_cov_avg_over_queries"
        f"  --top_k 150"
        f"  --experiment_name {exp_name}"
    )

    env = os.environ.copy()
    env["SWANLAB_API_KEY"] = SWANLAB_API_KEY
    env["SWANLAB_PROJECT"] = SWANLAB_PROJECT

    ret = os.system(cmd)
    if ret != 0:
        print(f"  [WARN] analyze failed for {ckpt_name}")


def analyze():
    print("\n" + "=" * 60)
    print("Step 3: Analyzing BIF results")
    print("=" * 60)

    bif_dir = str(OUT_BASE / "bif")
    if not os.path.isdir(bif_dir):
        print("No BIF results. Run 'run-bif' first.")
        return

    for ckpt_name in sorted(os.listdir(bif_dir)):
        ckpt_bif_dir = f"{bif_dir}/{ckpt_name}"
        if not os.path.isdir(ckpt_bif_dir):
            continue
        analyze_one(ckpt_name, ckpt_bif_dir)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    os.environ["SWANLAB_API_KEY"] = SWANLAB_API_KEY
    os.environ["SWANLAB_PROJECT"] = SWANLAB_PROJECT

    if cmd == "finetune":
        finetune()
    elif cmd == "run-bif":
        run_bif()
    elif cmd == "analyze":
        analyze()
    elif cmd == "all":
        finetune()
        run_bif()
    else:
        print(f"Unknown: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
