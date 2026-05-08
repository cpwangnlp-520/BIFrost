#!/usr/bin/env python3
"""Generate BIF hyperparameter sweep configs and run script.

Reference: https://timaeus.co/research/2026-04-21-sampling-guide

Design:
  - Use run-bif directly (not pipeline) for flexibility
  - pool_800.jsonl = PT pool (800 samples, 5 sources)
  - query_gsm8k_20.jsonl = 20 GSM8K questions (query set)
  - Single trained model (pythia-70m-step1000)
  - Phase 1: coarse grid lr × gamma × nbeta
  - Phase 3: nbeta≈0 sanity check

Usage:
  python scripts/hp_sweep.py generate   # create run script
  bash scripts/tmux_sweep.sh            # execute on 8 GPUs
  python scripts/hp_sweep.py analyze    # summarize results
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"

MODEL_PATH = "/workspace/pku_percy/models/pythia-70m-step1000"
POOL_JSONL = "data/pool_800.jsonl"
QUERY_JSONL = "data/query_gsm8k_20_answer_only.jsonl"
SWEEP_BASE_DIR = "./runs/hp_sweep"

PHASE1_LR = [1e-6, 1e-5, 1e-4]
PHASE1_GAMMA = [1, 10, 100, 1000]
PHASE1_NBETA = [1, 10, 100]

DRAWS = 500
BURNIN = 100
CHAINS = 3
THINNING = 2
MAX_LENGTH = 512

SAMPLER_TYPE = "rmsprop_sgld"
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 64
BATCHES_PER_DRAW = 3
GRAD_ACCUM = 1
DTYPE = "bfloat16"


def _fmt_lr(lr: float) -> str:
    if lr == 0:
        return "0"
    e = int(math.log10(lr))
    m = lr / 10**e
    if m == 1.0:
        return f"1e{e}"
    return f"{m:.0f}e{e}"


def _fmt_nb(nb: float) -> str:
    if nb == int(nb):
        return str(int(nb))
    return f"{nb:.1f}"


def _sweep_name(lr: float, gamma: float, nbeta: float) -> str:
    return f"lr{_fmt_lr(lr)}_g{gamma}_nb{_fmt_nb(nbeta)}"


def _run_bif_cmd(lr, gamma, nbeta, out_dir, experiment_name):
    return (
        f"python -m bif.cli run-bif"
        f"  --model_name_or_path {MODEL_PATH}"
        f"  --pool_jsonl {POOL_JSONL}"
        f"  --query_jsonl {QUERY_JSONL}"
        f"  --out_dir {out_dir}"
        f"  --sampler_type {SAMPLER_TYPE}"
        f"  --num_chains {CHAINS}"
        f"  --draws_per_chain {DRAWS}"
        f"  --num_burnin_steps {BURNIN}"
        f"  --num_steps_bw_draws {THINNING}"
        f"  --train_batch_size {BATCH_SIZE}"
        f"  --eval_batch_size {EVAL_BATCH_SIZE}"
        f"  --batches_per_draw {BATCHES_PER_DRAW}"
        f"  --gradient_accumulation_steps {GRAD_ACCUM}"
        f"  --lr {lr}"
        f"  --gamma {gamma}"
        f"  --nbeta {nbeta}"
        f"  --nbeta_mode devinterp"
        f"  --beta 0.125"
        f"  --dtype {DTYPE}"
        f"  --max_length {MAX_LENGTH}"
        f"  --experiment_name {experiment_name}"
    )


def _analyze_cmd(bif_root, out_dir, experiment_name):
    analyze_exp_name = f"analyze-{experiment_name}"
    return (
        f"python -m bif.cli analyze-bif"
        f"  --bif_root {bif_root}"
        f"  --out_dir {out_dir}"
        f"  --score_col cross_cov_avg_over_queries"
        f"  --top_k 20"
        f"  --experiment_name {analyze_exp_name}"
    )


def generate():
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = []

    # Phase 1: coarse sweep
    for lr in PHASE1_LR:
        for gamma in PHASE1_GAMMA:
            for nbeta in PHASE1_NBETA:
                name = _sweep_name(lr, gamma, nbeta)
                out_dir = f"{SWEEP_BASE_DIR}/phase1/{name}"
                exp_name = f"sweep-p1-{name}"
                configs.append((
                    f"phase1/{name}",
                    _run_bif_cmd(lr, gamma, nbeta, out_dir, exp_name),
                    _analyze_cmd(out_dir, f"{out_dir}/analysis", exp_name),
                ))

    # Phase 3: nbeta≈0 sanity check
    for gamma in [10, 100, 1000]:
        lr = 1e-5
        name = f"lr{_fmt_lr(lr)}_g{gamma}_nb0"
        out_dir = f"{SWEEP_BASE_DIR}/phase3/{name}"
        exp_name = f"sweep-p3-{name}"
        configs.append((
            f"phase3/{name}",
            _run_bif_cmd(lr, gamma, 0.0001, out_dir, exp_name),
            _analyze_cmd(out_dir, f"{out_dir}/analysis", exp_name),
        ))

    _gen_tmux_script(configs)

    n_p1 = len(PHASE1_LR) * len(PHASE1_GAMMA) * len(PHASE1_NBETA)
    n_p3 = len([10, 100, 1000])
    print(f"[gen] {n_p1} phase1 + {n_p3} phase3 = {len(configs)} configs")
    print(f"[gen] Per config: {CHAINS} chains × {DRAWS} draws, max_length={MAX_LENGTH}")
    print(f"[gen] Run: bash scripts/tmux_sweep.sh")


def _gen_tmux_script(configs):
    N_GPUS = 8
    SESSION = "bif-sweep"
    API_KEY = "RGvhmcyaE940jILdlzCGg"

    gpu_scripts_dir = ROOT / "scripts" / "gpu_scripts"
    gpu_scripts_dir.mkdir(parents=True, exist_ok=True)

    for gpu in range(N_GPUS):
        tasks = [c for i, c in enumerate(configs) if i % N_GPUS == gpu]
        if not tasks:
            continue

        script_lines = [
            "#!/bin/bash",
            f"# GPU {gpu}: {len(tasks)} configs",
            "set -euo pipefail",
            f'ROOT="$(cd "$(dirname "$0")/../.." && pwd)"',
            "cd $ROOT",
            f"export CUDA_VISIBLE_DEVICES={gpu}",
            f"export SWANLAB_API_KEY={API_KEY}",
            "",
        ]

        for label, run_cmd, analyze_cmd in tasks:
            script_lines.append(f'echo "\\n=== GPU {gpu}: {label} ==="')
            script_lines.append(f"mkdir -p $(dirname {SWEEP_BASE_DIR}/{label})")
            script_lines.append(f"{run_cmd} || echo '[WARN] run-bif FAILED for {label}'")
            script_lines.append(f"{analyze_cmd} || echo '[WARN] analyze-bif FAILED for {label}'")
            script_lines.append("")

        script_lines.append(f'echo "\\n=== GPU {gpu}: all done ==="')

        script_path = gpu_scripts_dir / f"gpu{gpu}.sh"
        script_path.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
        os.chmod(script_path, 0o755)

    # Main tmux launcher
    lines = [
        "#!/bin/bash",
        "# BIF Hyperparameter Sweep — 8 GPU tmux launcher",
        "#",
        f"# {len(configs)} configs, {N_GPUS} GPUs",
        f"# Per config: {CHAINS} chains × {DRAWS} draws, max_len={MAX_LENGTH}",
        "#",
        "# Usage:",
        "#   bash scripts/tmux_sweep.sh          # launch",
        "#   bash scripts/tmux_sweep.sh attach   # watch",
        "#   bash scripts/tmux_sweep.sh kill     # stop all",
        "",
        "set -euo pipefail",
        "",
        f'SESSION="{SESSION}"',
        'ROOT="$(cd "$(dirname "$0")/.." && pwd)"',
        'cd "$ROOT"',
        "",
        'if [ "${1:-}" = "kill" ]; then',
        f"  tmux kill-session -t $SESSION 2>/dev/null || true",
        '  echo "Session killed."',
        "  exit 0",
        "fi",
        "",
        'if [ "${1:-}" = "attach" ]; then',
        f"  tmux attach -t $SESSION",
        "  exit 0",
        "fi",
        "",
        f"tmux new-session -d -s $SESSION -c $ROOT",
        "",
    ]

    for gpu in range(N_GPUS):
        tasks = [c for i, c in enumerate(configs) if i % N_GPUS == gpu]
        if not tasks:
            continue
        rel_path = f"scripts/gpu_scripts/gpu{gpu}.sh"
        if gpu == 0:
            lines.append(f'tmux send-keys -t $SESSION:0 "bash {rel_path}" Enter')
            lines.append(f'tmux rename-window -t $SESSION:0 "gpu0"')
        else:
            lines.append(f'tmux new-window -t $SESSION -n "gpu{gpu}" -c $ROOT')
            lines.append(f'tmux send-keys -t $SESSION:gpu{gpu} "bash {rel_path}" Enter')

        print(f"  GPU {gpu}: {len(tasks)} configs — {', '.join(label for label,_,_ in tasks)}")

    lines += [
        "",
        'echo ""',
        f'echo "=== Session $SESSION launched ==="',
        f'echo "  tmux attach -t $SESSION"',
        f'echo "  bash $0 kill"',
    ]

    script_path = SCRIPTS_DIR / "tmux_sweep.sh"
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(script_path, 0o755)


def analyze():
    import numpy as np
    import pandas as pd

    base = Path(SWEEP_BASE_DIR)
    if not base.exists():
        print(f"No sweep results at {base}")
        return

    rows = []
    for phase_dir in sorted(base.glob("phase*")):
        for run_dir in sorted(phase_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            for chain_dir in sorted(run_dir.glob("chain_*/")):
                trace_path = chain_dir / "observable_loss_trace.npz"
                if not trace_path.exists():
                    continue
                try:
                    data = np.load(trace_path)
                    seq_loss = data["seq_loss"]
                    mean_loss = seq_loss.mean(axis=1)
                    rows.append({
                        "run": run_dir.name,
                        "phase": phase_dir.name,
                        "chain": chain_dir.name,
                        "n_draws": len(mean_loss),
                        "loss_start": float(mean_loss[0]),
                        "loss_end": float(mean_loss[-1]),
                        "loss_mean": float(mean_loss.mean()),
                        "loss_std": float(mean_loss.std()),
                        "loss_min": float(mean_loss.min()),
                        "loss_max": float(mean_loss.max()),
                        "loss_range": float(mean_loss.max() - mean_loss.min()),
                        "loss_trend": float(mean_loss[-1] - mean_loss[0]),
                    })
                except Exception as e:
                    print(f"  [warn] {trace_path}: {e}")

    if not rows:
        print("No trace data found.")
        return

    df = pd.DataFrame(rows)
    agg = df.groupby("run").agg(
        phase=("phase", "first"),
        n_chains=("chain", "count"),
        loss_mean=("loss_mean", "mean"),
        loss_std=("loss_std", "mean"),
        loss_range=("loss_range", "mean"),
        loss_trend=("loss_trend", "mean"),
    ).reset_index()

    agg["snr"] = agg["loss_range"] / (agg["loss_std"] + 1e-12)
    agg["quality"] = "bad"
    agg.loc[(agg["loss_trend"] > 0) & (agg["snr"] > 0.5), "quality"] = "ok"
    agg.loc[
        (agg["loss_trend"] > 0) & (agg["snr"] > 1.0) & (agg["loss_range"] < 5.0),
        "quality",
    ] = "good"

    quality_order = {"good": 0, "ok": 1, "bad": 2}
    agg["_q"] = agg["quality"].map(quality_order)
    agg = agg.sort_values(["_q", "snr"], ascending=[True, False])

    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 200)
    print(f"\n{'='*80}")
    print(f"BIF Sweep Results — {len(df)} chains")
    print(f"{'='*80}\n")
    print(agg[["run", "phase", "quality", "loss_mean", "loss_range", "snr", "loss_trend"]].to_string(index=False))

    good = agg[agg["quality"] == "good"]
    if len(good) > 0:
        print(f"\n{'='*80}")
        print(f"RECOMMENDED ({len(good)} good):")
        for _, r in good.iterrows():
            print(f"  {r['run']:30s}  loss={r['loss_mean']:.4f}  range={r['loss_range']:.4f}  snr={r['snr']:.2f}")

    csv_path = base / "sweep_summary.csv"
    agg.drop(columns=["_q"]).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "generate":
        generate()
    elif cmd == "analyze":
        analyze()
    else:
        print(f"Unknown: {cmd}")
        sys.exit(1)
