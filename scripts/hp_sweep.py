#!/usr/bin/env python3
"""Generate BIF hyperparameter sweep configs and run script.

Reference: https://timaeus.co/research/2026-04-21-sampling-guide

Phase 1: Coarse grid sweep over lr × gamma × nbeta
  - lr:     [1e-6, 1e-5, 1e-4]         (3 values)
  - gamma:  [1, 10, 100, 1000]          (4 values)
  - nbeta:  [1, 10, 100]                (3 values)
  → 36 combinations, RMSprop-SGLD, 2 chains × 200 draws

Phase 2: Fine sweep around best region from Phase 1

Phase 3: nβ=0 sanity check (verify loss gradient actually affects results)

Usage:
  python scripts/hp_sweep.py generate   # create configs + run script
  bash scripts/run_hp_sweep.sh          # execute sweep
  python scripts/hp_sweep.py analyze    # summarize results
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs" / "hp_sweep"
SCRIPTS_DIR = ROOT / "scripts"

# ─── User-configurable ────────────────────────────────────────────────────────
MODEL_PATH = "/workspace/pku_percy/models/pythia-70m-step1000"
POOL_JSONL = "data/pool_800.jsonl"
QUERY_JSONL = "data/pool_gsm8k_20.jsonl"  # finetune pool (for prepare-finetune)
SWEEP_BASE_DIR = "./runs/hp_sweep"

# Phase 1: coarse sweep
PHASE1_LR = [1e-6, 1e-5, 1e-4]
PHASE1_GAMMA = [1, 10, 100, 1000]
PHASE1_NBETA = [1, 10, 100]
PHASE1_DRAWS = 200
PHASE1_BURNIN = 50
PHASE1_CHAINS = 2
PHASE1_THINNING = 1

# Shared BIF params
SAMPLER_TYPE = "rmsprop_sgld"
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
BATCHES_PER_DRAW = 3
GRAD_ACCUM = 1
DTYPE = "bfloat16"
MAX_LENGTH = 128

# Train params (shared, only run once)
TRAIN_LR = 2e-4
TRAIN_EPOCHS = 1
TRAIN_BS = 8
TRAIN_MAX_LEN = 256
TRAIN_N_CKPT = 3


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


def _make_train_config() -> dict:
    return {
        "tokenizer_path": MODEL_PATH,
        "base_model_path": MODEL_PATH,
        "work_dir": f"{SWEEP_BASE_DIR}/_shared_train",
        "project_name": "BIFrost-HP-Sweep",
        "steps": {
            "build-pool": {
                "pool_jsonl": POOL_JSONL,
                "finetune_pool_jsonl": QUERY_JSONL,
            },
            "prepare-finetune": {
                "train_ratio": 0.5,
                "query_ratio": 0.2,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "min_chars": 10,
                "min_token_count": 10,
                "max_token_count": 4096,
                "min_int_score": 0,
                "min_language_score": 0.0,
            },
            "train": {
                "num_train_epochs": TRAIN_EPOCHS,
                "learning_rate": TRAIN_LR,
                "per_device_train_batch_size": TRAIN_BS,
                "per_device_eval_batch_size": TRAIN_BS,
                "gradient_accumulation_steps": 1,
                "max_length": TRAIN_MAX_LEN,
                "target_num_checkpoints": TRAIN_N_CKPT,
                "warmup_ratio": 0.03,
                "logging_steps": 5,
                "bf16": True,
                "gradient_checkpointing": True,
            },
            "run-bif": None,
            "analyze-bif": None,
            "extract-top": None,
            "schedule-compare": None,
            "schedule-analyze": None,
        },
    }


def _make_bif_config(lr: float, gamma: float, nbeta: float, phase: str) -> dict:
    name = _sweep_name(lr, gamma, nbeta)
    return {
        "tokenizer_path": MODEL_PATH,
        "base_model_path": MODEL_PATH,
        "work_dir": f"{SWEEP_BASE_DIR}/{phase}/{name}",
        "experiment_name": f"sweep-{phase}-{name}",
        "project_name": "BIFrost-HP-Sweep",
        "data_root": f"{SWEEP_BASE_DIR}/_shared_train",
        "data_root_redo": ["run-bif", "analyze-bif", "extract-top"],
        "steps": {
            "run-bif": {
                "sampler_type": SAMPLER_TYPE,
                "max_length": MAX_LENGTH,
                "num_chains": PHASE1_CHAINS,
                "draws_per_chain": PHASE1_DRAWS,
                "burn_in": PHASE1_BURNIN,
                "thinning": PHASE1_THINNING,
                "lr": lr,
                "gamma": gamma,
                "nbeta": nbeta,
                "nbeta_mode": "devinterp",
                "beta": 0.125,
                "noise_scale": 1.0,
                "train_batch_size": BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "batches_per_draw": BATCHES_PER_DRAW,
                "gradient_accumulation_steps": GRAD_ACCUM,
                "dtype": DTYPE,
            },
            "analyze-bif": {
                "score_col": "cross_cov_avg_over_queries",
                "top_k": 20,
            },
        },
    }


def _make_nb0_config(lr: float, gamma: float) -> dict:
    name = f"lr{_fmt_lr(lr)}_g{gamma}_nb0"
    return {
        "tokenizer_path": MODEL_PATH,
        "base_model_path": MODEL_PATH,
        "work_dir": f"{SWEEP_BASE_DIR}/phase3_sanity/{name}",
        "experiment_name": f"sweep-p3-{name}",
        "project_name": "BIFrost-HP-Sweep",
        "data_root": f"{SWEEP_BASE_DIR}/_shared_train",
        "data_root_redo": ["run-bif", "analyze-bif", "extract-top"],
        "steps": {
            "run-bif": {
                "sampler_type": SAMPLER_TYPE,
                "max_length": MAX_LENGTH,
                "num_chains": PHASE1_CHAINS,
                "draws_per_chain": PHASE1_DRAWS,
                "burn_in": PHASE1_BURNIN,
                "thinning": PHASE1_THINNING,
                "lr": lr,
                "gamma": gamma,
                "nbeta": 0.0001,
                "nbeta_mode": "devinterp",
                "beta": 0.125,
                "noise_scale": 1.0,
                "train_batch_size": BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "batches_per_draw": BATCHES_PER_DRAW,
                "gradient_accumulation_steps": GRAD_ACCUM,
                "dtype": DTYPE,
            },
            "analyze-bif": {
                "score_col": "cross_cov_avg_over_queries",
                "top_k": 20,
            },
        },
    }


def generate():
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # 0. Shared train config
    train_cfg = _make_train_config()
    train_path = CONFIGS_DIR / "00_shared_train.yaml"
    _write_yaml(train_cfg, train_path)
    print(f"[gen] {train_path}")

    # 1. Phase 1: coarse sweep
    phase1_configs = []
    for lr in PHASE1_LR:
        for gamma in PHASE1_GAMMA:
            for nbeta in PHASE1_NBETA:
                cfg = _make_bif_config(lr, gamma, nbeta, "phase1_coarse")
                name = _sweep_name(lr, gamma, nbeta)
                path = CONFIGS_DIR / f"phase1_{name}.yaml"
                _write_yaml(cfg, path)
                phase1_configs.append((name, path))
                print(f"[gen] {path}")

    # 3. Phase 3: nβ≈0 sanity check (run alongside phase1 for best few)
    # We'll generate for the medium lr/gamma combinations
    nb0_configs = []
    for lr in [1e-5]:
        for gamma in [10, 100, 1000]:
            cfg = _make_nb0_config(lr, gamma)
            name = f"lr{_fmt_lr(lr)}_g{gamma}_nb0"
            path = CONFIGS_DIR / f"phase3_{name}.yaml"
            _write_yaml(cfg, path)
            nb0_configs.append((name, path))
            print(f"[gen] {path}")

    # Generate run script
    _gen_run_script(phase1_configs, nb0_configs)

    total = len(phase1_configs) + len(nb0_configs)
    print(f"\n[gen] Done: {total} configs generated in {CONFIGS_DIR}/")
    print(f"[gen] Run script: scripts/run_hp_sweep.sh")


def _write_yaml(cfg: dict, path: Path):
    import yaml

    # Remove None steps (write null)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _gen_run_script(
    phase1: list[tuple[str, Path]],
    nb0: list[tuple[str, Path]],
):
    lines = [
        "#!/bin/bash",
        "# BIFrost Hyperparameter Sweep",
        "# Reference: https://timaeus.co/research/2026-04-21-sampling-guide",
        "#",
        "# Phase 1: Coarse grid over lr × gamma × nbeta (loss trace diagnostics)",
        "# Phase 3: nβ≈0 sanity check (verify gradient affects results)",
        "#",
        "# Usage:",
        "#   bash scripts/run_hp_sweep.sh              # run all",
        "#   bash scripts/run_hp_sweep.sh train        # train only",
        "#   bash scripts/run_hp_sweep.sh phase1       # phase1 only",
        "#   bash scripts/run_hp_sweep.sh phase3       # phase3 only",
        "#   bash scripts/run_hp_sweep.sh phase1 0 3   # phase1 indices 0..2",
        "",
        "set -euo pipefail",
        "",
        'ROOT="$(cd "$(dirname "$0")/.." && pwd)"',
        "cd \"$ROOT\"",
        "",
        "MODE=${1:-all}",
        "START_IDX=${2:-0}",
        "END_IDX=${3:-999999}",
        "",
        "export SWANLAB_API_KEY=${SWANLAB_API_KEY:-}",
        "export SWANLAB_MODE=${SWANLAB_MODE:-disabled}",
        "",
        "# ─── Step 0: Train model (shared across all sweep runs) ────────────────",
        "train_model() {",
        "  if [ -d ./runs/hp_sweep/_shared_train/train/final_model ]; then",
        '    echo "[sweep] Model already trained, skipping."',
        "  else",
        '    echo "[sweep] Training model..."',
        "    python -m bif.cli pipeline run --config configs/hp_sweep/00_shared_train.yaml",
        "  fi",
        "}",
        "",
        "# ─── Phase 1: Coarse sweep ─────────────────────────────────────────────",
    ]

    phase1_paths = [str(p) for _, p in phase1]
    for i, (name, path) in enumerate(phase1):
        lines.append(f"run_phase1_{i}() {{")
        lines.append(f'  echo "[sweep] Phase1 [{i}/{len(phase1)}]: {name}"')
        lines.append(
            f"  python -m bif.cli pipeline run --config {path} --from run-bif"
        )
        lines.append("}")
    lines.append("")

    lines.append("run_phase1() {")
    lines.append(f"  total={len(phase1)}")
    lines.append('  echo "[sweep] Phase 1: $total combinations"')
    lines.append("  for i in $(seq $START_IDX $((END_IDX < total ? END_IDX : total - 1))); do")
    lines.append("    run_phase1_$i")
    lines.append("  done")
    lines.append("}")
    lines.append("")

    # Phase 3
    lines.append("# ─── Phase 3: nβ≈0 sanity check ──────────────────────────────────────")
    for i, (name, path) in enumerate(nb0):
        lines.append(f"run_phase3_{i}() {{")
        lines.append(f'  echo "[sweep] Phase3 sanity: {name}"')
        lines.append(
            f"  python -m bif.cli pipeline run --config {path} --from run-bif"
        )
        lines.append("}")
    lines.append("")

    lines.append("run_phase3() {")
    for i, (name, _) in enumerate(nb0):
        lines.append(f"  run_phase3_{i}")
    lines.append("}")
    lines.append("")

    # Main dispatch
    lines.append('case "$MODE" in')
    lines.append("  train)   train_model ;;")
    lines.append("  phase1)  train_model; run_phase1 ;;")
    lines.append("  phase3)  train_model; run_phase3 ;;")
    lines.append("  all)     train_model; run_phase1; run_phase3 ;;")
    lines.append('  *)       echo "Usage: $0 {train|phase1|phase3|all} [start_idx] [end_idx]" ;;')
    lines.append("esac")

    script_path = SCRIPTS_DIR / "run_hp_sweep.sh"
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(script_path, 0o755)


def analyze():
    """Summarize sweep results from loss traces."""
    import numpy as np

    base = Path(SWEEP_BASE_DIR)
    if not base.exists():
        print(f"No sweep results found at {base}")
        return

    rows = []
    for phase_dir in sorted(base.glob("phase*")):
        for run_dir in sorted(phase_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            # Find chain dirs
            for ckpt_dir in sorted(run_dir.glob("bif_traces/*/")):
                for chain_dir in sorted(ckpt_dir.glob("chain_*/")):
                    trace_path = chain_dir / "observable_loss_trace.npz"
                    if not trace_path.exists():
                        continue
                    try:
                        data = np.load(trace_path)
                        seq_loss = data["seq_loss"]  # (draws, samples)
                        mean_loss = seq_loss.mean(axis=1)  # (draws,)
                        rows.append(
                            {
                                "run": run_dir.name,
                                "phase": phase_dir.name,
                                "checkpoint": ckpt_dir.name,
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
                            }
                        )
                    except Exception as e:
                        print(f"  [warn] {trace_path}: {e}")

    if not rows:
        print("No trace data found.")
        return

    import pandas as pd

    df = pd.DataFrame(rows)

    print(f"\n{'='*80}")
    print(f"BIF Hyperparameter Sweep — {len(df)} chains analyzed")
    print(f"{'='*80}\n")

    # Per-run summary (average across chains)
    agg = (
        df.groupby("run")
        .agg(
            phase=("phase", "first"),
            n_chains=("chain", "count"),
            loss_mean=("loss_mean", "mean"),
            loss_std=("loss_std", "mean"),
            loss_range=("loss_range", "mean"),
            loss_trend=("loss_trend", "mean"),
            loss_start=("loss_start", "mean"),
            loss_end=("loss_end", "mean"),
        )
        .reset_index()
    )

    # Diagnostics from the guide:
    # ✓ loss should increase monotonically (trend > 0)
    # ✓ loss should level off (range not too large)
    # ✓ good signal-to-noise (mean >> std)
    # ✓ chains should agree (low std across chains)

    agg["snr"] = agg["loss_range"] / (agg["loss_std"] + 1e-12)
    agg["quality"] = "bad"
    agg.loc[
        (agg["loss_trend"] > 0) & (agg["snr"] > 0.5), "quality"
    ] = "ok"
    agg.loc[
        (agg["loss_trend"] > 0) & (agg["snr"] > 1.0) & (agg["loss_range"] < 2.0),
        "quality",
    ] = "good"

    # Sort by quality then SNR
    quality_order = {"good": 0, "ok": 1, "bad": 2}
    agg["_q"] = agg["quality"].map(quality_order)
    agg = agg.sort_values(["_q", "snr"], ascending=[True, False])

    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 200)
    print(agg[["run", "phase", "quality", "loss_mean", "loss_range", "snr", "loss_trend"]].to_string(index=False))

    good = agg[agg["quality"] == "good"]
    if len(good) > 0:
        print(f"\n{'='*80}")
        print(f"RECOMMENDED configs ({len(good)} good):")
        print(f"{'='*80}")
        for _, r in good.iterrows():
            print(f"  {r['run']:30s}  loss={r['loss_mean']:.4f}  range={r['loss_range']:.4f}  snr={r['snr']:.2f}")
    else:
        ok = agg[agg["quality"] == "ok"]
        print(f"\nNo 'good' configs found. {len(ok)} 'ok' configs available.")

    # Save CSV
    csv_path = base / "sweep_summary.csv"
    agg.drop(columns=["_q"]).to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")


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
        print(f"Unknown command: {cmd}")
        print("Usage: python hp_sweep.py {generate|analyze}")
        sys.exit(1)
