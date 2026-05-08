#!/bin/bash
# BIFrost Hyperparameter Sweep
# Reference: https://timaeus.co/research/2026-04-21-sampling-guide
#
# Phase 1: Coarse grid over lr × gamma × nbeta (loss trace diagnostics)
# Phase 3: nβ≈0 sanity check (verify gradient affects results)
#
# Usage:
#   bash scripts/run_hp_sweep.sh              # run all
#   bash scripts/run_hp_sweep.sh train        # train only
#   bash scripts/run_hp_sweep.sh phase1       # phase1 only
#   bash scripts/run_hp_sweep.sh phase3       # phase3 only
#   bash scripts/run_hp_sweep.sh phase1 0 3   # phase1 indices 0..2

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE=${1:-all}
START_IDX=${2:-0}
END_IDX=${3:-999999}

export SWANLAB_API_KEY=${SWANLAB_API_KEY:-}
export SWANLAB_MODE=${SWANLAB_MODE:-disabled}

# ─── Step 0: Train model (shared across all sweep runs) ────────────────
train_model() {
  if [ -d ./runs/hp_sweep/_shared_train/train/final_model ]; then
    echo "[sweep] Model already trained, skipping."
  else
    echo "[sweep] Training model..."
    python -m bif.cli pipeline run --config configs/hp_sweep/00_shared_train.yaml
  fi
}

# ─── Phase 1: Coarse sweep ─────────────────────────────────────────────
run_phase1_0() {
  echo "[sweep] Phase1 [0/36]: lr1e-6_g1_nb1"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-6_g1_nb1.yaml --from run-bif
}
run_phase1_1() {
  echo "[sweep] Phase1 [1/36]: lr1e-6_g1_nb10"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-6_g1_nb10.yaml --from run-bif
}
run_phase1_2() {
  echo "[sweep] Phase1 [2/36]: lr1e-6_g1_nb100"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-6_g1_nb100.yaml --from run-bif
}
run_phase1_3() {
  echo "[sweep] Phase1 [3/36]: lr1e-6_g10_nb1"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-6_g10_nb1.yaml --from run-bif
}
run_phase1_4() {
  echo "[sweep] Phase1 [4/36]: lr1e-6_g10_nb10"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-6_g10_nb10.yaml --from run-bif
}
run_phase1_5() {
  echo "[sweep] Phase1 [5/36]: lr1e-6_g10_nb100"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-6_g10_nb100.yaml --from run-bif
}
run_phase1_6() {
  echo "[sweep] Phase1 [6/36]: lr1e-6_g100_nb1"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-6_g100_nb1.yaml --from run-bif
}
run_phase1_7() {
  echo "[sweep] Phase1 [7/36]: lr1e-6_g100_nb10"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-6_g100_nb10.yaml --from run-bif
}
run_phase1_8() {
  echo "[sweep] Phase1 [8/36]: lr1e-6_g100_nb100"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-6_g100_nb100.yaml --from run-bif
}
run_phase1_9() {
  echo "[sweep] Phase1 [9/36]: lr1e-6_g1000_nb1"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-6_g1000_nb1.yaml --from run-bif
}
run_phase1_10() {
  echo "[sweep] Phase1 [10/36]: lr1e-6_g1000_nb10"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-6_g1000_nb10.yaml --from run-bif
}
run_phase1_11() {
  echo "[sweep] Phase1 [11/36]: lr1e-6_g1000_nb100"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-6_g1000_nb100.yaml --from run-bif
}
run_phase1_12() {
  echo "[sweep] Phase1 [12/36]: lr1e-5_g1_nb1"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-5_g1_nb1.yaml --from run-bif
}
run_phase1_13() {
  echo "[sweep] Phase1 [13/36]: lr1e-5_g1_nb10"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-5_g1_nb10.yaml --from run-bif
}
run_phase1_14() {
  echo "[sweep] Phase1 [14/36]: lr1e-5_g1_nb100"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-5_g1_nb100.yaml --from run-bif
}
run_phase1_15() {
  echo "[sweep] Phase1 [15/36]: lr1e-5_g10_nb1"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-5_g10_nb1.yaml --from run-bif
}
run_phase1_16() {
  echo "[sweep] Phase1 [16/36]: lr1e-5_g10_nb10"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-5_g10_nb10.yaml --from run-bif
}
run_phase1_17() {
  echo "[sweep] Phase1 [17/36]: lr1e-5_g10_nb100"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-5_g10_nb100.yaml --from run-bif
}
run_phase1_18() {
  echo "[sweep] Phase1 [18/36]: lr1e-5_g100_nb1"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-5_g100_nb1.yaml --from run-bif
}
run_phase1_19() {
  echo "[sweep] Phase1 [19/36]: lr1e-5_g100_nb10"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-5_g100_nb10.yaml --from run-bif
}
run_phase1_20() {
  echo "[sweep] Phase1 [20/36]: lr1e-5_g100_nb100"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-5_g100_nb100.yaml --from run-bif
}
run_phase1_21() {
  echo "[sweep] Phase1 [21/36]: lr1e-5_g1000_nb1"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-5_g1000_nb1.yaml --from run-bif
}
run_phase1_22() {
  echo "[sweep] Phase1 [22/36]: lr1e-5_g1000_nb10"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-5_g1000_nb10.yaml --from run-bif
}
run_phase1_23() {
  echo "[sweep] Phase1 [23/36]: lr1e-5_g1000_nb100"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-5_g1000_nb100.yaml --from run-bif
}
run_phase1_24() {
  echo "[sweep] Phase1 [24/36]: lr1e-4_g1_nb1"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-4_g1_nb1.yaml --from run-bif
}
run_phase1_25() {
  echo "[sweep] Phase1 [25/36]: lr1e-4_g1_nb10"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-4_g1_nb10.yaml --from run-bif
}
run_phase1_26() {
  echo "[sweep] Phase1 [26/36]: lr1e-4_g1_nb100"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-4_g1_nb100.yaml --from run-bif
}
run_phase1_27() {
  echo "[sweep] Phase1 [27/36]: lr1e-4_g10_nb1"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-4_g10_nb1.yaml --from run-bif
}
run_phase1_28() {
  echo "[sweep] Phase1 [28/36]: lr1e-4_g10_nb10"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-4_g10_nb10.yaml --from run-bif
}
run_phase1_29() {
  echo "[sweep] Phase1 [29/36]: lr1e-4_g10_nb100"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-4_g10_nb100.yaml --from run-bif
}
run_phase1_30() {
  echo "[sweep] Phase1 [30/36]: lr1e-4_g100_nb1"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-4_g100_nb1.yaml --from run-bif
}
run_phase1_31() {
  echo "[sweep] Phase1 [31/36]: lr1e-4_g100_nb10"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-4_g100_nb10.yaml --from run-bif
}
run_phase1_32() {
  echo "[sweep] Phase1 [32/36]: lr1e-4_g100_nb100"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-4_g100_nb100.yaml --from run-bif
}
run_phase1_33() {
  echo "[sweep] Phase1 [33/36]: lr1e-4_g1000_nb1"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-4_g1000_nb1.yaml --from run-bif
}
run_phase1_34() {
  echo "[sweep] Phase1 [34/36]: lr1e-4_g1000_nb10"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-4_g1000_nb10.yaml --from run-bif
}
run_phase1_35() {
  echo "[sweep] Phase1 [35/36]: lr1e-4_g1000_nb100"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase1_lr1e-4_g1000_nb100.yaml --from run-bif
}

run_phase1() {
  total=36
  echo "[sweep] Phase 1: $total combinations"
  for i in $(seq $START_IDX $((END_IDX < total ? END_IDX : total - 1))); do
    run_phase1_$i
  done
}

# ─── Phase 3: nβ≈0 sanity check ──────────────────────────────────────
run_phase3_0() {
  echo "[sweep] Phase3 sanity: lr1e-5_g10_nb0"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase3_lr1e-5_g10_nb0.yaml --from run-bif
}
run_phase3_1() {
  echo "[sweep] Phase3 sanity: lr1e-5_g100_nb0"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase3_lr1e-5_g100_nb0.yaml --from run-bif
}
run_phase3_2() {
  echo "[sweep] Phase3 sanity: lr1e-5_g1000_nb0"
  python -m bif.cli pipeline run --config /workspace/pku_percy/BIFrost/configs/hp_sweep/phase3_lr1e-5_g1000_nb0.yaml --from run-bif
}

run_phase3() {
  run_phase3_0
  run_phase3_1
  run_phase3_2
}

case "$MODE" in
  train)   train_model ;;
  phase1)  train_model; run_phase1 ;;
  phase3)  train_model; run_phase3 ;;
  all)     train_model; run_phase1; run_phase3 ;;
  *)       echo "Usage: $0 {train|phase1|phase3|all} [start_idx] [end_idx]" ;;
esac
