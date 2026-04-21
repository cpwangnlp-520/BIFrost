#!/bin/bash
# Launch 3 BIF experiments in parallel on separate GPUs
# Usage: bash scripts/run_all.sh          (run all from scratch)
#        bash scripts/run_all.sh --resume  (resume incomplete steps)

set -euo pipefail

cd /workspace/pku_percy/bif

CFG_DIR="/workspace/pku_percy/bif/configs"
LOG_DIR="/workspace/pku_percy/runs/logs"
mkdir -p "$LOG_DIR"

RESUME_FLAG="${1:-}"

echo "=== Launching 3 BIF experiments ==="
echo "  pythia-70m  → GPU 0,1   project=BIF-70m"
echo "  pythia-160m → GPU 2,3   project=BIF-160m"
echo "  pythia-410m → GPU 4,5   project=BIF-410m"
echo ""

CUDA_VISIBLE_DEVICES=0,1 python -m bif.cli pipeline run \
    --config "$CFG_DIR/pythia70m.yaml" $RESUME_FLAG \
    > "$LOG_DIR/pythia70m.log" 2>&1 &
PID_70m=$!

CUDA_VISIBLE_DEVICES=2,3 python -m bif.cli pipeline run \
    --config "$CFG_DIR/pythia160m.yaml" $RESUME_FLAG \
    > "$LOG_DIR/pythia160m.log" 2>&1 &
PID_160m=$!

CUDA_VISIBLE_DEVICES=4,5 python -m bif.cli pipeline run \
    --config "$CFG_DIR/pythia410m.yaml" $RESUME_FLAG \
    > "$LOG_DIR/pythia410m.log" 2>&1 &
PID_410m=$!

echo "PIDs: 70m=$PID_70m  160m=$PID_160m  410m=$PID_410m"
echo "Logs: $LOG_DIR/"
echo ""
echo "Monitor:  tail -f $LOG_DIR/pythia70m.log"
echo "          tail -f $LOG_DIR/pythia160m.log"
echo "          tail -f $LOG_DIR/pythia410m.log"
echo ""

wait $PID_70m  && echo "[done] pythia-70m"  || echo "[FAIL] pythia-70m"
wait $PID_160m && echo "[done] pythia-160m" || echo "[FAIL] pythia-160m"
wait $PID_410m && echo "[done] pythia-410m" || echo "[FAIL] pythia-410m"

echo ""
echo "=== All experiments complete ==="
