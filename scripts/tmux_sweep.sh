#!/bin/bash
# BIF Hyperparameter Sweep — 8 GPU tmux launcher
#
# 39 configs, 8 GPUs
# Per config: 3 chains × 500 draws, max_len=512
#
# Usage:
#   bash scripts/tmux_sweep.sh          # launch
#   bash scripts/tmux_sweep.sh attach   # watch
#   bash scripts/tmux_sweep.sh kill     # stop all

set -euo pipefail

SESSION="bif-sweep"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "${1:-}" = "kill" ]; then
  tmux kill-session -t $SESSION 2>/dev/null || true
  echo "Session killed."
  exit 0
fi

if [ "${1:-}" = "attach" ]; then
  tmux attach -t $SESSION
  exit 0
fi

tmux new-session -d -s $SESSION -c $ROOT

tmux send-keys -t $SESSION:0 "bash scripts/gpu_scripts/gpu0.sh" Enter
tmux rename-window -t $SESSION:0 "gpu0"
tmux new-window -t $SESSION -n "gpu1" -c $ROOT
tmux send-keys -t $SESSION:gpu1 "bash scripts/gpu_scripts/gpu1.sh" Enter
tmux new-window -t $SESSION -n "gpu2" -c $ROOT
tmux send-keys -t $SESSION:gpu2 "bash scripts/gpu_scripts/gpu2.sh" Enter
tmux new-window -t $SESSION -n "gpu3" -c $ROOT
tmux send-keys -t $SESSION:gpu3 "bash scripts/gpu_scripts/gpu3.sh" Enter
tmux new-window -t $SESSION -n "gpu4" -c $ROOT
tmux send-keys -t $SESSION:gpu4 "bash scripts/gpu_scripts/gpu4.sh" Enter
tmux new-window -t $SESSION -n "gpu5" -c $ROOT
tmux send-keys -t $SESSION:gpu5 "bash scripts/gpu_scripts/gpu5.sh" Enter
tmux new-window -t $SESSION -n "gpu6" -c $ROOT
tmux send-keys -t $SESSION:gpu6 "bash scripts/gpu_scripts/gpu6.sh" Enter
tmux new-window -t $SESSION -n "gpu7" -c $ROOT
tmux send-keys -t $SESSION:gpu7 "bash scripts/gpu_scripts/gpu7.sh" Enter

echo ""
echo "=== Session $SESSION launched ==="
echo "  tmux attach -t $SESSION"
echo "  bash $0 kill"
