#!/bin/bash
# Re-organize sweep experiments under SwanLab project 'BIFrost-sweep'
# No GPU needed — runs on CPU only
#
# Usage:
#   bash scripts/tmux_reorganize.sh          # launch
#   bash scripts/tmux_reorganize.sh attach   # watch
#   bash scripts/tmux_reorganize.sh kill     # stop

set -euo pipefail

SESSION="bif-reorganize"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

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

tmux send-keys -t $SESSION:0 \
  "export SWANLAB_API_KEY=RGvhmcyaE940jILdlzCGg" Enter
tmux send-keys -t $SESSION:0 \
  "export SWANLAB_PROJECT=BIFrost-sweep" Enter
tmux send-keys -t $SESSION:0 \
  "export CUDA_VISIBLE_DEVICES=''" Enter
tmux send-keys -t $SESSION:0 \
  "cd $ROOT && python scripts/reorganize_sweep.py all" Enter

tmux rename-window -t $SESSION:0 "reorganize"

echo "=== Session $SESSION launched ==="
echo "  tmux attach -t $SESSION"
echo "  bash $0 kill"
