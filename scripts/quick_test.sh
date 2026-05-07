#!/bin/bash
# BIFrost Quick Test — RMSprop-SGLD with devinterp nbeta
# Runs BIF sampling only (~2 min on single GPU with Pythia-70M)
#
# Prerequisites:
#   1. Set MODEL to your local model path
#   2. Set SWANLAB_API_KEY (or use SWANLAB_MODE=disabled for offline)
#   3. pip install -e .

set -euo pipefail

# >>> SET YOUR MODEL PATH <<<
MODEL=/path/to/pythia-70m

POOL=data/pool_800.jsonl
QUERY=data/query_gsm8k_20.jsonl
OUT=./runs/quick_test_bif_$(date +%m%d_%H%M)

# Disable SwanLab cloud upload for quick local test
# Remove this line to enable SwanLab cloud tracking
export SWANLAB_MODE=disabled

echo "=== BIFrost Quick Test ==="
echo "Model:  $MODEL"
echo "Pool:   $POOL"
echo "Query:  $QUERY"
echo "Output: $OUT"

python -m bif.cli run-bif \
  --model_name_or_path "$MODEL" \
  --pool_jsonl "$POOL" \
  --query_jsonl "$QUERY" \
  --out_dir "$OUT" \
  --sampler_type rmsprop_sgld \
  --num_chains 2 \
  --draws_per_chain 50 \
  --num_burnin_steps 10 \
  --num_steps_bw_draws 5 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --batches_per_draw 3 \
  --gradient_accumulation_steps 1 \
  --lr 5e-6 \
  --gamma 1000 \
  --beta 0.125 \
  --nbeta_mode devinterp \
  --dtype bfloat16 \
  --max_length 128

echo ""
echo "=== Verify output ==="
python3 -c "
import numpy as np, os, json

out = '$OUT'
chains = sorted([d for d in os.listdir(out) if d.startswith('chain_')])
print(f'Chains: {chains}')

for c in chains:
    cd = os.path.join(out, c)
    pool = np.load(os.path.join(cd, 'observable_loss_trace.npz'))
    query = np.load(os.path.join(cd, 'query_loss_trace.npz'))
    print(f'  {c}: pool_seq {pool[\"seq_loss\"].shape}, query_seq {query[\"seq_loss\"].shape}')
    with open(os.path.join(cd, 'observable_meta.json')) as f:
        meta = json.load(f)
    print(f'    pool samples: {len(meta[\"sample_ids\"])}, query samples: {len(json.load(open(os.path.join(cd, \"query_meta.json\")))[\"sample_ids\"])}')

print()
print('Quick test PASSED')
"
