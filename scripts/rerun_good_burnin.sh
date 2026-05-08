#!/bin/bash
# Re-run 5 GOOD eps-sweep configs with burn-in loss trace logging
# Project: BIFrost-burnin-trace
#
# These configs showed GOOD quality loss traces with eps=1e-4.
# Now re-running with burn-in logging enabled so the loss trace
# shows the full trajectory from w0 (like the Timaeus guide 2.1).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd $ROOT

export SWANLAB_API_KEY=RGvhmcyaE940jILdlzCGg
export SWANLAB_PROJECT=BIFrost-burnin-trace

MODEL="/workspace/pku_percy/models/pythia-70m-step1000"
POOL="data/pool_800.jsonl"
QUERY="data/query_gsm8k_20_answer_only.jsonl"
OUT_BASE="./runs/burnin_trace"

run_one() {
    local lr=$1 gamma=$2 nbeta=$3 eps=$4 gpu=$5
    local tag="lr${lr}_g${gamma}_nb${nbeta}_eps${eps}"
    local out_dir="${OUT_BASE}/${tag}"
    local exp_name="burnin-${tag}"

    echo ""
    echo "=== GPU $gpu: $tag ==="
    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES=$gpu python -m bif.cli run-bif \
        --model_name_or_path $MODEL \
        --pool_jsonl $POOL \
        --query_jsonl $QUERY \
        --out_dir "$out_dir" \
        --sampler_type rmsprop_sgld \
        --num_chains 3 \
        --draws_per_chain 500 \
        --num_burnin_steps 100 \
        --num_steps_bw_draws 2 \
        --train_batch_size 32 \
        --eval_batch_size 256 \
        --batches_per_draw 3 \
        --gradient_accumulation_steps 1 \
        --lr $lr \
        --gamma $gamma \
        --nbeta $nbeta \
        --nbeta_mode devinterp \
        --beta 0.125 \
        --rmsprop_eps $eps \
        --dtype bfloat16 \
        --max_length 512 \
        --experiment_name "$exp_name" \
        || echo "[WARN] run-bif FAILED for $tag"

    CUDA_VISIBLE_DEVICES="" python -m bif.cli analyze-bif \
        --bif_root "$out_dir" \
        --out_dir "$out_dir/analysis" \
        --score_col cross_cov_avg_over_queries \
        --top_k 20 \
        --experiment_name "analyze-$exp_name" \
        || echo "[WARN] analyze-bif FAILED for $tag"
}

# GPU 0: lr1e-5_g10_nb10 + lr1e-6_g100_nb10
(
    run_one 1e-5 10  10 1e-4 0
    run_one 1e-6 100 10 1e-4 0
) &

# GPU 1: lr1e-6_g100_nb1 + lr1e-6_g10_nb10
(
    run_one 1e-6 100 1  1e-4 1
    run_one 1e-6 10  10 1e-4 1
) &

# GPU 2: lr1e-6_g10_nb1
(
    run_one 1e-6 10  1  1e-4 2
) &

wait
echo ""
echo "=== All done ==="
