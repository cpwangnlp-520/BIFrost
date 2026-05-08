#!/bin/bash
# GPU 4: 5 configs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd $ROOT
export CUDA_VISIBLE_DEVICES=4
export SWANLAB_API_KEY=RGvhmcyaE940jILdlzCGg

echo "\n=== GPU 4: phase1/lr1e-6_g10_nb10 ==="
mkdir -p $(dirname ./runs/hp_sweep/phase1/lr1e-6_g10_nb10)
python -m bif.cli run-bif  --model_name_or_path /workspace/pku_percy/models/pythia-70m-step1000  --pool_jsonl data/pool_800.jsonl  --query_jsonl data/query_gsm8k_20_answer_only.jsonl  --out_dir ./runs/hp_sweep/phase1/lr1e-6_g10_nb10  --sampler_type rmsprop_sgld  --num_chains 3  --draws_per_chain 500  --num_burnin_steps 100  --num_steps_bw_draws 2  --train_batch_size 8  --eval_batch_size 64  --batches_per_draw 3  --gradient_accumulation_steps 1  --lr 1e-06  --gamma 10  --nbeta 10  --nbeta_mode devinterp  --beta 0.125  --dtype bfloat16  --max_length 512  --experiment_name sweep-p1-lr1e-6_g10_nb10
python -m bif.cli analyze-bif  --bif_root ./runs/hp_sweep/phase1/lr1e-6_g10_nb10  --out_dir ./runs/hp_sweep/phase1/lr1e-6_g10_nb10/analysis  --score_col cross_cov_avg_over_queries  --top_k 20  --experiment_name sweep-p1-lr1e-6_g10_nb10

echo "\n=== GPU 4: phase1/lr1e-5_g1_nb1 ==="
mkdir -p $(dirname ./runs/hp_sweep/phase1/lr1e-5_g1_nb1)
python -m bif.cli run-bif  --model_name_or_path /workspace/pku_percy/models/pythia-70m-step1000  --pool_jsonl data/pool_800.jsonl  --query_jsonl data/query_gsm8k_20_answer_only.jsonl  --out_dir ./runs/hp_sweep/phase1/lr1e-5_g1_nb1  --sampler_type rmsprop_sgld  --num_chains 3  --draws_per_chain 500  --num_burnin_steps 100  --num_steps_bw_draws 2  --train_batch_size 8  --eval_batch_size 64  --batches_per_draw 3  --gradient_accumulation_steps 1  --lr 1e-05  --gamma 1  --nbeta 1  --nbeta_mode devinterp  --beta 0.125  --dtype bfloat16  --max_length 512  --experiment_name sweep-p1-lr1e-5_g1_nb1
python -m bif.cli analyze-bif  --bif_root ./runs/hp_sweep/phase1/lr1e-5_g1_nb1  --out_dir ./runs/hp_sweep/phase1/lr1e-5_g1_nb1/analysis  --score_col cross_cov_avg_over_queries  --top_k 20  --experiment_name sweep-p1-lr1e-5_g1_nb1

echo "\n=== GPU 4: phase1/lr1e-5_g100_nb100 ==="
mkdir -p $(dirname ./runs/hp_sweep/phase1/lr1e-5_g100_nb100)
python -m bif.cli run-bif  --model_name_or_path /workspace/pku_percy/models/pythia-70m-step1000  --pool_jsonl data/pool_800.jsonl  --query_jsonl data/query_gsm8k_20_answer_only.jsonl  --out_dir ./runs/hp_sweep/phase1/lr1e-5_g100_nb100  --sampler_type rmsprop_sgld  --num_chains 3  --draws_per_chain 500  --num_burnin_steps 100  --num_steps_bw_draws 2  --train_batch_size 8  --eval_batch_size 64  --batches_per_draw 3  --gradient_accumulation_steps 1  --lr 1e-05  --gamma 100  --nbeta 100  --nbeta_mode devinterp  --beta 0.125  --dtype bfloat16  --max_length 512  --experiment_name sweep-p1-lr1e-5_g100_nb100
python -m bif.cli analyze-bif  --bif_root ./runs/hp_sweep/phase1/lr1e-5_g100_nb100  --out_dir ./runs/hp_sweep/phase1/lr1e-5_g100_nb100/analysis  --score_col cross_cov_avg_over_queries  --top_k 20  --experiment_name sweep-p1-lr1e-5_g100_nb100

echo "\n=== GPU 4: phase1/lr1e-4_g10_nb10 ==="
mkdir -p $(dirname ./runs/hp_sweep/phase1/lr1e-4_g10_nb10)
python -m bif.cli run-bif  --model_name_or_path /workspace/pku_percy/models/pythia-70m-step1000  --pool_jsonl data/pool_800.jsonl  --query_jsonl data/query_gsm8k_20_answer_only.jsonl  --out_dir ./runs/hp_sweep/phase1/lr1e-4_g10_nb10  --sampler_type rmsprop_sgld  --num_chains 3  --draws_per_chain 500  --num_burnin_steps 100  --num_steps_bw_draws 2  --train_batch_size 8  --eval_batch_size 64  --batches_per_draw 3  --gradient_accumulation_steps 1  --lr 0.0001  --gamma 10  --nbeta 10  --nbeta_mode devinterp  --beta 0.125  --dtype bfloat16  --max_length 512  --experiment_name sweep-p1-lr1e-4_g10_nb10
python -m bif.cli analyze-bif  --bif_root ./runs/hp_sweep/phase1/lr1e-4_g10_nb10  --out_dir ./runs/hp_sweep/phase1/lr1e-4_g10_nb10/analysis  --score_col cross_cov_avg_over_queries  --top_k 20  --experiment_name sweep-p1-lr1e-4_g10_nb10

echo "\n=== GPU 4: phase3/lr1e-5_g10_nb0 ==="
mkdir -p $(dirname ./runs/hp_sweep/phase3/lr1e-5_g10_nb0)
python -m bif.cli run-bif  --model_name_or_path /workspace/pku_percy/models/pythia-70m-step1000  --pool_jsonl data/pool_800.jsonl  --query_jsonl data/query_gsm8k_20_answer_only.jsonl  --out_dir ./runs/hp_sweep/phase3/lr1e-5_g10_nb0  --sampler_type rmsprop_sgld  --num_chains 3  --draws_per_chain 500  --num_burnin_steps 100  --num_steps_bw_draws 2  --train_batch_size 8  --eval_batch_size 64  --batches_per_draw 3  --gradient_accumulation_steps 1  --lr 1e-05  --gamma 10  --nbeta 0.0001  --nbeta_mode devinterp  --beta 0.125  --dtype bfloat16  --max_length 512  --experiment_name sweep-p3-lr1e-5_g10_nb0
python -m bif.cli analyze-bif  --bif_root ./runs/hp_sweep/phase3/lr1e-5_g10_nb0  --out_dir ./runs/hp_sweep/phase3/lr1e-5_g10_nb0/analysis  --score_col cross_cov_avg_over_queries  --top_k 20  --experiment_name sweep-p3-lr1e-5_g10_nb0

echo "\n=== GPU 4: all done ==="
