# BIFrost — Bayesian Influence Function

**BIFrost** is a data influence estimation toolkit for LLM fine-tuning. It uses Localized SGLD sampling to score training-pool examples by their influence on a held-out query set, then compares replay strategies (selected / random / none) via schedule experiments.

> **BIFrost** = **BIF** + **Frost** (the rainbow bridge in Norse mythology connecting realms) — bridging pretraining data and finetuning objectives through influence estimation.

---

## Installation

```bash
pip install -e ".[dev]"
```

---

## Quick Start

```bash
bifrost pipeline run --config configs/my_experiment.yaml
```

---

## Full Pipeline

### 1. Write an experiment config

Configs are YAML files. See [`configs/`] for full examples.

```yaml
tokenizer_path: /models/pythia-70m
base_model_path: /models/pythia-70m

work_dir: /workspace/my_experiment
experiment_name: my_experiment

steps:

  build-pool:
    total_tokens: 10M
    domains: nemotron_math,octothinker,fineweb_edu,c4,wikipedia
    ratios: nemotron_math:0.1,octothinker:0.2,fineweb_edu:0.3,c4:0.2,wikipedia:0.2
    min_chars: 200
    min_tokens: 50
    max_tokens: 4096
    seed: 42

    finetune:
      total_tokens: 10M
      domains: nemotron_math
      seed: 42

  prepare-finetune:
    train_ratio: 0.7
    query_ratio: 0.1
    val_ratio: 0.1
    test_ratio: 0.1
    min_int_score: 4

  train:
    num_train_epochs: 1.0
    learning_rate: 2e-4
    target_num_checkpoints: 6
    bf16: true

  run-bif:
    num_chains: 4
    draws_per_chain: 60
    dtype: bfloat16

  analyze-bif:
    score_col: raw_cov_avg_over_queries
    top_k: 500

  extract-top:
    top_k: 500

  schedule-compare:
    replay_modes: [selected, random]
    schedules: [sequential, mixed]
    mix_ratios: [0.2]
    learning_rate: 2e-4
    num_train_epochs: 1
    bf16: true

  schedule-analyze: {}
```

> **Automatic path resolution**: intermediate paths between steps (`pool_jsonl`, `train_jsonl`, etc.) are derived from `work_dir`. Only external inputs (model paths, raw data) need to be specified.

### 2. Run / Resume

```bash
# Run the full pipeline from scratch
bifrost pipeline run --config configs/my_experiment.yaml

# Completed steps are automatically skipped
bifrost pipeline run --config configs/my_experiment.yaml

# Start from a specific step
bifrost pipeline run --config configs/my_experiment.yaml --from run-bif

# Force a new SwanLab run ID
bifrost pipeline run --config configs/my_experiment.yaml --new-run

# Check completion status
bifrost pipeline status --config configs/my_experiment.yaml
```

### 3. Reuse data from a previous run

Set `data_root` to symlink shared data from another run, auto-completing steps 1–6:

```yaml
work_dir: /workspace/new_experiment
data_root: /workspace/previous_experiment
steps:
  schedule-compare:
    replay_modes: [selected, random]
    ...
```

---

## How It Works

BIFrost implements a 3-phase pipeline:

1. **SGLD Sampling** (`run-bif`): For each model checkpoint, run Localized SGLD to perturb parameters and collect per-sample loss traces on both the **pool** set and the **query** set.

2. **Influence Scoring** (`analyze-bif`): Compute influence scores from the traces. The primary metric is `raw_cov_avg_over_queries` — the average covariance between pool-sample loss changes and query-set loss changes across SGLD draws. High positive covariance means the pool sample is "aligned" with the query objective.

3. **Replay Comparison** (`schedule-compare`): Train with the top-K most influential samples (selected) vs. random samples at various mix ratios and schedules (sequential or mixed), then compare eval losses.

### Why `raw_cov_avg_over_queries`?

The z-score-normalized correlation (`corr_mean_over_queries`) is structurally ≈ 0 because per-draw normalization cancels out the shared parameter-perturbation signal. `raw_cov_avg_over_queries` preserves the sign and magnitude of the influence signal.

---

## Individual Steps (CLI)

Each step can be invoked independently.

### build-pool — Build a multi-domain data pool

```bash
bifrost build-pool-v2 \
    --total_tokens 10M \
    --domains nemotron_math,octothinker,fineweb_edu,c4,wikipedia \
    --ratios nemotron_math:0.1,octothinker:0.2,fineweb_edu:0.3,c4:0.2,wikipedia:0.2 \
    --out_dir /exp/pool
```

Supported domains: `nemotron_math`, `octothinker`, `finemath`, `fineweb_edu`, `c4`, `wikipedia`, `slimpajama`, `starcoder`, `flan`, `sft_chat`

### prepare-finetune — Clean and split fine-tuning data

```bash
bifrost prepare-finetune \
    --input_path     /exp/finetune_pool/finetune_pool.jsonl \
    --tokenizer_path /models/pythia-70m \
    --out_dir        /exp/finetune_data \
    --train_ratio 0.7 --query_ratio 0.1 --val_ratio 0.1 --test_ratio 0.1
```

### train — Fine-tuning with periodic checkpoints

```bash
bifrost train \
    --base_model_path /models/pythia-70m \
    --tokenizer_path  /models/pythia-70m \
    --train_jsonl     /exp/finetune_data/stage2_train_1846.jsonl \
    --val_jsonl       /exp/finetune_data/stage2_val_263.jsonl \
    --output_dir      /exp/train \
    --bf16 --gradient_checkpointing
```

### run-bif — SGLD sampling and loss trace collection

```bash
bifrost run-bif \
    --model_root          /exp/train \
    --run_all_checkpoints \
    --pool_jsonl          /exp/pool/pt_pool.jsonl \
    --query_jsonl         /exp/finetune_data/stage2_query_263.jsonl \
    --out_dir             /exp/bif_traces \
    --num_chains 1 --draws_per_chain 100 --dtype bfloat16
```

### analyze-bif — Compute influence scores

```bash
bifrost analyze-bif \
    --bif_root /exp/bif_traces \
    --out_dir  /exp/bif_analysis \
    --score_col raw_cov_avg_over_queries \
    --top_k    500
```

### extract-top — Extract highest-influence samples

```bash
bifrost extract-top \
    --pool_jsonl  /exp/pool/pt_pool.jsonl \
    --ranking_csv /exp/bif_analysis/final_model/pool_scores.csv \
    --out_dir     /exp/top_samples \
    --top_k       500
```

### schedule-compare — Replay schedule comparison

```bash
bifrost schedule-compare \
    --base_model_path    /models/pythia-70m \
    --target_train_jsonl /exp/finetune_data/stage2_train_1846.jsonl \
    --target_val_jsonl   /exp/finetune_data/stage2_val_263.jsonl \
    --replay_pool_jsonl  /exp/top_samples/top_500_full.jsonl \
    --schedule           mixed \
    --replay_mode        selected \
    --replay_ratio       0.2 \
    --bf16 --gradient_checkpointing
```

Replay modes: `selected` (BIF-ranked), `random`, `top_random`, `none` (baseline)

---

## Experiment Tracking (SwanLab)

BIFrost integrates with [SwanLab](https://swanlab.cn) for experiment tracking. All steps share a single SwanLab run in pipeline mode.

---

## Output Directory Layout

```
<work_dir>/
├── pool/
│   └── pt_pool.jsonl
├── finetune_pool/
│   └── finetune_pool.jsonl
├── finetune_data/
│   ├── stage2_train_<n>.jsonl
│   ├── stage2_query_<n>.jsonl
│   ├── stage2_val_<n>.jsonl
│   └── stage2_test_<n>.jsonl
├── train/
│   ├── checkpoint-<step>/
│   └── final_model/
├── bif_traces/
│   └── <checkpoint>/chain_<id>/
├── bif_analysis/
│   └── <checkpoint>/pool_scores.csv
├── top_samples/
│   └── top_500_full.jsonl
├── schedule_compare/
├── schedule_analysis/
└── pipeline_state.json
```

---

## Project Layout

```
src/bif/
├── cli.py                    # unified CLI entry point
├── pipeline.py               # full-pipeline orchestration with state persistence
├── config.py                 # SGLDConfig and ReplayTrainConfig
├── constants.py              # shared constants
├── io.py                     # shared IO utilities
├── data/
│   ├── build_pool.py         # multi-domain pool construction
│   ├── finetune.py           # data cleaning and splitting
│   └── dataset.py            # PyTorch Dataset and DataCollator
├── training/
│   ├── checkpoint_trainer.py # HuggingFace Trainer fine-tuning
│   ├── schedule_trainer.py   # replay schedule comparison
│   ├── sgld.py               # Localized SGLD sampler
│   ├── loss.py               # per-example causal LM loss
│   └── callbacks.py          # CPTTrainer, ReplayTrainer, SwanLab callbacks
├── analysis/
│   ├── bif_runner.py         # SGLD sampling loop
│   ├── bif_analyzer.py       # influence scoring
│   ├── extractor.py          # top-k extraction
│   └── schedule_analyzer.py  # schedule comparison analysis
└── utils/
    ├── tracker.py            # SwanLab integration
    └── logging.py            # logging utilities
```

---

## License

MIT
