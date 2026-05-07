# BIFrost — Bayesian Influence Function

**BIFrost** is a data influence estimation toolkit for LLM fine-tuning. It uses Localized SGLD sampling to score training-pool examples by their influence on a held-out query set, then compares replay strategies (selected / random / none) via schedule experiments.

---

## Installation

```bash
git clone <repo_url> && cd BIFrost
pip install -e .
```

### SwanLab Setup (Experiment Tracking)

BIFrost uses [SwanLab](https://swanlab.cn) for experiment tracking and visualization.

```bash
# 1. Register at https://swanlab.cn and get your API key
# 2. Set environment variable:
export SWANLAB_API_KEY=<your_api_key>

# For quick local tests without cloud upload:
export SWANLAB_MODE=disabled
```

SwanLab naming convention:
- **Project**: `BIFrost` (fixed)
- **Experiment name**: auto-generated from run-bif params, e.g. `70m-rmsgld-lr5e-6-g1000-d2000-b200`
- Override by setting `experiment_name` in the config YAML

---

## Quick Start

### 1. Set your model path

Edit `configs/quick_test.yaml` and replace `/path/to/pythia-70m` with your local model path:

```yaml
tokenizer_path: /your/model/path
base_model_path: /your/model/path
```

### 2. Run the quick test (~2 min on single GPU)

```bash
# Option A: via pipeline (recommended)
python -m bif.cli pipeline run --config configs/quick_test.yaml

# Option B: BIF sampling only (skip pipeline)
bash scripts/quick_test.sh
```

### 3. Run the full small-pool experiment

```bash
# Edit configs/small_pool_exp.yaml — set tokenizer_path and base_model_path
python -m bif.cli pipeline run --config configs/small_pool_exp.yaml
```

### Resume after interruption

```bash
python -m bif.cli pipeline run --config configs/small_pool_exp.yaml --resume
```

### Start from a specific step

```bash
# Skip to BIF influence scoring (assumes train is done)
python -m bif.cli pipeline run --config configs/small_pool_exp.yaml --from run-bif --resume
```

### Check pipeline status

```bash
python -m bif.cli pipeline status --config configs/small_pool_exp.yaml
```

---

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampler_type` | `sgld` | SGLD sampler: `sgld` or `rmsprop_sgld` (recommended) |
| `nbeta_mode` | `devinterp` | N×β computation: `devinterp` (effective_bs/log(effective_bs)) or `dataset` (β×N) |
| `nbeta` | `0` | Override nbeta directly (0 = auto-compute from nbeta_mode) |
| `num_chains` | `4` | Number of independent SGLD chains |
| `draws_per_chain` | `60` | Number of samples to draw per chain |
| `burn_in` | `0` | Number of burn-in steps before collecting draws |
| `thinning` | `1` | Steps between consecutive draws |
| `batches_per_draw` | `0` | Mini-batches per SGLD step (0 = full dataset) |
| `lr` | `5e-6` | SGLD learning rate |
| `gamma` | `1e-3` | Localization strength (prior variance) |
| `beta` | `1.0` | Inverse temperature |
| `dtype` | `float32` | Model dtype: `float32`, `float16`, or `bfloat16` |

### nbeta_mode

The `nbeta` parameter controls gradient scaling in SGLD. Two modes are available:

- **`devinterp`** (default): `nbeta = effective_batch_size / log(effective_batch_size)`. Matches the [devinterp](https://github.com/mackelab/devinterp) reference implementation. Recommended for compatibility.

- **`dataset`**: `nbeta = beta * N` where N is the source dataset size. This is the exact Bayesian interpretation.

These differ significantly — for typical params (β=0.125, N=800, effective_bs=24), devinterp gives nbeta≈7.55 while dataset gives nbeta=100.

### Included Data Files

```
data/
├── pool_800.jsonl                    # 800 PT samples from 5 sources (BIF scoring pool)
├── pool_gsm8k_20.jsonl               # 20 GSM8K with CoT answers (finetune pool, split by pipeline)
├── query_gsm8k_20.jsonl              # 20 GSM8K questions (for standalone run-bif, not needed in pipeline)
└── query_gsm8k_20_with_answer.jsonl  # 20 GSM8K with answer_start_char (standalone use)
```

Pipeline mode only needs `pool_800.jsonl` + `pool_gsm8k_20.jsonl`. The pipeline automatically splits the finetune pool into train/query/val/test.

The other two query files are for standalone `run-bif` usage (e.g. `bifrost run-bif --query_jsonl data/query_gsm8k_20.jsonl`).

---

## Full Pipeline

### Pipeline Steps

| Step | What happens |
|------|-------------|
| `build-pool` | Build or symlink PT pool + optional finetune pool |
| `prepare-finetune` | Split finetune pool into train/query/val/test |
| `train` | Fine-tune base model, saving periodic checkpoints |
| `run-bif` | Run SGLD sampling at each checkpoint |
| `analyze-bif` | Compute influence scores and diagnostics |
| `extract-top` | Extract top-K most influential samples |
| `schedule-compare` | Train with selected vs random replay |
| `schedule-analyze` | Compare eval losses across strategies |

### Config File Format

Configs are YAML files. See [`configs/`] for examples.

```yaml
tokenizer_path: /models/pythia-70m
base_model_path: /models/pythia-70m
work_dir: ./runs/my_experiment
# experiment_name: my-custom-name   # auto-generated if not set, e.g. 70m-rmsgld-lr5e-6-g1000-d2000-b200
# project_name: BIFrost             # default: BIFrost

steps:
  build-pool:
    pool_jsonl: data/pool_800.jsonl
    finetune_pool_jsonl: data/pool_gsm8k_20.jsonl

  prepare-finetune:
    train_ratio: 0.7
    query_ratio: 0.1

  train:
    num_train_epochs: 1
    learning_rate: 2e-4
    bf16: true

  run-bif:
    sampler_type: rmsprop_sgld
    num_chains: 4
    draws_per_chain: 2000
    nbeta_mode: devinterp
    lr: 5e-6
    gamma: 1000
    beta: 0.125
    dtype: bfloat16

  analyze-bif:
    score_col: cross_cov_avg_over_queries
    top_k: 500

  extract-top:
    top_k: 500

  schedule-compare:
    schedules: [sequential, mixed]
    replay_modes: [selected, random, none]
    mix_ratios: [0.2]
    bf16: true

  schedule-analyze: {}
```

> **Automatic path resolution**: intermediate paths between steps are derived from `work_dir`. Only external inputs (model paths, raw data) need to be specified.

> **Using pre-built data**: Point `pool_jsonl` and `finetune_pool_jsonl` directly at existing JSONL files to skip pool construction.

---

## Individual Steps (CLI)

Each step can be invoked independently via `bifrost <step>`.

### run-bif — SGLD sampling and loss trace collection

```bash
bifrost run-bif \
    --model_name_or_path /models/pythia-70m \
    --pool_jsonl  data/pool_800.jsonl \
    --query_jsonl data/query_gsm8k_20.jsonl \
    --out_dir     ./runs/bif_traces \
    --sampler_type rmsprop_sgld \
    --num_chains 2 --draws_per_chain 50 \
    --nbeta_mode devinterp \
    --dtype bfloat16
```

### analyze-bif — Compute influence scores

```bash
bifrost analyze-bif \
    --bif_root  ./runs/bif_traces \
    --out_dir   ./runs/bif_analysis \
    --score_col cross_cov_avg_over_queries \
    --top_k     500
```

### extract-top — Extract highest-influence samples

```bash
bifrost extract-top \
    --pool_jsonl  data/pool_800.jsonl \
    --ranking_csv ./runs/bif_analysis/final_model/pool_scores.csv \
    --out_dir     ./runs/top_samples \
    --top_k       500
```

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
├── data/
│   ├── build_pool.py         # multi-domain pool construction
│   ├── finetune.py           # data cleaning and splitting
│   └── dataset.py            # PyTorch Dataset and DataCollator
├── training/
│   ├── checkpoint_trainer.py # HuggingFace Trainer fine-tuning
│   ├── schedule_trainer.py   # replay schedule comparison
│   ├── sgld.py               # Localized SGLD + RMSprop-SGLD samplers
│   ├── loss.py               # per-example causal LM loss
│   └── callbacks.py          # CPTTrainer, ReplayTrainer, SwanLab callbacks
├── analysis/
│   ├── bif_runner.py         # SGLD sampling loop
│   ├── bif_analyzer.py       # influence scoring + diagnostics
│   ├── extractor.py          # top-k extraction
│   └── schedule_analyzer.py  # schedule comparison analysis
└── utils/
    ├── tracker.py            # SwanLab integration
    └── logging.py            # logging utilities
```

---

## License

MIT
