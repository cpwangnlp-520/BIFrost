"""BIF full-pipeline runner with per-step state persistence.

Supports two modes:

1. Full pipeline (one-shot or resumable)
   bif pipeline run --config experiment.json
   bif pipeline run --config experiment.json --from run-bif --resume

2. Show status of a previous run
   bif pipeline status --config experiment.json

Config JSON format (example):
{
    "work_dir": "/workspace/my_experiment",
    "steps": {
        "build-pool": {
            "openwebmath_path": "/data/openwebmath.jsonl",
            "arxiv_path": "/data/arxiv.jsonl",
            "fineweb_edu_path": "/data/fineweb_edu.jsonl",
            "c4_path": "/data/c4.jsonl",
            "wikitext_path": "/data/wikitext.jsonl",
            "n_openwebmath": 1000,
            "n_arxiv": 1000,
            "n_fineweb_edu": 3500,
            "n_c4": 2500,
            "n_wikitext": 2000
        },
        "prepare-finetune": {
            "input_path": "/data/fineweb_edu.jsonl",
            "tokenizer_path": "/models/llama3-8b",
            "train_n": 2000,
            "query_n": 128,
            "val_n": 128,
            "test_n": 128
        },
        "train": {
            "base_model_path": "/models/llama3-8b",
            "tokenizer_path": "/models/llama3-8b",
            "num_train_epochs": 1.0,
            "learning_rate": 2e-5,
            "bf16": true,
            "target_num_checkpoints": 6
        },
        "run-bif": {
            "num_chains": 8,
            "draws_per_chain": 20,
            "dtype": "bfloat16"
        },
        "analyze-bif": {
            "score_col": "raw_cov_avg_over_queries",
            "top_k": 500
        },
        "extract-top": {
            "top_k": 500
        }
    }
}

Paths that depend on previous step outputs are automatically resolved
(e.g. train_jsonl, pool_jsonl) — you only need to specify the external
inputs in the config.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bif.io import ensure_dir, read_jsonl
from bif.utils.tracker import finish_pipeline as _swan_finish_pipeline
from bif.utils.tracker import init_run as _swan_init
from bif.utils.tracker import log as _swan_log
from bif.utils.tracker import log_bar as _swan_bar
from bif.utils.tracker import log_heatmap as _swan_heatmap
from bif.utils.tracker import log_table as _swan_table
from bif.utils.tracker import _ENV_PROJECT as _ENV_SWAN_PROJECT

# Environment variable names used to share a single SwanLab run across all
# pipeline sub-processes.  The parent sets these before spawning any step;
# tracker.py reads them to resume the same run instead of creating a new one.
_ENV_PIPELINE_RUN_ID = "SWANLAB_PIPELINE_RUN_ID"
_ENV_PIPELINE_EXPERIMENT = "SWANLAB_PIPELINE_EXPERIMENT"


# ─── Config loader ────────────────────────────────────────────────────────────


def _load_config(config_path: str) -> dict[str, Any]:
    """Load a pipeline config from a JSON or YAML file.

    The format is inferred from the file extension:
      ``.yaml`` / ``.yml``  → parsed with PyYAML
      anything else         → parsed as JSON (original behaviour)
    """
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        import yaml  # lazy import — not required for JSON-only users

        return yaml.safe_load(text)
    return json.loads(text)


STEPS: list[str] = [
    "build-pool",
    "prepare-finetune",
    "train",
    "run-bif",
    "analyze-bif",
    "extract-top",
    "schedule-compare",
    "schedule-analyze",
]

_STATE_FILE = "pipeline_state.json"


# ─── State persistence ────────────────────────────────────────────────────────


@dataclass
class PipelineState:
    """Records per-step completion status, persisted to disk after each step."""

    work_dir: str
    completed: dict[str, bool] = field(default_factory=dict)

    # ── IO ──────────────────────────────────────────────────────────────────
    def save(self) -> None:
        path = Path(self.work_dir) / _STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"completed": self.completed}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load_or_create(cls, work_dir: str) -> PipelineState:
        path = Path(work_dir) / _STATE_FILE
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(work_dir=work_dir, completed=data.get("completed", {}))
        return cls(work_dir=work_dir)

    # ── Queries ─────────────────────────────────────────────────────────────
    def is_done(self, step: str) -> bool:
        return bool(self.completed.get(step))

    def mark_done(self, step: str) -> None:
        self.completed[step] = True
        self.save()

    def print_status(self) -> None:
        print(f"\nPipeline status  ({self.work_dir})")
        print("─" * 50)
        for step in STEPS:
            icon = "✓" if self.is_done(step) else "○"
            print(f"  {icon}  {step}")
        print()


# ─── Path resolver ────────────────────────────────────────────────────────────


# Steps that produce shared data reusable across mix-ratio experiments.
# When data_root is set, these steps are auto-completed by symlinking.
_SHARED_DATA_STEPS = STEPS[:6]  # build-pool through extract-top

# Directories that are shared across runs (produced by _SHARED_DATA_STEPS).
_SHARED_DIRS = [
    "pool",
    "finetune_pool",
    "finetune_data",
    "train",
    "bif_traces",
    "bif_analysis",
    "top_samples",
]


class _Paths:
    """Centralises all derived paths so each step knows where to read/write."""

    def __init__(self, work_dir: str) -> None:
        self.work_dir = Path(work_dir)

    @property
    def pool_dir(self) -> Path:
        return self.work_dir / "pool"

    @property
    def pool_jsonl(self) -> Path:
        """Find the PT pool JSONL file.

        Priority:
          1. ``pool/pt_pool.jsonl``   (build-pool-v2 default name)
          2. ``pool/pool_10k_rebalanced.jsonl``  (legacy build-pool name)
          3. First ``*.jsonl`` file found under ``pool/``
        """
        for name in ("pt_pool.jsonl", "pool_10k_rebalanced.jsonl"):
            p = self.pool_dir / name
            if p.exists():
                return p
        # Fallback: any .jsonl in the pool dir
        candidates = sorted(self.pool_dir.glob("*.jsonl"))
        if candidates:
            return candidates[0]
        # Return the v2 default even if absent (will raise a clear error later)
        return self.pool_dir / "pt_pool.jsonl"

    @property
    def finetune_pool_dir(self) -> Path:
        return self.work_dir / "finetune_pool"

    @property
    def finetune_pool_jsonl(self) -> Path:
        """Find the finetune pool JSONL file (used as prepare-finetune input)."""
        for name in ("finetune_pool.jsonl",):
            p = self.finetune_pool_dir / name
            if p.exists():
                return p
        candidates = sorted(self.finetune_pool_dir.glob("*.jsonl"))
        if candidates:
            return candidates[0]
        return self.finetune_pool_dir / "finetune_pool.jsonl"

    @property
    def finetune_dir(self) -> Path:
        return self.work_dir / "finetune_data"

    def finetune_jsonl(self, split: str, n: int | None = None) -> Path:
        """Return path to a finetune split file.

        If n is given, uses the exact filename (legacy).
        If n is None, auto-discovers the file by glob pattern.
        """
        if n is not None:
            return self.finetune_dir / f"stage2_{split}_{n}.jsonl"
        import glob as _glob
        pattern = str(self.finetune_dir / f"stage2_{split}_*.jsonl")
        matches = sorted(_glob.glob(pattern))
        if matches:
            return Path(matches[0])
        return self.finetune_dir / f"stage2_{split}_0.jsonl"

    @property
    def train_dir(self) -> Path:
        return self.work_dir / "train"

    @property
    def bif_traces_dir(self) -> Path:
        return self.work_dir / "bif_traces"

    @property
    def bif_analysis_dir(self) -> Path:
        return self.work_dir / "bif_analysis"

    @property
    def top_samples_dir(self) -> Path:
        return self.work_dir / "top_samples"

    @property
    def schedule_dir(self) -> Path:
        return self.work_dir / "schedule_compare"

    @property
    def schedule_analysis_dir(self) -> Path:
        return self.work_dir / "schedule_analysis"


# ─── Step runners ─────────────────────────────────────────────────────────────


def _run_bif_cmd(
    args: list[str], nproc_per_node: int = 1, master_port: int = 29500
) -> None:
    """Invoke ``bif <args>`` as a subprocess; raise on non-zero exit.

    When *nproc_per_node* > 1 the command is launched via ``torchrun`` so
    that GPU chains are distributed across the requested number of processes.
    """
    if nproc_per_node > 1:
        torchrun = (
            shutil.which("torchrun") or f"{sys.executable.rsplit('/', 1)[0]}/torchrun"
        )
        cmd = [
            torchrun,
            f"--nproc_per_node={nproc_per_node}",
            f"--master_port={master_port}",
            "-m",
            "bif.cli",
        ] + args
    else:
        cmd = [sys.executable, "-m", "bif.cli"] + args
    print(f"\n[pipeline] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False, env=os.environ.copy())
    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed with exit code {result.returncode}: {' '.join(args)}"
        )


_POOL_STATS_CACHE = "_pool_stats.json"


def _compute_pool_stats(pool_path: str) -> dict[str, Any] | None:
    """Compute per-domain stats for a pool JSONL.  Returns None on failure."""
    rows = read_jsonl(pool_path)
    if not rows:
        return None

    import numpy as _np

    source_counts: dict[str, int] = {}
    source_charlens: dict[str, list[int]] = {}
    char_lens: list[int] = []
    for r in rows:
        src = str(r.get("source", r.get("subtype", "unknown")))
        source_counts[src] = source_counts.get(src, 0) + 1
        cl = len(str(r.get("text", "")))
        char_lens.append(cl)
        source_charlens.setdefault(src, []).append(cl)

    total = len(rows)
    est_tokens = sum(char_lens) / 4.0
    sources = sorted(source_counts, key=source_counts.get, reverse=True)

    percents = ["p10", "p25", "p50", "p75", "p90", "mean"]
    pct_mat: list[list[float]] = []
    for src in sources:
        tok_arr = _np.array(source_charlens[src], dtype=float) / 4.0
        row: list[float] = []
        for p in percents:
            if p == "mean":
                row.append(float(tok_arr.mean()))
            else:
                row.append(float(_np.percentile(tok_arr, int(p[1:]))))
        pct_mat.append(row)

    return {
        "total": total,
        "est_tokens": est_tokens,
        "sources": sources,
        "source_counts": source_counts,
        "source_est_tokens": {
            s: sum(source_charlens[s]) / 4.0 for s in sources
        },
        "source_charlens_raw": {
            s: source_charlens[s] for s in sources
        },
        "percentiles": percents,
        "pct_matrix": pct_mat,
    }


def _load_cached_stats(pool_path: str, label: str) -> dict[str, Any] | None:
    """Load cached stats from <pool_dir>/_pool_stats_<label>.json if available."""
    cache_path = Path(pool_path).parent / f"_pool_stats_{label}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _save_cached_stats(pool_path: str, label: str, stats: dict[str, Any]) -> None:
    """Save stats cache next to the pool file."""
    cache_path = Path(pool_path).parent / f"_pool_stats_{label}.json"
    try:
        cache_path.write_text(
            json.dumps(stats, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def _replay_pool_stats(stats: dict[str, Any], label: str) -> None:
    """Log pre-computed stats to SwanLab (no file I/O)."""
    import numpy as _np

    total = stats["total"]
    est_tokens = stats["est_tokens"]
    sources = stats["sources"]
    source_counts = stats["source_counts"]
    source_est_tokens = stats["source_est_tokens"]
    pct_matrix = stats["pct_matrix"]
    percents = stats["percentiles"]

    source_charlens = stats.get("source_charlens_raw", {})

    _swan_table(
        f"1_data/overview_{label}",
        headers=["domain", "samples", "est_tokens", "p50_tok", "p90_tok", "mean_tok"],
        rows=[
            [
                "(total)",
                str(total),
                f"{est_tokens / 1e6:.2f}M",
                "",
                "",
                "",
            ]
        ]
        + [
            [
                s,
                str(source_counts[s]),
                f"{source_est_tokens[s] / 1e6:.2f}M",
                f"{_np.percentile(_np.array(source_charlens[s]) / 4, 50):.0f}" if s in source_charlens else "",
                f"{_np.percentile(_np.array(source_charlens[s]) / 4, 90):.0f}" if s in source_charlens else "",
                f"{_np.mean(_np.array(source_charlens[s]) / 4):.0f}" if s in source_charlens else "",
            ]
            for s in sources
        ],
    )

    _swan_bar(
        f"1_data/domain_distribution_{label}",
        xaxis=sources,
        series={
            "samples": [source_counts[s] for s in sources],
            "est_tokens_M": [round(source_est_tokens[s] / 1e6, 2) for s in sources],
        },
    )

    mat = _np.array(pct_matrix, dtype=float).T
    _swan_heatmap(
        f"1_data/domain_token_length_{label}",
        xaxis=sources,
        yaxis=percents,
        matrix=mat,
        value_label="tokens",
    )

    print(
        f"[pipeline] {label}: {total} samples, ~{est_tokens / 1e6:.2f}M tokens, {len(source_counts)} sources"
    )


def _log_pool_data_stats(pool_path: str, label: str) -> None:
    """Log data distribution stats for a pool JSONL to SwanLab.

    Uses a JSON cache file next to the pool to avoid re-reading the full
    JSONL on every pipeline run.  Cache is written on first computation.
    """
    if not os.path.exists(pool_path):
        return

    stats = _load_cached_stats(pool_path, label)
    if stats is None:
        stats = _compute_pool_stats(pool_path)
        if stats is None:
            return
        _save_cached_stats(pool_path, label, stats)

    _replay_pool_stats(stats, label)


def _log_shared_data_stats(paths: _Paths) -> None:
    """Log data distribution stats for all data dirs.

    Called in the main process after SwanLab init.  For data_root runs this
    ensures the user can see what data is being reused.  For fresh runs this
    is idempotent (duplicates are harmless since echarts charts replace by key).
    """
    pt_pool = str(paths.pool_jsonl)
    if os.path.exists(pt_pool):
        _log_pool_data_stats(pt_pool, "pt_pool")

    finetune_pool = str(paths.finetune_pool_jsonl)
    if os.path.exists(finetune_pool):
        _log_pool_data_stats(finetune_pool, "finetune_pool")

    if paths.finetune_dir.exists():
        for split in ["train", "query", "val", "test"]:
            for jsonl in sorted(paths.finetune_dir.glob(f"stage2_{split}_*.jsonl")):
                _log_pool_data_stats(str(jsonl), f"finetune_{split}")
                break

    if paths.top_samples_dir.exists():
        for jsonl in sorted(paths.top_samples_dir.glob("top_*_full.jsonl")):
            _log_pool_data_stats(str(jsonl), "top_samples")
            break


def _step_build_pool(cfg: dict[str, Any], paths: _Paths) -> None:
    """Build the PT pool.

    Supports:
    - ``pool_jsonl``: skip build, use existing file directly (symlink/copy)
    - ``data_sources``: custom data source paths
    - ``total_tokens`` / ``domains`` / ``pool_type``: standard build-pool-v2
    - ``finetune`` sub-section: also build finetune pool (backward compatible)
    - ``finetune_pool_jsonl``: skip finetune pool build, use existing file

    Auto-skip: if both pt_pool.jsonl and finetune_pool.jsonl already exist
    (from a previous run), the build step is skipped entirely.
    """
    step_cfg = cfg.get("steps", {}).get("build-pool", {})
    ft_cfg = step_cfg.get("finetune", step_cfg.get("sft", {}))
    ft_pool_override = step_cfg.get("finetune_pool_jsonl")

    needs_ft = bool(ft_cfg or ft_pool_override)
    pt_exists = paths.pool_jsonl.exists()
    ft_exists = paths.finetune_pool_jsonl.exists() if needs_ft else True

    if pt_exists and ft_exists and "pool_jsonl" not in step_cfg:
        print(f"[pipeline] build-pool: pool already exists at {paths.pool_jsonl}, skipping")
        if needs_ft and ft_exists:
            print(f"[pipeline] build-pool: finetune pool already exists at {paths.finetune_pool_jsonl}, skipping")
        return True

    # ── Skip PT pool build if pool_jsonl is specified ────────────────────
    pool_jsonl_override = step_cfg.get("pool_jsonl")
    if pool_jsonl_override:
        src = Path(pool_jsonl_override)
        if not src.exists():
            raise FileNotFoundError(f"build-pool: pool_jsonl={pool_jsonl_override} does not exist")
        dst = paths.pool_dir / "pt_pool.jsonl"
        ensure_dir(str(paths.pool_dir))
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if src.is_absolute():
            dst.symlink_to(src)
        else:
            dst.symlink_to(src.resolve())
        print(f"[pipeline] build-pool: symlinked {src} → {dst}")
    elif "total_tokens" in step_cfg or "domains" in step_cfg or "pool_type" in step_cfg:
        # ── New build-pool-v2 interface ───────────────────────────────────
        args = [
            "build-pool-v2",
            "--out_dir",
            str(paths.pool_dir),
            "--out_name",
            "pt_pool.jsonl",
            "--seed",
            str(step_cfg.get("seed", 42)),
        ]
        if step_cfg.get("pool_type"):
            args += ["--pool_type", str(step_cfg["pool_type"])]
        if step_cfg.get("total_tokens"):
            args += ["--total_tokens", str(step_cfg["total_tokens"])]
        if step_cfg.get("domains"):
            args += ["--domains", str(step_cfg["domains"])]
        if step_cfg.get("ratios"):
            args += ["--ratios", str(step_cfg["ratios"])]
        if step_cfg.get("min_chars") is not None:
            args += ["--min_chars", str(step_cfg["min_chars"])]
        if step_cfg.get("min_tokens") is not None:
            args += ["--min_tokens", str(step_cfg["min_tokens"])]
        if step_cfg.get("max_tokens") is not None:
            args += ["--max_tokens", str(step_cfg["max_tokens"])]
        if step_cfg.get("min_int_score") is not None:
            args += ["--min_int_score", str(step_cfg["min_int_score"])]
        if step_cfg.get("min_lang_score") is not None:
            args += ["--min_lang_score", str(step_cfg["min_lang_score"])]
        if step_cfg.get("tokenizer_path"):
            args += ["--tokenizer_path", str(step_cfg["tokenizer_path"])]
        if step_cfg.get("calibrate_n") is not None:
            args += ["--calibrate_n", str(step_cfg["calibrate_n"])]
        if step_cfg.get("data_sources"):
            import json as _json
            args += ["--data_sources", _json.dumps(step_cfg["data_sources"])]
        # Explicit per-domain counts (n_nemotron_math, n_c4, etc.)
        from bif.data.build_pool import ALL_DOMAINS as _ALL_DOMAINS

        for d in _ALL_DOMAINS:
            val = step_cfg.get(f"n_{d}")
            if val is not None:
                args += [f"--n_{d}", str(val)]
        _run_bif_cmd(args)
    else:
        print(
            "[pipeline] build-pool: config must have 'total_tokens', 'domains', 'pool_jsonl', or 'data_sources', skipping"
        )
        return None

    # ── Optional: also build finetune pool ────────────────────────────────
    ft_cfg = step_cfg.get("finetune", step_cfg.get("sft", {}))
    ft_pool_override = step_cfg.get("finetune_pool_jsonl")

    if ft_pool_override:
        src = Path(ft_pool_override)
        if not src.exists():
            raise FileNotFoundError(f"build-pool: finetune_pool_jsonl={ft_pool_override} does not exist")
        dst = paths.finetune_pool_dir / "finetune_pool.jsonl"
        ensure_dir(str(paths.finetune_pool_dir))
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if src.is_absolute():
            dst.symlink_to(src)
        else:
            dst.symlink_to(src.resolve())
        print(f"[pipeline] build-pool: symlinked finetune pool {src} → {dst}")
    elif ft_cfg:
        _build_finetune_pool(ft_cfg, paths)

    # ── Log data stats to SwanLab ────────────────────────────────────────
    pt_pool = str(paths.pool_jsonl)
    finetune_pool = (
        str(paths.finetune_pool_jsonl) if paths.finetune_pool_jsonl.exists() else ""
    )
    if os.path.exists(pt_pool):
        _log_pool_data_stats(pt_pool, "pt_pool")
    if finetune_pool and os.path.exists(finetune_pool):
        _log_pool_data_stats(finetune_pool, "finetune_pool")


def _build_finetune_pool(ft_cfg: dict[str, Any], paths: _Paths) -> None:
    """Build the finetune pool using build-pool (called from _step_build_pool).

    Supports both CPT (plain text, e.g. nemotron_math) and SFT (sft_chat)
    domains.  The finetune pool is just another pool built with a different
    domain / ratio configuration.
    """
    print("\n[pipeline] Building finetune pool …")
    args = [
        "build-pool-v2",
        "--out_dir",
        str(paths.finetune_pool_dir),
        "--out_name",
        "finetune_pool.jsonl",
        "--seed",
        str(ft_cfg.get("seed", 42)),
    ]
    if ft_cfg.get("pool_type"):
        args += ["--pool_type", str(ft_cfg["pool_type"])]
    if ft_cfg.get("total_tokens"):
        args += ["--total_tokens", str(ft_cfg["total_tokens"])]
    if ft_cfg.get("domains"):
        args += ["--domains", str(ft_cfg["domains"])]
    if ft_cfg.get("ratios"):
        args += ["--ratios", str(ft_cfg["ratios"])]
    if ft_cfg.get("min_chars") is not None:
        args += ["--min_chars", str(ft_cfg["min_chars"])]
    if ft_cfg.get("min_tokens") is not None:
        args += ["--min_tokens", str(ft_cfg["min_tokens"])]
    if ft_cfg.get("max_tokens") is not None:
        args += ["--max_tokens", str(ft_cfg["max_tokens"])]
    if ft_cfg.get("min_int_score") is not None:
        args += ["--min_int_score", str(ft_cfg["min_int_score"])]
    if ft_cfg.get("min_lang_score") is not None:
        args += ["--min_lang_score", str(ft_cfg["min_lang_score"])]
    if ft_cfg.get("tokenizer_path"):
        args += ["--tokenizer_path", str(ft_cfg["tokenizer_path"])]
    if ft_cfg.get("calibrate_n") is not None:
        args += ["--calibrate_n", str(ft_cfg["calibrate_n"])]
    _run_bif_cmd(args)


def _global_tokenizer(cfg: dict[str, Any]) -> str | None:
    """Resolve tokenizer_path: global → train → None."""
    g = cfg.get("tokenizer_path")
    if g:
        return g
    return cfg.get("steps", {}).get("train", {}).get("tokenizer_path")


def _global_base_model(cfg: dict[str, Any]) -> str | None:
    """Resolve base_model_path: global → train → None."""
    g = cfg.get("base_model_path")
    if g:
        return g
    return cfg.get("steps", {}).get("train", {}).get("base_model_path")


def _step_prepare_finetune(cfg: dict[str, Any], paths: _Paths) -> None:
    step_cfg = cfg.get("steps", {}).get("prepare-finetune", {})

    # Auto-skip if all 4 splits already exist
    existing_splits = []
    for split in ["train", "query", "val", "test"]:
        p = paths.finetune_jsonl(split)
        if p.exists():
            existing_splits.append(split)
    if len(existing_splits) == 4:
        print(f"[pipeline] prepare-finetune: all 4 splits already exist, skipping")
        return True

    input_path = step_cfg.get("input_path")
    if not input_path:
        ft_jsonl = paths.finetune_pool_jsonl
        if ft_jsonl.exists():
            input_path = str(ft_jsonl)
            print(f"[pipeline] prepare-finetune: auto-resolved input_path={input_path}")
        else:
            raise FileNotFoundError(
                "prepare-finetune: 'input_path' not set in config and finetune pool not found at "
                f"{ft_jsonl}.\n"
                "Either:\n"
                "  (a) add 'input_path' to the prepare-finetune config section, or\n"
                "  (b) add a 'finetune' sub-section to 'build-pool' so the pipeline builds it."
            )

    args = [
        "prepare-finetune",
        "--input_path",
        input_path,
        "--tokenizer_path",
        step_cfg.get("tokenizer_path", _global_tokenizer(cfg) or ""),
        "--out_dir",
        str(paths.finetune_dir),
    ]
    if step_cfg.get("train_n") is not None:
        args += ["--train_n", str(step_cfg["train_n"])]
    if step_cfg.get("query_n") is not None:
        args += ["--query_n", str(step_cfg["query_n"])]
    if step_cfg.get("val_n") is not None:
        args += ["--val_n", str(step_cfg["val_n"])]
    if step_cfg.get("test_n") is not None:
        args += ["--test_n", str(step_cfg["test_n"])]
    for rkey in ("train_ratio", "query_ratio", "val_ratio", "test_ratio"):
        if step_cfg.get(rkey) is not None:
            args += [f"--{rkey}", str(step_cfg[rkey])]
    args += [
        "--seed",
        str(step_cfg.get("seed", 42)),
        "--min_chars",
        str(step_cfg.get("min_chars", 100)),
        "--min_token_count",
        str(step_cfg.get("min_token_count", 200)),
        "--max_token_count",
        str(step_cfg.get("max_token_count", 2000)),
        "--min_int_score",
        str(step_cfg.get("min_int_score", 4)),
        "--min_score",
        str(step_cfg.get("min_score", 0.0)),
        "--min_language_score",
        str(step_cfg.get("min_language_score", 0.9)),
    ]
    if step_cfg.get("text_key"):
        args += ["--text_key", step_cfg["text_key"]]
    if step_cfg.get("require_english"):
        args += ["--require_english"]
    _run_bif_cmd(args)


def _relog_schedule_metrics(schedule_analysis_dir: str, schedule_compare_dir: str) -> None:
    """Read schedule analysis CSV and re-log to the pipeline's SwanLab run."""
    from bif.analysis.schedule_analyzer import _do_swanlog, _discover_run_dirs, _collect_run, _summarize_runs

    runs_root = schedule_compare_dir
    if not os.path.exists(runs_root):
        print(f"[pipeline] relog_schedule_metrics: {runs_root} not found, skipping")
        return

    run_dirs = _discover_run_dirs(runs_root)
    runs = [_collect_run(rd) for rd in run_dirs]
    summary_df = _summarize_runs(runs)
    n = len(runs)
    print(f"[pipeline] relog_schedule_metrics: {n} runs, writing to SwanLab")
    _do_swanlog(runs, summary_df, schedule_analysis_dir, data_dirs=None, runs_root=runs_root)


def _step_train(cfg: dict[str, Any], paths: _Paths, experiment_name: str) -> None:
    step_cfg = cfg.get("steps", {}).get("train", {})

    # Auto-skip if model already trained (final_model or last checkpoint exists)
    if (paths.train_dir / "final_model").exists() or (paths.train_dir / "checkpoint-1").exists():
        print(f"[pipeline] train: model already exists at {paths.train_dir}, skipping")
        return True

    train_jsonl = str(paths.finetune_jsonl("train"))
    val_jsonl = str(paths.finetune_jsonl("val"))
    base_model = step_cfg.get("base_model_path", _global_base_model(cfg) or "")
    tokenizer = step_cfg.get("tokenizer_path", _global_tokenizer(cfg) or base_model)
    args = [
        "train",
        "--base_model_path",
        base_model,
        "--tokenizer_path",
        tokenizer,
        "--train_jsonl",
        train_jsonl,
        "--val_jsonl",
        val_jsonl,
        "--output_dir",
        str(paths.train_dir),
        "--num_train_epochs",
        str(step_cfg.get("num_train_epochs", 1.0)),
        "--learning_rate",
        str(step_cfg.get("learning_rate", 2e-5)),
        "--target_num_checkpoints",
        str(step_cfg.get("target_num_checkpoints", 6)),
        "--per_device_train_batch_size",
        str(step_cfg.get("per_device_train_batch_size", 8)),
        "--per_device_eval_batch_size",
        str(step_cfg.get("per_device_eval_batch_size", 8)),
        "--gradient_accumulation_steps",
        str(step_cfg.get("gradient_accumulation_steps", 4)),
        "--max_length",
        str(step_cfg.get("max_length", 512)),
        "--weight_decay",
        str(step_cfg.get("weight_decay", 0.01)),
        "--warmup_ratio",
        str(step_cfg.get("warmup_ratio", 0.03)),
        "--logging_steps",
        str(step_cfg.get("logging_steps", 10)),
        "--min_eval_steps",
        str(step_cfg.get("min_eval_steps", 20)),
        "--eval_steps",
        str(step_cfg.get("eval_steps", 0)),
        "--experiment_name",
        experiment_name,
        "--run_name",
        "basemodel",
    ]
    args += _ds_fsdp_args(step_cfg)
    nproc = int(step_cfg.get("nproc_per_node", 1))
    master_port = int(step_cfg.get("master_port", 29500))
    _run_bif_cmd(args, nproc_per_node=nproc, master_port=master_port)


def _step_run_bif(
    cfg: dict[str, Any], paths: _Paths, resume: bool, experiment_name: str
) -> None:
    step_cfg = cfg.get("steps", {}).get("run-bif", {})
    query_jsonl = str(paths.finetune_jsonl("query"))
    nproc = int(step_cfg.get("nproc_per_node", 1))
    master_port = int(step_cfg.get("master_port", 29500))
    args = [
        "run-bif",
        "--model_root",
        str(paths.train_dir),
        "--run_all_checkpoints",
        "--pool_jsonl",
        str(paths.pool_jsonl),
        "--query_jsonl",
        query_jsonl,
        "--out_dir",
        str(paths.bif_traces_dir),
        "--num_chains",
        str(step_cfg.get("num_chains", 4)),
        "--draws_per_chain",
        str(step_cfg.get("draws_per_chain", 60)),
        "--burn_in",
        str(step_cfg.get("burn_in", 0)),
        "--thinning",
        str(step_cfg.get("thinning", 1)),
        "--train_batch_size",
        str(step_cfg.get("train_batch_size", 16)),
        "--eval_batch_size",
        str(step_cfg.get("eval_batch_size", 32)),
        "--pool_eval_subset",
        str(step_cfg.get("pool_eval_subset", 0)),
        "--max_length",
        str(step_cfg.get("max_length", 1024)),
        "--lr",
        str(step_cfg.get("lr", 5e-6)),
        "--gamma",
        str(step_cfg.get("gamma", 1e-3)),
        "--noise_scale",
        str(step_cfg.get("noise_scale", 1.0)),
        "--dtype",
        step_cfg.get("dtype", "float32"),
        "--experiment_name",
        experiment_name,  # shared across all pipeline steps
        "--run_name",
        "bif",
    ]
    if step_cfg.get("beta") is not None:
        args += ["--beta", str(step_cfg["beta"])]
    if step_cfg.get("grad_clip") is not None:
        args += ["--grad_clip", str(step_cfg["grad_clip"])]
    if step_cfg.get("weight_decay") is not None:
        args += ["--weight_decay", str(step_cfg["weight_decay"])]
    if resume:
        args.append("--resume")
    else:
        # Warn if partial traces already exist: without --resume they will be
        # overwritten.  Users restarting mid-step should pass --resume.
        existing = list(paths.bif_traces_dir.glob("*/chain_*/pool_loss_trace.jsonl"))
        if existing:
            print(
                "[pipeline] WARNING: existing BIF traces found in "
                f"{paths.bif_traces_dir} but --resume was not set. "
                "Pass --resume to skip already-finished checkpoints."
            )
    _run_bif_cmd(args, nproc_per_node=nproc, master_port=master_port)


def _step_analyze_bif(cfg: dict[str, Any], paths: _Paths, experiment_name: str) -> None:
    step_cfg = cfg.get("steps", {}).get("analyze-bif", {})
    args = [
        "analyze-bif",
        "--bif_root",
        str(paths.bif_traces_dir),
        "--out_dir",
        str(paths.bif_analysis_dir),
        "--score_col",
        step_cfg.get("score_col", "raw_cov_avg_over_queries"),
        "--top_k",
        str(step_cfg.get("top_k", 500)),
        "--experiment_name",
        experiment_name,  # shared across all pipeline steps
        "--run_name",
        "analysis",
    ]
    _run_bif_cmd(args)


def _resolve_ranking_csv(
    step_cfg: dict[str, Any], bif_analysis_dir: Path
) -> str:
    explicit_ckpt = step_cfg.get("checkpoint")
    if explicit_ckpt:
        ranking_csv = str(bif_analysis_dir / explicit_ckpt / "pool_scores.csv")
        if not os.path.exists(ranking_csv):
            raise FileNotFoundError(
                f"extract-top: pool_scores.csv not found for checkpoint "
                f"{explicit_ckpt!r} at {ranking_csv}. "
                f"Run analyze-bif first, or check the checkpoint name."
            )
        return ranking_csv

    ranking_csv = str(bif_analysis_dir / "final_model" / "pool_scores.csv")
    if os.path.exists(ranking_csv):
        return ranking_csv

    def _ckpt_step(p: Path) -> int:
        digits = "".join(filter(str.isdigit, p.parent.name))
        return int(digits) if digits else 0

    candidates = sorted(
        bif_analysis_dir.glob("*/pool_scores.csv"),
        key=_ckpt_step,
        reverse=True,
    )
    if candidates:
        ranking_csv = str(candidates[0])
        print(
            f"[pipeline] extract-top: no 'final_model' found, "
            f"using {Path(ranking_csv).parent.name}/pool_scores.csv. "
            f'Set "checkpoint": "<name>" in extract-top config to pin a specific one.'
        )
    return ranking_csv


def _step_extract_top(cfg: dict[str, Any], paths: _Paths, experiment_name: str) -> None:
    step_cfg = cfg.get("steps", {}).get("extract-top", {})
    top_k = step_cfg.get("top_k", 500)
    ranking_csv = _resolve_ranking_csv(step_cfg, paths.bif_analysis_dir)
    score_cols = step_cfg.get("score_cols", ["raw_cov_avg_over_queries"])
    if isinstance(score_cols, str):
        score_cols = [score_cols]

    for sc in score_cols:
        sc_tag = "corr" if "corr" in sc else "raw"
        if len(score_cols) == 1:
            out_dir = str(paths.top_samples_dir)
            run_name = f"extraction_{sc_tag}"
        else:
            out_dir = str(paths.top_samples_dir / sc_tag)
            run_name = f"extraction_{sc_tag}"

        args = [
            "extract-top",
            "--pool_jsonl",
            str(paths.pool_jsonl),
            "--ranking_csv",
            ranking_csv,
            "--out_dir",
            out_dir,
            "--score_col",
            sc,
            "--top_k",
            str(top_k),
            "--experiment_name",
            experiment_name,
            "--run_name",
            run_name,
        ]
        print(f"[pipeline] extract-top: score_col={sc} → {out_dir}")
        _run_bif_cmd(args)


def _ds_fsdp_args(step_cfg: dict[str, Any]) -> list[str]:
    args: list[str] = []
    if step_cfg.get("bf16"):
        args.append("--bf16")
    if step_cfg.get("fp16"):
        args.append("--fp16")
    if step_cfg.get("gradient_checkpointing"):
        args.append("--gradient_checkpointing")
    if step_cfg.get("deepspeed"):
        args += ["--deepspeed", str(step_cfg["deepspeed"])]
    if step_cfg.get("fsdp"):
        args += ["--fsdp", str(step_cfg["fsdp"])]
    if step_cfg.get("fsdp_transformer_layer_cls_to_wrap"):
        args += [
            "--fsdp_transformer_layer_cls_to_wrap",
            str(step_cfg["fsdp_transformer_layer_cls_to_wrap"]),
        ]
    return args


def _step_schedule_compare(cfg: dict[str, Any], paths: _Paths, experiment_name: str = "") -> None:
    """Run schedule-comparison training with selected/random replay at mix ratios.

    Config format (under ``schedule-compare``):

        schedule-compare:
          base_model_path: /path/to/base/model   (optional, defaults to train step's)
          tokenizer_path: /path/to/tokenizer      (optional, defaults to train step's)
          val_jsonl: /path/to/val.jsonl           (optional, auto-resolved)
          top_k_jsonl: top_500_full.jsonl          (default, from extract-top)
          replay_modes: [selected, random]
          schedules: [sequential, mixed]
          mix_ratios: [0.1, 0.2, 0.3]
          score_types: [corr, raw]                 (optional, BIF score types to compare)
          num_train_epochs: 1.0
          learning_rate: 5e-5
          per_device_train_batch_size: 2
          per_device_eval_batch_size: 8
          gradient_accumulation_steps: 2
          max_length: 256
          eval_steps: 50
          logging_steps: 10
          bf16: false
          seed: 42
    """
    step_cfg = cfg.get("steps", {}).get("schedule-compare", {})
    if not step_cfg:
        print("[pipeline] schedule-compare: no config, skipping.")
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    train_cfg = cfg.get("steps", {}).get("train", {})
    ftune_cfg = cfg.get("steps", {}).get("prepare-finetune", {})

    data_root = cfg.get("data_root")
    if data_root and (not train_cfg or not ftune_cfg):
        src_cfg_path = Path(data_root) / "pipeline_config.json"
        if src_cfg_path.exists():
            src_cfg = json.loads(src_cfg_path.read_text(encoding="utf-8"))
            if not train_cfg:
                train_cfg = src_cfg.get("steps", {}).get("train", {})
            if not ftune_cfg:
                ftune_cfg = src_cfg.get("steps", {}).get("prepare-finetune", {})
            print(
                f"[pipeline] schedule-compare: loaded shared config from {src_cfg_path}"
            )

    base_model = step_cfg.get("base_model_path", _global_base_model(cfg) or train_cfg.get("base_model_path"))
    tokenizer_path = step_cfg.get(
        "tokenizer_path", _global_tokenizer(cfg) or train_cfg.get("tokenizer_path", base_model)
    )
    train_jsonl = str(paths.finetune_jsonl("train"))
    val_jsonl = step_cfg.get("val_jsonl", str(paths.finetune_jsonl("val")))

    if not base_model:
        raise ValueError(
            "schedule-compare: 'base_model_path' not found.  Either:\n"
            "  (a) add 'base_model_path' to the schedule-compare config section, or\n"
            "  (b) add a 'train' section with 'base_model_path', or\n"
            "  (c) set 'data_root' pointing to a completed pipeline run."
        )

    # Auto-inherit train config for missing schedule-compare fields
    inherit_keys = [
        "max_length", "bf16", "gradient_checkpointing", "deepspeed", "fsdp",
        "fsdp_transformer_layer_cls_to_wrap",
    ]
    for k in inherit_keys:
        if k not in step_cfg and k in train_cfg:
            step_cfg[k] = train_cfg[k]

    extract_cfg = cfg.get("steps", {}).get("extract-top", {})
    if not extract_cfg and data_root:
        src_cfg_path2 = Path(data_root) / "pipeline_config.json"
        if src_cfg_path2.exists():
            src_cfg2 = json.loads(src_cfg_path2.read_text(encoding="utf-8"))
            extract_cfg = src_cfg2.get("steps", {}).get("extract-top", {})
    top_k = extract_cfg.get("top_k", 500)
    top_k_jsonl_name = step_cfg.get("top_k_jsonl", f"top_{top_k}_full.jsonl")

    score_cols = extract_cfg.get("score_cols", ["raw_cov_avg_over_queries"])
    if isinstance(score_cols, str):
        score_cols = [score_cols]
    score_types = step_cfg.get("score_types", None)
    if score_types is None:
        score_types = []
        for sc in score_cols:
            score_types.append("corr" if "corr" in sc else "raw")
    if isinstance(score_types, str):
        score_types = [score_types]

    replay_modes = step_cfg.get("replay_modes", ["selected", "random"])
    schedules = step_cfg.get("schedules", ["mixed"])
    mix_ratios = step_cfg.get("mix_ratios", [0.1, 0.2, 0.3])

    ensure_dir(str(paths.schedule_dir))

    if "none" in replay_modes:
        for schedule in schedules:
            run_name = f"{schedule}_none"
            run_dir = str(paths.schedule_dir / run_name)
            if os.path.exists(os.path.join(run_dir, "run_summary.json")):
                print(f"[pipeline] schedule-compare: skip {run_name} (done)")
                continue
            args = [
                "schedule-compare",
                "--output_dir",
                run_dir,
                "--run_name",
                run_name,
                "--schedule",
                schedule,
                "--replay_mode",
                "none",
                "--replay_ratio",
                "0.0",
                "--base_model_path",
                base_model,
                "--tokenizer_path",
                tokenizer_path,
                "--target_train_jsonl",
                train_jsonl,
                "--target_val_jsonl",
                val_jsonl,
                "--replay_pool_jsonl",
                str(paths.pool_jsonl),
                "--max_length",
                str(step_cfg.get("max_length", 256)),
                "--num_train_epochs",
                str(step_cfg.get("num_train_epochs", 1.0)),
                "--learning_rate",
                str(step_cfg.get("learning_rate", 5e-5)),
                "--per_device_train_batch_size",
                str(step_cfg.get("per_device_train_batch_size", 2)),
                "--per_device_eval_batch_size",
                str(step_cfg.get("per_device_eval_batch_size", 8)),
                "--gradient_accumulation_steps",
                str(step_cfg.get("gradient_accumulation_steps", 2)),
                "--logging_steps",
                str(step_cfg.get("logging_steps", 10)),
                "--eval_steps",
                str(step_cfg.get("eval_steps", 0)),
                "--seed",
                str(step_cfg.get("seed", 42)),
            ] + _ds_fsdp_args(step_cfg)
            if experiment_name:
                args += ["--experiment_name", experiment_name]
            _run_bif_cmd(args)

    for st_idx, score_type in enumerate(score_types):
        if len(score_cols) == 1:
            top_dir = paths.top_samples_dir
            sc_prefix = ""
        else:
            top_dir = paths.top_samples_dir / score_type
            sc_prefix = f"{score_type}_"

        replay_pool_jsonl = str(top_dir / top_k_jsonl_name)
        if not os.path.exists(replay_pool_jsonl):
            if st_idx == 0:
                fallback = str(paths.top_samples_dir / top_k_jsonl_name)
                if os.path.exists(fallback):
                    replay_pool_jsonl = fallback
                    print(
                        f"[pipeline] schedule-compare: using fallback replay pool {fallback}"
                    )
                else:
                    raise FileNotFoundError(
                        f"schedule-compare: replay pool not found at {replay_pool_jsonl} "
                        f"or {fallback}. Run extract-top first."
                    )
            else:
                print(
                    f"[pipeline] schedule-compare: skipping score_type={score_type} "
                    f"(no {replay_pool_jsonl})"
                )
                continue

        def _common_schedule_args() -> list[str]:
            args = [
                "--base_model_path",
                base_model,
                "--tokenizer_path",
                tokenizer_path,
                "--target_train_jsonl",
                train_jsonl,
                "--target_val_jsonl",
                val_jsonl,
                "--replay_pool_jsonl",
                replay_pool_jsonl,
                "--max_length",
                str(step_cfg.get("max_length", 256)),
                "--num_train_epochs",
                str(step_cfg.get("num_train_epochs", 1.0)),
                "--learning_rate",
                str(step_cfg.get("learning_rate", 5e-5)),
                "--per_device_train_batch_size",
                str(step_cfg.get("per_device_train_batch_size", 2)),
                "--per_device_eval_batch_size",
                str(step_cfg.get("per_device_eval_batch_size", 8)),
                "--gradient_accumulation_steps",
                str(step_cfg.get("gradient_accumulation_steps", 2)),
                "--logging_steps",
                str(step_cfg.get("logging_steps", 10)),
                "--eval_steps",
                str(step_cfg.get("eval_steps", 0)),
                "--seed",
                str(step_cfg.get("seed", 42)),
            ] + _ds_fsdp_args(step_cfg)
            if experiment_name:
                args += ["--experiment_name", experiment_name]
            return args

        other_modes = [m for m in replay_modes if m != "none"]
        for ratio in mix_ratios:
            ratio_str = str(ratio).replace(".", "")
            group_label = f"{sc_prefix}ratio{ratio_str}"

            group_run_id_path = paths.schedule_dir / f".swanlab_{group_label}_run_id"

            run_id = None
            if group_run_id_path.exists():
                run_id = group_run_id_path.read_text(encoding="utf-8").strip()
                if not run_id:
                    run_id = None

            is_first = run_id is None
            if is_first and local_rank == 0:
                import swanlab
                run_obj = swanlab.init(
                    project=os.environ.get("SWANLAB_PROJECT", "bif"),
                    experiment_name=group_label,
                    config={"score_type": score_type, "replay_ratio": ratio},
                )
                run_id = run_obj.id
                group_run_id_path.write_text(run_id, encoding="utf-8")

            for schedule in schedules:
                for replay_mode in other_modes:
                    sched_short = "seq" if schedule == "sequential" else "mix"
                    mode_short = "sel" if replay_mode == "selected" else "rnd"
                    metric_prefix = f"{sched_short}_{mode_short}"

                    run_name = f"{group_label}_{schedule}_{replay_mode}"
                    run_dir = str(paths.schedule_dir / run_name)
                    if os.path.exists(os.path.join(run_dir, "run_summary.json")):
                        print(f"[pipeline] schedule-compare: skip {run_name} (done)")
                        continue
                    args = [
                        "schedule-compare",
                        "--output_dir",
                        run_dir,
                        "--run_name",
                        run_name,
                        "--schedule",
                        schedule,
                        "--replay_mode",
                        replay_mode,
                        "--replay_ratio",
                        str(ratio),
                    ] + _common_schedule_args()
                    if run_id:
                        args += ["--swanlab_run_id", run_id]
                    args += ["--metric_prefix", metric_prefix]
                    _run_bif_cmd(args)

    # Log finetune split stats
    for split in ["train", "query", "val", "test"]:
        split_path = str(paths.finetune_jsonl(split))
        if os.path.exists(split_path):
            _log_pool_data_stats(split_path, f"finetune_{split}")


def _step_schedule_analyze(cfg: dict[str, Any], paths: _Paths) -> None:
    """Analyze schedule comparison results and compare losses across mix ratios."""
    if not paths.schedule_dir.exists():
        print("[pipeline] schedule-analyze: no schedule_compare dir, skipping.")
        return

    data_dirs: dict[str, str] = {}
    if os.path.exists(str(paths.pool_jsonl)):
        data_dirs["pt_pool"] = str(paths.pool_jsonl)
    if os.path.exists(str(paths.finetune_pool_jsonl)):
        data_dirs["finetune_pool"] = str(paths.finetune_pool_jsonl)
    if paths.top_samples_dir.exists():
        for jsonl in sorted(paths.top_samples_dir.glob("top_*_full.jsonl")):
            data_dirs["top_samples"] = str(jsonl)
            break

    args = [
        "schedule-analyze",
        "--runs_root",
        str(paths.schedule_dir),
        "--out_dir",
        str(paths.schedule_analysis_dir),
    ]
    for label, path in data_dirs.items():
        args += ["--data_dirs", f"{label}={path}"]

    _run_bif_cmd(args)


_STEP_RUNNERS: dict[str, Any] = {
    "build-pool": _step_build_pool,
    "prepare-finetune": _step_prepare_finetune,
    "train": _step_train,
    "run-bif": _step_run_bif,
    "analyze-bif": _step_analyze_bif,
    "extract-top": _step_extract_top,
    "schedule-compare": _step_schedule_compare,
    "schedule-analyze": _step_schedule_analyze,
}

# Steps that accept an experiment_name argument
_STEPS_WITH_EXPERIMENT_NAME = {"train", "run-bif", "analyze-bif", "extract-top", "schedule-compare"}


def _symlink_shared_data(data_root: str, work_dir: str) -> list[str]:
    """Create symlinks from data_root into work_dir for shared data dirs.

    Only symlinks directories that exist in data_root.  Returns the list of
    directory names that were symlinked.
    """
    src = Path(data_root).resolve()
    dst = Path(work_dir)
    dst.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []
    for d in _SHARED_DIRS:
        src_dir = src / d
        dst_link = dst / d
        if not src_dir.exists():
            continue
        if dst_link.exists() or dst_link.is_symlink():
            if dst_link.is_symlink():
                dst_link.unlink()
            else:
                continue
        dst_link.symlink_to(src_dir)
        linked.append(d)
    return linked


# ─── Public API ───────────────────────────────────────────────────────────────


def run_pipeline(
    config_path: str,
    from_step: str | None = None,
    resume: bool = False,
    new_run: bool = False,
) -> None:
    """Execute the BIF pipeline according to *config_path*.

    Args:
        config_path: Path to the experiment JSON config.
        from_step: If set, skip all steps that appear before this step in
            STEPS (even if they are not marked complete in state).  Useful
            for re-running from a specific stage after a failure.
        resume: If True, skip steps already marked as completed in
            pipeline_state.json so the run can continue after a crash.
        new_run: If True, force a new SwanLab run ID instead of resuming
            the previous one.  Use this when you want a fresh experiment
            in the SwanLab UI.
    """
    cfg = _load_config(config_path)
    work_dir = cfg["work_dir"]
    paths = _Paths(work_dir)
    state = PipelineState.load_or_create(work_dir)

    # ── data_root: reuse shared data from a previous run ───────────────────
    # When data_root is set, symlink the shared dirs (pool, train, bif_traces,
    # etc.) from data_root into work_dir and auto-complete steps 1-6 so only
    # schedule-compare / schedule-analyze run.  This avoids rebuilding the
    # entire PT/SFT pool and re-running BIF when only the mix_ratio changes.
    data_root = cfg.get("data_root")
    if data_root:
        linked = _symlink_shared_data(data_root, work_dir)
        print(f"[pipeline] data_root={data_root}")
        print(f"[pipeline] Symlinked {len(linked)} dirs: {', '.join(linked)}")
        for step in _SHARED_DATA_STEPS:
            if not state.is_done(step):
                state.mark_done(step)
                print(f"[pipeline] Auto-completed (shared data): {step}")

    # All pipeline steps share ONE SwanLab run.  We generate a unique run ID
    # here and pass it via environment variables so every subprocess can
    # resume/join the same cloud run instead of creating a new one.
    # On restart (pipeline_run_log.json already exists), reuse the previous
    # run_id so all metrics land in the same SwanLab run.
    experiment_name = cfg.get("experiment_name", f"pipeline_{Path(work_dir).name}")
    run_log_path = Path(work_dir) / "pipeline_run_log.json"
    if new_run:
        pipeline_run_id = uuid.uuid4().hex
        print(
            f"[pipeline] --new-run: forcing new SwanLab run id={pipeline_run_id[:8]}…"
        )
    elif run_log_path.exists():
        try:
            prev_log = json.loads(run_log_path.read_text(encoding="utf-8"))
            prev_exp = prev_log.get("experiment_name", "")
            if prev_exp == experiment_name:
                pipeline_run_id = prev_log["pipeline_run_id"]
                print(
                    f"[pipeline] Reusing existing SwanLab run id={pipeline_run_id[:8]}…"
                )
            else:
                pipeline_run_id = uuid.uuid4().hex
                print(
                    f"[pipeline] experiment_name changed ({prev_exp!r} → {experiment_name!r}), "
                    f"new SwanLab run id={pipeline_run_id[:8]}…"
                )
        except (KeyError, json.JSONDecodeError):
            pipeline_run_id = uuid.uuid4().hex
    else:
        pipeline_run_id = uuid.uuid4().hex
    os.environ[_ENV_PIPELINE_RUN_ID] = pipeline_run_id
    os.environ[_ENV_PIPELINE_EXPERIMENT] = experiment_name

    project_name = cfg.get("project_name", "bif")
    os.environ[_ENV_SWAN_PROJECT] = project_name

    cuda_vis = cfg.get("cuda_visible_devices")
    if cuda_vis:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_vis)
        print(f"[pipeline] CUDA_VISIBLE_DEVICES={cuda_vis}")

    # Record the experiment name and full config in work_dir so downstream
    # steps (or data_root configs) can recover shared settings.
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    run_log_path.write_text(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "pipeline_run_id": pipeline_run_id,
                "config": config_path,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config_dump_path = Path(work_dir) / "pipeline_config.json"
    config_dump_path.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[pipeline] SwanLab run: {experiment_name!r}  project={project_name!r}  (all steps share one run, id={pipeline_run_id[:8]}…)"
    )
    print(f"[pipeline] Run log: {run_log_path}")

    _swan_init(
        experiment_name=experiment_name,
        config={"config_path": config_path, "work_dir": work_dir},
        tags=["pipeline"],
    )
    _log_shared_data_stats(paths)

    # Determine which steps to run
    start_idx = 0
    if from_step is not None:
        if from_step not in STEPS:
            valid = ", ".join(STEPS)
            raise ValueError(f"Unknown step {from_step!r}. Valid steps: {valid}")
        start_idx = STEPS.index(from_step)

    for step in STEPS[start_idx:]:
        if state.is_done(step):
            print(f"[pipeline] Skipping (already done): {step}")
            continue

        print(f"\n{'=' * 60}")
        print(f"[pipeline] Starting step: {step}")
        print(f"{'=' * 60}")

        runner = _STEP_RUNNERS[step]
        if step == "run-bif":
            result = runner(cfg, paths, resume=resume, experiment_name=experiment_name)
        elif step in _STEPS_WITH_EXPERIMENT_NAME:
            result = runner(cfg, paths, experiment_name=experiment_name)
        else:
            result = runner(cfg, paths)

        if step == "schedule-analyze":
            _relog_schedule_metrics(str(paths.schedule_analysis_dir), str(paths.schedule_dir))

        if result is not None:
            state.mark_done(step)
            print(f"[pipeline] Step complete: {step}")
        else:
            print(f"[pipeline] Step returned None — NOT marking as done")

    print("\n[pipeline] All steps finished.")
    state.print_status()
    _swan_finish_pipeline()
