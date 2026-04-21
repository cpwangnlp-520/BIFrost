"""Analyze BIF traces: compute influence scores and generate plots.

Supports single-process and multi-GPU (torchrun) execution.

Multi-GPU strategy
------------------
Checkpoints are distributed across ranks (rank handles ckpts[rank::world_size]).
Each rank independently loads traces, computes scores, saves per-checkpoint CSVs
and plots.  After a barrier, rank 0 loads all CSVs and runs the global
trajectory / stability analysis and uploads everything to SwanLab.

Usage (single GPU / CPU):
    python -m bif.analysis.bif_analyzer --bif_root ... --out_dir ...

Usage (multi-GPU, e.g. 8 cards):
    torchrun --nproc_per_node=8 -m bif.analysis.bif_analyzer \\
        --bif_root ... --out_dir ...
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from bif.io import ensure_dir, read_jsonl, save_json
from bif.utils.tracker import finish as swan_finish
from bif.utils.tracker import init_run, log_bar, log_heatmap, log_line, log_table
from bif.utils.tracker import log as swan_log

# ─── Distributed helpers ──────────────────────────────────────────────────────


def _get_dist_context() -> tuple[int, int]:
    """Return (rank, world_size). Works with or without torchrun."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _init_dist_if_needed() -> None:
    """Init NCCL/Gloo process group when launched via torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _checkpoint_sort_key(name: str) -> tuple[int, str]:
    if name == "base_model":
        return (-1, name)
    if name == "final_model":
        return (10**9, name)
    m = re.fullmatch(r"checkpoint-(\d+)", name)
    if m:
        return (int(m.group(1)), name)
    return (10**8, name)


def discover_checkpoint_dirs(root: str) -> list[tuple[str, str]]:
    entries = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full) and (
            name in ("base_model", "final_model")
            or re.fullmatch(r"checkpoint-\d+", name)
        ):
            entries.append((name, full))
    entries.sort(key=lambda x: _checkpoint_sort_key(x[0]))
    if not entries:
        raise ValueError(f"No checkpoint dirs under {root}")
    return entries


def _discover_chain_dirs(checkpoint_dir: str) -> list[str]:
    out = []
    for name in os.listdir(checkpoint_dir):
        full = os.path.join(checkpoint_dir, name)
        if os.path.isdir(full) and re.fullmatch(r"chain_\d+", name):
            out.append(full)
    out.sort()
    if not out:
        raise ValueError(f"No chain dirs under {checkpoint_dir}")
    return out


def rows_to_loss_matrix(
    rows: list[dict[str, Any]], dataset_name: str
) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
    rows = [r for r in rows if r.get("dataset") == dataset_name]
    if not rows:
        raise ValueError(f"No rows for dataset={dataset_name}")
    rows.sort(
        key=lambda r: (
            int(r["chain_id"]),
            int(r["draw_in_chain"]),
        )
    )

    template = None
    for r in rows:
        ids = r.get("sample_ids", [])
        losses = r.get("losses", [])
        if (
            isinstance(ids, list)
            and isinstance(losses, list)
            and ids
            and len(ids) == len(losses)
        ):
            template = r
            break
    if template is None:
        raise ValueError(f"No valid rows for dataset={dataset_name}")

    sample_ids = list(template["sample_ids"])
    n = len(sample_ids)
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    template_id_set = set(sample_ids)

    first_src = template.get("source_types", [None] * n)
    first_sub = template.get("subtypes", [None] * n)
    first_task = template.get("task_types", [None] * n)

    valid_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for r in rows:
        ids = r.get("sample_ids", [])
        losses = r.get("losses", [])
        if not isinstance(ids, list) or not isinstance(losses, list):
            dropped.append({"reason": "not_list", "chain_id": r.get("chain_id")})
            continue
        if not ids or not losses:
            dropped.append({"reason": "empty", "chain_id": r.get("chain_id")})
            continue
        if len(ids) != len(losses):
            dropped.append({"reason": "length_mismatch", "chain_id": r.get("chain_id")})
            continue
        if set(ids) != template_id_set:
            dropped.append({"reason": "id_set_mismatch", "chain_id": r.get("chain_id")})
            continue
        if len(set(ids)) != len(ids):
            dropped.append({"reason": "duplicate_ids", "chain_id": r.get("chain_id")})
            continue
        valid_rows.append(r)

    if not valid_rows:
        raise ValueError(f"All rows dropped for dataset={dataset_name}")

    mat = np.full((len(valid_rows), n), np.nan, dtype=np.float64)
    draw_meta = []

    for draw_idx, r in enumerate(valid_rows):
        for sid, loss in zip(r["sample_ids"], r["losses"]):
            mat[draw_idx, id_to_idx[sid]] = float(loss)
        draw_meta.append(
            {
                "chain_id": int(r["chain_id"]),
                "draw_in_chain": int(r["draw_in_chain"]),
                "global_draw": int(r["global_draw"]),
            }
        )

    good_mask = ~np.isnan(mat).any(axis=1)
    if not np.all(good_mask):
        mat = mat[good_mask]
        draw_meta = [dm for dm, g in zip(draw_meta, good_mask) if g]

    if mat.shape[0] == 0:
        raise ValueError(f"All rows invalid for dataset={dataset_name}")

    meta: dict[str, Any] = {
        "source_type": list(first_src),
        "subtype": list(first_sub),
        "task_type": list(first_task),
        "draw_meta": draw_meta,
        "num_rows_valid": int(mat.shape[0]),
        "num_rows_dropped": len(dropped),
        "dropped_rows": dropped[:200],
    }
    return sample_ids, mat, meta


def _align_by_draw_key(
    pool_mat: np.ndarray,
    pool_meta: dict[str, Any],
    query_mat: np.ndarray,
    query_meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    pool_keys = [(d["chain_id"], d["draw_in_chain"]) for d in pool_meta["draw_meta"]]
    query_keys = [(d["chain_id"], d["draw_in_chain"]) for d in query_meta["draw_meta"]]
    common = sorted(set(pool_keys) & set(query_keys))
    if not common:
        raise ValueError("No common draws between pool and query")
    pool_idx = {k: i for i, k in enumerate(pool_keys)}
    query_idx = {k: i for i, k in enumerate(query_keys)}
    pi = [pool_idx[k] for k in common]
    qi = [query_idx[k] for k in common]
    return pool_mat[pi], query_mat[qi]


def load_checkpoint_traces(checkpoint_dir: str) -> dict[str, Any]:
    chain_dirs = _discover_chain_dirs(checkpoint_dir)
    pool_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    for cdir in chain_dirs:
        pool_path = f"{cdir}/pool_loss_trace.jsonl"
        query_path = f"{cdir}/query_loss_trace.jsonl"
        pool_rows.extend(read_jsonl(pool_path))
        query_rows.extend(read_jsonl(query_path))

    pool_ids, pool_mat, pool_meta = rows_to_loss_matrix(pool_rows, "pool")
    query_ids, query_mat, query_meta = rows_to_loss_matrix(query_rows, "query")
    pool_mat, query_mat = _align_by_draw_key(pool_mat, pool_meta, query_mat, query_meta)

    common_keys = sorted(
        set((d["chain_id"], d["draw_in_chain"]) for d in pool_meta["draw_meta"])
        & set((d["chain_id"], d["draw_in_chain"]) for d in query_meta["draw_meta"])
    )
    for meta in (pool_meta, query_meta):
        meta["draw_meta"] = [
            d
            for d in meta["draw_meta"]
            if (d["chain_id"], d["draw_in_chain"]) in common_keys
        ]

    return {
        "pool_ids": pool_ids,
        "pool_loss_mat": pool_mat,
        "pool_meta": pool_meta,
        "query_ids": query_ids,
        "query_loss_mat": query_mat,
        "query_meta": query_meta,
        "num_draws": pool_mat.shape[0],
    }


def compute_bif_scores(
    pool_loss_mat: np.ndarray,
    query_loss_mat: np.ndarray,
) -> dict[str, np.ndarray]:
    pool_signal = -pool_loss_mat
    query_signal = -query_loss_mat

    pool_center = pool_signal - pool_signal.mean(axis=0, keepdims=True)
    query_center = query_signal - query_signal.mean(axis=0, keepdims=True)

    cov_matrix = (pool_center.T @ query_center) / pool_center.shape[0]
    raw_cov_avg = cov_matrix.mean(axis=1)

    pool_z = _safe_zscore_rows(pool_signal)
    query_z = _safe_zscore_rows(query_signal)
    corr_matrix = (pool_z.T @ query_z) / pool_z.shape[0]
    corr_avg = corr_matrix.mean(axis=1)
    corr_absmean = np.abs(corr_matrix).mean(axis=1)

    draw_idx = np.arange(pool_signal.shape[0], dtype=np.float64)
    draw_idx = (draw_idx - draw_idx.mean()) / (draw_idx.std() + 1e-12)
    draw_trend = ((pool_z.T @ draw_idx) / len(draw_idx)).reshape(-1)

    return {
        "raw_cov_avg_over_queries": raw_cov_avg,
        "corr_mean_over_queries": corr_avg,
        "corr_absmean_over_queries": corr_absmean,
        "draw_trend": draw_trend,
        "cov_matrix": cov_matrix,
        "query_pair_corr_matrix": corr_matrix,
    }


def _safe_zscore_rows(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (mat - mu) / sd


def average_rank(scores: np.ndarray, descending: bool = True) -> np.ndarray:
    order = np.argsort(-scores if descending else scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return ranks


def spearman_from_scores(a: np.ndarray, b: np.ndarray) -> float:
    ra = average_rank(a, descending=True)
    rb = average_rank(b, descending=True)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.mean(ra * rb))


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    a_top = set(np.argsort(-a)[:k].tolist())
    b_top = set(np.argsort(-b)[:k].tolist())
    return len(a_top & b_top) / float(k)


def build_pool_score_df(
    pool_ids: list[Any],
    pool_meta: dict[str, Any],
    score_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sample_id": pool_ids,
            "source": pool_meta.get("source_type", [None] * len(pool_ids)),
            "subtype": pool_meta.get("subtype", [None] * len(pool_ids)),
            "task_type": pool_meta.get("task_type", [None] * len(pool_ids)),
        }
    )
    for k, v in score_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(pool_ids):
            df[k] = v
    return df


def make_global_trajectory_df(
    per_ckpt_df: dict[str, pd.DataFrame], score_col: str
) -> pd.DataFrame:
    names = list(per_ckpt_df.keys())
    base_ids = per_ckpt_df[names[0]]["sample_id"].tolist()
    merged = pd.DataFrame({"sample_id": base_ids})
    source_map = per_ckpt_df[names[0]][
        ["sample_id", "source", "subtype", "task_type"]
    ].copy()
    merged = merged.merge(source_map, on="sample_id", how="left")

    for ck in names:
        cur = per_ckpt_df[ck][["sample_id", score_col]].copy()
        cur = cur.rename(columns={score_col: f"score__{ck}"})
        merged = merged.merge(cur, on="sample_id", how="left")

    score_cols = [f"score__{ck}" for ck in names]
    merged["traj_mean"] = merged[score_cols].mean(axis=1)
    merged["traj_std"] = merged[score_cols].std(axis=1)
    merged["traj_min"] = merged[score_cols].min(axis=1)
    merged["traj_max"] = merged[score_cols].max(axis=1)
    merged["emergence_last_minus_first"] = (
        merged[score_cols[-1]] - merged[score_cols[0]]
    )
    arr = merged[score_cols].to_numpy(dtype=np.float64)
    merged["num_positive_deltas"] = (np.diff(arr, axis=1) > 0).sum(axis=1)
    return merged


# ---- Native SwanLab echarts chart builders ──────────────────────────────────
# These functions build chart data structures consumed by log_heatmap /
# log_bar / log_line in tracker.py.  No matplotlib is needed here.


def _score_histogram_bars(
    scores: np.ndarray, bins: int = 40
) -> tuple[list[str], list[int]]:
    """Return (bin_labels, counts) for a histogram bar chart."""
    counts, edges = np.histogram(scores, bins=bins)
    labels = [f"{edges[i]:.3f}" for i in range(len(edges) - 1)]
    return labels, counts.tolist()


def _source_shift_series(
    names: list[str], top_dfs: dict[str, pd.DataFrame], source_col: str
) -> dict[str, list[float]]:
    """Return {source: [fraction_per_checkpoint]} for a stacked bar chart."""
    all_sources = sorted(
        {
            s
            for ck in names
            for s in top_dfs[ck][source_col].fillna("unknown").astype(str).unique()
        }
    )
    series: dict[str, list[float]] = {}
    for src in all_sources:
        series[src] = [
            float(
                top_dfs[ck][source_col]
                .fillna("unknown")
                .astype(str)
                .value_counts(normalize=True)
                .get(src, 0.0)
            )
            for ck in names
        ]
    return series


def _trajectory_stats_series(
    traj_df: pd.DataFrame, names: list[str], sort_by: str, top_n: int
) -> dict[str, list[float]]:
    """Return percentile-band series for the top-N trajectories.

    Instead of plotting one line per sample (slow, overlapping), we compute
    p25 / median / p75 across the top-N samples at each checkpoint — this
    gives a compact band chart that loads instantly and is much easier to read.
    """
    score_cols = [f"score__{ck}" for ck in names]
    sub = traj_df.sort_values(sort_by, ascending=False).head(top_n)
    arr = sub[score_cols].to_numpy(dtype=np.float64)  # shape (top_n, n_ckpts)
    series: dict[str, list[float]] = {
        "p75": [
            round(float(np.percentile(arr[:, j], 75)), 6) for j in range(len(names))
        ],
        "median": [
            round(float(np.percentile(arr[:, j], 50)), 6) for j in range(len(names))
        ],
        "p25": [
            round(float(np.percentile(arr[:, j], 25)), 6) for j in range(len(names))
        ],
        "mean": [round(float(arr[:, j].mean()), 6) for j in range(len(names))],
    }
    return series


def _checkpoint_sort_index(name: str, all_names: list[str]) -> int:
    sorted_names = sorted(all_names, key=_checkpoint_sort_key)
    return sorted_names.index(name) if name in sorted_names else 0


def _log_sample_table(
    traj_df: pd.DataFrame,
    names: list[str],
    score_col: str,
    top_k: int,
    pool_df: pd.DataFrame | None = None,
) -> None:
    """Log sample preview tables to SwanLab.

    Three tables are logged:
      1. Top samples by traj_mean — highest average influence across checkpoints
      2. Top emergent samples — largest score increase from first to last checkpoint
      3. Consistently positive samples — score increased at every checkpoint step
    Each table shows rank, source, score columns per checkpoint,
    traj_mean, emergence, and text preview (when pool_df available).
    """
    n_preview = min(50, len(traj_df))

    # Merge text from pool_df if available
    if pool_df is not None and "id" in pool_df.columns:
        text_map = dict(
            zip(pool_df["id"].astype(str), pool_df.get("text", [""] * len(pool_df)))
        )
    else:
        text_map = {}

    def _fmt_text(t: str) -> str:
        return str(t).strip().replace("\n", " ")

    def _build_rows(sub_df: pd.DataFrame) -> list[list[Any]]:
        score_cols = [
            f"score__{ck}" for ck in names if f"score__{ck}" in sub_df.columns
        ]
        rows = []
        for rank_i, (_, row) in enumerate(sub_df.head(n_preview).iterrows(), 1):
            r = [rank_i, str(row.get("source", ""))]
            for sc in score_cols:
                v = row.get(sc)
                r.append(f"{v:.4f}" if pd.notna(v) else "")
            r.append(f"{row.get('traj_mean', 0):.4f}")
            r.append(f"{row.get('emergence_last_minus_first', 0):.4f}")
            sid = str(row.get("sample_id", ""))
            r.append(_fmt_text(text_map.get(sid, "")))
            rows.append(r)
        return rows

    ck_short = [
        ck.replace("checkpoint-", "ck").replace("final_model", "final") for ck in names
    ]
    headers = ["rank", "source"] + ck_short + ["traj_mean", "emergence", "text"]

    top_mean = traj_df.head(n_preview)
    log_table(
        "4_2_influence/samples/top",
        headers=headers,
        rows=_build_rows(top_mean),
    )


def _process_one_checkpoint(
    ck_name: str,
    ck_dir: str,
    out_dir: str,
    score_col: str,
    top_k: int,
    save_full_query_matrix: bool,
    ck_step: int = 0,
) -> dict[str, Any]:
    """Load traces, compute scores, save CSVs and plots for one checkpoint.

    Returns a summary dict.  Called by every rank in parallel.

    Args:
        ck_step: Integer index of this checkpoint in the sorted checkpoint
            sequence.  Used as the SwanLab step so that metrics from
            different checkpoints form a continuous time series instead of
            each appearing as a single isolated point.
    """
    t0 = time.monotonic()
    loaded = load_checkpoint_traces(ck_dir)
    scores = compute_bif_scores(loaded["pool_loss_mat"], loaded["query_loss_mat"])
    df = build_pool_score_df(loaded["pool_ids"], loaded["pool_meta"], scores)
    if score_col not in df.columns:
        raise ValueError(f"score_col={score_col!r} not in {df.columns.tolist()}")
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    ck_out = f"{out_dir}/{ck_name}"
    ensure_dir(ck_out)
    df.to_csv(f"{ck_out}/pool_scores.csv", index=False)
    df.head(top_k).to_csv(f"{ck_out}/top_{top_k}.csv", index=False)

    save_json(
        f"{ck_out}/ckpt_meta.json",
        {
            "checkpoint": ck_name,
            "num_draws": int(loaded["num_draws"]),
            "pool_size": int(loaded["pool_loss_mat"].shape[1]),
            "query_size": int(loaded["query_loss_mat"].shape[1]),
            "num_rows_dropped_pool": int(
                loaded["pool_meta"].get("num_rows_dropped", 0)
            ),
            "num_rows_dropped_query": int(
                loaded["query_meta"].get("num_rows_dropped", 0)
            ),
        },
    )

    if save_full_query_matrix:
        np.save(
            f"{ck_out}/query_pair_corr_matrix.npy", scores["query_pair_corr_matrix"]
        )

    scores_arr = df[score_col].to_numpy()

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        labels, counts = _score_histogram_bars(scores_arr, bins=40)
        log_bar(
            f"4_2_influence/scores/distribution/{ck_name}",
            xaxis=labels,
            series={"count": counts},
        )

        swan_log(
            {
                "4_2_influence/scores/mean": float(scores_arr.mean()),
                "4_2_influence/scores/std": float(scores_arr.std()),
                "4_2_influence/scores/p10": float(np.percentile(scores_arr, 10)),
                "4_2_influence/scores/p50": float(np.percentile(scores_arr, 50)),
                "4_2_influence/scores/p90": float(np.percentile(scores_arr, 90)),
                "4_2_influence/scores/positive_frac": float((scores_arr > 0).mean()),
                "4_2_influence/scores/min": float(scores_arr.min()),
                "4_2_influence/scores/max": float(scores_arr.max()),
                "4_2_influence/num_draws": int(loaded["num_draws"]),
            },
        step=ck_step,
    )

    # ── SGLD sampling quality (per checkpoint, time-series) ───────────────
    pool_mat = loaded["pool_loss_mat"]
    query_mat = loaded["query_loss_mat"]
    draw_meta = loaded["pool_meta"].get("draw_meta", [])

    pool_mean_per_draw = pool_mat.mean(axis=1)
    query_mean_per_draw = query_mat.mean(axis=1)
    gap_per_draw = pool_mean_per_draw - query_mean_per_draw

    if rank == 0:
        swan_log(
            {
                "4_2_influence/sgld_pool_query_gap": float(gap_per_draw.mean()),
                "4_2_influence/sgld_pool_loss_mean": float(pool_mean_per_draw.mean()),
                "4_2_influence/sgld_query_loss_mean": float(query_mean_per_draw.mean()),
            },
            step=ck_step,
        )

        if draw_meta:
            chain_ids = sorted(set(d["chain_id"] for d in draw_meta))
            draws_per_chain = len(draw_meta) // len(chain_ids) if chain_ids else 1
            last_draw_pool = [
                float(pool_mean_per_draw[c * draws_per_chain + draws_per_chain - 1])
                for c in range(len(chain_ids))
                if c * draws_per_chain + draws_per_chain - 1 < len(pool_mean_per_draw)
            ]
            last_draw_query = [
                float(query_mean_per_draw[c * draws_per_chain + draws_per_chain - 1])
                for c in range(len(chain_ids))
                if c * draws_per_chain + draws_per_chain - 1 < len(query_mean_per_draw)
            ]
            if len(last_draw_pool) > 1:
                    swan_log(
                        {
                            "4_2_influence/sgld_chain_variance_pool": float(np.var(last_draw_pool)),
                            "4_2_influence/sgld_chain_variance_query": float(np.var(last_draw_query)),
                        },
                    step=ck_step,
                )

        elapsed = time.monotonic() - t0
        swan_log({"4_2_influence/seconds_per_checkpoint": round(elapsed, 2)}, step=ck_step)

    return {
        "checkpoint": ck_name,
        "num_draws": int(loaded["num_draws"]),
        "pool_size": len(loaded["pool_ids"]),
        "query_size": len(loaded["query_ids"]),
        "score_mean": float(df[score_col].mean()),
        "score_std": float(df[score_col].std()),
    }


def _global_analysis(
    out_dir: str,
    names: list[str],
    score_col: str,
    top_k: int,
    summary_rows: list[dict[str, Any]],
    pool_df: pd.DataFrame | None = None,
) -> None:
    """Cross-checkpoint trajectory / stability analysis.  Runs on rank 0 only."""
    pd.DataFrame(summary_rows).to_csv(f"{out_dir}/checkpoint_summary.csv", index=False)

    # Re-load per-checkpoint CSVs (written by all ranks) so rank 0 has full data
    per_ckpt_df: dict[str, pd.DataFrame] = {}
    score_vecs: dict[str, np.ndarray] = {}
    per_ckpt_top: dict[str, pd.DataFrame] = {}
    for ck_name in names:
        csv_path = f"{out_dir}/{ck_name}/pool_scores.csv"
        df = pd.read_csv(csv_path)
        per_ckpt_df[ck_name] = df
        score_vecs[ck_name] = df[score_col].to_numpy()
        per_ckpt_top[ck_name] = df.head(top_k).copy()

    traj_df = make_global_trajectory_df(per_ckpt_df, score_col)
    traj_df = traj_df.sort_values("traj_mean", ascending=False).reset_index(drop=True)
    traj_df.to_csv(f"{out_dir}/trajectory_scores.csv", index=False)

    # ── SwanLab: Top-K sample score table with text preview ───
    _log_sample_table(traj_df, names, score_col, top_k, pool_df=pool_df)

    # ── SwanLab: trajectory & emergence scalars as time-series ───────────
    # All logged with step=checkpoint_index so they form continuous curves
    # instead of single isolated points.
    for ck_idx, ck_name in enumerate(names):
        ck_col = f"score__{ck_name}"
        if ck_col in traj_df.columns:
            swan_log(
                {
                    "4_2_influence/trajectory/topk_mean": float(
                        traj_df.head(top_k)[ck_col].mean()
                    ),
                    "4_2_influence/trajectory/topk_std": float(
                        traj_df.head(top_k)[ck_col].std()
                    ),
                },
                step=ck_idx,
            )

    if "source" in traj_df.columns:
        # ── Source × checkpoint mean-score heatmap ────────────────────────
        sources = sorted(traj_df["source"].dropna().unique().tolist())
        ck_short = [
            n.replace("checkpoint-", "ck").replace("final_model", "final")
            for n in names
        ]
        if sources and len(names) > 1:
            src_ck_mat = np.zeros((len(sources), len(names)))
            for ck_idx, ck_name in enumerate(names):
                ck_col = f"score__{ck_name}"
                if ck_col not in traj_df.columns:
                    continue
                for src_idx, src in enumerate(sources):
                    src_vals = traj_df[traj_df["source"] == src][ck_col]
                    if not src_vals.empty:
                        src_ck_mat[src_idx, ck_idx] = float(src_vals.mean())
            log_heatmap(
                "4_2_influence/source/score_vs_checkpoint",
                xaxis=ck_short,
                yaxis=sources,
                matrix=src_ck_mat,
                value_label="mean_score",
            )

        # ── Source × checkpoint top-K count heatmap ──────────────────────
        if sources and len(names) > 1:
            count_mat = np.zeros((len(sources), len(names)))
            for ck_idx, ck_name in enumerate(names):
                top_ck = per_ckpt_top.get(ck_name)
                if top_ck is None or "source" not in top_ck.columns:
                    continue
                vc = top_ck["source"].fillna("unknown").value_counts()
                for src_idx, src in enumerate(sources):
                    count_mat[src_idx, ck_idx] = int(vc.get(src, 0))
            log_heatmap(
                "4_2_influence/source/topk_count_vs_checkpoint",
                xaxis=ck_short,
                yaxis=sources,
                matrix=count_mat,
                value_label="count_in_topK",
            )

    # ── Native echarts: Trajectory band lines ────────────────────────────────
    # Each chart shows p25/median/p75/mean across the top-20 samples — compact,
    # fast to render, and overlap-free compared to 20 individual lines.
    if len(names) > 1:
        top_series = _trajectory_stats_series(
            traj_df, names, sort_by="traj_mean", top_n=20
        )
        log_line(
            "4_2_influence/trajectory/top20_by_mean",
            xaxis=names,
            series=top_series,
            smooth=True,
        )

        emergent_series = _trajectory_stats_series(
            traj_df, names, sort_by="emergence_last_minus_first", top_n=20
        )
        log_line(
            "4_2_influence/trajectory/top20_emergent",
            xaxis=names,
            series=emergent_series,
            smooth=True,
        )

    # ── Native echarts: Source shift stacked bar ───────────────────────────────
    if "source" in traj_df.columns:
        shift_series = _source_shift_series(names, per_ckpt_top, "source")
        log_bar(
            "4_2_influence/source/shift_topk",
            xaxis=names,
            series=shift_series,
            stack=True,
        )

    # ── Source enrichment across checkpoints (restored from original) ──────
    source_rows = []
    for ck in names:
        cur = per_ckpt_df[ck]
        top = cur.head(top_k)
        top_counts = top["source"].fillna("unknown").value_counts(normalize=True)
        all_counts = cur["source"].fillna("unknown").value_counts(normalize=True)
        all_sources = sorted(
            set(top_counts.index.tolist()) | set(all_counts.index.tolist())
        )
        for src in all_sources:
            source_rows.append(
                {
                    "checkpoint": ck,
                    "source": src,
                    "top_fraction": float(top_counts.get(src, 0.0)),
                    "all_fraction": float(all_counts.get(src, 0.0)),
                    "enrichment_ratio": float(
                        (top_counts.get(src, 0.0) + 1e-12)
                        / (all_counts.get(src, 0.0) + 1e-12)
                    ),
                }
            )
    pd.DataFrame(source_rows).to_csv(
        f"{out_dir}/source_enrichment_topk.csv", index=False
    )


def analyze_bif_results(
    bif_root: str,
    out_dir: str,
    score_col: str = "raw_cov_avg_over_queries",
    top_k: int = 500,
    save_full_query_matrix: bool = False,
    experiment_name: str = "bif_analysis",
    run_name: str | None = None,
    manage_tracking: bool = True,
) -> None:
    """Run full BIF analysis across all checkpoints.

    Supports single-process and multi-GPU (torchrun) execution.

    Multi-GPU strategy
    ------------------
    Each rank processes a disjoint subset of checkpoints in parallel.
    After a barrier, rank 0 loads all per-checkpoint CSVs and performs
    the global trajectory / stability analysis.

    Single-GPU / CPU
    ----------------
    Call directly or via ``python -m bif.analysis.bif_analyzer``.

    Multi-GPU
    ---------
    ``torchrun --nproc_per_node=N -m bif.analysis.bif_analyzer --bif_root ...``
    """
    rank, world_size = _get_dist_context()
    ensure_dir(out_dir)

    # Only rank 0 initialises SwanLab; others silently no-op via tracker
    if rank == 0 and manage_tracking:
        init_run(
            experiment_name=experiment_name,
            run_name=run_name,
            config={"bif_root": bif_root, "score_col": score_col, "top_k": top_k},
            tags=["analysis"],
        )

    all_ckpts = discover_checkpoint_dirs(bif_root)
    names = [x[0] for x in all_ckpts]

    # Load pool text for sample previews
    pool_df: pd.DataFrame | None = None
    for pool_name in ("pt_pool.jsonl", "pool_10k_rebalanced.jsonl"):
        pool_path = os.path.join(bif_root, "..", "pool", pool_name)
        if os.path.exists(pool_path):
            pool_df = pd.DataFrame(read_jsonl(pool_path))
            break

    # ── Phase 1: parallel per-checkpoint processing ───────────────────────
    # Each rank handles every world_size-th checkpoint starting at its rank.
    # We pass ck_step (the checkpoint's index in the sorted list) so that
    # SwanLab scalar metrics form continuous time-series across checkpoints.
    assigned = all_ckpts[rank::world_size]
    summary_rows_local: list[dict[str, Any]] = []
    for ck_name, ck_dir in assigned:
        ck_step = names.index(ck_name) if ck_name in names else 0
        row = _process_one_checkpoint(
            ck_name,
            ck_dir,
            out_dir,
            score_col,
            top_k,
            save_full_query_matrix,
            ck_step=ck_step,
        )
        summary_rows_local.append(row)

    # ── Synchronise: wait for all ranks to finish writing CSVs ───────────
    _barrier()

    # ── Phase 2: global analysis on rank 0 only ───────────────────────────
    if rank == 0:
        # Collect summary rows in checkpoint order (all_ckpts order).
        # Read num_draws from the per-checkpoint ckpt_meta.json written by
        # whichever rank processed that checkpoint — avoids the bug where
        # summary_rows_local only contains entries for THIS rank's checkpoints.
        summary_rows: list[dict[str, Any]] = []
        for ck_name, _ in all_ckpts:
            csv_path = f"{out_dir}/{ck_name}/pool_scores.csv"
            meta_path = f"{out_dir}/{ck_name}/ckpt_meta.json"
            df = pd.read_csv(csv_path)
            meta: dict[str, Any] = {}
            if os.path.exists(meta_path):
                import json as _json

                with open(meta_path, encoding="utf-8") as _f:
                    meta = _json.load(_f)
            summary_rows.append(
                {
                    "checkpoint": ck_name,
                    "score_mean": float(df[score_col].mean()),
                    "score_std": float(df[score_col].std()),
                    "num_draws": int(meta.get("num_draws", 0)),
                    "pool_size": len(df),
                }
            )

        _global_analysis(
            out_dir, names, score_col, top_k, summary_rows, pool_df=pool_df
        )

        save_json(
            f"{out_dir}/analysis_config.json",
            {
                "bif_root": bif_root,
                "score_col": score_col,
                "top_k": top_k,
                "checkpoint_names": names,
                "world_size": world_size,
            },
        )

    # Barrier 1: keep non-zero ranks alive while rank 0 finishes writing files.
    # swan_finish() can block on network I/O; do it AFTER this barrier so that
    # if it hangs it does NOT cause other ranks to time out waiting for barrier 2.
    _barrier()

    if rank == 0 and manage_tracking:
        swan_finish()

    # Barrier 2: prevent non-zero ranks from calling destroy_process_group()
    # before rank 0 returns from swan_finish().  Without this, the process group
    # may be torn down while rank 0 is still in a collective inside swan_finish.
    _barrier()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze BIF results (single or multi-GPU via torchrun).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU / CPU
  python -m bif.analysis.bif_analyzer --bif_root TRACES --out_dir OUT

  # Multi-GPU (e.g. 8 cards) — checkpoints distributed across ranks
  torchrun --nproc_per_node=8 -m bif.analysis.bif_analyzer \\
      --bif_root TRACES --out_dir OUT
""",
    )
    parser.add_argument("--bif_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--score_col", default="raw_cov_avg_over_queries")
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--save_full_query_matrix", action="store_true")
    parser.add_argument(
        "--experiment_name",
        default="bif_analysis",
        help="SwanLab experiment name for this analysis run.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="SwanLab run display name within the experiment (e.g. 'analysis' in pipeline mode).",
    )
    args = parser.parse_args()

    _init_dist_if_needed()
    try:
        analyze_bif_results(
            bif_root=args.bif_root,
            out_dir=args.out_dir,
            score_col=args.score_col,
            top_k=args.top_k,
            save_full_query_matrix=args.save_full_query_matrix,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )
        rank, _ = _get_dist_context()
        if rank == 0:
            print("Analysis complete.")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
