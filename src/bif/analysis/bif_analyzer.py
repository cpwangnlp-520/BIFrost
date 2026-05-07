"""Analyze BIF traces: compute influence scores and generate plots.

Aligned with devinterp's compute_bif() approach:
- BIF = pairwise Pearson correlation between loss traces across sequences
  within the same observable (pool×pool), computed over the chain_draw axis
- Both token-level and sequence-level BIF are supported
- Chain reduction: stack (recommended) or mean across chains

Supports single-process and multi-GPU (torchrun) execution.
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
from bif.utils.naming import guess_model_tag, make_analyze_name
from bif.utils.tracker import finish as swan_finish
from bif.utils.tracker import (
    init_run,
    log_bar,
    log_boxplot,
    log_heatmap,
    log_line,
    log_pie,
    log_scatter,
    log_table,
)
from bif.utils.tracker import log as swan_log


def _get_dist_context() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _init_dist_if_needed() -> None:
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
    if not entries:
        has_chains = any(
            os.path.isdir(os.path.join(root, d))
            and re.fullmatch(r"chain_\d+", d)
            for d in os.listdir(root)
        )
        if has_chains:
            entries = [("final_model", root)]
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


def load_checkpoint_traces(checkpoint_dir: str) -> dict[str, Any]:
    """Load loss traces from a checkpoint directory.

    Supports both new (.npz) and legacy (.jsonl) formats.

    Returns dict with:
        pool_ids, pool_seq_loss, pool_token_loss, pool_meta,
        query_ids, query_seq_loss, query_token_loss, query_meta,
        num_draws
    """
    chain_dirs = _discover_chain_dirs(checkpoint_dir)

    npz_path = os.path.join(chain_dirs[0], "observable_loss_trace.npz")
    if os.path.isfile(npz_path):
        return _load_traces_npz(chain_dirs)

    return _load_traces_legacy(chain_dirs)


def _load_traces_npz(chain_dirs: list[str]) -> dict[str, Any]:
    """Load from new .npz format."""
    pool_seq_parts = []
    pool_token_parts = []
    query_seq_parts = []
    query_token_parts = []
    pool_meta = None
    query_meta = None

    for cdir in chain_dirs:
        pool_npz = np.load(os.path.join(cdir, "observable_loss_trace.npz"))
        pool_seq_parts.append(pool_npz["seq_loss"])
        pool_token_parts.append(pool_npz["token_loss"])

        query_npz = np.load(os.path.join(cdir, "query_loss_trace.npz"))
        query_seq_parts.append(query_npz["seq_loss"])
        query_token_parts.append(query_npz["token_loss"])

        if pool_meta is None:
            import json as _json
            with open(os.path.join(cdir, "observable_meta.json")) as f:
                pool_meta = _json.load(f)
            with open(os.path.join(cdir, "query_meta.json")) as f:
                query_meta = _json.load(f)

    pool_seq = np.concatenate(pool_seq_parts, axis=0)
    pool_token = np.concatenate(pool_token_parts, axis=0)
    query_seq = np.concatenate(query_seq_parts, axis=0)
    query_token = np.concatenate(query_token_parts, axis=0)

    num_chains = len(chain_dirs)
    draws_per_chain = pool_seq.shape[0] // num_chains

    for meta in (pool_meta, query_meta):
        if "source_type" not in meta and "source_types" in meta:
            meta["source_type"] = meta["source_types"]
        if "task_type" not in meta and "task_types" in meta:
            meta["task_type"] = meta["task_types"]

    return {
        "pool_ids": pool_meta["sample_ids"],
        "pool_seq_loss": pool_seq,
        "pool_token_loss": pool_token,
        "pool_meta": pool_meta,
        "query_ids": query_meta["sample_ids"],
        "query_seq_loss": query_seq,
        "query_token_loss": query_token,
        "query_meta": query_meta,
        "num_draws": pool_seq.shape[0],
        "num_chains": num_chains,
        "draws_per_chain": draws_per_chain,
    }


def _load_traces_legacy(chain_dirs: list[str]) -> dict[str, Any]:
    """Load from legacy .jsonl format."""
    pool_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    for cdir in chain_dirs:
        pool_path = os.path.join(cdir, "pool_loss_trace.jsonl")
        query_path = os.path.join(cdir, "query_loss_trace.jsonl")
        if os.path.isfile(pool_path):
            pool_rows.extend(read_jsonl(pool_path))
        if os.path.isfile(query_path):
            query_rows.extend(read_jsonl(query_path))

    pool_ids, pool_mat, pool_meta = rows_to_loss_matrix(pool_rows, "pool")
    query_ids, query_mat, query_meta = rows_to_loss_matrix(query_rows, "query")
    pool_mat, query_mat = _align_by_draw_key(pool_mat, pool_meta, query_mat, query_meta)

    num_draws = pool_mat.shape[0]

    draw_meta = pool_meta.get("draw_meta", [])
    if draw_meta:
        num_chains = len(set(d["chain_id"] for d in draw_meta))
        draws_per_chain = num_draws // num_chains if num_chains > 0 else num_draws
    else:
        num_chains = 1
        draws_per_chain = num_draws

    pool_token_loss = pool_mat[:, :, np.newaxis]
    query_token_loss = query_mat[:, :, np.newaxis]

    return {
        "pool_ids": pool_ids,
        "pool_seq_loss": pool_mat,
        "pool_token_loss": pool_token_loss,
        "pool_meta": pool_meta,
        "query_ids": query_ids,
        "query_seq_loss": query_mat,
        "query_token_loss": query_token_loss,
        "query_meta": query_meta,
        "num_draws": num_draws,
        "num_chains": num_chains,
        "draws_per_chain": draws_per_chain,
    }


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


# ─── BIF Computation (aligned with devinterp) ──────────────────────────────


def compute_bif_pairwise(
    seq_loss: np.ndarray,
    num_chains: int = 1,
    reduce_chains: str = "stack",
) -> np.ndarray:
    """Compute pairwise BIF correlation matrix (aligned with devinterp compute_bif).

    This computes the Pearson correlation between loss traces for all pairs
    of samples within the same observable, across the chain_draw axis.

    This is the correct BIF definition from the paper: BIF(i,j) = corr(L_i, L_j)
    where L_i and L_j are the loss traces of samples i and j across SGLD draws.

    Args:
        seq_loss: Shape (num_draws, n_samples) — sequence-level loss per draw.
            num_draws = num_chains × draws_per_chain when chains are stacked.
        num_chains: Number of chains. Used to reshape for chain reduction.
        reduce_chains: "stack" (recommended) or "mean".

    Returns:
        BIF correlation matrix of shape (n_samples, n_samples).
    """
    if reduce_chains == "stack":
        loss = seq_loss.T  # (n_samples, num_draws)
    elif reduce_chains == "mean":
        draws_per_chain = seq_loss.shape[0] // num_chains
        reshaped = seq_loss.reshape(num_chains, draws_per_chain, -1)
        loss = reshaped.mean(axis=0).T  # (n_samples, draws_per_chain)
    else:
        raise ValueError(f"Unknown reduce_chains: {reduce_chains}")

    loss_t = torch.as_tensor(loss, dtype=torch.float32)
    if torch.cuda.is_available():
        loss_t = loss_t.cuda()

    corr = torch.corrcoef(loss_t)
    return corr.cpu().numpy()


def compute_bif_tokenwise(
    token_loss: np.ndarray,
    num_chains: int = 1,
    reduce_chains: str = "stack",
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Compute token-level BIF (aligned with devinterp _tokenwise_bif).

    Args:
        token_loss: Shape (num_draws, n_samples, context_length).
        num_chains: Number of chains.
        reduce_chains: "stack" or "mean".
        batch_size: Batch size for block processing.
        device: Torch device.

    Returns:
        Token-level BIF of shape (n_samples, n_samples, context_length, context_length).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if reduce_chains == "stack":
        loss = token_loss.transpose(1, 0, 2)
    elif reduce_chains == "mean":
        draws_per_chain = token_loss.shape[0] // num_chains
        reshaped = token_loss.reshape(num_chains, draws_per_chain, -1, token_loss.shape[2])
        loss = reshaped.mean(axis=0).transpose(1, 0, 2)
    else:
        raise ValueError(f"Unknown reduce_chains: {reduce_chains}")

    n_samples = loss.shape[0]
    n_tokens = loss.shape[2]
    result = np.empty((n_samples, n_tokens, n_samples, n_tokens), dtype=np.float32)

    for i in range(0, n_samples, batch_size):
        for j in range(0, n_samples, batch_size):
            bi = min(batch_size, n_samples - i)
            bj = min(batch_size, n_samples - j)
            block_i = torch.as_tensor(loss[i:i+bi], device=device)
            block_j = torch.as_tensor(loss[j:j+bj], device=device)
            block_corr = _batch_corrcoef_tokenwise(block_i, block_j)
            result[i:i+bi, :, j:j+bj, :] = block_corr.cpu().numpy()
            del block_i, block_j, block_corr

    return result.transpose(0, 2, 1, 3)


def _batch_corrcoef_tokenwise(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Batched token-wise correlation.

    Args:
        a: shape (n_a, series_a, observations) — (batch, tokens, draws)
        b: shape (n_b, series_b, observations)

    Returns:
        shape (n_a, n_b, series_a, series_b) cross-correlation block.
    """
    n_a, series_a, n_obs = a.shape
    n_b, series_b, _ = b.shape

    a_centered = a - a.mean(dim=2, keepdim=True)
    b_centered = b - b.mean(dim=2, keepdim=True)

    a_broadcast = a_centered[:, None, :, :].expand(n_a, n_b, series_a, n_obs)
    b_broadcast = b_centered[None, :, :, :].expand(n_a, n_b, series_b, n_obs)
    combined = torch.cat([a_broadcast, b_broadcast], dim=2)

    cov = combined @ combined.transpose(-1, -2) / (n_obs - 1)

    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    std = torch.sqrt(diag)
    cov /= std.unsqueeze(-1) * std.unsqueeze(-2)

    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    cov *= 1 - eye
    cov += eye

    return cov[:, :, :series_a, series_a:]


def compute_bif_scores(
    pool_seq_loss: np.ndarray,
    query_seq_loss: np.ndarray,
    num_chains: int = 1,
    reduce_chains: str = "stack",
    negate_scores: bool = False,
) -> dict[str, np.ndarray]:
    """Compute BIF influence scores.

    Aligned with devinterp's compute_bif() approach:
    - pool_bif_matrix: pairwise BIF within pool (N_pool × N_pool)
    - query_bif_matrix: pairwise BIF within query (N_query × N_query)
    - cross_corr: pool × query cross-correlation (for backward compat)

    The primary score is the mean BIF correlation with other samples,
    which measures how "influential" a sample is in the loss landscape.
    """
    pool_bif_matrix = compute_bif_pairwise(pool_seq_loss, num_chains, reduce_chains)
    query_bif_matrix = compute_bif_pairwise(query_seq_loss, num_chains, reduce_chains)

    n_pool = pool_bif_matrix.shape[0]
    n_query = query_bif_matrix.shape[0]

    np.fill_diagonal(pool_bif_matrix, 0.0)
    np.fill_diagonal(query_bif_matrix, 0.0)

    pool_bif_mean = pool_bif_matrix.mean(axis=1)
    query_bif_mean = query_bif_matrix.mean(axis=1)

    pool_centered = pool_seq_loss - pool_seq_loss.mean(axis=0, keepdims=True)
    query_centered = query_seq_loss - query_seq_loss.mean(axis=0, keepdims=True)
    cross_cov = (pool_centered.T @ query_centered) / pool_centered.shape[0]

    pool_z = _safe_zscore_cols(pool_seq_loss)
    query_z = _safe_zscore_cols(query_seq_loss)
    cross_corr = (pool_z.T @ query_z) / pool_z.shape[0]

    sign = -1.0 if negate_scores else 1.0

    mean_loss = pool_seq_loss.mean(axis=0)

    draw_idx = np.arange(pool_seq_loss.shape[0], dtype=np.float64)
    draw_idx = (draw_idx - draw_idx.mean()) / (draw_idx.std() + 1e-12)
    draw_trend = ((pool_z.T @ draw_idx) / len(draw_idx)).reshape(-1)

    return {
        "bif_mean": sign * pool_bif_mean,
        "bif_matrix": pool_bif_matrix,
        "query_bif_matrix": query_bif_matrix,
        "cross_corr_mean_over_queries": sign * cross_corr.mean(axis=1),
        "cross_corr_matrix": cross_corr,
        "cross_cov_avg_over_queries": sign * cross_cov.mean(axis=1),
        "mean_loss": mean_loss,
        "draw_trend": draw_trend,
    }


def _safe_zscore_cols(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
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
    src_list = pool_meta.get("source_type", [None] * len(pool_ids))
    sub_list = pool_meta.get("subtype", [None] * len(pool_ids))
    task_list = pool_meta.get("task_type", [None] * len(pool_ids))

    if isinstance(src_list, (list, tuple)) and len(src_list) == len(pool_ids):
        pass
    else:
        src_list = [None] * len(pool_ids)

    if isinstance(sub_list, (list, tuple)) and len(sub_list) == len(pool_ids):
        pass
    else:
        sub_list = [None] * len(pool_ids)

    if isinstance(task_list, (list, tuple)) and len(task_list) == len(pool_ids):
        pass
    else:
        task_list = [None] * len(pool_ids)

    df = pd.DataFrame(
        {
            "sample_id": pool_ids,
            "source": src_list,
            "subtype": sub_list,
            "task_type": task_list,
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


def _score_histogram_bars(
    scores: np.ndarray, bins: int = 40
) -> tuple[list[str], list[int]]:
    counts, edges = np.histogram(scores, bins=bins)
    labels = [f"{edges[i]:.3f}" for i in range(len(edges) - 1)]
    return labels, counts.tolist()


def _source_shift_series(
    names: list[str], top_dfs: dict[str, pd.DataFrame], source_col: str
) -> dict[str, list[float]]:
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
    score_cols = [f"score__{ck}" for ck in names]
    sub = traj_df.sort_values(sort_by, ascending=False).head(top_n)
    arr = sub[score_cols].to_numpy(dtype=np.float64)
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
    n_preview = min(50, len(traj_df))

    text_map = {}
    if pool_df is not None:
        id_col = "id" if "id" in pool_df.columns else "sample_id"
        if id_col in pool_df.columns and "text" in pool_df.columns:
            text_map = dict(zip(pool_df[id_col].astype(str), pool_df["text"]))

    def _fmt_text(t: str, max_len: int = 200) -> str:
        s = str(t).strip().replace("\n", " ").replace("\r", " ")
        return s[:max_len] + "..." if len(s) > max_len else s

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


def _log_checkpoint_sample_table(
    df: pd.DataFrame,
    score_col: str,
    top_k: int,
    ck_name: str,
    pool_df: pd.DataFrame | None = None,
) -> None:
    n_preview = min(50, len(df))

    text_map: dict[str, str] = {}
    if pool_df is not None:
        id_col = "id" if "id" in pool_df.columns else "sample_id"
        if id_col in pool_df.columns and "text" in pool_df.columns:
            text_map = dict(zip(pool_df[id_col].astype(str), pool_df["text"]))

    def _fmt(t: str) -> str:
        return str(t).strip().replace("\n", " ").replace("\r", " ")

    headers = ["rank", "sample_id", "source", score_col, "cross_corr", "mean_loss", "text"]
    rows = []
    for rank_i, (_, row) in enumerate(df.head(n_preview).iterrows(), 1):
        sid = str(row.get("sample_id", ""))
        rows.append([
            rank_i,
            sid,
            str(row.get("source", "")),
            f"{row.get(score_col, 0):.4f}",
            f"{row.get('cross_corr_mean_over_queries', 0):.4f}",
            f"{row.get('mean_loss', 0):.4f}",
            _fmt(text_map.get(sid, "")),
        ])

    log_table(
        f"4_2_influence/samples/top/{ck_name}",
        headers=headers,
        rows=rows,
    )

    bottom = df.tail(n_preview).iloc[::-1]
    rows_bot = []
    for rank_i, (_, row) in enumerate(bottom.iterrows(), 1):
        sid = str(row.get("sample_id", ""))
        rows_bot.append([
            rank_i,
            sid,
            str(row.get("source", "")),
            f"{row.get(score_col, 0):.4f}",
            f"{row.get('cross_corr_mean_over_queries', 0):.4f}",
            f"{row.get('mean_loss', 0):.4f}",
            _fmt(text_map.get(sid, "")),
        ])

    log_table(
        f"4_2_influence/samples/bottom/{ck_name}",
        headers=headers,
        rows=rows_bot,
    )


def _process_one_checkpoint(
    ck_name: str,
    ck_dir: str,
    out_dir: str,
    score_col: str,
    top_k: int,
    save_full_query_matrix: bool,
    ck_step: int = 0,
    negate_scores: bool = True,
    pool_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    t0 = time.monotonic()
    try:
        loaded = load_checkpoint_traces(ck_dir)
    except ValueError as exc:
        print(f"[analyze] Skipping {ck_name}: {exc}")
        return {"checkpoint": ck_name, "error": str(exc)}

    num_chains = loaded.get("num_chains", 1)
    scores = compute_bif_scores(
        loaded["pool_seq_loss"],
        loaded["query_seq_loss"],
        num_chains=num_chains,
        reduce_chains="stack",
        negate_scores=negate_scores,
    )
    df = build_pool_score_df(loaded["pool_ids"], loaded["pool_meta"], scores)
    if score_col not in df.columns:
        raise ValueError(f"score_col={score_col!r} not in {df.columns.tolist()}")
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    ck_out = f"{out_dir}/{ck_name}"
    ensure_dir(ck_out)
    df.to_csv(f"{ck_out}/pool_scores.csv", index=False)
    df.head(top_k).to_csv(f"{ck_out}/top_{top_k}.csv", index=False)

    bif_matrix_path = f"{ck_out}/bif_matrix.npy"
    np.save(bif_matrix_path, scores["bif_matrix"])

    save_json(
        f"{ck_out}/ckpt_meta.json",
        {
            "checkpoint": ck_name,
            "num_draws": int(loaded["num_draws"]),
            "pool_size": int(loaded["pool_seq_loss"].shape[1]),
            "query_size": int(loaded["query_seq_loss"].shape[1]),
            "num_chains": num_chains,
        },
    )

    if save_full_query_matrix:
        np.save(
            f"{ck_out}/query_bif_matrix.npy", scores["query_bif_matrix"]
        )

    scores_arr = df[score_col].to_numpy()
    bif_mat = scores["bif_matrix"]
    n_pool = bif_mat.shape[0]
    pool_mat = loaded["pool_seq_loss"]
    query_mat = loaded["query_seq_loss"]

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        _log_loss_traces(pool_mat, query_mat, num_chains, ck_name)
        _log_score_summary(scores_arr, loaded["num_draws"], ck_step)
        _log_score_histogram(scores_arr, ck_name)
        _log_score_vs_selfvar_scatter(
            scores_arr, scores["cross_cov_avg_over_queries"],
            pool_mat, query_mat, ck_name,
        )
        _log_cross_cov_heatmap(
            scores["cross_corr_matrix"], loaded["pool_ids"],
            loaded["query_ids"], ck_name,
            pool_sources=df["source"].fillna("unknown").tolist() if "source" in df.columns else None,
        )
        _log_bif_heatmap_topk(
            bif_mat, df, loaded["pool_ids"], top_k, ck_name,
            score_col=score_col,
        )
        _log_score_by_source(df, score_col, top_k, ck_name)
        _log_eigenvalue_spectrum(bif_mat, ck_name)
        _log_convergence(pool_mat, num_chains, ck_name)
        if num_chains > 1:
            _log_rhat(pool_mat, num_chains, ck_name, ck_step)
            _log_chain_consistency(pool_mat, num_chains, scores_arr, ck_name, ck_step)
            _log_chain_scatter(pool_mat, num_chains, scores_arr, ck_name)

        _log_checkpoint_sample_table(df, score_col, top_k, ck_name, pool_df=pool_df)

        elapsed = time.monotonic() - t0
        swan_log({"4_2_influence/seconds_per_checkpoint": round(elapsed, 2)}, step=ck_step)

    return {
        "checkpoint": ck_name,
        "num_draws": int(loaded["num_draws"]),
        "pool_size": len(loaded["pool_ids"]),
        "query_size": len(loaded.get("query_ids", [])),
        "score_mean": float(df[score_col].mean()),
        "score_std": float(df[score_col].std()),
    }


def _log_loss_traces(
    pool_seq_loss: np.ndarray,
    query_seq_loss: np.ndarray,
    num_chains: int,
    ck_name: str,
) -> None:
    """Loss trace per draw — native swanlab.log time-series with zoom/align."""
    total = pool_seq_loss.shape[0]
    pool_mean = pool_seq_loss.mean(axis=1)
    pool_std = pool_seq_loss.std(axis=1)
    query_mean = query_seq_loss.mean(axis=1)
    query_std = query_seq_loss.std(axis=1)

    if num_chains > 1:
        dpc = total // num_chains
        for draw_idx in range(dpc):
            data = {}
            pool_cvals = []
            query_cvals = []
            for c in range(num_chains):
                offset = c * dpc
                pv = float(pool_seq_loss[offset + draw_idx].mean())
                qv = float(query_seq_loss[offset + draw_idx].mean())
                data[f"1_diag/{ck_name}/pool_loss/chain_{c}"] = pv
                data[f"1_diag/{ck_name}/query_loss/chain_{c}"] = qv
                pool_cvals.append(pv)
                query_cvals.append(qv)
            data[f"1_diag/{ck_name}/pool_loss/chains_mean"] = float(np.mean(pool_cvals))
            data[f"1_diag/{ck_name}/query_loss/chains_mean"] = float(np.mean(query_cvals))
            swan_log(data, step=draw_idx)
    else:
        for draw_idx in range(total):
            swan_log({
                f"1_diag/{ck_name}/pool_loss/mean": float(pool_mean[draw_idx]),
                f"1_diag/{ck_name}/pool_loss/std": float(pool_std[draw_idx]),
                f"1_diag/{ck_name}/query_loss/mean": float(query_mean[draw_idx]),
                f"1_diag/{ck_name}/query_loss/std": float(query_std[draw_idx]),
            }, step=draw_idx)


def _log_score_summary(
    scores_arr: np.ndarray, num_draws: int, ck_step: int,
) -> None:
    swan_log(
        {
            "2_scores/mean": float(scores_arr.mean()),
            "2_scores/std": float(scores_arr.std()),
            "2_scores/p10": float(np.percentile(scores_arr, 10)),
            "2_scores/p50": float(np.percentile(scores_arr, 50)),
            "2_scores/p90": float(np.percentile(scores_arr, 90)),
            "2_scores/positive_frac": float((scores_arr > 0).mean()),
            "2_scores/min": float(scores_arr.min()),
            "2_scores/max": float(scores_arr.max()),
            "2_scores/num_draws": int(num_draws),
        },
        step=ck_step,
    )


def _log_score_histogram(scores_arr: np.ndarray, ck_name: str) -> None:
    labels, counts = _score_histogram_bars(scores_arr, bins=40)
    log_bar(f"2_scores/distribution/{ck_name}", xaxis=labels, series={"count": counts})


def _log_score_vs_selfvar_scatter(
    scores_arr: np.ndarray,
    cross_cov_arr: np.ndarray,
    pool_seq_loss: np.ndarray,
    query_seq_loss: np.ndarray,
    ck_name: str,
) -> None:
    """Cross-cov (influence) vs self-variance: distinguish real influence from noise.

    X = pool sample loss variance over draws (self-var)
    Y = cross-cov with query set (influence score)
    If high influence is just due to high variance, it's noise, not signal.
    Color by score rank: top-20 red, bottom-20 blue, rest gray.
    """
    pool_var = pool_seq_loss.var(axis=0)
    query_mean = query_seq_loss.mean(axis=0)
    n = len(scores_arr)
    max_pts = min(300, n)
    if n > max_pts:
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(n, max_pts, replace=False))
    else:
        idx = np.arange(n)

    log_scatter(
        f"2_scores/cross_cov_vs_selfvar/{ck_name}",
        xaxis_name="pool_self_variance",
        yaxis_name="cross_cov_avg_over_queries",
        series={
            "samples": [(float(pool_var[i]), float(cross_cov_arr[i])) for i in idx],
        },
    )

    loss_mean = pool_seq_loss.mean(axis=0)
    log_scatter(
        f"2_scores/cross_cov_vs_mean_loss/{ck_name}",
        xaxis_name="pool_mean_loss",
        yaxis_name="cross_cov_avg_over_queries",
        series={
            "samples": [(float(loss_mean[i]), float(cross_cov_arr[i])) for i in idx],
        },
    )


def _log_cross_cov_heatmap(
    cross_corr_matrix: np.ndarray,
    pool_ids: list,
    query_ids: list,
    ck_name: str,
    pool_sources: list | None = None,
) -> None:
    """Pool × Query cross-correlation heatmap, aggregated by source when available."""
    n_pool = cross_corr_matrix.shape[0]
    n_query = cross_corr_matrix.shape[1]
    max_query = min(20, n_query)
    query_labels = [f"q{j}" for j in range(max_query)]

    if pool_sources is not None and len(pool_sources) == n_pool:
        sources = sorted(set(pool_sources))
        source_to_indices = {src: [] for src in sources}
        for i in range(n_pool):
            source_to_indices[pool_sources[i]].append(i)

        source_query_mat = np.zeros((len(sources), max_query))
        for i, src in enumerate(sources):
            idx = source_to_indices[src]
            source_query_mat[i] = cross_corr_matrix[idx, :max_query].mean(axis=0)

        source_mean_corr = source_query_mat.mean(axis=1)
        sorted_idx = np.argsort(-source_mean_corr)
        sources_sorted = [sources[i] for i in sorted_idx]
        mat_sorted = source_query_mat[sorted_idx]

        log_heatmap(
            f"3_influence/source_x_query_heatmap/{ck_name}",
            xaxis=query_labels,
            yaxis=sources_sorted,
            matrix=mat_sorted,
            value_label="mean_cross_corr",
        )
    else:
        max_pool = min(50, n_pool)
        pool_labels = [f"p{i}" for i in range(max_pool)]
        log_heatmap(
            f"3_influence/pool_x_query_heatmap/{ck_name}",
            xaxis=query_labels,
            yaxis=pool_labels,
            matrix=cross_corr_matrix[:max_pool, :max_query],
            value_label="cross_corr",
        )


def _log_bif_heatmap_topk(
    bif_mat: np.ndarray,
    df: pd.DataFrame,
    pool_ids: list,
    top_k: int,
    ck_name: str,
    score_col: str = "bif_mean",
) -> None:
    """Top-K pool samples BIF submatrix heatmap + source×source block sorted by score."""
    n_pool = bif_mat.shape[0]
    k = min(top_k, n_pool)
    top_idx = np.arange(k)

    sub_labels = []
    for i in top_idx:
        src = str(df.iloc[i].get("source", ""))[:12] if "source" in df.columns else ""
        sub_labels.append(f"r{i+1}[{src}]" if src else f"r{i+1}")

    log_heatmap(
        f"3_influence/bif_top{k}_heatmap/{ck_name}",
        xaxis=sub_labels,
        yaxis=sub_labels,
        matrix=bif_mat[np.ix_(top_idx, top_idx)],
        value_label="BIF corr",
    )

    if "source" in df.columns:
        sources = sorted(df["source"].fillna("unknown").unique().tolist())
        if 2 <= len(sources) <= 30:
            source_score = {}
            for src in sources:
                mask = (df["source"].fillna("unknown") == src).to_numpy()
                if score_col in df.columns:
                    source_score[src] = float(df.loc[mask, score_col].mean())
                else:
                    source_score[src] = 0.0

            sources_sorted = sorted(sources, key=lambda s: -source_score.get(s, 0))

            source_block = np.zeros((len(sources_sorted), len(sources_sorted)))
            for i, src_i in enumerate(sources_sorted):
                mask_i = (df["source"].fillna("unknown") == src_i).to_numpy()
                for j, src_j in enumerate(sources_sorted):
                    mask_j = (df["source"].fillna("unknown") == src_j).to_numpy()
                    if i == j:
                        ii = np.where(mask_i)[0]
                        if len(ii) > 1:
                            sub = bif_mat[np.ix_(ii, ii)]
                            source_block[i, j] = float(sub[~np.eye(len(ii), dtype=bool)].mean())
                        else:
                            source_block[i, j] = 0.0
                    else:
                        source_block[i, j] = float(bif_mat[np.ix_(mask_i, mask_j)].mean())
            log_heatmap(
                f"3_influence/bif_source_blocks/{ck_name}",
                xaxis=sources_sorted,
                yaxis=sources_sorted,
                matrix=source_block,
                value_label="mean BIF",
            )


def _log_score_by_source(
    df: pd.DataFrame, score_col: str, top_k: int, ck_name: str,
) -> None:
    """Score distribution by source + enrichment metrics."""
    if "source" not in df.columns or score_col not in df.columns:
        return

    sources = sorted(df["source"].fillna("unknown").unique().tolist())
    if len(sources) < 2 or len(sources) > 20:
        return

    box_data = []
    labels = []
    for src in sources:
        vals = df[df["source"].fillna("unknown") == src][score_col].dropna().values
        if len(vals) < 5:
            continue
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        lower = max(float(vals.min()), q1 - 1.5 * iqr)
        upper = min(float(vals.max()), q3 + 1.5 * iqr)
        box_data.append([round(lower, 6), round(q1, 6), round(median, 6), round(q3, 6), round(upper, 6)])
        labels.append(str(src)[:20])

    if len(box_data) >= 2:
        log_boxplot(
            f"3_influence/score_by_source/{ck_name}",
            xaxis=labels,
            series={score_col: box_data},
        )

    topk_src_frac = df.head(top_k)["source"].fillna("unknown").value_counts(normalize=True)
    pool_src_frac = df["source"].fillna("unknown").value_counts(normalize=True)
    enrichment_data = {}
    for src in sources:
        safe_src = src.replace(" ", "_").replace("/", "_")[:30]
        t_frac = float(topk_src_frac.get(src, 0))
        p_frac = float(pool_src_frac.get(src, 0))
        enrichment_data[f"3_influence/enrichment/{safe_src}"] = (t_frac + 1e-9) / (p_frac + 1e-9)
    swan_log(enrichment_data)


def _log_eigenvalue_spectrum(bif_mat: np.ndarray, ck_name: str) -> None:
    """Eigenvalue spectrum of BIF matrix: effective dimensionality of influence."""
    n = bif_mat.shape[0]
    if n > 800:
        return
    symmetric = (bif_mat + bif_mat.T) / 2.0
    np.fill_diagonal(symmetric, 0.0)
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
        eigenvalues = np.sort(eigenvalues)[::-1]
        n_ev = min(20, len(eigenvalues))
        ev_labels = [f"ev{i}" for i in range(n_ev)]
        log_bar(
            f"2_scores/eigenvalue_spectrum/{ck_name}",
            xaxis=ev_labels,
            series={"eigenvalue": [round(v, 6) for v in eigenvalues[:n_ev].tolist()]},
        )
        total_var = float(eigenvalues.sum())
        if total_var > 0:
            top_k_var = [float(eigenvalues[:k].sum()) / total_var for k in [1, 3, 5, 10]]
            swan_log(
                {
                    "2_scores/top1_ev_variance_ratio": top_k_var[0],
                    "2_scores/top3_ev_variance_ratio": top_k_var[1],
                    "2_scores/top5_ev_variance_ratio": top_k_var[2],
                    "2_scores/top10_ev_variance_ratio": top_k_var[3],
                },
            )
    except Exception:
        pass


def _log_convergence(
    pool_seq_loss: np.ndarray,
    num_chains: int,
    ck_name: str,
) -> None:
    """BIF score convergence: native swanlab.log time-series vs n_draws."""
    total_draws = pool_seq_loss.shape[0]
    checkpoints = sorted(set([5, 10, 20, 30, 50, 80, 100, 150, 200] + [total_draws]))
    checkpoints = [c for c in checkpoints if c <= total_draws and c >= 3]
    if len(checkpoints) < 2:
        return

    for n_draws in checkpoints:
        sub_loss = pool_seq_loss[:n_draws]
        try:
            sub_scores = compute_bif_pairwise(sub_loss, num_chains=num_chains, reduce_chains="stack")
            np.fill_diagonal(sub_scores, 0.0)
            bif_mean = sub_scores.mean(axis=1)
            swan_log({
                f"1_diag/{ck_name}/convergence/bif_mean_avg": float(bif_mean.mean()),
                f"1_diag/{ck_name}/convergence/bif_mean_std": float(bif_mean.std()),
            }, step=n_draws)
        except Exception:
            pass


def _log_rhat(
    pool_seq_loss: np.ndarray,
    num_chains: int,
    ck_name: str,
    ck_step: int,
) -> None:
    """Gelman-Rubin R-hat: are chains converged? R-hat < 1.1 means OK."""
    if num_chains < 2:
        return
    draws_per_chain = pool_seq_loss.shape[0] // num_chains
    if draws_per_chain < 5:
        return

    n_samples = pool_seq_loss.shape[1]
    max_samples = min(50, n_samples)

    rhat_values = []
    for s in range(max_samples):
        chains_data = []
        for c in range(num_chains):
            chain_loss = pool_seq_loss[c * draws_per_chain:(c + 1) * draws_per_chain, s]
            chains_data.append(chain_loss)
        chains_arr = np.array(chains_data)

        chain_means = chains_arr.mean(axis=1)
        chain_vars = chains_arr.var(axis=1, ddof=1)

        grand_mean = chain_means.mean()
        B = draws_per_chain / (num_chains - 1) * np.sum((chain_means - grand_mean) ** 2)
        W = chain_vars.mean()

        if W < 1e-12:
            continue
        var_hat = (1 - 1.0 / draws_per_chain) * W + B / draws_per_chain
        rhat = float(np.sqrt(var_hat / W))
        rhat_values.append(rhat)

    if not rhat_values:
        return

    rhat_arr = np.array(rhat_values)
    swan_log(
        {
            "1_diag/rhat/mean": float(rhat_arr.mean()),
            "1_diag/rhat/max": float(rhat_arr.max()),
            "1_diag/rhat/frac_below_1.1": float((rhat_arr < 1.1).mean()),
            "1_diag/rhat/frac_below_1.2": float((rhat_arr < 1.2).mean()),
        },
        step=ck_step,
    )


def _log_chain_consistency(
    pool_seq_loss: np.ndarray,
    num_chains: int,
    full_scores: np.ndarray,
    ck_name: str,
    ck_step: int,
) -> None:
    """Inter-chain Spearman correlation of BIF rankings."""
    draws_per_chain = pool_seq_loss.shape[0] // num_chains
    if draws_per_chain < 3:
        return

    chain_scores = []
    for c in range(num_chains):
        chain_loss = pool_seq_loss[c * draws_per_chain:(c + 1) * draws_per_chain]
        try:
            chain_bif = compute_bif_pairwise(chain_loss, num_chains=1, reduce_chains="stack")
            np.fill_diagonal(chain_bif, 0.0)
            chain_scores.append(chain_bif.mean(axis=1))
        except Exception:
            return

    if len(chain_scores) < 2:
        return

    spearman_pairs = []
    for i in range(len(chain_scores)):
        for j in range(i + 1, len(chain_scores)):
            sp = spearman_from_scores(chain_scores[i], chain_scores[j])
            spearman_pairs.append(sp)

    swan_log(
        {
            "1_diag/chain_spearman/mean": float(np.mean(spearman_pairs)),
            "1_diag/chain_spearman/min": float(np.min(spearman_pairs)),
        },
        step=ck_step,
    )


def _log_chain_scatter(
    pool_seq_loss: np.ndarray,
    num_chains: int,
    scores_arr: np.ndarray,
    ck_name: str,
) -> None:
    """Each chain's score vs mean-of-other-chains — scales to N chains."""
    if num_chains < 2:
        return
    draws_per_chain = pool_seq_loss.shape[0] // num_chains
    if draws_per_chain < 3:
        return

    chain_scores = []
    for c in range(num_chains):
        chain_loss = pool_seq_loss[c * draws_per_chain:(c + 1) * draws_per_chain]
        try:
            chain_bif = compute_bif_pairwise(chain_loss, num_chains=1, reduce_chains="stack")
            np.fill_diagonal(chain_bif, 0.0)
            chain_scores.append(chain_bif.mean(axis=1))
        except Exception:
            return

    chain_scores_arr = np.array(chain_scores)
    max_points = 300
    n = chain_scores_arr.shape[1]
    if n > max_points:
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(n, max_points, replace=False))
    else:
        idx = np.arange(n)

    for c in range(num_chains):
        others = [j for j in range(num_chains) if j != c]
        others_mean = chain_scores_arr[others].mean(axis=0)
        log_scatter(
            f"1_diag/chain_vs_rest/chain_{c}/{ck_name}",
            xaxis_name=f"chain_{c}_score",
            yaxis_name="mean_other_chains",
            series={
                "samples": [(float(chain_scores_arr[c][k]), float(others_mean[k])) for k in idx]
            },
        )


def _global_analysis(
    out_dir: str,
    names: list[str],
    score_col: str,
    top_k: int,
    summary_rows: list[dict[str, Any]],
    pool_df: pd.DataFrame | None = None,
) -> None:
    pd.DataFrame(summary_rows).to_csv(f"{out_dir}/checkpoint_summary.csv", index=False)

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

    _log_sample_table(traj_df, names, score_col, top_k, pool_df=pool_df)

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

    if "source" in traj_df.columns:
        shift_series = _source_shift_series(names, per_ckpt_top, "source")
        log_bar(
            "4_2_influence/source/shift_topk",
            xaxis=names,
            series=shift_series,
            stack=True,
        )

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
    score_col: str = "bif_mean",
    top_k: int = 500,
    save_full_query_matrix: bool = False,
    experiment_name: str | None = None,
    run_name: str | None = None,
    manage_tracking: bool = True,
    negate_scores: bool = False,
) -> None:
    rank, world_size = _get_dist_context()
    ensure_dir(out_dir)

    if rank == 0 and manage_tracking:
        auto_name = make_analyze_name(
            guess_model_tag(bif_root), score_col, top_k,
        )
        init_run(
            experiment_name=experiment_name or auto_name,
            run_name=run_name,
            config={"bif_root": bif_root, "score_col": score_col, "top_k": top_k},
            tags=["analysis"],
        )

    all_ckpts = discover_checkpoint_dirs(bif_root)
    names = [x[0] for x in all_ckpts]

    pool_df: pd.DataFrame | None = None
    for pool_name in ("pt_pool.jsonl", "pool_10k_rebalanced.jsonl"):
        pool_path = os.path.join(bif_root, "..", "pool", pool_name)
        if os.path.exists(pool_path):
            pool_df = pd.DataFrame(read_jsonl(pool_path))
            break

    if pool_df is None:
        search_dirs = [bif_root]
        parent = os.path.dirname(bif_root)
        for _ in range(3):
            if parent and parent != "/":
                search_dirs.append(parent)
                parent = os.path.dirname(parent)
        import json as _json
        for search_dir in search_dirs:
            run_cfg_path = os.path.join(search_dir, "run_config.json")
            if os.path.exists(run_cfg_path):
                with open(run_cfg_path) as _f:
                    _run_cfg = _json.load(_f)
                _pool_jsonl = _run_cfg.get("pool_jsonl", "")
                if _pool_jsonl and os.path.exists(_pool_jsonl):
                    pool_df = pd.DataFrame(read_jsonl(_pool_jsonl))
                    break

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
            negate_scores=negate_scores,
            pool_df=pool_df,
        )
        summary_rows_local.append(row)

    _barrier()

    if rank == 0:
        summary_rows: list[dict[str, Any]] = []
        valid_names: list[str] = []
        for ck_name, _ in all_ckpts:
            csv_path = f"{out_dir}/{ck_name}/pool_scores.csv"
            if not os.path.exists(csv_path):
                print(f"[analyze] Skipping {ck_name} in global analysis (no pool_scores.csv)")
                continue
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
            valid_names.append(ck_name)

        _global_analysis(
            out_dir, valid_names, score_col, top_k, summary_rows, pool_df=pool_df
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

    _barrier()

    if rank == 0 and manage_tracking:
        swan_finish()

    _barrier()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze BIF results (single or multi-GPU via torchrun).",
    )
    parser.add_argument("--bif_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--score_col", default="bif_mean")
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--save_full_query_matrix", action="store_true")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--run_name", default=None)
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
