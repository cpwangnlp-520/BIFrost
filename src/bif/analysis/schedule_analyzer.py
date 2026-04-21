"""Analyze schedule comparison training results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from bif.io import ensure_dir, read_json
from bif.utils.tracker import finish as swan_finish
from bif.utils.tracker import init_run, log_bar, log_heatmap, log_table
from bif.utils.tracker import log as swan_log


def _discover_run_dirs(root: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full) and os.path.exists(
            os.path.join(full, "run_summary.json")
        ):
            out.append(full)
    if not out:
        raise ValueError(f"No run dirs found under {root}")
    return out


def _classify_run(run_name: str) -> dict[str, Any]:
    import re
    name = run_name.lower()
    out: dict[str, Any] = {
        "run_name": run_name,
        "ratio": None,
        "schedule": "unknown",
        "replay_type": "unknown",
        "score_type": None,
        "group_key": "unknown",
    }
    if name.startswith("corr_"):
        out["score_type"] = "corr"
        name = name[len("corr_"):]
    elif name.startswith("raw_"):
        out["score_type"] = "raw"
        name = name[len("raw_"):]
    if "_none" in name or name.endswith("none"):
        out["replay_type"] = "none"
    else:
        ratio_match = re.search(r"ratio(\d+)", name)
        if ratio_match:
            digits = ratio_match.group(1)
            out["ratio"] = int(digits) / 10 ** len(digits)
    if "sequential" in name:
        out["schedule"] = "sequential"
    elif "mixed" in name:
        out["schedule"] = "mixed"
    if out["replay_type"] == "unknown":
        if "selected" in name:
            out["replay_type"] = "selected"
        elif "top_random" in name:
            out["replay_type"] = "top_random"
        elif "random" in name:
            out["replay_type"] = "random"
    st = out["score_type"] or ""
    out["group_key"] = f"{st}_{out['replay_type']}_{out['schedule']}" if st else f"{out['replay_type']}_{out['schedule']}"
    return out


def _collect_run(run_dir: str) -> dict[str, Any]:
    run_name = os.path.basename(os.path.abspath(run_dir))
    read_json(f"{run_dir}/run_config.json")
    read_json(f"{run_dir}/run_summary.json")
    final_eval = read_json(f"{run_dir}/final_eval_metrics.json")
    ds_summary = read_json(f"{run_dir}/dataset_summary.json")

    train_log_path = f"{run_dir}/logs/train_log_history.json"
    eval_log_path = f"{run_dir}/logs/eval_log_history.json"
    log_rows: list[dict[str, Any]] = []
    if os.path.exists(train_log_path):
        try:
            train_obj = read_json(train_log_path)
            if isinstance(train_obj, list):
                log_rows.extend(train_obj)
            elif isinstance(train_obj, dict) and "rows" in train_obj:
                log_rows.extend(train_obj["rows"])
        except Exception:
            pass
    if os.path.exists(eval_log_path):
        try:
            eval_obj = read_json(eval_log_path)
            if isinstance(eval_obj, list):
                log_rows.extend(eval_obj)
            elif isinstance(eval_obj, dict) and "rows" in eval_obj:
                log_rows.extend(eval_obj["rows"])
        except Exception:
            pass
    log_df = pd.DataFrame(log_rows) if log_rows else pd.DataFrame()

    best_eval_loss = None
    best_eval_step = None
    if not log_df.empty and "eval_loss" in log_df.columns:
        eval_df = log_df.dropna(subset=["eval_loss"])
        if not eval_df.empty:
            best_idx = eval_df["eval_loss"].idxmin()
            best_eval_loss = float(eval_df.loc[best_idx, "eval_loss"])
            best_eval_step = int(eval_df.loc[best_idx, "step"])

    meta = _classify_run(run_name)
    return {
        "run_dir": run_dir,
        "run_name": run_name,
        "log_df": log_df,
        "best_eval_loss": best_eval_loss,
        "best_eval_step": best_eval_step,
        "final_eval_loss": final_eval.get("eval_loss"),
        **meta,
        "dataset_summary": ds_summary,
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runs:
        ds = r.get("dataset_summary", {})
        rows.append(
            {
                "run_name": r["run_name"],
                "ratio": r["ratio"],
                "schedule": r["schedule"],
                "replay_type": r["replay_type"],
                "score_type": r.get("score_type"),
                "group_key": r["group_key"],
                "best_eval_loss": r["best_eval_loss"],
                "final_eval_loss": r["final_eval_loss"],
                "replay_examples_used": ds.get("replay_examples_used"),
                "train_examples_total": ds.get("train_examples_total"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["ratio", "schedule", "replay_type"],
            na_position="last",
        ).reset_index(drop=True)
    return df


def _format_run_label(r: dict[str, Any]) -> str:
    st_prefix = f"{r['score_type']}_" if r.get("score_type") else ""
    if r["replay_type"] == "none":
        sched_str = "seq" if r["schedule"] == "sequential" else "mix"
        return f"{st_prefix}no_replay_{sched_str}"
    ratio_str = f"r{r['ratio']}" if r["ratio"] is not None else "r?"
    sched_str = "seq" if r["schedule"] == "sequential" else "mix"
    replay_str = "sel" if r["replay_type"] == "selected" else "rnd"
    return f"{st_prefix}{ratio_str}_{sched_str}_{replay_str}"


def _build_eval_map(r: dict[str, Any]) -> dict[int, float]:
    df = r["log_df"]
    if df.empty or "eval_loss" not in df.columns:
        return {}
    sub = df.dropna(subset=["eval_loss"])
    return dict(zip(sub["step"].astype(int), sub["eval_loss"]))


def _log_replay_domain_analysis(
    pool_path: str,
    top_samples_path: str | None,
    data_dirs: dict[str, str],
) -> None:
    """Log domain composition analysis: pool vs top-k selected vs target.

    This is the key visualization for showing that non-target-domain PT data
    can help the target domain.  Three comparisons:
    1. Pool domain distribution vs top-k domain distribution
    2. Target-domain enrichment in top-k vs pool
    3. Non-target contribution to improvement
    """
    from bif.io import read_jsonl

    pool_rows = read_jsonl(pool_path)
    if not pool_rows:
        return

    import numpy as _np

    pool_src: dict[str, int] = {}
    for r in pool_rows:
        src = str(r.get("source", "unknown"))
        pool_src[src] = pool_src.get(src, 0) + 1

    # Identify target domain from finetune_pool
    target_domain = None
    ft_path = data_dirs.get("finetune_pool")
    if ft_path and os.path.exists(ft_path):
        ft_rows = read_jsonl(ft_path)
        if ft_rows:
            ft_src: dict[str, int] = {}
            for r in ft_rows:
                src = str(r.get("source", "unknown"))
                ft_src[src] = ft_src.get(src, 0) + 1
            if ft_src:
                target_domain = max(ft_src, key=ft_src.get)

    sources = sorted(pool_src, key=pool_src.get, reverse=True)
    pool_total = sum(pool_src.values())

    # Top-k domain distribution
    top_src: dict[str, int] = {}
    top_rows = []
    if top_samples_path and os.path.exists(top_samples_path):
        top_rows = read_jsonl(top_samples_path)
    elif top_samples_path:
        for f in sorted(Path(top_samples_path).parent.glob("top_*_full.jsonl")):
            top_rows = read_jsonl(str(f))
            break

    if not top_rows:
        for f in sorted(
            Path(pool_path).parent.parent.glob("top_samples/top_*_full.jsonl")
        ):
            top_rows = read_jsonl(str(f))
            break

    if top_rows:
        for r in top_rows:
            src = str(r.get("source", "unknown"))
            top_src[src] = top_src.get(src, 0) + 1

    top_total = sum(top_src.values()) if top_src else 1

    # 1. Pool vs Top-k domain fraction comparison
    pool_fracs = [pool_src.get(s, 0) / pool_total for s in sources]
    top_fracs = [top_src.get(s, 0) / top_total for s in sources]

    log_bar(
        "3_schedule/domain_pool_vs_selected",
        xaxis=sources,
        series={
            "pool_frac": [round(f, 4) for f in pool_fracs],
            "topk_frac": [round(f, 4) for f in top_fracs],
        },
    )

    # 2. Enrichment factor: top-k fraction / pool fraction
    enrichment = []
    for i, s in enumerate(sources):
        ef = top_fracs[i] / pool_fracs[i] if pool_fracs[i] > 0 else 0.0
        enrichment.append(round(ef, 2))
    log_bar(
        "3_schedule/domain_enrichment",
        xaxis=sources,
        series={"enrichment_factor": enrichment},
    )

    # 3. Target vs non-target breakdown
    if target_domain:
        target_in_pool = pool_src.get(target_domain, 0) / pool_total
        target_in_topk = top_src.get(target_domain, 0) / top_total
        nontarget_in_topk = 1 - target_in_topk

        log_table(
            "3_schedule/replay_domain_insight",
            headers=["metric", "value"],
            rows=[
                ["target_domain", target_domain],
                ["target_in_pool", f"{target_in_pool:.1%}"],
                ["target_in_topk", f"{target_in_topk:.1%}"],
                ["nontarget_in_topk", f"{nontarget_in_topk:.1%}"],
                [
                    "target_enrichment",
                    f"{target_in_topk / target_in_pool:.2f}x"
                    if target_in_pool > 0
                    else "N/A",
                ],
                ["top_k_size", str(top_total)],
            ]
            + [
                [
                    f"topk_{s}",
                    f"{top_src.get(s, 0) / top_total:.1%} ({top_src.get(s, 0)}/{top_total})",
                ]
                for s in sources
                if top_src.get(s, 0) > 0
            ],
        )

    # 4. Domain heatmap: pool% vs topk% side by side
    if top_src:
        mat = _np.zeros((2, len(sources)))
        for j, s in enumerate(sources):
            mat[0, j] = pool_src.get(s, 0) / pool_total * 100
            mat[1, j] = top_src.get(s, 0) / top_total * 100
        log_heatmap(
            "3_schedule/domain_pool_vs_selected_heatmap",
            xaxis=sources,
            yaxis=["pool%", "topk%"],
            matrix=mat,
            value_label="%",
            precision=1,
        )


def analyze_schedule_compare(
    runs_root: str,
    out_dir: str,
    data_dirs: dict[str, str] | None = None,
) -> None:
    """Analyze and visualize schedule comparison results.

    In pipeline mode, this is the ONLY step that writes schedule metrics to
    SwanLab.  Each schedule-compare run only writes JSON files; this function
    reads them all and logs scalar time-series (with smooth button) into one
    SwanLab run.

    Args:
        data_dirs: Optional mapping of label→JSONL path for data stats logging.
            In pipeline mode this is used to log data distribution even when
            earlier steps (build-pool, prepare-finetune) were skipped via
            data_root.
    """
    ensure_dir(out_dir)

    in_pipeline = bool(os.environ.get("SWANLAB_PIPELINE_RUN_ID"))

    if not in_pipeline:
        init_run(
            experiment_name="schedule_analysis",
            config={"runs_root": runs_root},
            tags=["schedule_compare", "analysis"],
        )

    run_dirs = _discover_run_dirs(runs_root)
    runs = [_collect_run(rd) for rd in run_dirs]

    summary_df = _summarize_runs(runs)
    summary_df.to_csv(f"{out_dir}/run_summary_table.csv", index=False)

    if not in_pipeline:
        _do_swanlog(runs, summary_df, out_dir, data_dirs, runs_root=runs_root)
        swan_finish()


def _do_swanlog(
    runs: list[dict[str, Any]],
    summary_df: pd.DataFrame,
    out_dir: str,
    data_dirs: dict[str, str] | None,
    runs_root: str = "",
) -> None:
    """Log all schedule metrics to SwanLab. Called by main process in pipeline mode."""
    none_runs = [r for r in runs if r["replay_type"] == "none"]
    none_eval_map: dict[int, float] = {}
    if none_runs:
        none_eval_map = _build_eval_map(none_runs[0])

    # ── Per-run native scalar time-series (SwanLab auto-charts) ─────────────
    for r in runs:
        label = _format_run_label(r)
        df = r["log_df"]
        if df.empty:
            continue
        loss_col = "train_loss" if "train_loss" in df.columns else "loss"
        eval_df = (
            df.dropna(subset=["eval_loss"])
            if "eval_loss" in df.columns
            else pd.DataFrame()
        )
        for _, row in eval_df.iterrows():
            metrics: dict[str, float] = {}
            metrics[f"3_schedule/{label}/eval_loss"] = float(row["eval_loss"])
            swan_log(metrics, step=int(row["step"]))
        train_df = (
            df.dropna(subset=[loss_col])
            if loss_col in df.columns
            else pd.DataFrame()
        )
        for _, row in train_df.iterrows():
            metrics: dict[str, float] = {}
            if pd.notna(row.get(loss_col)):
                metrics[f"3_schedule/{label}/train_loss"] = float(row[loss_col])
            if "train_loss_target" in row and pd.notna(row.get("train_loss_target")):
                metrics[f"3_schedule/{label}/train_loss_target"] = float(
                    row["train_loss_target"]
                )
            elif r["replay_type"] == "none" and pd.notna(row.get(loss_col)):
                metrics[f"3_schedule/{label}/train_loss_target"] = float(row[loss_col])
            if "train_loss_replay" in row and pd.notna(row.get("train_loss_replay")):
                metrics[f"3_schedule/{label}/train_loss_replay"] = float(
                    row["train_loss_replay"]
                )
            if "grad_norm" in row and pd.notna(row.get("grad_norm")):
                metrics[f"3_schedule/{label}/grad_norm"] = float(row["grad_norm"])
            if "learning_rate" in row and pd.notna(row.get("learning_rate")):
                metrics[f"3_schedule/{label}/learning_rate"] = float(row["learning_rate"])
            if metrics:
                swan_log(metrics, step=int(row["step"]))

    unique_ratios = sorted(set(r["ratio"] for r in runs if r["ratio"] is not None))

    # ── Per-ratio delta-over-baseline (scalar time-series) ─────────────────
    for ratio in unique_ratios:
        ratio_str = str(ratio).replace(".", "")
        ratio_runs = [r for r in runs if r["ratio"] == ratio]

        sel_runs = [r for r in ratio_runs if r["replay_type"] == "selected"]
        rnd_runs = [r for r in ratio_runs if r["replay_type"] == "random"]

        sel_eval_map = _build_eval_map(sel_runs[0]) if sel_runs else {}
        rnd_eval_map = _build_eval_map(rnd_runs[0]) if rnd_runs else {}

        all_steps = sorted(
            set(sel_eval_map.keys())
            | set(rnd_eval_map.keys())
            | set(none_eval_map.keys())
        )

        for eval_idx, step_val in enumerate(all_steps):
            metrics: dict[str, float] = {}
            sv = sel_eval_map.get(step_val)
            rv = rnd_eval_map.get(step_val)
            nv = none_eval_map.get(step_val)

            if sv is not None and nv is not None:
                metrics[f"3_schedule/delta/ratio{ratio_str}_selected_vs_base"] = sv - nv
            if rv is not None and nv is not None:
                metrics[f"3_schedule/delta/ratio{ratio_str}_random_vs_base"] = rv - nv
            if sv is not None and rv is not None:
                metrics[f"3_schedule/delta/ratio{ratio_str}_sel_vs_rnd_gap"] = sv - rv
            if metrics:
                swan_log(metrics, step=eval_idx)

    # ── Per-ratio final eval_loss scalar (native SwanLab) ──────────────────
    for ratio in unique_ratios:
        ratio_str = str(ratio).replace(".", "")
        for r in runs:
            if r["ratio"] != ratio:
                continue
            v = r.get("final_eval_loss")
            if v is not None:
                swan_log(
                    {f"3_schedule/final_eval_loss/ratio{ratio_str}": float(v)},
                    step=0,
                )
    if none_runs:
        nv = none_runs[0].get("final_eval_loss")
        if nv is not None:
            swan_log({"3_schedule/final_eval_loss/no_replay": float(nv)}, step=0)

    # ── Comparison table ───────────────────────────────────────────────────
    if not summary_df.empty:
        headers = [
            "run_name",
            "ratio",
            "schedule",
            "replay_type",
            "best_eval_loss",
            "final_eval_loss",
            "delta_vs_base",
            "improvement",
        ]
        rows = []
        baseline_loss = none_runs[0].get("final_eval_loss") if none_runs else None
        for _, row in summary_df.iterrows():
            v = row.get("final_eval_loss")
            delta_str = ""
            pct_str = ""
            if baseline_loss is not None and pd.notna(v) and baseline_loss > 0:
                delta = float(v - baseline_loss)
                delta_str = f"{delta:+.4f}"
                pct_str = f"{delta / baseline_loss * 100:+.1f}%"
            rows.append(
                [
                    _format_run_label(row) if pd.notna(row.get("run_name")) else "",
                    str(row.get("ratio", "")),
                    str(row.get("schedule", "")),
                    str(row.get("replay_type", "")),
                    f"{row['best_eval_loss']:.4f}"
                    if pd.notna(row.get("best_eval_loss"))
                    else "",
                    f"{row['final_eval_loss']:.4f}"
                    if pd.notna(row.get("final_eval_loss"))
                    else "",
                    delta_str,
                    pct_str,
                ]
            )
        log_table("3_schedule/comparison_table", headers=headers, rows=rows)

    # ── 2×2 comparison table: (sequential/mixed) × (selected/random) ─────────
    unique_schedules = sorted(set(r["schedule"] for r in runs if r["schedule"] != "unknown"))
    unique_replay = ["selected", "random"]
    unique_score_types = sorted(set(r.get("score_type") for r in runs if r.get("score_type")))
    if not unique_score_types:
        unique_score_types = [None]

    for ratio in unique_ratios:
        for st in unique_score_types:
            st_tag = f"{st}_" if st else ""
            headers_2x2 = [""] + [f"{rt}" for rt in unique_replay]
            rows_2x2 = []
            for sched in unique_schedules:
                row_data = [sched]
                for rt in unique_replay:
                    match = [
                        r for r in runs
                        if r["ratio"] == ratio
                        and r["schedule"] == sched
                        and r["replay_type"] == rt
                        and (st is None or r.get("score_type") == st)
                    ]
                    if match:
                        v = match[0].get("final_eval_loss")
                        row_data.append(f"{v:.4f}" if v is not None else "N/A")
                    else:
                        row_data.append("-")
                rows_2x2.append(row_data)
            none_for_st = [
                r for r in none_runs
                if st is None or r.get("score_type") == st
            ]
            if none_for_st:
                row_data = ["no_replay"]
                for sched in unique_schedules:
                    match = [r for r in none_for_st if r["schedule"] == sched]
                    if match:
                        v = match[0].get("final_eval_loss")
                        row_data.append(f"{v:.4f}" if v is not None else "N/A")
                    else:
                        row_data.append("-")
                rows_2x2.append(row_data)
            log_table(
                f"3_schedule/2x2_{st_tag}ratio{str(ratio).replace('.', '')}",
                headers=headers_2x2,
                rows=rows_2x2,
            )

    # ── Replay domain composition vs pool (influence analysis) ────────────
    replay_pool_path = None
    top_samples_path = None
    if data_dirs:
        for label, path in data_dirs.items():
            if label == "pt_pool":
                replay_pool_path = path
            elif label == "top_samples":
                top_samples_path = path

    if replay_pool_path is None:
        for candidate in [
            os.path.join(os.path.dirname(runs_root), "pool", "pt_pool.jsonl"),
            os.path.join(os.path.dirname(runs_root), "top_samples"),
        ]:
            if os.path.exists(candidate) and candidate.endswith(".jsonl"):
                if "top" in candidate:
                    top_samples_path = candidate
                else:
                    replay_pool_path = candidate

    if replay_pool_path and os.path.exists(replay_pool_path):
        _log_replay_domain_analysis(replay_pool_path, top_samples_path, data_dirs or {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze schedule comparison results.")
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--data_dirs",
        action="append",
        default=[],
        help="label=path pairs for data stats logging (e.g. pt_pool=/path/to/pool.jsonl)",
    )
    args = parser.parse_args()

    data_dirs: dict[str, str] = {}
    for item in args.data_dirs:
        if "=" in item:
            k, v = item.split("=", 1)
            data_dirs[k] = v

    analyze_schedule_compare(args.runs_root, args.out_dir, data_dirs=data_dirs or None)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
