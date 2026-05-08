#!/usr/bin/env python3
"""Re-organize sweep experiments under SwanLab project 'BIFrost-sweep'.

Two modes:
  1. relog-training  — Re-log training metrics from .npz trace files
  2. re-analyze      — Re-run analyze-bif for all experiments
  3. all             — Run both

Naming convention on SwanLab:
  Training:  sweep-p1-lr1e-5_g100_nb10   (phase1/phase3 prefix)
  Analysis:  analyze-sweep-p1-lr1e-5_g100_nb10

Usage:
  export SWANLAB_API_KEY=...
  CUDA_VISIBLE_DEVICES="" python scripts/reorganize_sweep.py relog-training
  CUDA_VISIBLE_DEVICES="" python scripts/reorganize_sweep.py re-analyze
  CUDA_VISIBLE_DEVICES="" python scripts/reorganize_sweep.py all
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWEEP_BASE = ROOT / "runs" / "hp_sweep"
PROJECT = "BIFrost-sweep"


def _discover_runs():
    rows = []
    for phase_dir in sorted(SWEEP_BASE.glob("phase*")):
        for run_dir in sorted(phase_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            chains = sorted(run_dir.glob("chain_*"))
            chain_count = sum(
                1 for c in chains
                if (c / "observable_loss_trace.npz").exists()
            )
            has_analysis = (run_dir / "analysis" / "checkpoint_summary.csv").exists()
            phase = phase_dir.name
            name = run_dir.name
            rows.append({
                "phase": phase,
                "name": name,
                "run_dir": run_dir,
                "chain_count": chain_count,
                "has_analysis": has_analysis,
            })
    return rows


def _sweep_exp_name(phase: str, name: str) -> str:
    return f"sweep-{phase}-{name}"


def _analyze_exp_name(phase: str, name: str) -> str:
    return f"analyze-sweep-{phase}-{name}"


def _load_run_config(run_dir: Path) -> dict | None:
    p = run_dir / "run_config.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _load_chain_config(chain_dir: Path) -> dict | None:
    p = chain_dir / "chain_config.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def relog_training():
    import numpy as np
    import swanlab

    runs = _discover_runs()
    completed = [r for r in runs if r["chain_count"] >= 2]
    print(f"\n[relog] Found {len(runs)} runs, {len(completed)} with >=2 chains")

    for i, r in enumerate(completed):
        exp_name = _sweep_exp_name(r["phase"], r["name"])
        run_dir = r["run_dir"]
        run_cfg = _load_run_config(run_dir)

        print(f"\n[{i+1}/{len(completed)}] {exp_name}  ({r['chain_count']} chains)")

        try:
            swanlab.init(
                project=PROJECT,
                experiment_name=exp_name,
                config=run_cfg or {},
                tags=["sweep", r["phase"], "training"],
                description=f"Re-logged training metrics from {run_dir}",
            )
        except Exception as e:
            print(f"  [ERROR] init failed: {e}")
            continue

        time.sleep(1)

        for chain_dir in sorted(run_dir.glob("chain_*")):
            chain_id = chain_dir.name
            trace_path = chain_dir / "observable_loss_trace.npz"
            if not trace_path.exists():
                continue

            try:
                data = np.load(trace_path)
                seq_loss = data["seq_loss"]
            except Exception as e:
                print(f"  [WARN] {chain_id}: {e}")
                continue

            chain_cfg = _load_chain_config(chain_dir)
            sgld_cfg = chain_cfg.get("sgld_config", {}) if chain_cfg else {}
            cid = chain_cfg.get("chain_id", 0) if chain_cfg else 0

            pool_mean_per_draw = seq_loss.mean(axis=1)
            pool_std_per_draw = seq_loss.std(axis=1)
            n_draws = len(pool_mean_per_draw)

            query_trace_path = chain_dir / "query_loss_trace.npz"
            if query_trace_path.exists():
                qdata = np.load(query_trace_path)
                q_seq = qdata["seq_loss"]
                q_mean_per_draw = q_seq.mean(axis=1)
                q_std_per_draw = q_seq.std(axis=1)
            else:
                q_mean_per_draw = q_std_per_draw = None

            offset = cid * n_draws
            for d in range(n_draws):
                step = offset + d
                log_data = {
                    f"chain{cid}/pool_loss_mean": float(pool_mean_per_draw[d]),
                    f"chain{cid}/pool_loss_std": float(pool_std_per_draw[d]),
                }
                if q_mean_per_draw is not None:
                    log_data[f"chain{cid}/query_loss_mean"] = float(q_mean_per_draw[d])
                    log_data[f"chain{cid}/query_loss_std"] = float(q_std_per_draw[d])
                    log_data[f"chain{cid}/pool_query_gap"] = float(
                        pool_mean_per_draw[d] - q_mean_per_draw[d]
                    )

                if d == 0:
                    log_data["chain_id"] = cid
                    if sgld_cfg:
                        log_data["lr"] = sgld_cfg.get("lr", 0)
                        log_data["gamma"] = sgld_cfg.get("gamma", 0)
                        log_data["nbeta"] = sgld_cfg.get("nbeta", 0)

                swanlab.log(log_data, step=step)

            print(f"  {chain_id}: logged {n_draws} draws")

        try:
            swanlab.finish()
        except Exception:
            pass
        time.sleep(2)

    print(f"\n[relog] Done — {len(completed)} experiments logged to project '{PROJECT}'")


def re_analyze():
    runs = _discover_runs()
    completable = [r for r in runs if r["chain_count"] >= 2]
    print(f"\n[analyze] Found {len(runs)} runs, {len(completable)} analyable")

    for i, r in enumerate(completable):
        exp_name = _analyze_exp_name(r["phase"], r["name"])
        run_dir = r["run_dir"]
        out_dir = run_dir / "analysis"

        print(f"\n[{i+1}/{len(completable)}] {exp_name}")

        cmd = (
            f"python -m bif.cli analyze-bif"
            f"  --bif_root {run_dir}"
            f"  --out_dir {out_dir}"
            f"  --score_col cross_cov_avg_over_queries"
            f"  --top_k 20"
            f"  --experiment_name {exp_name}"
        )

        env = os.environ.copy()
        env["SWANLAB_PROJECT"] = PROJECT
        env["CUDA_VISIBLE_DEVICES"] = ""

        ret = os.system(cmd)
        if ret != 0:
            print(f"  [WARN] analyze failed (ret={ret})")
        else:
            print(f"  OK")

        time.sleep(2)

    print(f"\n[analyze] Done — {len(completable)} experiments analyzed")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))

    if cmd == "relog-training":
        relog_training()
    elif cmd == "re-analyze":
        re_analyze()
    elif cmd == "all":
        relog_training()
        re_analyze()
    else:
        print(f"Unknown: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
