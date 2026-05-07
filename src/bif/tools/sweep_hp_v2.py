"""BIF Hyperparameter Sweep — Using devinterp's sample() + compute_bif().

Replaces custom QuickObservable + run_one_combo with devinterp's proven pipeline:
  - devinterp.slt.sampling.sample() for SGLD sampling + zarr output
  - devinterp.slt.bif.compute_bif() for BIF correlation computation
  - 9 per-step metrics (grad_norm, noise_norm, localization, distance, etc.)

Usage:
    python -m bif.tools.sweep_hp_v2 --gpus 8 --phase 1
    python -m bif.tools.sweep_hp_v2 --gpus 8 --phase 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from bif.utils.tracker import (
    finish as swan_finish,
    init_run,
    log_bar,
    log_heatmap,
    log_line,
    log_pie,
    log_scatter,
    log_table,
    log as swan_log,
)


MODEL_PATH = "/workspace/pku_percy/models/pythia-70m-step1000"
POOL_JSONL = "/workspace/pku_percy/runs/small_pool_exp/data/pool_800.jsonl"
QUERY_JSONL = "/workspace/pku_percy/runs/small_pool_exp/data/query_gsm8k_20_answer.jsonl"
DATASET_SIZE = 800
BATCH_SIZE = 64
MAX_LENGTH = 1024
SWANLAB_KEY = "RGvhmcyaE940jILdlzCGg"


@dataclass(frozen=True)
class Combo:
    lr: float
    n_beta: float
    gamma: float
    sampler_type: str = "rmsprop_sgld"
    rmsprop_eps: float = 0.1

    @property
    def tag(self) -> str:
        parts = [f"lr{self.lr:.0e}", f"nb{self.n_beta:.0f}", f"g{self.gamma:.0e}"]
        if self.sampler_type == "sgmcmc_sgld":
            parts.append("sgld")
        if self.rmsprop_eps != 0.1:
            parts.append(f"re{self.rmsprop_eps:.2f}")
        return "_".join(parts)


def exp1_rmsprop_coarse() -> list[Combo]:
    combos = []
    for lr in [1e-5, 1e-4, 1e-3]:
        for nb in [1, 5, 10, 50, 100]:
            for g in [0, 1, 5, 10, 100, 1000]:
                combos.append(Combo(lr=lr, n_beta=nb, gamma=g))
    return combos


def exp2_sgld_coarse() -> list[Combo]:
    combos = []
    for lr in [1e-6, 1e-5, 1e-4, 1e-3]:
        for nb in [1, 10, 50, 100]:
            for g in [1, 10, 100, 1000]:
                combos.append(Combo(lr=lr, n_beta=nb, gamma=g, sampler_type="sgmcmc_sgld"))
    return combos


def exp3_rmsprop_eps() -> list[Combo]:
    combos = []
    for lr in [1e-4, 1e-3]:
        for nb in [5, 10, 50]:
            for g in [10, 50, 100]:
                for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
                    combos.append(Combo(lr=lr, n_beta=nb, gamma=g, rmsprop_eps=eps))
    return combos


def exp4_rmsprop_fine() -> list[Combo]:
    combos = []
    for lr in [1e-4, 5e-4, 1e-3]:
        for nb in [2, 5, 10, 50]:
            for g in [5, 10, 50, 100]:
                combos.append(Combo(lr=lr, n_beta=nb, gamma=g))
    return combos


def exp5_nb0_baseline() -> list[Combo]:
    combos = []
    pairs = [
        (1e-4, 5, 10), (1e-4, 10, 50), (1e-4, 50, 100),
        (1e-3, 5, 10), (1e-3, 10, 50), (1e-3, 50, 100),
        (5e-4, 10, 20), (5e-4, 50, 50),
        (1e-3, 20, 100), (2e-3, 10, 50),
        (1e-4, 100, 100), (1e-3, 100, 100),
    ]
    for lr, nb, g in pairs:
        combos.append(Combo(lr=lr, n_beta=nb, gamma=g))
        combos.append(Combo(lr=lr, n_beta=0, gamma=g))
    return combos


PHASE1 = {
    1: ("p1-exp1-rmsprop-coarse", exp1_rmsprop_coarse, 2, 60, 5, 200,
        "Phase1 RMSprop coarse: lr×nβ×γ. 2c×60d, devinterp sample()."),
    2: ("p1-exp2-sgld-coarse", exp2_sgld_coarse, 2, 60, 5, 200,
        "Phase1 vanilla SGLD coarse: lr×nβ×γ. 2c×60d, devinterp sample()."),
    3: ("p1-exp3-rmsprop-eps", exp3_rmsprop_eps, 2, 60, 5, 200,
        "Phase1 RMSprop eps: lr×nβ×γ×eps. 2c×60d, devinterp sample()."),
}

PHASE2 = {
    4: ("p2-exp4-rmsprop-fine", exp4_rmsprop_fine, 3, 150, 8, 500,
        "Phase2 RMSprop fine: denser lr×nβ×γ. 3c×150d, devinterp sample()."),
    5: ("p2-exp5-nb0-baseline", exp5_nb0_baseline, 3, 150, 8, 500,
        "Phase2 nβ=0 baseline. 3c×150d, devinterp sample()."),
}


def _make_datasets(tokenizer, max_length=MAX_LENGTH):
    import json as _json

    def tokenize(texts):
        enc = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length")
        ds = Dataset.from_list([{"input_ids": ids} for ids in enc["input_ids"]])
        ds.set_format(type="torch", columns=["input_ids"])
        return ds

    with open(POOL_JSONL) as f:
        pool_texts = [_json.loads(line)["text"] for line in f]
    with open(QUERY_JSONL) as f:
        query_texts = [_json.loads(line)["text"] for line in f]

    pool_ds = tokenize(pool_texts)
    query_ds = tokenize(query_texts * 4)
    return pool_ds, query_ds


def run_one_combo(
    combo: Combo,
    model: torch.nn.Module,
    pool_ds: Dataset,
    query_ds: Dataset,
    num_chains: int,
    num_draws: int,
    thinning: int,
    burnin: int,
    seed: int,
    device: str,
    output_dir: str | None = None,
) -> dict[str, Any]:
    from devinterp.slt.sampling import sample
    from devinterp.slt.bif import compute_bif

    out_path = None
    if output_dir:
        out_path = os.path.join(output_dir, f"{combo.tag}.zarr")
        if os.path.exists(out_path):
            shutil.rmtree(out_path)

    t0 = time.monotonic()
    try:
        samples = sample(
            model=model,
            dataset=pool_ds,
            observables={
                "pool": (pool_ds, 3),
                "query": (query_ds, 1),
            },
            lr=combo.lr,
            n_beta=combo.n_beta,
            num_chains=num_chains,
            num_draws=num_draws,
            batch_size=BATCH_SIZE,
            num_burnin_steps=burnin,
            num_steps_bw_draws=thinning,
            init_seed=seed,
            localization=combo.gamma,
            noise_level=1.0,
            sampling_method=combo.sampler_type,
            rmsprop_eps=combo.rmsprop_eps if combo.sampler_type == "rmsprop_sgld" else None,
            device=device,
            save_metrics=True,
            output_path=out_path,
        )
    except Exception as e:
        return _error_result(combo, str(e))

    elapsed = time.monotonic() - t0
    ds = samples.dataset

    result = _extract_diagnostics(ds, combo, num_chains, num_draws)
    result["elapsed"] = elapsed

    bif_result = _compute_bif_safe(samples, device)
    result.update(bif_result)

    return result


def _extract_diagnostics(ds, combo, num_chains, num_draws):
    status = "ok"

    loss_pool = ds["loss_pool"].values
    loss_query = ds["loss_query"].values

    nan_frac = float(np.isnan(loss_pool).mean()) + float(np.isnan(loss_query).mean())
    inf_frac = float(np.isinf(loss_pool).mean()) + float(np.isinf(loss_query).mean())

    if inf_frac > 0.05:
        status = "diverged"
    elif nan_frac > 0.05:
        status = "nan"

    seq_loss = loss_pool.mean(axis=-1).mean(axis=-1)
    mean_per_chain = seq_loss.mean(axis=0)
    initial_loss = float(mean_per_chain[0])
    final_loss = float(mean_per_chain[-1])
    loss_increase = final_loss - initial_loss
    chain_std = float(seq_loss.std(axis=0).mean())

    mg = ds.get("metrics_scaled_grad")
    mn = ds.get("metrics_noise")
    ml = ds.get("metrics_localization")
    md = ds.get("metrics_distance")

    snr = 0.0
    mean_grad_norm = 0.0
    mean_noise_norm = 0.0
    mean_localization = 0.0
    mean_distance = 0.0
    if mg is not None and mn is not None:
        gv = mg.values
        nv = mn.values
        mean_grad_norm = float(gv.mean())
        mean_noise_norm = float(nv.mean())
        snr = mean_grad_norm / (mean_noise_norm + 1e-12)
    if ml is not None:
        mean_localization = float(ml.values.mean())
    if md is not None:
        mean_distance = float(md.values.mean())

    autocorr_lag1 = _autocorr_lag1(mean_per_chain) if len(mean_per_chain) > 2 else 0.0
    r_hat = _gelman_rubin(seq_loss) if num_chains >= 2 and status == "ok" else float("nan")

    pool_traces = seq_loss.tolist()

    return {
        "tag": combo.tag, "lr": combo.lr, "n_beta": combo.n_beta,
        "gamma": combo.gamma, "sampler_type": combo.sampler_type,
        "rmsprop_eps": combo.rmsprop_eps,
        "status": status,
        "initial_loss": initial_loss, "final_loss": final_loss,
        "loss_increase": loss_increase,
        "chain_std": chain_std, "snr": snr,
        "mean_grad_norm": mean_grad_norm, "mean_noise_norm": mean_noise_norm,
        "mean_localization": mean_localization, "mean_distance": mean_distance,
        "autocorr_lag1": autocorr_lag1, "r_hat": r_hat,
        "pool_traces": pool_traces,
    }


def _compute_bif_safe(samples, device):
    from devinterp.slt.bif import compute_bif
    try:
        bif_result = compute_bif(
            samples,
            correlation_method="sequence",
            reduce_chain_dimension_method="stack",
            device=device,
        )
        inf = bif_result["influences"].values
        np.fill_diagonal(inf, 0.0)
        off_diag = inf[~np.eye(inf.shape[0], dtype=bool)]
        return {
            "bif_score_mean": float(off_diag.mean()),
            "bif_score_std": float(off_diag.std()),
            "pos_frac": float((off_diag > 0.05).mean()),
        }
    except Exception:
        return {
            "bif_score_mean": float("nan"),
            "bif_score_std": float("nan"),
            "pos_frac": float("nan"),
        }


def _autocorr_lag1(x):
    if len(x) < 3:
        return 0.0
    x0 = x - x.mean()
    d = (x0 ** 2).sum()
    return float(np.sum(x0[:-1] * x0[1:]) / d) if d > 1e-12 else 0.0


def _gelman_rubin(arr):
    nc, nd = arr.shape
    if nc < 2 or nd < 2:
        return float("nan")
    cm = arr.mean(axis=1)
    cv = arr.var(axis=1, ddof=1)
    w = float(cv.mean())
    b = float(cm.var(ddof=1)) * nd
    if w < 1e-12:
        return float("nan")
    return float(math.sqrt(((1 - 1.0 / nd) * w + b / nd) / w))


def _error_result(combo, err):
    nan = float("nan")
    return {
        "tag": combo.tag, "lr": combo.lr, "n_beta": combo.n_beta,
        "gamma": combo.gamma, "sampler_type": combo.sampler_type,
        "rmsprop_eps": combo.rmsprop_eps,
        "status": f"error:{err[:60]}", "initial_loss": nan, "final_loss": nan,
        "loss_increase": nan, "chain_std": nan, "snr": 0.0,
        "mean_grad_norm": 0.0, "mean_noise_norm": 0.0,
        "mean_localization": 0.0, "mean_distance": 0.0,
        "autocorr_lag1": 0.0, "r_hat": nan,
        "bif_score_mean": nan, "bif_score_std": nan, "pos_frac": nan,
        "pool_traces": [],
    }


def classify(r):
    s = r["status"]
    if s != "ok":
        return s
    li = r["loss_increase"]
    snr = r["snr"]
    cs = r["chain_std"]
    if math.isnan(li):
        return "nan_loss"
    if li < -0.01:
        return "loss_decreased"
    if li < 0.01:
        return "too_little_signal"
    if snr < 0.01:
        return "too_much_noise"
    if cs > 1.0:
        return "chains_disagree"
    return "good"


def _gpu_worker(gpu_id, combos, result_dir,
                num_chains, num_draws, thinning, burnin, seed, zarr_base):
    try:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pool_ds, query_ds = _make_datasets(tokenizer)

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
        model.to(device)

        zarr_dir = os.path.join(zarr_base, f"gpu{gpu_id}")
        os.makedirs(zarr_dir, exist_ok=True)

        pbar = tqdm(combos, desc=f"GPU{gpu_id}", position=gpu_id, leave=True)
        for combo in pbar:
            pbar.set_postfix_str(combo.tag[:40])
            try:
                result = run_one_combo(
                    combo, model, pool_ds, query_ds,
                    num_chains, num_draws, thinning, burnin,
                    seed, device, output_dir=zarr_dir,
                )
            except Exception as e:
                result = _error_result(combo, str(e))

            path = os.path.join(result_dir, f"gpu{gpu_id}_{combo.tag}.json")
            with open(path, "w") as f:
                json.dump(result, f, default=_json_default)
                f.flush()
                os.fsync(f.fileno())

        marker = os.path.join(result_dir, f"_done_gpu{gpu_id}")
        with open(marker, "w") as f:
            f.write(f"completed {len(combos)} combos\n")

        del model
        torch.cuda.empty_cache()
        print(f"GPU{gpu_id} DONE — {len(combos)} combos", file=sys.stderr)
    except Exception as e:
        print(f"GPU{gpu_id} FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def _collect_results(result_dir):
    results = []
    for fname in sorted(os.listdir(result_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(result_dir, fname)) as f:
                r = json.load(f)
            r["_source_file"] = fname
            results.append(r)
        except Exception:
            pass
    results.sort(key=lambda r: (r.get("lr", 0), r.get("n_beta", 0), r.get("gamma", 0)))
    return results


def _safe(v):
    return 0.0 if isinstance(v, float) and math.isnan(v) else v


def _fmt(v):
    return "nan" if isinstance(v, float) and math.isnan(v) else f"{v:.4f}"


def log_experiment(results, exp_name, exp_desc, sampling_config):
    lrs = sorted(set(r["lr"] for r in results))
    nbs = sorted(set(r["n_beta"] for r in results))
    gammas = sorted(set(r["gamma"] for r in results))

    swan_config = {
        "model": "pythia-70m-step1000",
        "pool_size": 800, "query_size": 20, "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "num_combos": len(results),
        "lr_range": [f"{v:.0e}" for v in lrs],
        "nbeta_range": [f"{v:.0f}" for v in nbs],
        "gamma_range": [f"{v:.0e}" for v in gammas],
        "engine": "devinterp",
        "guide": "https://timaeus.co/research/2026-04-21-sampling-guide",
        **sampling_config,
    }

    init_run(
        experiment_name=exp_name,
        description=exp_desc,
        config=swan_config,
        tags=["sweep", exp_name],
        project="bif-sweep",
    )

    for i, r in enumerate(results):
        tag = r["tag"]
        swan_log({
            f"m/{tag}/dloss": _safe(r.get("loss_increase", 0)),
            f"m/{tag}/snr": _safe(r.get("snr", 0)),
            f"m/{tag}/cstd": _safe(r.get("chain_std", 0)),
            f"m/{tag}/bif": _safe(r.get("bif_score_mean", 0)),
            f"m/{tag}/rhat": _safe(r.get("r_hat", 0)),
            f"m/{tag}/ac1": _safe(r.get("autocorr_lag1", 0)),
            f"m/{tag}/dist": _safe(r.get("mean_distance", 0)),
            f"m/{tag}/loc": _safe(r.get("mean_localization", 0)),
        }, step=i)

    for r in results:
        traces = r.get("pool_traces", [])
        if not traces or not traces[0]:
            continue
        nd = len(traces[0])
        xa = [str(j) for j in range(nd)]
        series = {f"c{c}": [round(v, 6) for v in ch] for c, ch in enumerate(traces)}
        if len(traces) > 1:
            series["mean"] = [round(v, 6) for v in np.mean(traces, axis=0).tolist()]
        log_line(f"trace/{r['tag']}", xaxis=xa, series=series)

    _log_charts(results)
    _log_tables(results)
    _log_pie(results)

    swan_finish()


def _log_charts(results):
    ok = [r for r in results if r.get("status") == "ok"]
    good = [r for r in results if classify(r) == "good"]

    for stype, sname in [("rmsprop_sgld", "rms"), ("sgmcmc_sgld", "sgld")]:
        sub = [r for r in ok if r.get("sampler_type") == stype]
        if not sub:
            continue
        for lr_val in sorted(set(r["lr"] for r in sub)):
            lr_sub = [r for r in sub if r["lr"] == lr_val]
            nb_list = sorted(set(r["n_beta"] for r in lr_sub))
            g_list = sorted(set(r["gamma"] for r in lr_sub))
            if len(nb_list) >= 2 and len(g_list) >= 2:
                mat_li = np.zeros((len(g_list), len(nb_list)))
                mat_snr = np.zeros((len(g_list), len(nb_list)))
                for r in lr_sub:
                    i, j = g_list.index(r["gamma"]), nb_list.index(r["n_beta"])
                    mat_li[i, j] = _safe(r.get("loss_increase", 0))
                    mat_snr[i, j] = _safe(r.get("snr", 0))
                log_heatmap(f"hm/{sname}/dloss_lr{lr_val:.0e}",
                            [f"nb{v:.0f}" for v in nb_list],
                            [f"g{v:.0e}" for v in g_list], mat_li, "dloss")
                log_heatmap(f"hm/{sname}/snr_lr{lr_val:.0e}",
                            [f"nb{v:.0f}" for v in nb_list],
                            [f"g{v:.0e}" for v in g_list], mat_snr, "SNR")

    if good:
        pts = [(r["loss_increase"], r["snr"]) for r in good]
        log_scatter("scatter/dloss_vs_snr", "dloss", "SNR", {"good": pts})


def _log_tables(results):
    headers = ["#", "lr", "nβ", "γ", "sampler", "eps", "status", "dloss", "SNR", "Rhat", "bif", "cstd"]
    rows = []
    for i, r in enumerate(results):
        rows.append([
            i, f"{r['lr']:.0e}", f"{r['n_beta']:.0f}", f"{r['gamma']:.0e}",
            r.get("sampler_type", "").replace("rmsprop_sgld", "rms").replace("sgmcmc_sgld", "sgld"),
            f"{r.get('rmsprop_eps', 0.1):.2f}", classify(r),
            _fmt(r.get("loss_increase", float("nan"))),
            _fmt(r.get("snr", 0)),
            _fmt(r.get("r_hat", float("nan"))),
            _fmt(r.get("bif_score_mean", float("nan"))),
            _fmt(r.get("chain_std", float("nan"))),
        ])
    log_table("table/all", headers=headers, rows=rows)

    good = sorted([r for r in results if classify(r) == "good"],
                   key=lambda x: x.get("snr", 0), reverse=True)
    if good:
        g_rows = []
        for r in good[:100]:
            g_rows.append([
                f"{r['lr']:.0e}", f"{r['n_beta']:.0f}", f"{r['gamma']:.0e}",
                r.get("sampler_type", "").replace("rmsprop_sgld", "rms").replace("sgmcmc_sgld", "sgld"),
                _fmt(r.get("loss_increase", float("nan"))),
                _fmt(r.get("snr", 0)),
                _fmt(r.get("r_hat", float("nan"))),
                _fmt(r.get("bif_score_mean", float("nan"))),
                _fmt(r.get("chain_std", float("nan"))),
                _fmt(r.get("autocorr_lag1", 0)),
            ])
        log_table("table/top100_by_snr",
                  headers=["lr", "nβ", "γ", "sampler", "dloss", "SNR", "Rhat", "bif", "cstd", "ac1"],
                  rows=g_rows)


def _log_pie(results):
    sc = {}
    for r in results:
        c = classify(r)
        sc[c] = sc.get(c, 0) + 1
    if sc:
        log_pie("pie/classification", "combos", list(sc.items()))


def run_experiment(exp_id, combos, num_chains, num_draws, thinning, burnin,
                   num_gpus, seed=42):
    entry = PHASE1.get(exp_id, PHASE2.get(exp_id))
    exp_name = entry[0]
    exp_desc = entry[6]
    total_steps = burnin + num_draws * thinning

    print(f"\n{'='*70}")
    print(f"  {exp_name}")
    print(f"  {exp_desc}")
    print(f"  {len(combos)} combos × {num_chains}c × {num_draws}d × thin{thinning} + burn{burnin} = {total_steps} steps")
    print(f"{'='*70}")

    out_dir = f"/workspace/pku_percy/runs/hp_sweep/{exp_name}"
    result_dir = os.path.join(out_dir, "results")
    zarr_base = os.path.join(out_dir, "zarr")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(zarr_base, exist_ok=True)

    n_gpu = min(num_gpus, torch.cuda.device_count())
    chunks = [[] for _ in range(n_gpu)]
    for i, c in enumerate(combos):
        chunks[i % n_gpu].append(c)

    print(f"  Distributing {len(combos)} combos across {n_gpu} GPUs:")
    for gid in range(n_gpu):
        if chunks[gid]:
            print(f"    GPU{gid}: {len(chunks[gid])} combos")

    ctx = mp.get_context("spawn")
    procs = []
    for gid in range(n_gpu):
        if not chunks[gid]:
            continue
        p = ctx.Process(target=_gpu_worker, args=(
            gid, chunks[gid], result_dir,
            num_chains, num_draws, thinning, burnin, seed, zarr_base,
        ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    results = _collect_results(result_dir)
    print(f"  Collected {len(results)}/{len(combos)} results")

    sc = {}
    for r in results:
        s = classify(r)
        sc[s] = sc.get(s, 0) + 1
    print(f"  Status: {dict(sorted(sc.items()))}")

    good = sorted([r for r in results if classify(r) == "good"],
                   key=lambda x: x.get("snr", 0), reverse=True)
    if good:
        print(f"  Top 5 by SNR:")
        for r in good[:5]:
            print(f"    lr={r['lr']:.0e} nβ={r['n_beta']:.0f} γ={r['gamma']:.0e} "
                  f"Δloss={r['loss_increase']:.4f} SNR={r['snr']:.4f} "
                  f"bif={r.get('bif_score_mean', 'nan'):.4f} R̂={r.get('r_hat', 'nan'):.4f}")

    print(f"  Logging to SwanLab (project=bif-sweep)...")
    log_experiment(results, exp_name, exp_desc, {
        "num_chains": num_chains, "draws_per_chain": num_draws,
        "thinning": thinning, "burnin": burnin, "total_steps": total_steps,
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="BIF HP Sweep v2 (devinterp)")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None)
    args = parser.parse_args()

    import swanlab
    swanlab.login(api_key=SWANLAB_KEY)

    t0 = time.monotonic()

    if args.phase is None or args.phase == 1:
        for eid in sorted(PHASE1.keys()):
            entry = PHASE1[eid]
            run_experiment(eid, entry[1](), entry[2], entry[3], entry[4], entry[5],
                           args.gpus, args.seed)

    if args.phase is None or args.phase == 2:
        for eid in sorted(PHASE2.keys()):
            entry = PHASE2[eid]
            run_experiment(eid, entry[1](), entry[2], entry[3], entry[4], entry[5],
                           args.gpus, args.seed)

    elapsed = time.monotonic() - t0
    print(f"\n{'='*70}")
    print(f"  ALL DONE  ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
