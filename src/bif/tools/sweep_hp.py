"""BIF Hyperparameter Sweep — Following Timaeus Sampling Guide.

Guide: https://timaeus.co/research/2026-04-21-sampling-guide
Paper: https://arxiv.org/abs/2509.26544

Two-phase approach (as recommended by guide §3):

  Phase 1 — Quick Screening  (~2 hr on 8×H200)
    Exp1  RMSprop coarse:  ε × nβ × γ       90 combos
    Exp2  SGLD coarse:     ε × nβ × γ       64 combos
    Exp3  RMSprop eps:     ε × nβ × γ × a   90 combos
    → 4 chains × 100 draws, thinning=5, burn-in=200, total=700 steps

  Phase 2 — Deep Run  (~6.6 hr on 8×H200)
    Exp4  RMSprop fine:    denser grid       80 combos
    Exp5  nβ=0 baseline:   verify gradient   24 combos
    Exp6  Collection:      burn-in & thin    36 combos
    → 4 chains × 200 draws, thinning=10, burn-in=500, total=2500 steps

Workers write JSON results per combo. Main process collects & logs to SwanLab.

Usage:
    python -m bif.tools.sweep_hp --gpus 8
    python -m bif.tools.sweep_hp --gpus 8 --phase 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from bif.config import SGLDConfig
from bif.data.dataset import (
    JsonlSequenceDataset,
    get_batch_by_indices,
    move_batch_to_device,
)
from bif.io import ensure_dir
from bif.training.loss import per_example_causal_lm_loss
from bif.training.sgld import create_sampler
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


# ─── Constants ──────────────────────────────────────────────────────────────

MODEL_PATH = "/workspace/pku_percy/models/pythia-70m-step1000"
POOL_JSONL = "/workspace/pku_percy/runs/small_pool_exp/data/pool_800.jsonl"
QUERY_JSONL = "/workspace/pku_percy/runs/small_pool_exp/data/query_gsm8k_20_answer.jsonl"
DATASET_SIZE = 800
EVAL_BATCH = 64
TRAIN_BATCH = 64
MAX_LENGTH = 1024


# ─── Combo ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Combo:
    lr: float
    n_beta: float
    gamma: float
    sampler_type: str = "rmsprop_sgld"
    rmsprop_eps: float = 0.1
    num_burnin_steps: int = 0
    num_steps_bw_draws: int = 1

    @property
    def tag(self) -> str:
        parts = [f"lr{self.lr:.0e}", f"nb{self.n_beta:.0f}", f"g{self.gamma:.0e}"]
        if self.sampler_type == "sgld":
            parts.append("sgld")
        if self.rmsprop_eps != 0.1:
            parts.append(f"re{self.rmsprop_eps:.2f}")
        if self.num_burnin_steps > 0:
            parts.append(f"b{self.num_burnin_steps}")
        if self.num_steps_bw_draws > 1:
            parts.append(f"t{self.num_steps_bw_draws}")
        return "_".join(parts)


# ─── Experiment grids ───────────────────────────────────────────────────────


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
                combos.append(Combo(lr=lr, n_beta=nb, gamma=g, sampler_type="sgld"))
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


def exp6_collection_params() -> list[Combo]:
    combos = []
    settings = [
        (1e-3, 10, 50),
        (1e-3, 50, 100),
        (5e-4, 10, 20),
    ]
    for lr, nb, g in settings:
        for bi in [0, 200, 500, 1000]:
            for th in [5, 10, 20]:
                combos.append(Combo(lr=lr, n_beta=nb, gamma=g, num_burnin_steps=bi, num_steps_bw_draws=th))
    return combos


PHASE1 = {
    1: ("p1-exp1-rmsprop-coarse", exp1_rmsprop_coarse, 2, 60, 5, 200,
        "Phase1 RMSprop coarse sweep: lr×nβ×γ. 2 chains, 60 draws, pool_subset=200.",
        200),
    2: ("p1-exp2-sgld-coarse", exp2_sgld_coarse, 2, 60, 5, 200,
        "Phase1 vanilla SGLD coarse sweep: lr×nβ×γ. 2 chains, 60 draws, pool_subset=200.",
        200),
    3: ("p1-exp3-rmsprop-eps", exp3_rmsprop_eps, 2, 60, 5, 200,
        "Phase1 RMSprop eps sweep: lr×nβ×γ×rmsprop_eps. 2 chains, 60 draws, pool_subset=200.",
        200),
}

PHASE2 = {
    4: ("p2-exp4-rmsprop-fine", exp4_rmsprop_fine, 3, 150, 8, 500,
        "Phase2 RMSprop fine grid: denser lr×nβ×γ. 3 chains, 150 draws, pool_subset=300.",
        300),
    5: ("p2-exp5-nb0-baseline", exp5_nb0_baseline, 3, 150, 8, 500,
        "Phase2 nβ=0 baseline: verify gradient signal vs noise. 3 chains, 150 draws, pool_subset=300.",
        300),
    6: ("p2-exp6-collection", exp6_collection_params, 3, 150, 8, 500,
        "Phase2 collection params: burn-in & thinning sweep. 3 chains, 150 draws, pool_subset=300.",
        300),
}


# ─── Quick observable ───────────────────────────────────────────────────────


class QuickObservable:
    def __init__(self, name: str, dataset: JsonlSequenceDataset,
                 eval_batch_size: int, device: torch.device,
                 max_samples: int = 0, seed: int = 1337):
        self.name = name
        self.eval_batch_size = eval_batch_size
        self.device = device

        n_total = len(dataset)
        if max_samples > 0 and max_samples < n_total:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(seed)
            indices = torch.randperm(n_total, generator=rng)[:max_samples].sort()[0].tolist()
        else:
            indices = list(range(n_total))

        self.n_samples = len(indices)
        all_ids, all_mask = [], []
        for start in range(0, self.n_samples, eval_batch_size):
            bi = indices[start:start + eval_batch_size]
            batch = get_batch_by_indices(dataset, bi)
            all_ids.append(batch["input_ids"])
            all_mask.append(batch["attention_mask"])
        self.input_ids = torch.cat(all_ids, dim=0).to(device)
        self.attention_mask = torch.cat(all_mask, dim=0).to(device)

    def compute_seq_loss(self, model: torch.nn.Module) -> torch.Tensor:
        all_losses = []
        with torch.no_grad():
            for s in range(0, self.n_samples, self.eval_batch_size):
                e = min(s + self.eval_batch_size, self.n_samples)
                ids, mask = self.input_ids[s:e], self.attention_mask[s:e]
                out = model(input_ids=ids, attention_mask=mask)
                labels = ids.clone()
                labels[mask == 0] = -100
                all_losses.append(
                    per_example_causal_lm_loss(labels=labels, logits=out.logits).cpu()
                )
        return torch.cat(all_losses, dim=0)


# ─── Single combo runner ────────────────────────────────────────────────────


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_one_combo(
    combo: Combo,
    model: torch.nn.Module,
    anchor_params: dict[str, torch.Tensor],
    pool_ds: JsonlSequenceDataset,
    pool_obs: QuickObservable,
    query_obs: QuickObservable,
    dataset_size: int,
    num_chains: int,
    draws_per_chain: int,
    thinning: int,
    burnin: int,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    beta = combo.n_beta / float(dataset_size) if combo.n_beta > 0 else 0.0
    cfg = SGLDConfig(
        lr=combo.lr, beta=beta, gamma=combo.gamma, noise_level=1.0,
        num_chains=num_chains, draws_per_chain=draws_per_chain,
        num_burnin_steps=burnin,
        num_steps_bw_draws=thinning,
        seed=seed, sampler_type=combo.sampler_type,
        rmsprop_eps=combo.rmsprop_eps,
    )

    _set_seed(seed)
    for name, p in model.named_parameters():
        if p.requires_grad:
            p.data.copy_(anchor_params[name])

    sampler = create_sampler(model, anchor_params, cfg, dataset_size)
    chain_pool_traces: list[list[float]] = []
    chain_query_traces: list[list[float]] = []
    all_grad_norm: list[float] = []
    all_noise_norm: list[float] = []
    all_param_dist: list[float] = []
    step_loss_trace: list[float] = []

    for cid in range(num_chains):
        sampler.reset_to_anchor()
        _set_seed(seed + 10000 * cid)
        bg = torch.Generator(device="cpu")
        bg.manual_seed(seed + 10000 * cid + 17)
        ng = None
        if device and device.type == "cuda":
            ng = torch.Generator(device=device)
            ng.manual_seed(seed + 10000 * cid + 23)

        pool_trace = []
        query_trace = []
        total_steps = burnin + draws_per_chain * thinning

        for step in range(total_steps):
            is_burnin = step < burnin
            is_draw = (not is_burnin and (step - burnin) % thinning == 0)

            if is_draw:
                model.eval()
                pool_seq = pool_obs.compute_seq_loss(model)
                query_seq = query_obs.compute_seq_loss(model)
                pool_trace.append(float(pool_seq.mean()))
                query_trace.append(float(query_seq.mean()))
                with torch.no_grad():
                    pd_sq = sum((p.data - anchor_params[n]).float().norm().item() ** 2
                                for n, p in sampler.params)
                    all_param_dist.append(pd_sq ** 0.5)

            bi = torch.randperm(len(pool_ds), generator=bg)[:TRAIN_BATCH].tolist()
            batch = move_batch_to_device(get_batch_by_indices(pool_ds, bi), device)
            si = sampler.step(batch, step_generator=ng)
            all_grad_norm.append(si["grad_norm"])
            all_noise_norm.append(si["noise_norm"])
            step_loss_trace.append(si["loss"])

        chain_pool_traces.append(pool_trace)
        chain_query_traces.append(query_trace)

    pool_arr = np.array(chain_pool_traces, dtype=np.float64)
    query_arr = np.array(chain_query_traces, dtype=np.float64)

    nan_frac = float(np.isnan(pool_arr).mean()) + float(np.isnan(query_arr).mean())
    inf_frac = float(np.isinf(pool_arr).mean()) + float(np.isinf(query_arr).mean())

    status = "ok"
    if inf_frac > 0.05:
        status = "diverged"
    elif nan_frac > 0.05:
        status = "nan"
    if pool_arr.size == 0:
        status = "no_draws"

    if status == "ok" and pool_arr.shape[1] > 0:
        mean_per_draw = pool_arr.mean(axis=0)
        initial_loss = float(mean_per_draw[0])
        final_loss = float(mean_per_draw[-1])
        loss_increase = final_loss - initial_loss
        loss_monotone = int(np.all(np.diff(mean_per_draw) >= 0))
        chain_std = float(pool_arr.std(axis=0).mean())
    else:
        initial_loss = final_loss = loss_increase = float("nan")
        loss_monotone = 0
        chain_std = float("nan")
        mean_per_draw = np.array([])

    snr = float(np.mean(all_grad_norm) / (np.mean(all_noise_norm) + 1e-12)) if all_noise_norm else 0.0
    autocorr_lag1 = _autocorr_lag1(mean_per_draw) if len(mean_per_draw) > 2 else 0.0

    bif_diag = _compute_lightweight_bif(pool_arr, query_arr) if status == "ok" and pool_arr.shape[1] >= 5 else {
        k: float("nan") for k in ["bif_score_mean", "bif_score_std", "off_diag_mean", "off_diag_std", "pos_frac"]
    }

    r_hat = _gelman_rubin(pool_arr) if num_chains >= 2 and status == "ok" else float("nan")

    return {
        "tag": combo.tag,
        "lr": combo.lr, "n_beta": combo.n_beta, "gamma": combo.gamma,
        "sampler_type": combo.sampler_type, "rmsprop_eps": combo.rmsprop_eps,
        "num_burnin_steps": burnin, "num_steps_bw_draws": thinning,
        "beta_used": beta,
        "status": status,
        "initial_loss": initial_loss, "final_loss": final_loss,
        "loss_increase": loss_increase,
        "loss_monotone_increase": loss_monotone,
        "chain_std": chain_std, "snr": snr,
        "mean_param_dist": float(np.mean(all_param_dist)) if all_param_dist else 0.0,
        "mean_grad_norm": float(np.mean(all_grad_norm)) if all_grad_norm else 0.0,
        "mean_noise_norm": float(np.mean(all_noise_norm)) if all_noise_norm else 0.0,
        "autocorr_lag1": autocorr_lag1, "r_hat": r_hat,
        "pool_traces": chain_pool_traces,
        "query_traces": chain_query_traces,
        "step_loss_trace": step_loss_trace,
        "total_steps": burnin + draws_per_chain * thinning,
        **bif_diag,
    }


def _autocorr_lag1(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    x0 = x - x.mean()
    d = (x0 ** 2).sum()
    return float(np.sum(x0[:-1] * x0[1:]) / d) if d > 1e-12 else 0.0


def _gelman_rubin(arr: np.ndarray) -> float:
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


def _compute_lightweight_bif(pool_arr: np.ndarray, query_arr: np.ndarray) -> dict[str, float]:
    n_chains, n_draws = pool_arr.shape
    if n_draws < 3:
        return {k: float("nan") for k in ["bif_score_mean", "bif_score_std", "off_diag_mean", "off_diag_std", "pos_frac"]}
    try:
        loss_t = torch.as_tensor(pool_arr.T, dtype=torch.float32)
        if torch.cuda.is_available():
            loss_t = loss_t.cuda()
        corr = torch.corrcoef(loss_t).cpu().numpy()
        np.fill_diagonal(corr, 0.0)
        bif_mean = corr.mean(axis=1)
        off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
        return {
            "bif_score_mean": float(bif_mean.mean()),
            "bif_score_std": float(bif_mean.std()),
            "off_diag_mean": float(off_diag.mean()),
            "off_diag_std": float(off_diag.std()),
            "pos_frac": float((off_diag > 0.05).mean()),
        }
    except Exception:
        return {k: float("nan") for k in ["bif_score_mean", "bif_score_std", "off_diag_mean", "off_diag_std", "pos_frac"]}


def classify(r: dict[str, Any]) -> str:
    s = r["status"]
    if s != "ok":
        return s
    li = r["loss_increase"]
    snr = r["snr"]
    cs = r["chain_std"]
    mono = r["loss_monotone_increase"]
    if li < -0.01:
        return "loss_decreased"
    if li < 0.01:
        return "too_little_signal"
    if snr < 0.01:
        return "too_much_noise"
    if cs > 1.0:
        return "chains_disagree"
    if not mono and li > 0.5:
        return "spiky"
    return "good"


# ─── GPU worker — writes JSON per combo ─────────────────────────────────────


def _gpu_worker(gpu_id, combos, model_path, pool_jsonl, query_jsonl,
                result_dir, num_chains, draws_per_chain, thinning, burnin,
                seed, pool_eval_subset=0):
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        _set_seed(seed + gpu_id)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        model.to(device)

        pool_ds = JsonlSequenceDataset(
            pool_jsonl, tokenizer, max_length=MAX_LENGTH,
            text_key="text", id_key="id",
            source_type_key="source", subtype_key="subtype",
        )
        query_ds = JsonlSequenceDataset(
            query_jsonl, tokenizer, max_length=MAX_LENGTH,
            text_key="text", id_key="id",
            source_type_key="source", subtype_key="subtype",
            task_type_key="task_type",
        )
        pool_obs = QuickObservable("pool", pool_ds, EVAL_BATCH, device,
                                    max_samples=pool_eval_subset, seed=42)
        query_obs = QuickObservable("query", query_ds, EVAL_BATCH, device,
                                     max_samples=0, seed=43)
        anchor_params = {n: p.detach().clone().to(device)
                         for n, p in model.named_parameters() if p.requires_grad}
        ds_size = len(pool_ds)

        pbar = tqdm(combos, desc=f"GPU{gpu_id}", position=gpu_id, leave=True)
        for combo in pbar:
            pbar.set_postfix_str(combo.tag[:40])
            t0 = time.monotonic()
            try:
                result = run_one_combo(
                    combo, model, anchor_params, pool_ds,
                    pool_obs, query_obs, ds_size,
                    num_chains, draws_per_chain, thinning, burnin,
                    seed, device,
                )
            except Exception as e:
                result = _error_result(combo, ds_size, str(e))
                result["num_burnin_steps"] = burnin
                result["num_steps_bw_draws"] = thinning
            result["elapsed"] = time.monotonic() - t0
            result["gpu_id"] = gpu_id

            safe_result = {k: v for k, v in result.items()
                           if k not in ("pool_traces", "query_traces", "step_loss_trace")}
            safe_result["pool_traces"] = result.get("pool_traces", [])
            safe_result["query_traces"] = result.get("query_traces", [])

            path = os.path.join(result_dir, f"gpu{gpu_id}_{combo.tag}.json")
            try:
                with open(path, "w") as f:
                    json.dump(safe_result, f, default=_json_default)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                print(f"GPU{gpu_id} WRITE FAIL {combo.tag}: {e}", file=sys.stderr)

        marker = os.path.join(result_dir, f"_done_gpu{gpu_id}")
        with open(marker, "w") as f:
            f.write(f"completed {len(combos)} combos\n")

        del model
        torch.cuda.empty_cache()
        print(f"GPU{gpu_id} DONE — {len(combos)} combos completed", file=sys.stderr)
    except Exception as e:
        print(f"GPU {gpu_id} FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        marker = os.path.join(result_dir, f"_fail_gpu{gpu_id}")
        try:
            with open(marker, "w") as f:
                f.write(f"error: {e}\n")
        except Exception:
            pass


def _error_result(combo: Combo, ds_size: int, err: str) -> dict[str, Any]:
    nan = float("nan")
    return {
        "tag": combo.tag, "lr": combo.lr, "n_beta": combo.n_beta,
        "gamma": combo.gamma, "sampler_type": combo.sampler_type,
        "rmsprop_eps": combo.rmsprop_eps,
        "beta_used": combo.n_beta / max(ds_size, 1),
        "status": f"error:{err[:50]}", "initial_loss": nan, "final_loss": nan,
        "loss_increase": nan, "loss_monotone_increase": 0, "chain_std": nan,
        "snr": 0.0, "mean_param_dist": 0.0, "mean_grad_norm": 0.0,
        "mean_noise_norm": 0.0, "autocorr_lag1": 0.0, "r_hat": nan,
        "bif_score_mean": nan, "bif_score_std": nan,
        "off_diag_mean": nan, "off_diag_std": nan, "pos_frac": nan,
        "pool_traces": [], "query_traces": [],
    }


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


# ─── Collect results from JSON files ────────────────────────────────────────


def _collect_results(result_dir: str) -> list[dict[str, Any]]:
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
    results.sort(key=lambda r: (r.get("lr", 0), r.get("n_beta", 0), r.get("gamma", 0), r.get("sampler_type", "")))
    return results


# ─── SwanLab logging (main process only) ────────────────────────────────────


def _safe(v):
    if isinstance(v, float) and math.isnan(v):
        return 0.0
    return v


def _fmt(v):
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return f"{v:.4f}"


def log_experiment(results: list[dict[str, Any]], exp_name: str, exp_desc: str = "",
                   sampling_config: dict[str, Any] | None = None):
    """Log all results for one experiment to SwanLab (main process)."""
    lrs = sorted(set(r["lr"] for r in results))
    nbs = sorted(set(r["n_beta"] for r in results))
    gammas = sorted(set(r["gamma"] for r in results))
    sampler_types = sorted(set(r.get("sampler_type", "") for r in results))

    swan_config = {
        "model": "pythia-70m-step1000",
        "pool_size": 800, "query_size": 20, "max_length": MAX_LENGTH,
        "eval_batch": EVAL_BATCH, "train_batch": TRAIN_BATCH,
        "num_combos": len(results),
        "lr_range": [f"{v:.0e}" for v in lrs],
        "nbeta_range": [f"{v:.0f}" for v in nbs],
        "gamma_range": [f"{v:.0e}" for v in gammas],
        "sampler_types": sampler_types,
        "guide": "https://timaeus.co/research/2026-04-21-sampling-guide",
    }
    if sampling_config:
        swan_config.update(sampling_config)

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
        }, step=i)

    for r in results:
        tag = r["tag"]
        traces = r.get("pool_traces", [])
        if not traces or not traces[0]:
            continue
        nd = len(traces[0])
        xa = [str(i) for i in range(nd)]
        series = {f"c{c}": [round(v, 6) for v in ch] for c, ch in enumerate(traces)}
        if len(traces) > 1:
            series["mean"] = [round(v, 6) for v in np.mean(traces, axis=0).tolist()]
        log_line(f"trace/{tag}", xaxis=xa, series=series)

    _log_summary_charts(results)
    _log_summary_tables(results)
    _log_classification_pie(results)

    swan_finish()


def _log_summary_charts(results):
    ok = [r for r in results if r.get("status") == "ok"]
    good = [r for r in results if classify(r) == "good"]

    for stype, sname in [("rmsprop_sgld", "rms"), ("sgld", "sgld")]:
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
                            [f"g{v:.0e}" for v in g_list], mat_li, "Δloss")
                log_heatmap(f"hm/{sname}/snr_lr{lr_val:.0e}",
                            [f"nb{v:.0f}" for v in nb_list],
                            [f"g{v:.0e}" for v in g_list], mat_snr, "SNR")

    if good:
        pts = [(r["loss_increase"], r["snr"]) for r in good]
        if len(pts) > 200:
            idx = np.random.RandomState(42).choice(len(pts), 200, replace=False)
            pts = [pts[i] for i in sorted(idx)]
        log_scatter("scatter/dloss_vs_snr", "Δloss", "SNR", {"good": pts})

    nb0 = [r for r in ok if r.get("n_beta") == 0]
    nb_pos = [r for r in ok if r.get("n_beta", 0) > 0]
    if nb0 and nb_pos:
        log_bar("bar/nb0_vs_pos/dloss", ["nβ=0", "nβ>0"], {
            "Δloss": [
                round(float(np.mean([r["loss_increase"] for r in nb0])), 4),
                round(float(np.mean([r["loss_increase"] for r in nb_pos])), 4),
            ]
        })
        n0b = [r["bif_score_mean"] for r in nb0 if not math.isnan(r.get("bif_score_mean", float("nan")))]
        npb = [r["bif_score_mean"] for r in nb_pos if not math.isnan(r.get("bif_score_mean", float("nan")))]
        if n0b and npb:
            log_bar("bar/nb0_vs_pos/bif", ["nβ=0", "nβ>0"], {
                "bif_mean": [round(float(np.mean(n0b)), 4), round(float(np.mean(npb)), 4)]
            })


def _log_classification_pie(results):
    sc = {}
    for r in results:
        c = classify(r)
        sc[c] = sc.get(c, 0) + 1
    if sc:
        log_pie("pie/classification", "combos", list(sc.items()))


def _log_summary_tables(results):
    headers = ["#", "lr", "nβ", "γ", "sampler", "eps", "burn", "thin",
               "status", "Δloss", "SNR", "R̂", "bif", "cstd", "ac1"]
    rows = []
    for i, r in enumerate(results):
        rows.append([
            i, f"{r['lr']:.0e}", f"{r['n_beta']:.0f}", f"{r['gamma']:.0e}",
            r.get("sampler_type", "").replace("rmsprop_sgld", "rms"),
            f"{r.get('rmsprop_eps', 0.1):.2f}", r.get("num_burnin_steps", 0),
            r.get("num_steps_bw_draws", 1), classify(r),
            _fmt(r.get("loss_increase", float("nan"))),
            _fmt(r.get("snr", 0)),
            _fmt(r.get("r_hat", float("nan"))),
            _fmt(r.get("bif_score_mean", float("nan"))),
            _fmt(r.get("chain_std", float("nan"))),
            _fmt(r.get("autocorr_lag1", 0)),
        ])
    log_table("table/all", headers=headers, rows=rows)

    good = sorted([r for r in results if classify(r) == "good"],
                   key=lambda x: x.get("snr", 0), reverse=True)
    if good:
        g_rows = []
        for r in good[:150]:
            g_rows.append([
                f"{r['lr']:.0e}", f"{r['n_beta']:.0f}", f"{r['gamma']:.0e}",
                r.get("sampler_type", "").replace("rmsprop_sgld", "rms"),
                f"{r.get('rmsprop_eps', 0.1):.2f}", r.get("num_burnin_steps", 0),
                r.get("num_steps_bw_draws", 1),
                _fmt(r.get("loss_increase", float("nan"))),
                _fmt(r.get("snr", 0)),
                _fmt(r.get("r_hat", float("nan"))),
                _fmt(r.get("bif_score_mean", float("nan"))),
                _fmt(r.get("bif_score_std", float("nan"))),
                _fmt(r.get("off_diag_mean", float("nan"))),
                _fmt(r.get("chain_std", float("nan"))),
                _fmt(r.get("autocorr_lag1", 0)),
            ])
        log_table("table/top150_by_snr",
                  headers=["lr", "nβ", "γ", "sampler", "eps", "burn", "thin",
                           "Δloss", "SNR", "R̂", "bif_mean", "bif_std",
                           "off_diag", "cstd", "ac1"],
                  rows=g_rows)


# ─── Run experiment ─────────────────────────────────────────────────────────


def run_experiment(exp_id, combos, num_chains, draws_per_chain, thinning, burnin,
                   model_path, pool_jsonl, query_jsonl, out_dir, num_gpus, seed=42,
                   pool_eval_subset=0):
    entry = PHASE1.get(exp_id, PHASE2.get(exp_id))
    exp_name = entry[0] if entry else f"exp{exp_id}"
    exp_desc = entry[6] if entry and len(entry) > 6 else ""
    total_steps = burnin + draws_per_chain * thinning
    print(f"\n{'='*70}")
    print(f"  {exp_name}")
    print(f"  {exp_desc}")
    print(f"  {len(combos)} combos × {num_chains}c × {draws_per_chain}d × thin{thinning} + burn{burnin} = {total_steps} steps")
    print(f"  pool_eval_subset={pool_eval_subset}")
    est_min = len(combos) * (num_chains * total_steps * 0.25 + num_chains * draws_per_chain * (pool_eval_subset or 800) / EVAL_BATCH * 0.15) / 60 / min(num_gpus, torch.cuda.device_count())
    print(f"  Est. total: ~{est_min:.0f} min on {min(num_gpus, torch.cuda.device_count())} GPUs")
    print(f"{'='*70}")

    result_dir = os.path.join(out_dir, exp_name)
    ensure_dir(result_dir)

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
            gid, chunks[gid], model_path, pool_jsonl, query_jsonl,
            result_dir, num_chains, draws_per_chain, thinning, burnin, seed,
            pool_eval_subset,
        ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    results = _collect_results(result_dir)
    print(f"  Collected {len(results)}/{len(combos)} results from {result_dir}")

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
                  f"bif={r.get('bif_score_mean', 'nan')} R̂={r.get('r_hat', 'nan')}")

    print(f"  Logging to SwanLab (project=bif-sweep)...")
    log_experiment(
        results, exp_name=exp_name, exp_desc=exp_desc,
        sampling_config={
            "num_chains": num_chains, "draws_per_chain": draws_per_chain,
            "thinning": thinning, "burnin": burnin, "total_steps": total_steps,
            "pool_eval_subset": pool_eval_subset,
        },
    )

    return results


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="BIF HP Sweep")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--pool_jsonl", default=POOL_JSONL)
    parser.add_argument("--query_jsonl", default=QUERY_JSONL)
    parser.add_argument("--out_dir", default="/workspace/pku_percy/runs/hp_sweep")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Run only phase 1 (screening) or 2 (deep). Default: both.")
    args = parser.parse_args()

    import swanlab
    swanlab.login(api_key="RGvhmcyaE940jILdlzCGg")

    t0 = time.monotonic()

    if args.phase is None or args.phase == 1:
        for eid in sorted(PHASE1.keys()):
            entry = PHASE1[eid]
            name, fn = entry[0], entry[1]
            nc, nd, th, bi = entry[2], entry[3], entry[4], entry[5]
            desc = entry[6] if len(entry) > 6 else ""
            pes = entry[7] if len(entry) > 7 else 0
            run_experiment(eid, fn(), nc, nd, th, bi,
                           args.model_path, args.pool_jsonl, args.query_jsonl,
                           args.out_dir, args.gpus, args.seed,
                           pool_eval_subset=pes)

    if args.phase is None or args.phase == 2:
        for eid in sorted(PHASE2.keys()):
            entry = PHASE2[eid]
            name, fn = entry[0], entry[1]
            nc, nd, th, bi = entry[2], entry[3], entry[4], entry[5]
            desc = entry[6] if len(entry) > 6 else ""
            pes = entry[7] if len(entry) > 7 else 0
            run_experiment(eid, fn(), nc, nd, th, bi,
                           args.model_path, args.pool_jsonl, args.query_jsonl,
                           args.out_dir, args.gpus, args.seed,
                           pool_eval_subset=pes)

    elapsed = time.monotonic() - t0
    print(f"\n{'='*70}")
    print(f"  ALL DONE  ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
