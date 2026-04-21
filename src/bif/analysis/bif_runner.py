"""BIF trace runner: SGLD-based influence function trace collection."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from bif.config import SGLDConfig
from bif.data.dataset import (
    JsonlSequenceDataset,
    get_batch_by_indices,
    move_batch_to_device,
)
from bif.io import ensure_dir, save_json
from bif.training.loss import per_example_causal_lm_loss
from bif.training.sgld import LocalizedSGLDSampler
from bif.utils.logging import get_logger
from bif.utils.tracker import finish as swan_finish
from bif.utils.tracker import init_run
from bif.utils.tracker import log as swan_log

logger = get_logger("bif.runner")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _broadcast_plan(
    plan: list[tuple[str, str]], rank: int, world_size: int
) -> list[tuple[str, str]]:
    """Broadcast the checkpoint plan from rank 0 to all other ranks.

    rank 0 calls this with the real plan; other ranks call it with an empty list.
    All ranks leave with the same plan, preventing barrier mismatches when
    --resume causes different ranks to see different filesystem states.
    """
    if world_size <= 1:
        return plan

    import pickle

    # NCCL backend only supports CUDA tensors — put broadcast tensors on GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = (
        torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if rank == 0:
        data = pickle.dumps(plan)
        size_t = torch.tensor([len(data)], dtype=torch.long, device=device)
    else:
        size_t = torch.tensor([0], dtype=torch.long, device=device)

    dist.broadcast(size_t, src=0)
    n_bytes = int(size_t.item())

    buf = torch.zeros(n_bytes, dtype=torch.uint8, device=device)
    if rank == 0:
        buf[:] = torch.frombuffer(data, dtype=torch.uint8).to(device)
    dist.broadcast(buf, src=0)

    if rank != 0:
        plan = pickle.loads(bytes(buf.cpu().tolist()))
    return plan


def _is_checkpoint_complete(
    out_dir: str, expected_world_size: int, draws_per_chain: int, num_chains: int
) -> bool:
    """Return True if this checkpoint's bif_traces look fully written.

    Criteria (all must hold):
    - manifest_rank{i}.json exists for every rank 0..world_size-1
    - every chain assigned to at least one rank has a pool_loss_trace.jsonl
      with exactly draws_per_chain lines
    """
    for rank in range(expected_world_size):
        manifest_path = os.path.join(out_dir, f"manifest_rank{rank:03d}.json")
        if not os.path.isfile(manifest_path):
            return False

    # Check total draws across all chain dirs
    total_draws = 0
    names = os.listdir(out_dir) if os.path.isdir(out_dir) else []
    for name in names:
        chain_dir = os.path.join(out_dir, name)
        if not (os.path.isdir(chain_dir) and re.fullmatch(r"chain_\d+", name)):
            continue
        trace = os.path.join(chain_dir, "pool_loss_trace.jsonl")
        if not os.path.isfile(trace):
            return False
        with open(trace) as f:
            lines = sum(1 for _ in f)
        if lines < draws_per_chain:
            return False
        total_draws += lines

    expected_draws = num_chains * draws_per_chain
    return total_draws >= expected_draws


def _discover_checkpoint_plan(
    model_root: str,
    base_model_path: str | None = None,
    include_final_model: bool = True,
    resume_out_dir: str | None = None,
    world_size: int = 1,
    draws_per_chain: int = 60,
    num_chains: int = 4,
) -> list[tuple[str, str]]:
    """Build the ordered list of (name, path) checkpoints to process.

    When *resume_out_dir* is given, any checkpoint whose output directory
    already contains complete traces (all manifests + full draws) is silently
    skipped so the pipeline can resume after a crash without re-running
    finished work.
    """
    plan: list[tuple[str, str]] = []
    if base_model_path is not None:
        if not os.path.isdir(base_model_path):
            raise FileNotFoundError(base_model_path)
        plan.append(("base_model", base_model_path))

    entries = []
    for name in os.listdir(model_root):
        full = os.path.join(model_root, name)
        if os.path.isdir(full) and re.fullmatch(r"checkpoint-\d+", name):
            entries.append((int(name.split("-")[-1]), name, full))
    for _, name, full in sorted(entries):
        plan.append((name, full))

    final_path = os.path.join(model_root, "final_model")
    if include_final_model and os.path.isdir(final_path):
        plan.append(("final_model", final_path))

    if not plan:
        raise ValueError(f"No checkpoints under {model_root}")

    if resume_out_dir is None:
        return plan

    # Filter out already-completed checkpoints
    remaining = []
    for ckpt_name, ckpt_path in plan:
        ckpt_out = os.path.join(resume_out_dir, ckpt_name)
        if _is_checkpoint_complete(ckpt_out, world_size, draws_per_chain, num_chains):
            logger.info("Skipping completed checkpoint: %s", ckpt_name)
        else:
            remaining.append((ckpt_name, ckpt_path))

    if not remaining:
        logger.info("All checkpoints already complete — nothing to do.")
    return remaining


class LossTraceWriter:
    """Write per-draw loss traces to JSONL.

    Supports use as a context manager so the underlying file is always
    closed — even if an exception is raised mid-chain:

        with LossTraceWriter(path, name) as writer:
            writer.write_draw(...)
    """

    def __init__(self, out_path: str, dataset_name: str):
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self.out_path = out_path
        self.dataset_name = dataset_name
        self._f = open(out_path, "w", encoding="utf-8")

    def __enter__(self) -> "LossTraceWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def write_draw(
        self,
        chain_id: int,
        draw_in_chain: int,
        global_draw: int,
        sample_ids: list[Any],
        dataset_indices: list[int],
        source_types: list[Any],
        subtypes: list[Any],
        task_types: list[Any],
        losses: torch.Tensor,
    ) -> None:
        row = {
            "chain_id": chain_id,
            "draw_in_chain": draw_in_chain,
            "global_draw": global_draw,
            "dataset": self.dataset_name,
            "sample_ids": sample_ids,
            "dataset_indices": dataset_indices,
            "source_types": source_types,
            "subtypes": subtypes,
            "task_types": task_types,
            "losses": [float(x) for x in losses.cpu().tolist()],
        }
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def _collect_losses_over_dataset(
    model: torch.nn.Module,
    dataset: JsonlSequenceDataset,
    eval_batch_size: int,
    device: torch.device,
    show_progress: bool = True,
) -> tuple[list[Any], list[int], list[Any], list[Any], list[Any], torch.Tensor]:
    model.eval()
    all_ids, all_indices, all_src, all_sub, all_task = [], [], [], [], []
    all_losses: list[torch.Tensor] = []

    batches = list(range(0, len(dataset), eval_batch_size))
    with torch.no_grad():
        for start in tqdm(batches, desc="Eval", leave=False, disable=not show_progress):
            indices = list(range(start, min(start + eval_batch_size, len(dataset))))
            batch = get_batch_by_indices(dataset, indices)
            all_ids.extend(batch["sample_ids"])
            all_indices.extend(batch["dataset_indices"].tolist())
            all_src.extend(batch["source_types"])
            all_sub.extend(batch["subtypes"])
            all_task.extend(batch["task_types"])
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            losses = per_example_causal_lm_loss(
                labels=batch["labels"], logits=outputs.logits
            )
            all_losses.append(losses.cpu())

    return (
        all_ids,
        all_indices,
        all_src,
        all_sub,
        all_task,
        torch.cat(all_losses, dim=0),
    )


def run_bif(
    model_name_or_path: str,
    pool_jsonl: str,
    query_jsonl: str,
    out_dir: str,
    sgld_cfg: SGLDConfig | None = None,
    tokenizer_path: str | None = None,
    max_length: int = 256,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    pool_eval_subset: int = 0,
    device: str | None = None,
    dtype: str = "float32",
    pool_text_key: str = "text",
    pool_id_key: str = "id",
    pool_source_type_key: str = "source",
    pool_subtype_key: str = "subtype",
    query_text_key: str = "text",
    query_id_key: str = "id",
    query_source_type_key: str = "source",
    query_subtype_key: str = "subtype",
    query_task_type_key: str = "task_type",
    experiment_name: str | None = None,
    run_name: str | None = None,
    manage_tracking: bool = True,
    global_step_offset: int = 0,
) -> None:
    """Run BIF trace collection for a single model checkpoint.

    Args:
        pool_eval_subset: If > 0, only evaluate on this many random pool
            samples (without replacement, same subset per chain) instead of
            the full pool.  This dramatically speeds up BIF for large pools
            while still providing enough candidates for top-K selection.
            Recommended: 5-10x your top_k.
        tokenizer_path: Where to load the tokenizer from.  HuggingFace
            Trainer only saves tokenizer files alongside the *final* model,
            not inside intermediate checkpoint directories.  Pass the base
            model path (or any directory that contains tokenizer files) here
            so that per-checkpoint runs can still load the tokenizer.
            Defaults to *model_name_or_path* when not provided.
        manage_tracking: When True (default, single-checkpoint mode), this
            function owns the SwanLab run lifetime: it calls init_run() at
            the start and finish() at the end.  Set to False when the caller
            (e.g. a full pipeline loop) has already opened a SwanLab run and
            wants all checkpoints logged into that single run — in that case
            the caller is responsible for calling init_run() before the loop
            and finish() after it.
        global_step_offset: Added to every logged step so that, in pipeline
            mode, metrics from consecutive checkpoints appear on a continuous
            x-axis rather than all starting from 0.
    """
    if sgld_cfg is None:
        sgld_cfg = SGLDConfig()

    rank, world_size, local_rank = _get_distributed_context()

    # Pin each rank to its own GPU before any cuda op / model load
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    _set_seed(sgld_cfg.seed + rank)

    ckpt_name = os.path.basename(model_name_or_path)
    if manage_tracking and rank == 0:
        init_run(
            experiment_name=experiment_name or f"bif_{ckpt_name}",
            run_name=run_name,
            config={
                "checkpoint": ckpt_name,
                "model": model_name_or_path,
                "sgld": asdict(sgld_cfg),
                "max_length": max_length,
            },
            tags=["bif", ckpt_name],
        )

    if device is None:
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    ensure_dir(out_dir)

    if rank == 0:
        save_json(
            f"{out_dir}/run_config.json",
            {
                "model_name_or_path": model_name_or_path,
                "max_length": max_length,
                "sgld_config": asdict(sgld_cfg),
            },
        )

    logger.info("Loading tokenizer and model (rank=%d)", rank)
    # HF Trainer only saves tokenizer files in the final model directory, not
    # in intermediate checkpoints.  Resolve: explicit tokenizer_path > the
    # checkpoint dir itself (works when tokenizer files are present there).
    tok_src = tokenizer_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pt_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=pt_dtype if device.type == "cuda" else torch.float32,
    )
    model.to(device)

    pool_ds = JsonlSequenceDataset(
        pool_jsonl,
        tokenizer,
        max_length=max_length,
        text_key=pool_text_key,
        id_key=pool_id_key,
        source_type_key=pool_source_type_key,
        subtype_key=pool_subtype_key,
    )
    query_ds = JsonlSequenceDataset(
        query_jsonl,
        tokenizer,
        max_length=max_length,
        text_key=query_text_key,
        id_key=query_id_key,
        source_type_key=query_source_type_key,
        subtype_key=query_subtype_key,
        task_type_key=query_task_type_key,
    )

    anchor_params = {
        name: p.detach().clone().to(device)
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    sampler = LocalizedSGLDSampler(
        model,
        anchor_params,
        sgld_cfg,
        source_dataset_size=len(pool_ds),
    )

    # ── Pool subsampling for faster BIF eval ──────────────────────────────
    if pool_eval_subset > 0 and pool_eval_subset < len(pool_ds):
        rng = torch.Generator(device="cpu")
        rng.manual_seed(sgld_cfg.seed + 999)
        subset_indices = torch.randperm(len(pool_ds), generator=rng)[
            :pool_eval_subset
        ].sort()[0]
        pool_ds_eval = pool_ds.subset(subset_indices.tolist())
        logger.info(
            "Pool eval subsample: %d → %d samples (seed=%d)",
            len(pool_ds),
            len(pool_ds_eval),
            sgld_cfg.seed + 999,
        )
    else:
        pool_ds_eval = pool_ds

    assigned_chains = list(range(rank, sgld_cfg.num_chains, world_size))
    if not assigned_chains:
        logger.info("No chains assigned to rank %d", rank)
        # Still need to write an empty manifest so _is_checkpoint_complete can
        # count all world_size manifests.  Then fall through to the shared
        # _barrier() below — every rank must hit exactly the same barrier count.
        save_json(
            f"{out_dir}/manifest_rank{rank:03d}.json",
            {"rank": rank, "assigned_chains": []},
        )
        _barrier()
        return

    for chain_id in assigned_chains:
        logger.info("Starting chain %d on rank %d", chain_id, rank)
        sampler.reset_to_anchor()

        batch_gen = torch.Generator(device="cpu")
        batch_gen.manual_seed(sgld_cfg.seed + 10000 * chain_id + 17)
        noise_gen = None
        if device.type == "cuda":
            noise_gen = torch.Generator(device=device)
            noise_gen.manual_seed(sgld_cfg.seed + 10000 * chain_id + 23)

        chain_dir = f"{out_dir}/chain_{chain_id:03d}"
        ensure_dir(chain_dir)

        draw_in_chain = 0
        global_draw_base = chain_id * sgld_cfg.draws_per_chain

        with (
            LossTraceWriter(
                f"{chain_dir}/pool_loss_trace.jsonl", "pool"
            ) as pool_writer,
            LossTraceWriter(
                f"{chain_dir}/query_loss_trace.jsonl", "query"
            ) as query_writer,
        ):
            for step in range(sgld_cfg.total_sampling_steps):
                retain = (
                    step >= sgld_cfg.burn_in
                    and (step - sgld_cfg.burn_in) % sgld_cfg.thinning == 0
                )
                if retain:
                    pool_ids, pool_idx, pool_src, pool_sub, pool_task, pool_losses = (
                        _collect_losses_over_dataset(
                            model,
                            pool_ds_eval,
                            eval_batch_size,
                            device,
                            show_progress=(rank == 0),
                        )
                    )
                    q_ids, q_idx, q_src, q_sub, q_task, q_losses = (
                        _collect_losses_over_dataset(
                            model,
                            query_ds,
                            eval_batch_size,
                            device,
                            show_progress=(rank == 0),
                        )
                    )
                    global_draw = global_draw_base + draw_in_chain
                    pool_writer.write_draw(
                        chain_id,
                        draw_in_chain,
                        global_draw,
                        pool_ids,
                        pool_idx,
                        pool_src,
                        pool_sub,
                        pool_task,
                        pool_losses,
                    )
                    query_writer.write_draw(
                        chain_id,
                        draw_in_chain,
                        global_draw,
                        q_ids,
                        q_idx,
                        q_src,
                        q_sub,
                        q_task,
                        q_losses,
                    )
                    logger.info(
                        "chain=%d draw=%d pool_mean=%.4f query_mean=%.4f",
                        chain_id,
                        draw_in_chain,
                        pool_losses.mean().item(),
                        q_losses.mean().item(),
                    )
                    if rank == 0:
                        pool_np = pool_losses.cpu().float().numpy()
                        q_np = q_losses.cpu().float().numpy()
                        log_step = global_step_offset + global_draw
                        swan_log(
                            {
                                f"4_1_bif/chain{chain_id}_pool_loss_mean": float(
                                    pool_np.mean()
                                ),
                                f"4_1_bif/chain{chain_id}_query_loss_mean": float(
                                    q_np.mean()
                                ),
                                f"4_1_bif/chain{chain_id}_pool_query_loss_gap": float(
                                    pool_np.mean() - q_np.mean()
                                ),
                            },
                            step=log_step,
                        )
                    draw_in_chain += 1
                    if draw_in_chain >= sgld_cfg.draws_per_chain:
                        break

                batch_indices = torch.randperm(len(pool_ds), generator=batch_gen)[
                    :train_batch_size
                ].tolist()
                batch = move_batch_to_device(
                    get_batch_by_indices(pool_ds, batch_indices), device
                )
                sampler.step(batch, step_generator=noise_gen)
        # LossTraceWriters closed automatically by context manager above
        save_json(
            f"{chain_dir}/chain_config.json",
            {
                "chain_id": chain_id,
                "draws_written": draw_in_chain,
                "sgld_config": asdict(sgld_cfg),
            },
        )

    save_json(
        f"{out_dir}/manifest_rank{rank:03d}.json",
        {
            "rank": rank,
            "assigned_chains": assigned_chains,
        },
    )
    logger.info("All chains completed on rank %d", rank)
    # Barrier BEFORE swan_finish(): rank 0's swan_finish() blocks waiting for
    # the swanlab cloud response (potentially many seconds).  If the barrier
    # were placed after, every other rank would have already exited run_bif()
    # and raced into the next checkpoint's model-load + its own barrier,
    # causing seqnum 2 to be enqueued on those ranks while rank 0 is still
    # stuck in the cloud upload.  That mismatch is what triggers the NCCL
    # watchdog timeout (600 s) and SIGABRT seen in the logs.
    _barrier()
    if manage_tracking and rank == 0:
        swan_finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BIF trace collection.")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--model_root", default=None)
    parser.add_argument("--base_model_path", default=None)
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help=(
            "Path to tokenizer files. HF Trainer does not copy tokenizer files "
            "into intermediate checkpoint dirs, so pass the base model path here "
            "when running --run_all_checkpoints. Auto-detected from "
            "model_root/final_model or base_model_path when not set."
        ),
    )
    parser.add_argument("--run_all_checkpoints", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip checkpoints whose output directory already has complete traces.",
    )
    parser.add_argument("--pool_jsonl", required=True)
    parser.add_argument("--query_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--pool_eval_subset", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--gamma", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--num_chains", type=int, default=4)
    parser.add_argument("--draws_per_chain", type=int, default=60)
    parser.add_argument("--burn_in", type=int, default=0)
    parser.add_argument("--thinning", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument(
        "--experiment_name",
        default=None,
        help=(
            "SwanLab experiment name. Defaults to 'bif_pipeline' (multi-checkpoint) "
            "or 'bif_{checkpoint_name}' (single-checkpoint) when not set."
        ),
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="SwanLab run display name within the experiment (e.g. 'bif' in pipeline mode).",
    )
    args = parser.parse_args()

    if args.run_all_checkpoints and not args.model_root:
        raise ValueError("--model_root is required with --run_all_checkpoints")
    if not args.run_all_checkpoints and not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required")

    rank, world_size, _ = _get_distributed_context()
    # Init process group before any dist.barrier() calls inside run_bif()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    cfg = SGLDConfig(
        lr=args.lr,
        gamma=args.gamma,
        beta=args.beta,
        noise_scale=args.noise_scale,
        num_chains=args.num_chains,
        draws_per_chain=args.draws_per_chain,
        burn_in=args.burn_in,
        thinning=args.thinning,
        seed=args.seed,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
    )

    try:
        if args.run_all_checkpoints:
            rank, world_size, _ = _get_distributed_context()
            # Only rank 0 scans the filesystem to build the plan, then
            # broadcasts it.  This guarantees every rank has the identical
            # list, preventing barrier count mismatches when --resume causes
            # different ranks to observe different filesystem states.
            if rank == 0:
                plan = _discover_checkpoint_plan(
                    args.model_root,
                    args.base_model_path,
                    resume_out_dir=args.out_dir if args.resume else None,
                    world_size=world_size,
                    draws_per_chain=cfg.draws_per_chain,
                    num_chains=cfg.num_chains,
                )
            else:
                plan = []
            plan = _broadcast_plan(plan, rank, world_size)
            if not plan:
                logger.info("Nothing left to run.")
                return

            # Resolve tokenizer source once for all checkpoints.
            # Priority: --tokenizer_path > model_root/final_model > base_model_path
            # Intermediate HF checkpoints do NOT contain tokenizer files, so we
            # must point to a directory that does.
            tokenizer_path = args.tokenizer_path
            if tokenizer_path is None:
                final_model = os.path.join(args.model_root, "final_model")
                if os.path.isdir(final_model) and os.path.exists(
                    os.path.join(final_model, "tokenizer_config.json")
                ):
                    tokenizer_path = final_model
                    logger.info("Auto-detected tokenizer at: %s", tokenizer_path)
                elif args.base_model_path and os.path.isdir(args.base_model_path):
                    tokenizer_path = args.base_model_path
                    logger.info(
                        "Using base_model_path as tokenizer: %s", tokenizer_path
                    )
                else:
                    logger.warning(
                        "Could not auto-detect tokenizer path. Will try loading "
                        "from each checkpoint directory (may fail for intermediate "
                        "checkpoints). Pass --tokenizer_path to fix this."
                    )
            # Pipeline mode: open ONE swanlab run for the entire remaining
            # checkpoint sequence.  manage_tracking=False means each run_bif()
            # only logs — it does not open/close the run itself.
            # global_step_offset shifts each checkpoint's steps onto a
            # continuous x-axis (checkpoint-0 uses steps 0..N-1,
            # checkpoint-1 uses steps N..2N-1, etc.).
            draws_per_ckpt = cfg.num_chains * cfg.draws_per_chain
            if args.experiment_name:
                run_label = args.experiment_name
            else:
                run_label = "bif_pipeline_resume" if args.resume else "bif_pipeline"
            if rank == 0:
                ckpt_names = [name for name, _ in plan]
                init_run(
                    experiment_name=run_label,
                    run_name=args.run_name,
                    config={
                        "checkpoints": ckpt_names,
                        "resume": args.resume,
                        "sgld": asdict(cfg),
                        "max_length": args.max_length,
                    },
                    tags=["bif", "pipeline"] + (["resume"] if args.resume else []),
                )
            for ckpt_idx, (ckpt_name, ckpt_path) in enumerate(plan):
                logger.info("Checkpoint: %s", ckpt_name)
                run_bif(
                    model_name_or_path=ckpt_path,
                    tokenizer_path=tokenizer_path,
                    pool_jsonl=args.pool_jsonl,
                    query_jsonl=args.query_jsonl,
                    out_dir=f"{args.out_dir}/{ckpt_name}",
                    sgld_cfg=cfg,
                    max_length=args.max_length,
                    train_batch_size=args.train_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    pool_eval_subset=args.pool_eval_subset,
                    device=args.device,
                    dtype=args.dtype,
                    manage_tracking=False,
                    global_step_offset=ckpt_idx * draws_per_ckpt,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # All checkpoints done — now safe to close the single run.
            # _barrier() inside run_bif() already ensured all ranks are here.
            if rank == 0:
                swan_finish()
        else:
            # Single-checkpoint mode: run_bif() owns its own swanlab run.
            run_bif(
                model_name_or_path=args.model_name_or_path,
                tokenizer_path=args.tokenizer_path,
                pool_jsonl=args.pool_jsonl,
                query_jsonl=args.query_jsonl,
                out_dir=args.out_dir,
                sgld_cfg=cfg,
                max_length=args.max_length,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                pool_eval_subset=args.pool_eval_subset,
                device=args.device,
                dtype=args.dtype,
                experiment_name=args.experiment_name,
                run_name=args.run_name,
                manage_tracking=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
