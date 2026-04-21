"""CLI entry point for BIF."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bif",
        description="BIF: Bayesian Influence Function toolkit",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- data ---
    p_pool2 = sub.add_parser(
        "build-pool-v2",
        help="Build a token-budget-aware multi-domain pool (nemotron_math / octothinker / fineweb_edu / c4 / wikipedia / sft_chat)",
    )
    from bif.data.build_pool import ALL_DOMAINS as _ALL_DOMAINS

    p_pool2.add_argument(
        "--total_tokens", default=None, help="Token budget: '4M', '8M', '500K', etc."
    )
    p_pool2.add_argument(
        "--pool_type",
        default=None,
        choices=["pt", "target", "sft"],
        help="Domain preset: pt (general pretraining), target (rare/task-specific), sft (supervised)",
    )
    p_pool2.add_argument(
        "--ratios", default=None, help="Per-domain ratios: 'domain1:0.4,domain2:0.3'"
    )
    p_pool2.add_argument(
        "--domains", default=None, help="Comma-separated active domains (default: all)"
    )
    for _d in _ALL_DOMAINS:
        p_pool2.add_argument(
            f"--n_{_d}", type=int, default=None, help=f"Explicit sample count for {_d}"
        )
    p_pool2.add_argument("--min_chars", type=int, default=200)
    p_pool2.add_argument("--min_tokens", type=int, default=50)
    p_pool2.add_argument("--max_tokens", type=int, default=4096)
    p_pool2.add_argument("--min_int_score", type=int, default=0)
    p_pool2.add_argument("--min_lang_score", type=float, default=0.8)
    p_pool2.add_argument("--out_dir", required=True)
    p_pool2.add_argument("--out_name", default=None)
    p_pool2.add_argument("--seed", type=int, default=42)
    p_pool2.add_argument(
        "--tokenizer_path", default=None,
        help="Tokenizer for accurate token counting during calibration",
    )
    p_pool2.add_argument(
        "--calibrate_n", type=int, default=100,
        help="Samples per domain for calibration (default: 100)",
    )

    p_pool = sub.add_parser(
        "build-pool", help="Build data pool (alias for build-pool-v2)"
    )
    p_pool.add_argument("--out_dir", required=True)
    p_pool.add_argument("--out_name", default=None)
    p_pool.add_argument("--seed", type=int, default=42)
    p_pool.add_argument("--total_tokens", default=None)
    p_pool.add_argument("--domains", default=None)
    p_pool.add_argument("--ratios", default=None)
    p_pool.add_argument("--min_chars", type=int, default=200)
    p_pool.add_argument("--min_tokens", type=int, default=50)
    p_pool.add_argument("--max_tokens", type=int, default=4096)
    p_pool.add_argument("--min_int_score", type=int, default=0)
    p_pool.add_argument("--min_lang_score", type=float, default=0.8)

    p_ft = sub.add_parser("prepare-finetune", help="Prepare finetune data")
    p_ft.add_argument("--input_path", required=True)
    p_ft.add_argument("--tokenizer_path", required=True)
    p_ft.add_argument("--out_dir", required=True)
    p_ft.add_argument("--text_key", default="text")
    p_ft.add_argument("--train_n", type=int, default=None)
    p_ft.add_argument("--query_n", type=int, default=None)
    p_ft.add_argument("--val_n", type=int, default=None)
    p_ft.add_argument("--test_n", type=int, default=None)
    p_ft.add_argument("--train_ratio", type=float, default=0.7)
    p_ft.add_argument("--query_ratio", type=float, default=0.1)
    p_ft.add_argument("--val_ratio", type=float, default=0.1)
    p_ft.add_argument("--test_ratio", type=float, default=0.1)
    p_ft.add_argument("--min_chars", type=int, default=100)
    p_ft.add_argument("--min_token_count", type=int, default=200)
    p_ft.add_argument("--max_token_count", type=int, default=2000)
    p_ft.add_argument("--min_int_score", type=int, default=4)
    p_ft.add_argument("--min_score", type=float, default=0.0)
    p_ft.add_argument("--min_language_score", type=float, default=0.9)
    p_ft.add_argument("--require_english", action="store_true")
    p_ft.add_argument("--seed", type=int, default=42)

    # --- training ---
    p_train = sub.add_parser("train", help="Train stage-2 LM with checkpoints")
    p_train.add_argument("--base_model_path", required=True)
    p_train.add_argument("--tokenizer_path", required=True)
    p_train.add_argument("--train_jsonl", required=True)
    p_train.add_argument("--val_jsonl", required=True)
    p_train.add_argument("--output_dir", required=True)
    p_train.add_argument("--num_train_epochs", type=float, default=1.0)
    p_train.add_argument("--learning_rate", type=float, default=2e-5)
    p_train.add_argument("--per_device_train_batch_size", type=int, default=8)
    p_train.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p_train.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p_train.add_argument("--max_length", type=int, default=512)
    p_train.add_argument("--weight_decay", type=float, default=0.01)
    p_train.add_argument("--warmup_ratio", type=float, default=0.03)
    p_train.add_argument("--logging_steps", type=int, default=10)
    p_train.add_argument("--min_eval_steps", type=int, default=20)
    p_train.add_argument("--eval_steps", type=int, default=0)
    p_train.add_argument("--bf16", action="store_true")
    p_train.add_argument("--fp16", action="store_true")
    p_train.add_argument("--gradient_checkpointing", action="store_true")
    p_train.add_argument("--target_num_checkpoints", type=int, default=6)
    p_train.add_argument(
        "--experiment_name",
        default="stage2_train",
        help="SwanLab experiment name for this training run.",
    )
    p_train.add_argument(
        "--run_name",
        default=None,
        help="SwanLab run display name within the experiment.",
    )

    p_sched = sub.add_parser("schedule-compare", help="Schedule comparison training")
    p_sched.add_argument("--base_model_path", required=True)
    p_sched.add_argument("--tokenizer_path", required=True)
    p_sched.add_argument("--target_train_jsonl", required=True)
    p_sched.add_argument("--target_val_jsonl", required=True)
    p_sched.add_argument("--replay_pool_jsonl", default="")
    p_sched.add_argument("--output_dir", required=True)
    p_sched.add_argument("--run_name", required=True)
    p_sched.add_argument("--schedule", required=True, choices=["sequential", "mixed", "proportional"])
    p_sched.add_argument(
        "--replay_mode",
        required=True,
        choices=["selected", "random", "top_random", "none"],
    )
    p_sched.add_argument("--replay_ratio", type=float, default=0.0)
    p_sched.add_argument("--max_length", type=int, default=256)
    p_sched.add_argument("--num_train_epochs", type=float, default=1.0)
    p_sched.add_argument("--learning_rate", type=float, default=5e-5)
    p_sched.add_argument("--per_device_train_batch_size", type=int, default=2)
    p_sched.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p_sched.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p_sched.add_argument("--logging_steps", type=int, default=10)
    p_sched.add_argument("--eval_steps", type=int, default=50)
    p_sched.add_argument("--seed", type=int, default=42)
    p_sched.add_argument("--bf16", action="store_true")
    p_sched.add_argument("--fp16", action="store_true")
    p_sched.add_argument("--gradient_checkpointing", action="store_true")
    p_sched.add_argument(
        "--deepspeed", default=None,
        help="Path to DeepSpeed config JSON",
    )
    p_sched.add_argument(
        "--fsdp", default="",
        help="FSDP strategy string, e.g. 'full_shard auto_wrap'",
    )
    p_sched.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap", default=None,
        help="Transformer layer class name for FSDP auto-wrap",
    )
    p_sched.add_argument("--experiment_name", default="replay_train")
    p_sched.add_argument("--swanlab_run_id", default=None)
    p_sched.add_argument("--metric_prefix", default="")

    # --- analysis ---
    p_run = sub.add_parser("run-bif", help="Run BIF trace collection")
    p_run.add_argument("--model_name_or_path", default=None)
    p_run.add_argument("--model_root", default=None)
    p_run.add_argument("--base_model_path", default=None)
    p_run.add_argument(
        "--tokenizer_path",
        default=None,
        help="Tokenizer directory. Defaults to model_name_or_path. "
        "Set this to the base model path when running --run_all_checkpoints "
        "because HF checkpoint dirs do not contain tokenizer files.",
    )
    p_run.add_argument("--run_all_checkpoints", action="store_true")
    p_run.add_argument(
        "--resume",
        action="store_true",
        help="Skip checkpoints whose output directory already has complete traces.",
    )
    p_run.add_argument("--pool_jsonl", required=True)
    p_run.add_argument("--query_jsonl", required=True)
    p_run.add_argument("--out_dir", required=True)
    p_run.add_argument("--num_chains", type=int, default=4)
    p_run.add_argument("--draws_per_chain", type=int, default=60)
    p_run.add_argument("--max_length", type=int, default=256)
    p_run.add_argument("--train_batch_size", type=int, default=16)
    p_run.add_argument("--eval_batch_size", type=int, default=32)
    p_run.add_argument("--pool_eval_subset", type=int, default=0)
    p_run.add_argument("--lr", type=float, default=5e-6)
    p_run.add_argument("--gamma", type=float, default=1e-3)
    p_run.add_argument("--beta", type=float, default=1.0)
    p_run.add_argument("--noise_scale", type=float, default=1.0)
    p_run.add_argument("--burn_in", type=int, default=0)
    p_run.add_argument("--thinning", type=int, default=1)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--grad_clip", type=float, default=None)
    p_run.add_argument("--weight_decay", type=float, default=0.0)
    p_run.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    p_run.add_argument(
        "--experiment_name",
        default=None,
        help="SwanLab experiment name for this BIF run.",
    )
    p_run.add_argument(
        "--run_name",
        default=None,
        help="SwanLab run display name within the experiment.",
    )

    p_analyze = sub.add_parser("analyze-bif", help="Analyze BIF results")
    p_analyze.add_argument("--bif_root", required=True)
    p_analyze.add_argument("--out_dir", required=True)
    p_analyze.add_argument("--score_col", default="raw_cov_avg_over_queries")
    p_analyze.add_argument("--top_k", type=int, default=500)
    p_analyze.add_argument(
        "--experiment_name",
        default="bif_analysis",
        help="SwanLab experiment name for this analysis run.",
    )
    p_analyze.add_argument(
        "--run_name",
        default=None,
        help="SwanLab run display name within the experiment.",
    )

    p_extract = sub.add_parser("extract-top", help="Extract top-influence samples")
    p_extract.add_argument("--pool_jsonl", required=True)
    p_extract.add_argument("--ranking_csv", required=True)
    p_extract.add_argument("--out_dir", required=True)
    p_extract.add_argument("--id_col", default="sample_id")
    p_extract.add_argument("--source_col", default="source")
    p_extract.add_argument("--text_col", default="text")
    p_extract.add_argument("--score_col", default=None)
    p_extract.add_argument("--top_k", type=int, default=500)
    p_extract.add_argument("--top_n_per_source", type=int, default=3)
    p_extract.add_argument("--preview_chars", type=int, default=600)
    p_extract.add_argument("--restrict_source_topn_to_topk", action="store_true")
    p_extract.add_argument(
        "--experiment_name",
        default="bif_extraction",
        help="SwanLab experiment name for this extraction run.",
    )
    p_extract.add_argument(
        "--run_name",
        default=None,
        help="SwanLab run display name within the experiment.",
    )

    p_sched_a = sub.add_parser(
        "schedule-analyze", help="Analyze schedule comparison results"
    )
    p_sched_a.add_argument("--runs_root", required=True)
    p_sched_a.add_argument("--out_dir", required=True)
    p_sched_a.add_argument(
        "--data_dirs",
        action="append",
        default=[],
        help="label=path for data stats logging",
    )

    # --- pipeline ---
    p_pipe = sub.add_parser("pipeline", help="Full-pipeline runner with per-step state")
    pipe_sub = p_pipe.add_subparsers(
        dest="pipeline_command", help="Pipeline sub-commands"
    )

    p_pipe_run = pipe_sub.add_parser(
        "run", help="Run the full pipeline (or resume from a step)"
    )
    p_pipe_run.add_argument(
        "--config",
        required=True,
        metavar="EXPERIMENT.json",
        help="Path to the experiment JSON config file",
    )
    p_pipe_run.add_argument(
        "--from",
        dest="from_step",
        default=None,
        metavar="STEP",
        help="Start from this step (skip earlier steps)",
    )
    p_pipe_run.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps already marked complete in pipeline_state.json",
    )
    p_pipe_run.add_argument(
        "--new-run",
        action="store_true",
        help="Force a new SwanLab run instead of resuming the previous one",
    )

    p_pipe_status = pipe_sub.add_parser(
        "status", help="Show pipeline step completion status"
    )
    p_pipe_status.add_argument(
        "--config",
        required=True,
        metavar="EXPERIMENT.json",
        help="Path to the experiment JSON config file",
    )

    args = parser.parse_args()

    if args.command == "build-pool-v2":
        from bif.data.build_pool import (
            ALL_DOMAINS,
            POOL_PRESETS,
            _parse_token_count,
            _parse_ratios,
            _print_plan,
            build_domain_pool,
            calibrate_token_stats,
            compute_quotas,
        )

        if args.domains:
            domains = [d.strip() for d in args.domains.split(",") if d.strip()]
            unknown = [d for d in domains if d not in ALL_DOMAINS]
            if unknown:
                parser.error(f"Unknown domain(s): {unknown}. Supported: {ALL_DOMAINS}")
        elif args.pool_type:
            domains = POOL_PRESETS[args.pool_type]
        else:
            domains = list(ALL_DOMAINS)

        direct = {
            d: getattr(args, f"n_{d}")
            for d in domains
            if getattr(args, f"n_{d}") is not None
        }

        if direct and args.total_tokens:
            parser.error("--total_tokens and --n_<domain> cannot be used together.")

        calibrated_stats = None
        if args.tokenizer_path:
            calibrate_n = getattr(args, "calibrate_n", 100)
            print(f"[calibrate] Sampling {calibrate_n} examples per domain with tokenizer …")
            calibrated_stats = calibrate_token_stats(
                domains=domains,
                quota=calibrate_n,
                seed=args.seed,
                tokenizer_path=args.tokenizer_path,
                min_chars=args.min_chars,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
                min_int_score=args.min_int_score,
                min_lang_score=args.min_lang_score,
            )
            for d, s in calibrated_stats.items():
                print(
                    f"  {d:20s}  avg_tok/sample={s['avg_tokens_per_sample']}  "
                    f"chars/tok={s['chars_per_token']}  bpt={s['bytes_per_token']}"
                )

        if direct:
            quotas = direct
            print("[mode] explicit sample counts")
        elif args.total_tokens:
            total_tokens = _parse_token_count(args.total_tokens)
            ratios = _parse_ratios(args.ratios) if args.ratios else None
            quotas = compute_quotas(
                domains, total_tokens, ratios,
                calibrated_stats=calibrated_stats,
            )
            print(
                f"[mode] token budget  target={total_tokens:,} (~{total_tokens / 1e6:.1f}M)"
            )
        else:
            parser.error("Specify --total_tokens or at least one --n_<domain>.")

        print()
        total_tokens_val = _parse_token_count(args.total_tokens) if args.total_tokens else None
        _print_plan(quotas, total_tokens=total_tokens_val, calibrated_stats=calibrated_stats)

        out_path = build_domain_pool(
            quotas=quotas,
            out_dir=args.out_dir,
            seed=args.seed,
            min_chars=args.min_chars,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            min_int_score=args.min_int_score,
            min_lang_score=args.min_lang_score,
            output_filename=args.out_name,
            tokenizer_path=args.tokenizer_path,
            calibrated_stats=calibrated_stats,
        )
        print(f"\nPool written to: {out_path}")
        return

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "build-pool":
        print("Note: 'build-pool' is an alias for 'build-pool-v2'. Use 'build-pool-v2' instead.")
        parser.error("Use 'bif build-pool-v2' with the same arguments.")

    elif args.command == "prepare-finetune":
        from bif.data.finetune import prepare_finetune_data

        prepare_finetune_data(
            input_path=args.input_path,
            tokenizer_path=args.tokenizer_path,
            out_dir=args.out_dir,
            text_key=args.text_key,
            train_n=args.train_n,
            query_n=args.query_n,
            val_n=args.val_n,
            test_n=args.test_n,
            train_ratio=args.train_ratio,
            query_ratio=args.query_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            min_chars=args.min_chars,
            min_token_count=args.min_token_count,
            max_token_count=args.max_token_count,
            min_int_score=args.min_int_score,
            min_score=args.min_score,
            min_language_score=args.min_language_score,
            require_english=args.require_english,
            seed=args.seed,
        )

    elif args.command == "train":
        from bif.training.checkpoint_trainer import train_with_checkpoints

        train_with_checkpoints(
            base_model_path=args.base_model_path,
            tokenizer_path=args.tokenizer_path,
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            min_eval_steps=args.min_eval_steps,
            eval_steps=args.eval_steps,
            bf16=args.bf16,
            fp16=args.fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            target_num_checkpoints=args.target_num_checkpoints,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )

    elif args.command == "schedule-compare":
        from bif.training.schedule_trainer import train_schedule_compare

        train_schedule_compare(
            base_model_path=args.base_model_path,
            tokenizer_path=args.tokenizer_path,
            target_train_jsonl=args.target_train_jsonl,
            target_val_jsonl=args.target_val_jsonl,
            replay_pool_jsonl=args.replay_pool_jsonl,
            output_dir=args.output_dir,
            run_name=args.run_name,
            schedule=args.schedule,
            replay_mode=args.replay_mode,
            replay_ratio=args.replay_ratio,
            max_length=args.max_length,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            seed=args.seed,
            bf16=args.bf16,
            fp16=args.fp16,
            gradient_checkpointing=args.gradient_checkpointing,
            deepspeed=args.deepspeed,
            fsdp=args.fsdp,
            fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap,
            experiment_name=args.experiment_name,
        )

    elif args.command == "run-bif":
        # Delegate entirely to bif_runner.main() which owns all the logic for
        # --run_all_checkpoints, --resume, tokenizer auto-detection, distributed
        # process-group init, and multi-checkpoint SwanLab run management.
        import sys as _sys
        from bif.analysis import bif_runner as _bif_runner

        # Reconstruct argv so bif_runner.main()'s argparse sees the same flags.
        _argv = []
        if args.model_name_or_path:
            _argv += ["--model_name_or_path", args.model_name_or_path]
        if args.model_root:
            _argv += ["--model_root", args.model_root]
        if args.base_model_path:
            _argv += ["--base_model_path", args.base_model_path]
        if args.tokenizer_path:
            _argv += ["--tokenizer_path", args.tokenizer_path]
        if args.run_all_checkpoints:
            _argv += ["--run_all_checkpoints"]
        if getattr(args, "resume", False):
            _argv += ["--resume"]
        _argv += ["--pool_jsonl", args.pool_jsonl]
        _argv += ["--query_jsonl", args.query_jsonl]
        _argv += ["--out_dir", args.out_dir]
        _argv += ["--num_chains", str(args.num_chains)]
        _argv += ["--draws_per_chain", str(args.draws_per_chain)]
        _argv += ["--max_length", str(args.max_length)]
        _argv += ["--train_batch_size", str(args.train_batch_size)]
        _argv += ["--eval_batch_size", str(args.eval_batch_size)]
        _argv += ["--pool_eval_subset", str(args.pool_eval_subset)]
        _argv += ["--lr", str(args.lr)]
        _argv += ["--gamma", str(args.gamma)]
        _argv += ["--beta", str(args.beta)]
        _argv += ["--noise_scale", str(args.noise_scale)]
        _argv += ["--burn_in", str(args.burn_in)]
        _argv += ["--thinning", str(args.thinning)]
        _argv += ["--seed", str(args.seed)]
        if args.grad_clip is not None:
            _argv += ["--grad_clip", str(args.grad_clip)]
        _argv += ["--weight_decay", str(args.weight_decay)]
        _argv += ["--dtype", args.dtype]
        if args.experiment_name is not None:
            _argv += ["--experiment_name", args.experiment_name]
        if args.run_name is not None:
            _argv += ["--run_name", args.run_name]

        _saved = _sys.argv
        _sys.argv = ["bif run-bif"] + _argv
        try:
            _bif_runner.main()
        finally:
            _sys.argv = _saved

    elif args.command == "analyze-bif":
        from bif.analysis.bif_analyzer import analyze_bif_results

        analyze_bif_results(
            bif_root=args.bif_root,
            out_dir=args.out_dir,
            score_col=args.score_col,
            top_k=args.top_k,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )

    elif args.command == "extract-top":
        from bif.analysis.extractor import extract_top_samples

        extract_top_samples(
            pool_jsonl=args.pool_jsonl,
            ranking_csv=args.ranking_csv,
            out_dir=args.out_dir,
            id_col=args.id_col,
            source_col=args.source_col,
            text_col=args.text_col,
            score_col=args.score_col,
            top_k=args.top_k,
            top_n_per_source=args.top_n_per_source,
            preview_chars=args.preview_chars,
            restrict_source_topn_to_topk=args.restrict_source_topn_to_topk,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )

    elif args.command == "schedule-analyze":
        from bif.analysis.schedule_analyzer import analyze_schedule_compare

        data_dirs: dict[str, str] = {}
        for item in args.data_dirs:
            if "=" in item:
                k, v = item.split("=", 1)
                data_dirs[k] = v

        analyze_schedule_compare(
            runs_root=args.runs_root,
            out_dir=args.out_dir,
            data_dirs=data_dirs or None,
        )

    elif args.command == "pipeline":
        if args.pipeline_command is None:
            p_pipe.print_help()
            sys.exit(1)

        import json as _json
        from pathlib import Path as _Path

        if args.pipeline_command == "run":
            from bif.pipeline import run_pipeline

            run_pipeline(
                config_path=args.config,
                from_step=args.from_step,
                resume=args.resume,
                new_run=args.new_run,
            )

        elif args.pipeline_command == "status":
            from bif.pipeline import PipelineState, _load_config

            cfg = _load_config(args.config)
            PipelineState.load_or_create(cfg["work_dir"]).print_status()


if __name__ == "__main__":
    main()
