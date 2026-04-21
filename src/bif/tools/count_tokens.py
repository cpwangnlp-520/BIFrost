"""Token counter for training data files (JSONL / JSON / Parquet).

Usage examples
--------------
# Single file, default text key
python -m bif.tools.count_tokens \
    --tokenizer /workspace/models/Meta-Llama-3-8B \
    --input /workspace/pku_percy/datasets/sampled_10k/stage2_train_8000.jsonl

# Multiple files at once
python -m bif.tools.count_tokens \
    --tokenizer /workspace/models/Meta-Llama-3-8B \
    --input pool.jsonl train.jsonl val.jsonl

# Whole directory (scans .jsonl / .json / .parquet recursively)
python -m bif.tools.count_tokens \
    --tokenizer /workspace/models/Meta-Llama-3-8B \
    --input /workspace/pku_percy/datasets/sampled_10k/

# Custom text key + truncation at max_length
python -m bif.tools.count_tokens \
    --tokenizer /workspace/models/Meta-Llama-3-8B \
    --input train.jsonl \
    --text_key content \
    --max_length 2048

# Save per-sample details to CSV
python -m bif.tools.count_tokens \
    --tokenizer /workspace/models/Meta-Llama-3-8B \
    --input train.jsonl \
    --save_csv token_stats.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from bif.io import iter_records, list_input_files


# ─────────────────────────────────────────────────────────────
# Core counting logic
# ─────────────────────────────────────────────────────────────

def _count_file(
    file_path: str,
    tokenizer: Any,
    text_key: str,
    max_length: int | None,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Return per-sample dicts with token counts for one file."""
    rows = list(iter_records(file_path))
    results: list[dict[str, Any]] = []

    texts: list[str] = []
    meta: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        text = row.get(text_key)
        if text is None:
            # try common fallback keys
            for k in ("content", "body", "document", "article"):
                if row.get(k):
                    text = row[k]
                    break
        if text is None:
            results.append(
                {
                    "file": file_path,
                    "index": idx,
                    "id": row.get("id", idx),
                    "tokens": None,
                    "skipped": True,
                }
            )
            continue
        texts.append(str(text))
        meta.append({"file": file_path, "index": idx, "id": row.get("id", idx)})

    # Batch tokenise (no padding, no special tokens for raw counts)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=max_length is not None,
            max_length=max_length,
            padding=False,
        )
        for i, ids in enumerate(enc["input_ids"]):
            m = meta[start + i]
            results.append(
                {
                    "file": m["file"],
                    "index": m["index"],
                    "id": m["id"],
                    "tokens": len(ids),
                    "skipped": False,
                }
            )

    return results


# ─────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────

def _compute_stats(token_counts: list[int]) -> dict[str, Any]:
    if not token_counts:
        return {}
    arr = np.array(token_counts, dtype=np.float64)
    total = int(arr.sum())
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std())
    return {
        "samples": n,
        "total_tokens": total,
        "mean": round(mean, 1),
        "std": round(std, 1),
        "min": int(arr.min()),
        "p25": round(float(np.percentile(arr, 25)), 1),
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p75": round(float(np.percentile(arr, 75)), 1),
        "p90": round(float(np.percentile(arr, 90)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "p99": round(float(np.percentile(arr, 99)), 1),
        "max": int(arr.max()),
    }


# ─────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────

def _fmt_num(n: int | float) -> str:
    """Format large numbers with commas."""
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def _print_table(header: str, stats: dict[str, Any], skipped: int) -> None:
    col = 28
    print(f"\n{'─' * 60}")
    print(f"  {header}")
    print(f"{'─' * 60}")
    print(f"  {'Samples':<{col}} {_fmt_num(stats['samples'])}")
    if skipped:
        print(f"  {'Skipped (no text field)':<{col}} {skipped}")
    print(f"  {'Total tokens':<{col}} {_fmt_num(stats['total_tokens'])}")
    print(f"  {'Mean tokens / sample':<{col}} {_fmt_num(stats['mean'])}")
    print(f"  {'Std dev':<{col}} {_fmt_num(stats['std'])}")
    print(f"{'─' * 60}")
    print(f"  {'Min':<{col}} {_fmt_num(stats['min'])}")
    print(f"  {'p25':<{col}} {_fmt_num(stats['p25'])}")
    print(f"  {'Median (p50)':<{col}} {_fmt_num(stats['p50'])}")
    print(f"  {'p75':<{col}} {_fmt_num(stats['p75'])}")
    print(f"  {'p90':<{col}} {_fmt_num(stats['p90'])}")
    print(f"  {'p95':<{col}} {_fmt_num(stats['p95'])}")
    print(f"  {'p99':<{col}} {_fmt_num(stats['p99'])}")
    print(f"  {'Max':<{col}} {_fmt_num(stats['max'])}")
    print(f"{'─' * 60}")


def _print_histogram(token_counts: list[int], bins: int = 10) -> None:
    if not token_counts:
        return
    lo, hi = min(token_counts), max(token_counts)
    if lo == hi:
        print(f"\n  All samples have exactly {lo} tokens.\n")
        return
    step = (hi - lo) / bins
    buckets: list[int] = [0] * bins
    for v in token_counts:
        b = min(int((v - lo) / step), bins - 1)
        buckets[b] += 1
    bar_max = max(buckets)
    bar_width = 30
    print(f"\n  Token distribution (n={len(token_counts):,})")
    print(f"  {'Range':<22} {'Count':>7}  {'Bar'}")
    print(f"  {'─' * 60}")
    for i, count in enumerate(buckets):
        left = int(lo + i * step)
        right = int(lo + (i + 1) * step)
        bar_len = int(count / bar_max * bar_width) if bar_max else 0
        bar = "█" * bar_len
        label = f"{left:>6,} – {right:>6,}"
        print(f"  {label}   {count:>7,}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def count_tokens(
    inputs: list[str],
    tokenizer_path: str,
    text_key: str = "text",
    max_length: int | None = None,
    batch_size: int = 512,
    histogram: bool = True,
    save_csv: str | None = None,
    save_json_path: str | None = None,
) -> dict[str, Any]:
    """Count tokens across one or more JSONL/JSON/Parquet files.

    Returns a summary dict with per-file and overall stats.
    """
    t0 = time.time()

    # Expand directories → individual files
    all_files: list[str] = []
    for inp in inputs:
        all_files.extend(list_input_files(inp))
    all_files = sorted(set(all_files))

    if not all_files:
        raise ValueError(f"No supported files found in: {inputs}")

    print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    print(f"Tokenizer: {tokenizer.__class__.__name__}  vocab_size={tokenizer.vocab_size:,}")
    if max_length:
        print(f"Truncating at max_length={max_length}")
    print(f"Files to process: {len(all_files)}\n")

    all_results: list[dict[str, Any]] = []
    per_file_stats: list[dict[str, Any]] = []

    for fp in tqdm(all_files, desc="Files", unit="file"):
        file_results = _count_file(fp, tokenizer, text_key, max_length, batch_size)
        all_results.extend(file_results)

        valid = [r["tokens"] for r in file_results if not r["skipped"]]
        skip_n = sum(1 for r in file_results if r["skipped"])
        stats = _compute_stats(valid)
        stats["file"] = fp
        stats["skipped"] = skip_n
        per_file_stats.append(stats)

        if len(all_files) > 1:
            # Brief per-file summary line
            tqdm.write(
                f"  {Path(fp).name}: {stats.get('samples', 0):,} samples  "
                f"{stats.get('total_tokens', 0):,} tokens  "
                f"mean={stats.get('mean', 0):.0f}  "
                f"max={stats.get('max', 0):,}"
                + (f"  skipped={skip_n}" if skip_n else "")
            )

    # ── Per-file tables (only when >1 file) ──────────────────
    if len(all_files) > 1:
        for s in per_file_stats:
            _print_table(Path(s["file"]).name, s, s["skipped"])

    # ── Overall stats ─────────────────────────────────────────
    all_valid = [r["tokens"] for r in all_results if not r["skipped"]]
    total_skipped = sum(1 for r in all_results if r["skipped"])
    overall = _compute_stats(all_valid)

    title = Path(all_files[0]).name if len(all_files) == 1 else f"OVERALL ({len(all_files)} files)"
    _print_table(title, overall, total_skipped)

    if histogram:
        _print_histogram(all_valid)

    elapsed = time.time() - t0
    speed = len(all_valid) / elapsed if elapsed > 0 else 0
    print(f"  Elapsed: {elapsed:.1f}s  ({speed:,.0f} samples/s)\n")

    # ── Save CSV ──────────────────────────────────────────────
    if save_csv:
        import csv
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "index", "id", "tokens", "skipped"])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  Per-sample CSV saved → {save_csv}")

    # ── Save JSON summary ─────────────────────────────────────
    summary = {
        "tokenizer": tokenizer_path,
        "text_key": text_key,
        "max_length": max_length,
        "files": per_file_stats,
        "overall": overall,
        "total_skipped": total_skipped,
    }
    if save_json_path:
        Path(save_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  Summary JSON saved  → {save_json_path}\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count tokens in JSONL/JSON/Parquet training files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path or HuggingFace hub name of the tokenizer.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more file paths or directories to scan.",
    )
    parser.add_argument(
        "--text_key",
        default="text",
        help="JSON field containing the text (default: text).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Truncate tokens at this length (counts truncated length). "
             "Omit for raw counts.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Tokenizer batch size (default: 512).",
    )
    parser.add_argument(
        "--no_histogram",
        action="store_true",
        help="Skip the distribution histogram.",
    )
    parser.add_argument(
        "--save_csv",
        default=None,
        metavar="FILE.csv",
        help="Save per-sample token counts to a CSV file.",
    )
    parser.add_argument(
        "--save_json",
        default=None,
        metavar="FILE.json",
        help="Save summary statistics to a JSON file.",
    )
    args = parser.parse_args()

    try:
        count_tokens(
            inputs=args.input,
            tokenizer_path=args.tokenizer,
            text_key=args.text_key,
            max_length=args.max_length,
            batch_size=args.batch_size,
            histogram=not args.no_histogram,
            save_csv=args.save_csv,
            save_json_path=args.save_json,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
