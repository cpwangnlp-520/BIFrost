"""Prepare stage-2 fine-tuning data with quality filtering."""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer

from bif.io import ensure_dir, extract_text, iter_records, write_jsonl


def _summarize(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    lengths = [len(tokenizer.encode(r["text"], add_special_tokens=False)) for r in rows]
    return {
        "count": len(rows),
        "mean_tokens": float(sum(lengths) / max(1, len(lengths))),
        "min_tokens": int(min(lengths)) if lengths else 0,
        "max_tokens": int(max(lengths)) if lengths else 0,
        "total_tokens": int(sum(lengths)),
    }


def prepare_finetune_data(
    input_path: str,
    tokenizer_path: str,
    out_dir: str,
    text_key: str = "text",
    train_n: int | None = None,
    query_n: int | None = None,
    val_n: int | None = None,
    test_n: int | None = None,
    train_ratio: float = 0.7,
    query_ratio: float = 0.1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_chars: int = 100,
    min_token_count: int = 200,
    max_token_count: int = 2000,
    require_english: bool = False,
    min_language_score: float = 0.9,
    min_int_score: int = 4,
    min_score: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Filter and split source data into train/query/val/test sets.

    Split sizing supports two modes:
      Explicit:  pass train_n / query_n / val_n / test_n (absolute counts)
      Ratio:     leave *_n as None, use *_ratio to split the filtered pool
      Mixed:     set some *_n and leave the rest as None — the explicit counts
                 are allocated first, the remainder is split by ratio
    """
    random.seed(seed)
    ensure_dir(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    collected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    raw_count = 0
    filtered_counts: dict[str, int] = {
        k: 0
        for k in [
            "missing_text",
            "too_short_chars",
            "duplicate_url",
            "wrong_language",
            "low_language_score",
            "low_int_score",
            "low_score",
            "too_short_tokens",
            "too_long_tokens",
        ]
    }

    for row in tqdm(iter_records(input_path), desc="Reading source"):
        raw_count += 1
        text = extract_text(row, text_key)
        if text is None:
            filtered_counts["missing_text"] += 1
            continue
        if len(text) < min_chars:
            filtered_counts["too_short_chars"] += 1
            continue

        url = row.get("url", "") or ""
        url_str = str(url).strip()
        if url_str:
            if url_str in seen_urls:
                filtered_counts["duplicate_url"] += 1
                continue
            seen_urls.add(url_str)

        language = row.get("language") or row.get("lang")
        _ls_raw = (
            row.get("language_score")
            if row.get("language_score") is not None
            else row.get("lang_score")
        )
        language_score = float(_ls_raw) if _ls_raw is not None else 1.0
        int_score = int(row["int_score"]) if row.get("int_score") is not None else 0
        score = float(row["score"]) if row.get("score") is not None else 0.0
        _tc_raw = (
            row.get("token_count")
            if row.get("token_count") is not None
            else row.get("n_tokens")
        )
        token_count = int(_tc_raw) if _tc_raw is not None else -1

        if require_english and language != "en":
            filtered_counts["wrong_language"] += 1
            continue
        if language_score < min_language_score:
            filtered_counts["low_language_score"] += 1
            continue
        if int_score < min_int_score:
            filtered_counts["low_int_score"] += 1
            continue
        if score < min_score:
            filtered_counts["low_score"] += 1
            continue

        if token_count < 0:
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count < min_token_count:
            filtered_counts["too_short_tokens"] += 1
            continue
        if token_count > max_token_count:
            filtered_counts["too_long_tokens"] += 1
            continue

        subtype = row.get("snapshot_type")
        collected.append(
            {
                "id": f"stage2_{len(collected):07d}",
                "doc_id": url,
                "source": str(row.get("crawl", "finetune_corpus")),
                "subtype": str(subtype) if subtype else None,
                "url": url,
                "text": text,
                "token_count_meta": int(token_count),
                "language": language,
                "language_score": float(language_score),
                "score": float(score),
                "int_score": int(int_score),
            }
        )

    total_available = len(collected)

    # ── Resolve split sizes ────────────────────────────────────────────────
    _tn = train_n
    _qn = query_n
    _vn = val_n
    _xn = test_n

    # If all *_n are explicitly set, use them directly
    explicit_total = sum(x or 0 for x in (_tn, _qn, _vn, _xn))

    if _tn is not None and _qn is not None and _vn is not None and _xn is not None:
        need_total = explicit_total
        if total_available < need_total:
            raise RuntimeError(
                f"Not enough samples after filtering: need {need_total}, "
                f"got {total_available}"
            )
    else:
        # Mixed or pure-ratio mode: allocate explicit counts first, then
        # distribute the remainder proportionally.
        reserved = (_tn or 0) + (_qn or 0) + (_vn or 0) + (_xn or 0)
        remainder = total_available - reserved
        if remainder < 0:
            raise RuntimeError(
                f"Explicit *_n counts ({reserved}) exceed available samples ({total_available})"
            )
        ratio_total = train_ratio + query_ratio + val_ratio + test_ratio
        if ratio_total <= 0:
            raise ValueError(f"Split ratios must sum to > 0, got {ratio_total}")
        # Count how many splits are in ratio mode (no explicit *_n)
        ratio_splits = sum(1 for x in (_tn, _qn, _vn, _xn) if x is None)
        # Distribute remainder by ratio, rounding down; leftover goes to train
        leftover = remainder
        if _qn is None:
            _qn = int(remainder * query_ratio / ratio_total)
            leftover -= _qn
        if _vn is None:
            _vn = int(remainder * val_ratio / ratio_total)
            leftover -= _vn
        if _xn is None:
            _xn = int(remainder * test_ratio / ratio_total)
            leftover -= _xn
        if _tn is None:
            _tn = leftover
        if _qn < 1 and total_available > 0:
            _qn = 1
        if _vn < 1 and total_available > 0:
            _vn = 1
        if _xn < 1 and total_available > 0:
            _xn = 1
        if _tn < 1 and total_available > 0:
            _tn = 1

    random.shuffle(collected)
    query_rows = collected[:_qn]
    val_rows = collected[_qn : _qn + _vn]
    test_rows = collected[_qn + _vn : _qn + _vn + _xn]
    train_rows = collected[_qn + _vn + _xn : _qn + _vn + _xn + _tn]

    write_jsonl(f"{out_dir}/stage2_train_{len(train_rows)}.jsonl", train_rows)
    write_jsonl(f"{out_dir}/stage2_query_{len(query_rows)}.jsonl", query_rows)
    write_jsonl(f"{out_dir}/stage2_val_{len(val_rows)}.jsonl", val_rows)
    write_jsonl(f"{out_dir}/stage2_test_{len(test_rows)}.jsonl", test_rows)

    stats = {
        "input_path": input_path,
        "tokenizer_path": tokenizer_path,
        "raw_records_scanned": raw_count,
        "usable_records_after_filtering": total_available,
        "split_counts": {"train": _tn, "query": _qn, "val": _vn, "test": _xn},
        "filtered_counts": filtered_counts,
        "train_summary": _summarize(train_rows, tokenizer),
        "query_summary": _summarize(query_rows, tokenizer),
        "val_summary": _summarize(val_rows, tokenizer),
        "test_summary": _summarize(test_rows, tokenizer),
    }
    with open(f"{out_dir}/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare stage-2 fine-tuning data.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--train_n", type=int, default=None)
    parser.add_argument("--query_n", type=int, default=None)
    parser.add_argument("--val_n", type=int, default=None)
    parser.add_argument("--test_n", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--query_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--min_chars", type=int, default=100)
    parser.add_argument("--min_token_count", type=int, default=200)
    parser.add_argument("--max_token_count", type=int, default=2000)
    parser.add_argument("--require_english", action="store_true")
    parser.add_argument("--min_language_score", type=float, default=0.9)
    parser.add_argument("--min_int_score", type=int, default=4)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = prepare_finetune_data(
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
        require_english=args.require_english,
        min_language_score=args.min_language_score,
        min_int_score=args.min_int_score,
        min_score=args.min_score,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
