"""Build a token-budget-aware data pool from multiple raw dataset domains.

Supported domains (PT pool — common / general)
----------------------------------------------
fineweb_edu     fineweb-edu/data/**  (parquet, carries int_score / score fields)
c4              c4/en/*.json         (json-lines)
wikipedia       wikimedia/wikipedia/20231101.en/  (parquet)
slimpajama      SlimPajama-6B/data/ (parquet, mixed: C4/CommonCrawl/GitHub/Wikipedia/StackExchange)

Supported domains (Target pool — rare / task-specific)
------------------------------------------------------
nemotron_math   Nemotron-CC-Math-v1  (parquet, quality subsets: 3 / 4plus / 4plus_MIND)
octothinker     OctoThinker/MegaMath-Web-Pro-Max  (parquet)
starcoder       starcoderdata/       (parquet, code data)
flan            dolma/flan/          (json.gz, instruction-following data)

Two sampling modes
------------------
1. Token budget  (--total_tokens 4M)  — distribute quota by ratio or equally.
   Token estimation uses file-size sampling, NOT per-sample tokenizer encoding.
2. Explicit counts  (--n_<domain> N)  — skip ratio logic entirely.

Output schema (BIF-compatible)
-------------------------------
id, doc_id, source, subtype, text, n_tokens, url

CLI examples
------------
# PT pool 8 M tokens, equal split across all PT domains
bif build-pool-v2 --total_tokens 8M --out_dir /workspace/.../pool

# PT pool with explicit ratios
bif build-pool-v2 \\
    --total_tokens 8M \\
    --ratios nemotron_math:0.35,octothinker:0.25,fineweb_edu:0.20,c4:0.10,wikipedia:0.10 \\
    --out_dir /workspace/.../pool

# SFT / FT pool 4M tokens using only sft_chat domain
bif build-pool-v2 \\
    --total_tokens 4M \\
    --domains sft_chat \\
    --out_dir /workspace/.../ft_pool

# Mixed PT + SFT pool with explicit domain subset
bif build-pool-v2 \\
    --total_tokens 8M \\
    --domains nemotron_math,octothinker,fineweb_edu \\
    --out_dir /workspace/.../pt_pool

# Explicit sample counts (no token budget)
bif build-pool-v2 \\
    --n_nemotron_math 3000 --n_fineweb_edu 5000 --n_c4 2000 \\
    --out_dir /workspace/.../custom_pool
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from bif.io import ensure_dir, normalize_text, save_json, write_jsonl
from bif.utils.logging import get_logger

logger = get_logger("bif.data.build_pool")

# ---------------------------------------------------------------------------
# Dataset roots  (corrected to actual on-disk layout)
# ---------------------------------------------------------------------------

_PT_ROOT  = Path("/workspace/pku_percy/datasets/pt-data")
_SFT_ROOT = Path("/workspace/pku_percy/datasets/sft-data")

_DOMAIN_ROOTS: dict[str, Path] = {
    "nemotron_math": _PT_ROOT / "Nemotron-CC-Math-v1",
    "octothinker":   _PT_ROOT / "OctoThinker" / "MegaMath-Web-Pro-Max",
    "finemath":      _PT_ROOT / "finemath" / "finemath-3plus",
    "fineweb_edu":   _PT_ROOT / "fineweb-edu" / "data",
    "c4":            _PT_ROOT / "c4" / "en",
    "wikipedia":     _PT_ROOT / "wikimedia" / "wikipedia" / "20231101.en",
    "slimpajama":    _PT_ROOT / "SlimPajama-6B" / "data",
    "starcoder":     _PT_ROOT / "starcoderdata",
    "flan":          _PT_ROOT / "dolma" / "flan",
    "sft_chat":      _SFT_ROOT / "Nemotron-Post-Training-Dataset-v2" / "data",
}

ALL_DOMAINS: list[str] = list(_DOMAIN_ROOTS.keys())
PT_DOMAINS:  list[str] = ["fineweb_edu", "c4", "wikipedia", "slimpajama", "finemath"]
TARGET_DOMAINS: list[str] = ["nemotron_math", "octothinker", "starcoder", "flan"]
SFT_DOMAINS: list[str] = ["sft_chat"]

POOL_PRESETS: dict[str, list[str]] = {
    "pt": PT_DOMAINS,
    "target": TARGET_DOMAINS,
    "sft": SFT_DOMAINS,
}

# Quality subsets for Nemotron-CC-Math, ordered high → low quality.
_NEMOTRON_SUBSETS: list[str] = ["4plus_MIND", "4plus", "3"]

# ---------------------------------------------------------------------------
# File-size-based token estimation
# ---------------------------------------------------------------------------
# Bytes-per-token calibration for each domain.
# Derived empirically from sampled parquet/json files:
#   nemotron_math: parquet compressed ~0.37 bytes/char, ~4.5 chars/token → ~1.7 bytes/token
#   octothinker  : parquet ~1.05 bytes/char, ~4.5 chars/token → ~4.7 bytes/token
#   fineweb_edu  : parquet ~0.5 bytes/char (estimate) → ~2.3 bytes/token
#   c4           : json-lines ~38 bytes/raw-line but actual text≈2130 chars avg,
#                  file has overhead; raw file bytes / (chars/4) ≈ 38*10000/(2130/4)=714 — wrong;
#                  more carefully: 820MB file, ~356K rows, avg 2130 chars → ~4.5 chars/tok
#                  → ~820e6/(356e3*(2130/4.5)) ≈ 4.9 bytes/token. Use 5.
#   wikipedia    : parquet ~0.60 bytes/char, ~4.5 chars/token → ~2.7 bytes/token
#   sft_chat     : messages are nested; estimate ~6 bytes/token (conservative)
#
# These are used ONLY for converting a token budget → sample-count quota.
# Per-sample token estimation during filtering still uses len(text)//4.
_BYTES_PER_TOKEN: dict[str, float] = {
    "nemotron_math": 1.7,
    "octothinker":   4.7,
    "fineweb_edu":   2.3,
    "finemath":       4.0,
    "c4":            5.0,
    "wikipedia":     2.7,
    "slimpajama":    3.5,
    "starcoder":     3.0,
    "flan":          4.0,
    "sft_chat":      6.0,
}

_DEFAULT_BYTES_PER_TOKEN: float = 4.0

_CHARS_PER_TOKEN: int = 4


def calibrate_token_stats(
    domains: list[str] | None = None,
    quota: int = 100,
    seed: int = 42,
    tokenizer_path: str | None = None,
    min_chars: int = 200,
    min_tokens: int = 50,
    max_tokens: int = 4096,
    min_int_score: int = 0,
    min_lang_score: float = 0.8,
) -> dict[str, dict[str, float]]:
    """Sample a small batch per domain and measure actual token statistics.

    If *tokenizer_path* is provided, uses the real tokenizer for accurate
    counts; otherwise falls back to ``len(text) // 4``.

    Returns:
        Per-domain dict with keys: avg_chars, avg_tokens, chars_per_token,
        avg_tokens_per_sample, total_bytes, bytes_per_token, est_total_tokens.
    """
    from transformers import AutoTokenizer

    target_domains = domains or list(ALL_DOMAINS)
    tok = None
    if tokenizer_path:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    results: dict[str, dict[str, float]] = {}
    for domain in target_domains:
        rows = collect_domain(
            domain,
            quota=quota,
            seed=seed,
            min_chars=min_chars,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            min_int_score=min_int_score,
            min_lang_score=min_lang_score,
        )
        if not rows:
            continue
        chars_list = [len(r["text"]) for r in rows]
        if tok:
            tok_counts = [len(tok.encode(r["text"])) for r in rows]
        else:
            tok_counts = [c // _CHARS_PER_TOKEN for c in chars_list]
        avg_chars = sum(chars_list) / len(chars_list)
        avg_tok = sum(tok_counts) / len(tok_counts)
        cpt = avg_chars / avg_tok if avg_tok > 0 else _CHARS_PER_TOKEN
        total_bytes = _get_domain_bytes(domain)
        bpt = total_bytes / (avg_tok * len(rows)) if avg_tok > 0 and len(rows) > 0 else _DEFAULT_BYTES_PER_TOKEN
        est_total = total_bytes / bpt if bpt > 0 else 0
        results[domain] = {
            "avg_chars": round(avg_chars),
            "avg_tokens_per_sample": round(avg_tok),
            "chars_per_token": round(cpt, 2),
            "total_bytes": total_bytes,
            "bytes_per_token": round(bpt, 2),
            "est_total_tokens": int(est_total),
        }
    return results


def _domain_total_bytes(domain: str) -> int:
    root = _DOMAIN_ROOTS.get(domain)
    if root is None or not root.exists():
        return 0

    total = 0
    if domain == "nemotron_math":
        for subset in _NEMOTRON_SUBSETS:
            subdir = root / subset
            if subdir.is_dir():
                for fp in subdir.glob("*.parquet"):
                    total += fp.stat().st_size
    elif domain == "sft_chat":
        for fp in root.glob("chat-*.parquet"):
            total += fp.stat().st_size
    elif domain == "flan":
        for ext in (".json.gz", ".json", ".parquet"):
            for fp_str in glob.glob(str(root / f"**/*{ext}"), recursive=True):
                total += os.path.getsize(fp_str)
    elif domain == "starcoder":
        for ext in (".parquet", ".json", ".jsonl"):
            for fp_str in glob.glob(str(root / f"**/*{ext}"), recursive=True):
                total += os.path.getsize(fp_str)
    else:
        for ext in (".parquet", ".json", ".jsonl"):
            for fp_str in glob.glob(str(root / f"**/*{ext}"), recursive=True):
                total += os.path.getsize(fp_str)
    return total


_DOMAIN_BYTES_CACHE: dict[str, int] = {}


def _get_domain_bytes(domain: str) -> int:
    if domain not in _DOMAIN_BYTES_CACHE:
        _DOMAIN_BYTES_CACHE[domain] = _domain_total_bytes(domain)
    return _DOMAIN_BYTES_CACHE[domain]


def estimate_domain_tokens(domain: str) -> int:
    """Estimate total tokens available in *domain* from raw file sizes.

    Formula::

        est_tokens = total_bytes / bytes_per_token[domain]

    Much faster than reading all files; accurate to ±30% for planning purposes.
    """
    total_bytes = _get_domain_bytes(domain)
    bpt = _BYTES_PER_TOKEN.get(domain, _DEFAULT_BYTES_PER_TOKEN)
    return max(1, int(total_bytes / bpt))


# ---------------------------------------------------------------------------
# Low-level file iteration
# ---------------------------------------------------------------------------

def _list_files(root: Path, exts: tuple[str, ...] = (".parquet", ".json", ".jsonl", ".json.gz")) -> list[str]:
    files: list[str] = []
    for ext in exts:
        files.extend(glob.glob(str(root / f"**/*{ext}"), recursive=True))
    return sorted(set(files))


def _iter_parquet(path: str) -> Iterator[dict[str, Any]]:
    df = pd.read_parquet(path)
    yield from df.to_dict(orient="records")


def _iter_json(path: str) -> Iterator[dict[str, Any]]:
    """Handle both JSON-array and JSON-lines files robustly."""
    with open(path, encoding="utf-8") as f:
        head = f.read(8).lstrip()
    if head.startswith("["):
        with open(path, encoding="utf-8") as f:
            try:
                obj = json.load(f)
                for row in obj:
                    if isinstance(row, dict):
                        yield row
            except json.JSONDecodeError:
                pass
        return
    # JSON-lines fallback
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSON line %d in %s", line_no, path)
                continue


def _iter_file(path: str) -> Iterator[dict[str, Any]]:
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        yield from _iter_parquet(path)
    elif ext == ".gz":
        yield from _iter_json_gz(path)
    else:
        yield from _iter_json(path)


def _iter_json_gz(path: str) -> Iterator[dict[str, Any]]:
    """Read gzip-compressed JSON-lines file."""
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSON line %d in %s", line_no, path)
                continue


# ---------------------------------------------------------------------------
# Domain-specific record extractors
# ---------------------------------------------------------------------------

def _extract_nemotron_math(row: dict[str, Any]) -> dict[str, Any] | None:
    """Nemotron-CC-Math-v1 schema: id, text, metadata (dict)."""
    text = normalize_text(row.get("text") or "")
    if not text:
        return None
    meta = row.get("metadata") or {}
    url = meta.get("url", "") if isinstance(meta, dict) else ""
    return {
        "text": text,
        "doc_id": str(row.get("id", "")),
        "url": url,
        "subtype": row.get("_subtype"),  # injected by the file-walker
        "lang": "en",
        "lang_score": 1.0,
        "int_score": 4,
        "score": 4.0,
    }


def _extract_octothinker(row: dict[str, Any]) -> dict[str, Any] | None:
    """OctoThinker schema: text, id, domain, finemath_score, lang, lang_score."""
    text = normalize_text(row.get("text") or "")
    if not text:
        return None
    lang = str(row.get("lang") or "en")
    if lang and lang != "en":
        return None
    lang_score = float(row.get("lang_score") or 1.0)
    return {
        "text": text,
        "doc_id": str(row.get("id", "")),
        "url": str(row.get("url", "")),
        "subtype": str(row.get("domain") or ""),
        "lang": lang,
        "lang_score": lang_score,
        "int_score": 4,
        "score": float(row.get("finemath_score") or 0.5),
    }


def _extract_fineweb_edu(row: dict[str, Any]) -> dict[str, Any] | None:
    """FineWeb-Edu schema: text, id, dump, url, language, language_score, token_count, score, int_score."""
    text = normalize_text(row.get("text") or "")
    if not text:
        return None
    lang = str(row.get("language") or "en")
    if lang and lang != "en":
        return None
    lang_score = None
    try:
        lang_score = float(row.get("language_score") or 1.0)
    except (TypeError, ValueError):
        lang_score = 1.0
    token_count = -1
    try:
        tc = row.get("token_count")
        if tc is not None:
            token_count = int(tc)
    except (TypeError, ValueError):
        token_count = -1
    int_score = 0
    try:
        int_score = int(row.get("int_score") or 0)
    except (TypeError, ValueError):
        int_score = 0
    score = 0.0
    try:
        score = float(row.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "text": text,
        "doc_id": str(row.get("id", "")),
        "url": str(row.get("url", "")),
        "subtype": str(row.get("dump") or ""),
        "lang": lang,
        "lang_score": lang_score,
        "int_score": int_score,
        "score": score,
        "token_count": token_count,
    }


def _extract_c4(row: dict[str, Any]) -> dict[str, Any] | None:
    """C4 schema: text, timestamp, url."""
    text = normalize_text(row.get("text") or "")
    if not text:
        return None
    return {
        "text": text,
        "doc_id": str(row.get("url", "")),
        "url": str(row.get("url", "")),
        "subtype": None,
        "lang": "en",
        "lang_score": 1.0,
        "int_score": 3,
        "score": 3.0,
    }


def _extract_wikipedia(row: dict[str, Any]) -> dict[str, Any] | None:
    """Wikimedia schema: id, url, title, text."""
    body = normalize_text(row.get("text") or "")
    if not body:
        return None
    title = str(row.get("title") or "").strip()
    text = normalize_text(f"{title}\n\n{body}") if title else body
    return {
        "text": text,
        "doc_id": str(row.get("id", "")),
        "url": str(row.get("url", "")),
        "subtype": None,
        "lang": "en",
        "lang_score": 1.0,
        "int_score": 3,
        "score": 3.0,
    }


def _messages_to_text(messages: Any) -> str | None:
    """Flatten Nemotron SFT conversation into plain text."""
    try:
        parts: list[str] = []
        for m in messages:
            role = str(m.get("role", "")).strip()
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                parts.append(f"[System] {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                # Strip <think>...</think> reasoning traces, keep final answer only.
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if content:
                    parts.append(f"Assistant: {content}")
        return "\n\n".join(parts) if parts else None
    except Exception:
        return None


def _extract_sft_chat(row: dict[str, Any]) -> dict[str, Any] | None:
    """Nemotron SFT schema: uuid, messages, category, reasoning."""
    text = _messages_to_text(row.get("messages"))
    if not text:
        return None
    text = normalize_text(text)
    return {
        "text": text,
        "doc_id": str(row.get("uuid", "")),
        "url": "",
        "subtype": str(row.get("category") or ""),
        "lang": "en",
        "lang_score": 1.0,
        "int_score": 4,
        "score": 4.0,
    }


def _extract_slimpajama(row: dict[str, Any]) -> dict[str, Any] | None:
    """SlimPajama-6B schema: text, meta (dict with redpajama_set_name)."""
    text = normalize_text(row.get("text") or "")
    if not text:
        return None
    meta = row.get("meta") or {}
    source = meta.get("redpajama_set_name", "") if isinstance(meta, dict) else ""
    return {
        "text": text,
        "doc_id": str(row.get("__index_level_0__", "")),
        "url": "",
        "subtype": source,
        "lang": "en",
        "lang_score": 1.0,
        "int_score": 3,
        "score": 3.0,
    }


def _extract_starcoder(row: dict[str, Any]) -> dict[str, Any] | None:
    """StarCoder data schema: content / text, possibly with meta fields."""
    text = normalize_text(row.get("content") or row.get("text") or "")
    if not text:
        return None
    return {
        "text": text,
        "doc_id": str(row.get("id", "")),
        "url": str(row.get("repository_name") or row.get("url") or ""),
        "subtype": str(row.get("language") or row.get("max_stars_repo_name") or ""),
        "lang": "code",
        "lang_score": 1.0,
        "int_score": 3,
        "score": 3.0,
    }


def _extract_flan(row: dict[str, Any]) -> dict[str, Any] | None:
    """Dolma FLAN schema: text, possibly with id / source fields."""
    text = normalize_text(row.get("text") or "")
    if not text:
        return None
    return {
        "text": text,
        "doc_id": str(row.get("id", "")),
        "url": "",
        "subtype": "flan",
        "lang": "en",
        "lang_score": 1.0,
        "int_score": 3,
        "score": 3.0,
    }


def _extract_finemath(row: dict[str, Any]) -> dict[str, Any] | None:
    """FineMath-3plus schema: text, url, token_count, char_count, score, int_score, language, language_score."""
    text = normalize_text(row.get("text") or "")
    if not text:
        return None
    lang = str(row.get("language") or "en")
    if lang and lang != "en":
        return None
    lang_score = None
    try:
        lang_score = float(row.get("language_score") or 1.0)
    except (TypeError, ValueError):
        lang_score = 1.0
    int_score = 0
    try:
        int_score = int(row.get("int_score") or 0)
    except (TypeError, ValueError):
        int_score = 0
    score = 0.0
    try:
        score = float(row.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "text": text,
        "doc_id": str(row.get("url", "")),
        "url": str(row.get("url", "")),
        "subtype": str(row.get("crawl") or ""),
        "lang": lang,
        "lang_score": lang_score,
        "int_score": int_score,
        "score": score,
        "token_count": int(row.get("token_count") or -1),
    }


_EXTRACTORS: dict[str, Any] = {
    "nemotron_math": _extract_nemotron_math,
    "octothinker":   _extract_octothinker,
    "finemath":      _extract_finemath,
    "fineweb_edu":   _extract_fineweb_edu,
    "c4":            _extract_c4,
    "wikipedia":     _extract_wikipedia,
    "slimpajama":    _extract_slimpajama,
    "starcoder":     _extract_starcoder,
    "flan":          _extract_flan,
    "sft_chat":      _extract_sft_chat,
}


# ---------------------------------------------------------------------------
# File-walker per domain
# ---------------------------------------------------------------------------

def _domain_file_entries(domain: str) -> list[tuple[str, str | None]]:
    root = _DOMAIN_ROOTS[domain]
    entries: list[tuple[str, str | None]] = []

    if domain == "nemotron_math":
        for subset in _NEMOTRON_SUBSETS:
            subdir = root / subset
            if subdir.is_dir():
                for fp in sorted(subdir.glob("*.parquet")):
                    entries.append((str(fp), subset))
        return entries

    if domain == "sft_chat":
        for fp in sorted(root.glob("chat-*.parquet")):
            entries.append((str(fp), "chat"))
        return entries

    if domain == "flan":
        for fp in sorted(root.glob("*.json.gz")):
            entries.append((str(fp), "flan"))
        for fp in sorted(root.glob("*.parquet")):
            entries.append((str(fp), "flan"))
        return entries

    if domain == "slimpajama":
        for fp in sorted(root.glob("train-*.parquet")):
            entries.append((str(fp), None))
        return entries

    if domain == "finemath":
        for fp in sorted(root.glob("train-*.parquet")):
            entries.append((str(fp), None))
        return entries

    for fp in _list_files(root):
        entries.append((fp, None))
    return entries


# ---------------------------------------------------------------------------
# Core sampling
# ---------------------------------------------------------------------------

def collect_domain(
    domain: str,
    quota: int,
    seed: int,
    min_chars: int = 200,
    min_tokens: int = 50,
    max_tokens: int = 4096,
    min_int_score: int = 0,
    min_lang_score: float = 0.8,
    shuffle_files: bool = True,
) -> list[dict[str, Any]]:
    """Stream-read files for *domain* and collect up to *quota* passing samples.

    Token counts are estimated as ``len(text) // CHARS_PER_TOKEN`` to avoid
    loading a tokenizer.  This is used for per-sample filtering only; the
    global token budget is estimated via file-size ratios (much faster).

    Args:
        domain:         One of ``ALL_DOMAINS``.
        quota:          Target number of samples.
        seed:           Random seed for file-order shuffling.
        min_chars:      Minimum character length after normalisation.
        min_tokens:     Minimum estimated token count (chars // 4).
        max_tokens:     Maximum estimated token count (chars // 4).
        min_int_score:  Minimum ``int_score`` (fineweb_edu / octothinker).
                        Set to 0 to disable.
        min_lang_score: Minimum ``lang_score`` threshold.
        shuffle_files:  Randomise file read order before collecting.

    Returns:
        List of record dicts with keys id (None), doc_id, source, subtype,
        text, n_tokens, url — compatible with BIF pool format.
    """
    extractor = _EXTRACTORS[domain]
    rng = random.Random(seed)

    entries = _domain_file_entries(domain)
    if not entries:
        logger.warning("No files found for domain=%s under %s", domain, _DOMAIN_ROOTS[domain])
        return []

    if shuffle_files:
        rng.shuffle(entries)

    candidates: list[dict[str, Any]] = []
    pbar = tqdm(total=quota, desc=f"  [{domain}]", unit="sample", leave=False)

    for fp, subtype_hint in entries:
        if len(candidates) >= quota:
            break
        try:
            for row in _iter_file(fp):
                if len(candidates) >= quota:
                    break
                if domain == "nemotron_math" and subtype_hint:
                    row["_subtype"] = subtype_hint
                rec = extractor(row)
                if rec is None:
                    continue
                text: str = rec["text"]
                if len(text) < min_chars:
                    continue
                # Per-sample token estimation: chars // 4 (no tokenizer)
                est_tok = len(text) // _CHARS_PER_TOKEN
                if est_tok < min_tokens or est_tok > max_tokens:
                    continue
                # Language filter: skip if lang field is present and non-English
                # Exception: code data (lang="code") always passes
                lang = rec.get("lang")
                if lang and lang not in ("en", "", "code"):
                    continue
                # Language score filter
                ls = rec.get("lang_score")
                if ls is not None:
                    try:
                        if float(ls) < min_lang_score:
                            continue
                    except (TypeError, ValueError):
                        pass
                # Quality score filter for domains that carry int_score
                if min_int_score > 0:
                    is_ = rec.get("int_score", 99)
                    try:
                        if int(is_) < min_int_score:
                            continue
                    except (TypeError, ValueError):
                        pass
                candidates.append(rec)
                pbar.update(1)
        except Exception as exc:
            logger.warning("Skipping file %s: %s", fp, exc)
            continue

    pbar.close()

    if len(candidates) < quota:
        logger.warning(
            "%s: requested %d samples, collected %d after filtering",
            domain, quota, len(candidates),
        )

    rng.shuffle(candidates)
    return candidates[:quota]


# ---------------------------------------------------------------------------
# Quota computation — file-size-based token estimation
# ---------------------------------------------------------------------------

def _parse_token_count(s: str) -> int:
    """Parse human-readable token counts: '4M', '500K', '4000000'."""
    s = s.strip().upper().replace("_", "")
    if s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    return int(s)


def compute_quotas(
    domains: list[str],
    total_tokens: int,
    ratios: dict[str, float] | None = None,
    verbose: bool = True,
    calibrated_stats: dict[str, dict[str, float]] | None = None,
) -> dict[str, int]:
    """Convert a token budget into per-domain sample quotas.

    Token estimation strategy
    -------------------------
    If *calibrated_stats* is provided (from ``calibrate_token_stats()``),
    uses measured avg_tokens_per_sample for quota calculation.
    Otherwise falls back to the hardcoded lookup tables.

    Args:
        domains:           Active domain names.
        total_tokens:      Total token target (e.g. 8_000_000 for 8M).
        ratios:            Fractional share per domain (need not sum to 1).
        verbose:           Print file-size scan progress.
        calibrated_stats:  Per-domain stats from ``calibrate_token_stats()``.

    Returns:
        Mapping from domain name to sample count.
    """
    if ratios:
        ratios = {d: v for d, v in ratios.items() if d in domains}
        missing = [d for d in domains if d not in ratios]
        if missing:
            assigned = sum(ratios.values())
            per_missing = max((1.0 - assigned) / len(missing), 0.0)
            for d in missing:
                ratios[d] = per_missing
    else:
        ratios = {d: 1.0 / len(domains) for d in domains}

    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("ratios sum to zero")
    ratios = {d: v / total_ratio for d, v in ratios.items()}

    quotas: dict[str, int] = {}
    for d in domains:
        token_alloc = int(total_tokens * ratios[d])
        if calibrated_stats and d in calibrated_stats:
            avg_tok = calibrated_stats[d]["avg_tokens_per_sample"]
        else:
            avg_tok = _AVG_TOKENS_PER_SAMPLE.get(d, 400)
        quota = max(1, int(token_alloc / avg_tok))
        quotas[d] = quota
        if verbose:
            logger.info(
                "  %-16s  target=%s tokens  avg_tok=%d  → quota=%d samples",
                d, f"{token_alloc:,}", int(avg_tok), quota,
            )

    return quotas


# Empirical average tokens per sample (used only for quota → sample-count conversion).
# Derived from sampled files: avg_chars / 4  (chars per token ≈ 4).
_AVG_TOKENS_PER_SAMPLE: dict[str, int] = {
    "nemotron_math": 850,
    "octothinker":   470,
    "fineweb_edu":    880,
    "c4":             475,
    "wikipedia":      710,
    "slimpajama":     500,
    "starcoder":      600,
    "flan":           400,
    "sft_chat":       500,
}


def _parse_ratios(s: str) -> dict[str, float]:
    """Parse 'domain1:0.4,domain2:0.3' into a float dict."""
    result: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if ":" not in part:
            raise ValueError(f"Invalid ratio format: {part!r} (expected 'domain:value')")
        k, v = part.split(":", 1)
        result[k.strip()] = float(v.strip())
    return result


# ---------------------------------------------------------------------------
# Pool builder (public API)
# ---------------------------------------------------------------------------

def _summarize(rows: list[dict[str, Any]], tokenizer=None) -> dict[str, Any]:
    if tokenizer:
        lengths = [len(tokenizer.encode(r["text"], add_special_tokens=False)) for r in rows]
    else:
        lengths = [len(r["text"]) // _CHARS_PER_TOKEN for r in rows]
    return {
        "count": len(rows),
        "estimated_total_tokens": sum(lengths),
        "estimated_mean_tokens": round(sum(lengths) / max(1, len(lengths))),
        "estimated_min_tokens": min(lengths) if lengths else 0,
        "estimated_max_tokens": max(lengths) if lengths else 0,
    }


def build_domain_pool(
    quotas: dict[str, int],
    out_dir: str,
    seed: int = 42,
    min_chars: int = 200,
    min_tokens: int = 50,
    max_tokens: int = 4096,
    min_int_score: int = 0,
    min_lang_score: float = 0.8,
    output_filename: str | None = None,
    tokenizer_path: str | None = None,
    calibrated_stats: dict[str, dict[str, float]] | None = None,
    data_sources: list[dict[str, Any]] | None = None,
) -> str:
    """Build and persist a multi-domain data pool.

    Args:
        quotas:          Mapping from domain name to sample count.
        out_dir:         Output directory (created if absent).
        seed:            Global random seed.
        min_chars:       Minimum character length after normalisation.
        min_tokens:      Minimum estimated token count.
        max_tokens:      Maximum estimated token count.
        min_int_score:   Minimum int_score filter (0 = disabled).
        min_lang_score:  Minimum language_score threshold.
        output_filename: Override the auto-generated output filename.

    Returns:
        Absolute path to the written JSONL file.
    """
    ensure_dir(out_dir)
    all_rows: list[dict[str, Any]] = []
    stats_by_domain: dict[str, Any] = {}
    tokenizer = None
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    for i, (domain, quota) in enumerate(quotas.items()):
        logger.info("Collecting domain=%s  quota=%d", domain, quota)
        rows = collect_domain(
            domain=domain,
            quota=quota,
            seed=seed + i * 7,
            min_chars=min_chars,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            min_int_score=min_int_score,
            min_lang_score=min_lang_score,
        )
        for r in rows:
            r["source"] = domain
        stats_by_domain[domain] = _summarize(rows, tokenizer=tokenizer)
        logger.info(
            "  %s: collected=%d  est_tokens=%d",
            domain, len(rows), stats_by_domain[domain]["estimated_total_tokens"],
        )
        all_rows.extend(rows)

    if data_sources:
        for i, ds in enumerate(data_sources):
            logger.info("Collecting custom source: %s from %s", ds.get("source_name", "custom"), ds["path"])
            rows = collect_custom_source(
                src_cfg=ds,
                seed=seed + (len(quotas) + i) * 7,
                min_chars=min_chars,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                min_int_score=min_int_score,
                min_lang_score=min_lang_score,
            )
            stats_by_domain[ds.get("source_name", "custom")] = _summarize(rows, tokenizer=tokenizer)
            logger.info(
                "  %s: collected=%d  est_tokens=%d",
                ds.get("source_name", "custom"), len(rows),
                stats_by_domain[ds.get("source_name", "custom")]["estimated_total_tokens"],
            )
            all_rows.extend(rows)

    rng = random.Random(seed)
    rng.shuffle(all_rows)
    for idx, row in enumerate(all_rows):
        row["id"] = f"pool_{idx:07d}"
        # Store estimated token count in n_tokens field (BIF-compatible)
        row.setdefault("n_tokens", len(row["text"]) // _CHARS_PER_TOKEN)

    total_stats = _summarize(all_rows, tokenizer=tokenizer)
    n_tok = total_stats["estimated_total_tokens"]

    if output_filename is None:
        tok_label = (
            f"{n_tok // 1_000_000}M"
            if n_tok >= 1_000_000
            else f"{n_tok // 1_000}K"
        )
        output_filename = (
            f"pool_{len(all_rows)}samples_{tok_label}tok"
            f"_{len(quotas)}domains.jsonl"
        )

    out_path = os.path.join(out_dir, output_filename)
    write_jsonl(out_path, all_rows)

    save_json(
        os.path.join(out_dir, output_filename.replace(".jsonl", "_stats.json")),
        {
            "output": out_path,
            "total": total_stats,
            "by_domain": stats_by_domain,
            "quotas": quotas,
        },
    )
    logger.info("Pool written to: %s  (%d samples, ~%s tokens)", out_path, len(all_rows),
                f"{n_tok/1e6:.1f}M" if n_tok >= 1_000_000 else f"{n_tok//1000}K")

    if calibrated_stats and total_stats:
        actual_total = total_stats.get("estimated_total_tokens", 0)
        try:
            from bif.utils.tracker import _is_initialised, log_bar, log_table
            if _is_initialised():
                actual_per_domain = {}
                est_per_domain = {}
                for domain, st in stats_by_domain.items():
                    actual_per_domain[domain] = st.get("estimated_total_tokens", 0)
                    est_per_domain[domain] = int(quotas.get(domain, 0)) * int(calibrated_stats.get(domain, {}).get("avg_tokens_per_sample", 400))
                domains = sorted(set(list(actual_per_domain.keys()) + list(est_per_domain.keys())))
                log_table(
                    "1_data/token_estimate_vs_actual_summary",
                    headers=["metric", "value"],
                    rows=[
                        ["total_actual_tokens", str(actual_total)],
                        ["total_estimated_tokens", str(sum(est_per_domain.values()))],
                        ["token_accuracy", f"{actual_total / max(1, sum(est_per_domain.values())):.4f}"],
                    ],
                )
                log_bar(
                    "1_data/token_estimate_vs_actual",
                    xaxis=domains,
                    series={
                        "estimated": [est_per_domain.get(d, 0) for d in domains],
                        "actual": [actual_per_domain.get(d, 0) for d in domains],
                    },
                )
        except Exception:
            pass

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_custom_source(
    src_cfg: dict[str, Any],
    seed: int = 42,
    min_chars: int = 200,
    min_tokens: int = 50,
    max_tokens: int = 4096,
    min_int_score: int = 0,
    min_lang_score: float = 0.8,
) -> list[dict[str, Any]]:
    """Collect samples from a custom data source path.

    Args:
        src_cfg: Dict with keys: path (required), format, text_col,
                 source_name, n (quota).
    """
    path = src_cfg["path"]
    fmt = src_cfg.get("format", None)
    text_col = src_cfg.get("text_col", "text")
    source_name = src_cfg.get("source_name", Path(path).stem)
    quota = src_cfg.get("n", None)

    if fmt is None:
        if path.endswith(".parquet") or Path(path).is_dir():
            fmt = "parquet"
        else:
            fmt = "jsonl"

    rows: list[dict[str, Any]] = []
    if fmt == "parquet":
        if Path(path).is_dir():
            fps = sorted(Path(path).rglob("*.parquet"))
        else:
            fps = [Path(path)]
        rng = random.Random(seed)
        fps_list = list(fps)
        rng.shuffle(fps_list)
        for fp in fps_list:
            if quota and len(rows) >= quota:
                break
            try:
                df = pd.read_parquet(fp)
                for _, pd_row in df.iterrows():
                    if quota and len(rows) >= quota:
                        break
                    text = str(pd_row.get(text_col, "")).strip()
                    if not text or len(text) < min_chars:
                        continue
                    est_tok = len(text) // _CHARS_PER_TOKEN
                    if est_tok < min_tokens or est_tok > max_tokens:
                        continue
                    int_score = 0
                    try:
                        int_score = int(pd_row.get("int_score", 0))
                    except (TypeError, ValueError):
                        pass
                    if min_int_score > 0 and int_score < min_int_score:
                        continue
                    lang_score = 1.0
                    try:
                        lang_score = float(pd_row.get("language_score", pd_row.get("lang_score", 1.0)))
                    except (TypeError, ValueError):
                        pass
                    if lang_score < min_lang_score:
                        continue
                    rows.append({
                        "text": text,
                        "doc_id": str(pd_row.get("url", pd_row.get("doc_id", ""))),
                        "url": str(pd_row.get("url", "")),
                        "subtype": str(pd_row.get("crawl", pd_row.get("subtype", ""))),
                        "lang": str(pd_row.get("language", pd_row.get("lang", "en"))),
                        "lang_score": lang_score,
                        "int_score": int_score,
                        "score": float(pd_row.get("score", 0)),
                        "source": source_name,
                        "n_tokens": est_tok,
                    })
            except Exception as exc:
                logger.warning("Skipping %s: %s", fp, exc)
    else:
        import json as _json
        if Path(path).is_dir():
            fps = sorted(Path(path).rglob("*.jsonl")) + sorted(Path(path).rglob("*.json"))
        else:
            fps = [Path(path)]
        for fp in fps:
            if quota and len(rows) >= quota:
                break
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        if quota and len(rows) >= quota:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        obj = _json.loads(line)
                        text = str(obj.get(text_col, "")).strip()
                        if not text or len(text) < min_chars:
                            continue
                        est_tok = len(text) // _CHARS_PER_TOKEN
                        if est_tok < min_tokens or est_tok > max_tokens:
                            continue
                        int_score = 0
                        try:
                            int_score = int(obj.get("int_score", 0))
                        except (TypeError, ValueError):
                            pass
                        if min_int_score > 0 and int_score < min_int_score:
                            continue
                        rows.append({
                            "text": text,
                            "doc_id": str(obj.get("doc_id", obj.get("id", ""))),
                            "url": str(obj.get("url", "")),
                            "subtype": str(obj.get("subtype", "")),
                            "lang": str(obj.get("lang", obj.get("language", "en"))),
                            "lang_score": float(obj.get("lang_score", obj.get("language_score", 1.0))),
                            "int_score": int_score,
                            "score": float(obj.get("score", 0)),
                            "source": source_name,
                            "n_tokens": est_tok,
                        })
            except Exception as exc:
                logger.warning("Skipping %s: %s", fp, exc)

    rng = random.Random(seed + 1)
    rng.shuffle(rows)
    if quota:
        rows = rows[:quota]
    logger.info("Custom source '%s' from %s: collected %d samples", source_name, path, len(rows))
    return rows


def _print_plan(
    quotas: dict[str, int],
    total_tokens: int | None = None,
    calibrated_stats: dict[str, dict[str, float]] | None = None,
) -> None:
    header = f"  {'Domain':<22} {'Quota':>8}  {'Est. tokens':>14}  {'Avg tok/sample':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    total_tok = 0
    for domain, quota in quotas.items():
        if calibrated_stats and domain in calibrated_stats:
            avg = int(calibrated_stats[domain]["avg_tokens_per_sample"])
        else:
            avg = _AVG_TOKENS_PER_SAMPLE.get(domain, 400)
        est = quota * avg
        total_tok += est
        print(f"  {domain:<22} {quota:>8,}  {est:>14,}  {avg:>14,}")
    print("  " + "-" * (len(header) - 2))
    tok_label = f"{total_tok / 1e6:.2f}M" if total_tok >= 1_000_000 else f"{total_tok // 1_000}K"
    budget_str = ""
    if total_tokens is not None:
        budget_str = f"  (budget: {total_tokens/1e6:.1f}M)"
    print(f"  {'TOTAL':<22} {sum(quotas.values()):>8,}  {tok_label:>14}{budget_str}\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="bif build-pool-v2",
        description="Build a token-budget-aware multi-domain data pool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- custom data sources ---
    parser.add_argument(
        "--data_sources",
        default=None,
        help="Custom data source paths (JSON string or repeated flag). "
             "Each source: {\"path\": \"/data/dir\", \"format\": \"parquet|jsonl\", "
             "\"text_col\": \"text\", \"source_name\": \"mydata\", \"n\": 1000}. "
             "Example: --data_sources '[{\"path\":\"/data/math\",\"format\":\"parquet\",\"source_name\":\"math\"}]'",
    )

    # --- token-budget mode ---
    parser.add_argument(
        "--total_tokens",
        default=None,
        help="Token budget: '4M', '8M', '500000', etc. "
             "Token counts are estimated from raw file sizes (no tokenizer).",
    )
    parser.add_argument(
        "--ratios",
        default=None,
        help="Per-domain sampling ratios: 'domain1:0.4,domain2:0.3'. "
             "Unspecified domains share the remainder equally. "
             "Default: uniform split.",
    )
    parser.add_argument(
        "--pool_type",
        default=None,
        choices=list(POOL_PRESETS.keys()),
        help="Domain preset: 'pt' (general pretraining), 'target' (rare/task-specific), "
             "'sft' (supervised). Overrides --domains unless --domains is also set.",
    )
    parser.add_argument(
        "--domains",
        default=None,
        help="Comma-separated list of active domains. "
             f"Default: all ({', '.join(ALL_DOMAINS)}). "
             f"PT-only: {', '.join(PT_DOMAINS)}  "
             f"Target: {', '.join(TARGET_DOMAINS)}  "
             f"SFT-only: {', '.join(SFT_DOMAINS)}",
    )

    # --- explicit count mode (one flag per domain) ---
    for d in ALL_DOMAINS:
        parser.add_argument(
            f"--n_{d}",
            type=int,
            default=None,
            help=f"Sample count for {d} (explicit-count mode).",
        )

    # --- quality filters ---
    parser.add_argument("--min_chars",      type=int,   default=200,
                        help="Minimum character count after normalization (default: 200).")
    parser.add_argument("--min_tokens",     type=int,   default=50,
                        help="Minimum estimated token count (chars//4, default: 50).")
    parser.add_argument("--max_tokens",     type=int,   default=4096,
                        help="Maximum estimated token count (chars//4, default: 4096).")
    parser.add_argument("--min_int_score",  type=int,   default=0,
                        help="Minimum int_score for fineweb_edu/octothinker (0 = disabled).")
    parser.add_argument("--min_lang_score", type=float, default=0.8,
                        help="Minimum language_score threshold (default: 0.8).")

    # --- output ---
    parser.add_argument("--out_dir",  required=True,
                        help="Output directory (created if absent).")
    parser.add_argument("--out_name", default=None,
                        help="Override auto-generated output filename.")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--dry_run",  action="store_true",
                        help="Print sampling plan and exit without collecting data.")
    parser.add_argument("--tokenizer_path", default=None,
                        help="Tokenizer for accurate token counting during calibration. "
                             "If set, samples a small batch first to measure actual "
                             "chars/token and avg_tokens_per_sample. "
                             "Without this, uses len(text)//4 estimation.")
    parser.add_argument("--calibrate_n", type=int, default=100,
                        help="Number of samples per domain for calibration (default: 100).")

    args = parser.parse_args()

    # Resolve active domains
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        unknown = [d for d in domains if d not in ALL_DOMAINS]
        if unknown:
            parser.error(f"Unknown domain(s): {unknown}. Supported: {ALL_DOMAINS}")
    elif args.pool_type:
        domains = POOL_PRESETS[args.pool_type]
        print(f"[preset] pool_type={args.pool_type} → domains={domains}")
    else:
        domains = list(ALL_DOMAINS)

    # Detect explicit-count mode
    direct: dict[str, int] = {
        d: getattr(args, f"n_{d}")
        for d in domains
        if getattr(args, f"n_{d}") is not None
    }

    if direct and args.total_tokens:
        parser.error("--total_tokens and --n_<domain> cannot be used together.")

    # Parse custom data sources
    data_sources: list[dict[str, Any]] = []
    if args.data_sources:
        import json as _json
        ds_raw = args.data_sources
        try:
            data_sources = _json.loads(ds_raw)
        except _json.JSONDecodeError:
            parser.error(f"--data_sources: invalid JSON: {ds_raw[:100]}")
        if not isinstance(data_sources, list):
            data_sources = [data_sources]
        for ds in data_sources:
            if "path" not in ds:
                parser.error("--data_sources: each source must have 'path'")

    total_tokens_int: int | None = None
    calibrated_stats: dict[str, dict[str, float]] | None = None

    if args.tokenizer_path:
        print(f"[calibrate] Sampling {args.calibrate_n} examples per domain with tokenizer …")
        calibrated_stats = calibrate_token_stats(
            domains=domains,
            quota=args.calibrate_n,
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
        total_tokens_int = _parse_token_count(args.total_tokens)
        ratios = _parse_ratios(args.ratios) if args.ratios else None
        print(f"[mode] token budget  target={total_tokens_int:,} (~{total_tokens_int / 1e6:.1f}M tokens)")
        if args.ratios:
            print(f"[ratios] {args.ratios}")
        else:
            print(f"[ratios] uniform ({1/len(domains)*100:.1f}% each)")
        print()
        quotas = compute_quotas(
            domains, total_tokens_int, ratios, verbose=True,
            calibrated_stats=calibrated_stats,
        )
    else:
        parser.error("Specify --total_tokens or at least one --n_<domain>.")

    print()
    _print_plan(quotas, total_tokens=total_tokens_int, calibrated_stats=calibrated_stats)

    if args.dry_run:
        print("[dry_run] Exiting without collecting data.")
        return

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
        data_sources=data_sources if data_sources else None,
    )
    print(f"\nPool written to: {out_path}")


if __name__ == "__main__":
    main()
