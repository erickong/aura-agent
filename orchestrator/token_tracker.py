"""Token usage tracking for cost optimization and cache hit monitoring.

Supports both DeepSeek (prompt_cache_hit_tokens / prompt_cache_miss_tokens)
and Anthropic (cache_read_input_tokens / cache_creation_input_tokens) usage formats.

Usage:
    from .token_tracker import log_usage, get_stats, reset_stats

    response = client.messages.create(...)
    log_usage("run_cycle", response)

    stats = get_stats()
    print(stats["estimated_cost_saved"])
"""

import json
import os
import time
from datetime import datetime
from typing import Any

# ── Pricing (USD per 1M tokens) ─────────────────────────────────────
# Load from config/env, fall back to DeepSeek V4 Pro defaults.


def _get_prices():
    """Resolve token prices from config (lazy to avoid circular imports)."""
    try:
        from .config import TOKEN_PRICE_CACHE_HIT, TOKEN_PRICE_CACHE_MISS, TOKEN_PRICE_OUTPUT
        return TOKEN_PRICE_CACHE_HIT, TOKEN_PRICE_CACHE_MISS, TOKEN_PRICE_OUTPUT
    except Exception:
        return 0.145, 1.74, 1.74


# Anthropic Claude pricing (fallback estimates — not configurable yet)
_CLAUDE_CACHE_WRITE_PRICE = 3.75  # per 1M tokens
_CLAUDE_CACHE_READ_PRICE = 0.30   # per 1M tokens
_CLAUDE_INPUT_PRICE = 15.0        # per 1M tokens
_CLAUDE_OUTPUT_PRICE = 75.0       # per 1M tokens

# ── Storage ──────────────────────────────────────────────────────────

# Lazy path resolution — uses config.DATA_DIR after imports are ready.
# _resolve_stats_path() is the canonical accessor; _stats_path is only
# used as a fallback if config import fails (defensive).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_stats_path = ""

# In-memory accumulator — populated from disk at first call, then kept in sync
_usage_records: list[dict] = []
_loaded_from_disk: bool = False
_skip_count: int = 0
_skip_loaded: bool = False


def _resolve_stats_path() -> str:
    """Resolve token_stats.jsonl path from config after imports are ready."""
    try:
        from .config import DATA_DIR
        return os.path.join(DATA_DIR, "token_stats.jsonl")
    except Exception:
        return _stats_path


def _resolve_skip_path() -> str:
    """Resolve skip_stats.json path."""
    try:
        from .config import DATA_DIR
        return os.path.join(DATA_DIR, "skip_stats.json")
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", ".aura", "skip_stats.json",
        )


def _detect_provider() -> str:
    """Return the API provider. Uses stored config value (set at setup time).
    Falls back to URL inspection for backward compatibility."""
    try:
        from .config import AURA_API_PROVIDER
        if AURA_API_PROVIDER:
            return AURA_API_PROVIDER
    except Exception:
        pass
    # Fallback: inspect URL
    try:
        from .config import AURA_API_BASE_URL
    except Exception:
        return "unknown"
    if "deepseek" in AURA_API_BASE_URL.lower():
        return "deepseek"
    if "anthropic" in AURA_API_BASE_URL.lower():
        return "anthropic"
    return "unknown"


def _ensure_loaded() -> None:
    """Load records from disk if not yet loaded this process."""
    global _usage_records, _loaded_from_disk, _skip_count, _skip_loaded
    if _loaded_from_disk:
        return

    # Load token records
    try:
        path = _resolve_stats_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            _usage_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    except OSError:
        pass
    _loaded_from_disk = True

    # Load skip count
    try:
        skip_path = _resolve_skip_path()
        if os.path.exists(skip_path):
            with open(skip_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                _skip_count = data.get("skip_count", 0)
    except (OSError, json.JSONDecodeError):
        pass
    _skip_loaded = True


def extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from an API response.

    Handles both DeepSeek and Anthropic usage formats.
    DeepSeek API returns: prompt_cache_hit_tokens, prompt_cache_miss_tokens
    Anthropic API returns: cache_read_input_tokens, cache_creation_input_tokens, input_tokens
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    result: dict[str, int] = {}

    # Standard fields present in both APIs
    for field in ("input_tokens", "output_tokens"):
        val = getattr(usage, field, None)
        if val is not None:
            result[field] = int(val)

    # DeepSeek cache fields
    for field in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        val = getattr(usage, field, None)
        if val is not None:
            result[field] = int(val)

    # Anthropic cache fields
    for field in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        val = getattr(usage, field, None)
        if val is not None:
            result[field] = int(val)

    return result


def log_usage(source: str, response: Any, extra: dict | None = None) -> dict:
    """Log token usage from an API response. Returns the extracted usage dict.

    Args:
        source: Label like 'run_cycle', 'review_cycle', 'compress_memory', 'extract_skill'
        response: The API response object
        extra: Optional extra info to store (cycle number, task_id, etc.)
    """
    _ensure_loaded()

    usage = extract_usage(response)
    if not usage:
        return {}

    record: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "provider": _detect_provider(),
        "usage": usage,
    }
    if extra:
        record["extra"] = extra

    _usage_records.append(record)

    # Append to disk for persistence
    try:
        path = _resolve_stats_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # Non-critical: stats survive in memory

    return usage


def get_stats() -> dict:
    """Get aggregate token usage statistics.

    Returns a dict with:
        total_calls, total_input_tokens, total_output_tokens,
        total_cache_hit_tokens, total_cache_miss_tokens,
        cache_hit_rate, estimated_cost, estimated_cost_saved,
        by_source, recent_calls, provider
    """
    _ensure_loaded()
    records = list(_usage_records)

    if not records:
        return {
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_hit_tokens": 0,
            "total_cache_miss_tokens": 0,
            "cache_hit_rate": 0.0,
            "estimated_cost": 0.0,
            "estimated_cost_saved": 0.0,
            "by_source": {},
            "recent_calls": [],
            "provider": _detect_provider(),
        }

    provider = records[0].get("provider", _detect_provider())
    total_input = 0
    total_output = 0
    total_cache_hit = 0
    total_cache_miss = 0
    total_cache_read = 0
    total_cache_creation = 0
    by_source: dict[str, dict] = {}

    for rec in records:
        source = rec.get("source", "unknown")
        usage = rec.get("usage", {})

        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)

        total_input += inp
        total_output += out

        if provider == "deepseek":
            hit = usage.get("prompt_cache_hit_tokens", 0)
            miss = usage.get("prompt_cache_miss_tokens", 0)
            total_cache_hit += hit
            total_cache_miss += miss
        else:
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_creation = usage.get("cache_creation_input_tokens", 0)
            total_cache_read += cache_read
            total_cache_creation += cache_creation

        if source not in by_source:
            by_source[source] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_hit_tokens": 0,
                "cache_miss_tokens": 0,
            }
        bs = by_source[source]
        bs["calls"] += 1
        bs["input_tokens"] += inp
        bs["output_tokens"] += out
        if provider == "deepseek":
            bs["cache_hit_tokens"] += usage.get("prompt_cache_hit_tokens", 0)
            bs["cache_miss_tokens"] += usage.get("prompt_cache_miss_tokens", 0)
        else:
            bs["cache_hit_tokens"] += usage.get("cache_read_input_tokens", 0)
            bs["cache_miss_tokens"] += usage.get("cache_creation_input_tokens", 0)

    if provider == "deepseek":
        cache_hit_price, cache_miss_price, output_price = _get_prices()
        estimated_cost = (
            (total_cache_hit / 1_000_000) * cache_hit_price
            + (total_cache_miss / 1_000_000) * cache_miss_price
            + (total_output / 1_000_000) * output_price
        )
        full_input_cost = (total_input / 1_000_000) * cache_miss_price
        estimated_saved = full_input_cost - (
            (total_cache_hit / 1_000_000) * cache_hit_price
            + (total_cache_miss / 1_000_000) * cache_miss_price
        )
        total_cache = total_cache_hit
    else:
        estimated_cost = (
            (total_cache_read / 1_000_000) * _CLAUDE_CACHE_READ_PRICE
            + (total_cache_creation / 1_000_000) * _CLAUDE_CACHE_WRITE_PRICE
            + (max(0, total_input - total_cache_read - total_cache_creation) / 1_000_000) * _CLAUDE_INPUT_PRICE
            + (total_output / 1_000_000) * _CLAUDE_OUTPUT_PRICE
        )
        full_input_cost = (total_input / 1_000_000) * _CLAUDE_INPUT_PRICE
        estimated_saved = full_input_cost - (
            (total_cache_read / 1_000_000) * _CLAUDE_CACHE_READ_PRICE
            + (total_cache_creation / 1_000_000) * _CLAUDE_CACHE_WRITE_PRICE
            + (max(0, total_input - total_cache_read - total_cache_creation) / 1_000_000) * _CLAUDE_INPUT_PRICE
        )
        total_cache = total_cache_read

    cache_hit_rate = (total_cache / max(total_input, 1)) if total_input > 0 else 0.0

    return {
        "total_calls": len(records),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_hit_tokens": total_cache_hit,
        "total_cache_miss_tokens": total_cache_miss,
        "total_cache_read_tokens": total_cache_read,
        "total_cache_creation_tokens": total_cache_creation,
        "cache_hit_rate": round(cache_hit_rate, 4),
        "estimated_cost": round(estimated_cost, 6),
        "estimated_cost_saved": round(max(0, estimated_saved), 6),
        "by_source": by_source,
        "provider": provider,
        "recent_calls": [
            {
                "timestamp": r.get("timestamp", "")[:19],
                "source": r.get("source", "?"),
                "hit": r.get("usage", {}).get("prompt_cache_hit_tokens", 0)
                    or r.get("usage", {}).get("cache_read_input_tokens", 0),
                "miss": r.get("usage", {}).get("prompt_cache_miss_tokens", 0),
                "input": r.get("usage", {}).get("input_tokens", 0),
                "output": r.get("usage", {}).get("output_tokens", 0),
            }
            for r in records[-10:]
        ],
    }


def reset_stats() -> None:
    """Reset in-memory and on-disk token stats."""
    global _usage_records, _loaded_from_disk, _skip_count, _skip_loaded
    _usage_records = []
    _loaded_from_disk = True  # Don't reload after reset
    _skip_count = 0
    _skip_loaded = True
    try:
        path = _resolve_stats_path()
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
    try:
        skip_path = _resolve_skip_path()
        if os.path.exists(skip_path):
            os.remove(skip_path)
    except OSError:
        pass


def format_stats(stats: dict) -> str:
    """Format token stats as a human-readable string."""
    if stats["total_calls"] == 0:
        return "No token usage recorded yet."

    provider = stats.get("provider", "unknown")
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║                    Token Usage Statistics                   ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  Provider: {provider:<20}                                  ║",
        f"║  Total API calls:        {stats['total_calls']:>6}                                ║",
        f"║  Total input tokens:     {stats['total_input_tokens']:>10,}                        ║",
        f"║  Total output tokens:    {stats['total_output_tokens']:>10,}                        ║",
    ]

    if stats["total_cache_hit_tokens"] > 0:
        lines.append(
            f"║  Cache hit tokens:       {stats['total_cache_hit_tokens']:>10,}                        ║"
        )
    if stats["total_cache_miss_tokens"] > 0:
        lines.append(
            f"║  Cache miss tokens:      {stats['total_cache_miss_tokens']:>10,}                        ║"
        )
    if stats["total_cache_read_tokens"] > 0:
        lines.append(
            f"║  Cache read tokens:      {stats['total_cache_read_tokens']:>10,}                        ║"
        )
    if stats["total_cache_creation_tokens"] > 0:
        lines.append(
            f"║  Cache creation tokens:  {stats['total_cache_creation_tokens']:>10,}                        ║"
        )

    lines.extend([
        f"║  Cache hit rate:         {stats['cache_hit_rate']:>8.1%}                          ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  Estimated cost:         ${stats['estimated_cost']:>10.4f}                       ║",
        f"║  Estimated cost saved:   ${stats['estimated_cost_saved']:>10.4f}                       ║",
        "╠══════════════════════════════════════════════════════════════╣",
    ])

    by_source = stats.get("by_source", {})
    for source, bs in sorted(by_source.items()):
        hit_rate = bs["cache_hit_tokens"] / max(bs["input_tokens"], 1) if bs["input_tokens"] > 0 else 0.0
        lines.append(
            f"║  {source:<18} {bs['calls']:>3} calls, "
            f"in={bs['input_tokens']:,}, hit={hit_rate:.0%}     ║"
        )

    lines.append("╚══════════════════════════════════════════════════════════════╝")
    return "\n".join(lines)


def format_cycle_stats(usage: dict) -> str:
    """Format a single cycle's token usage for inline display."""
    if not usage:
        return ""

    provider = _detect_provider()
    hit = usage.get("prompt_cache_hit_tokens", 0) or usage.get("cache_read_input_tokens", 0)
    miss = usage.get("prompt_cache_miss_tokens", 0)
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)

    if provider == "deepseek" and (hit or miss):
        cache_hit_price, cache_miss_price, output_price = _get_prices()
        hit_rate = hit / max(hit + miss, 1)
        cost = (
            (hit / 1_000_000) * cache_hit_price
            + (miss / 1_000_000) * cache_miss_price
            + (out / 1_000_000) * output_price
        )
        return (
            f"tokens: in={inp:,} out={out:,} | "
            f"cache: hit={hit:,} miss={miss:,} ({hit_rate:.0%}) | "
            f"~${cost:.4f}"
        )
    elif provider == "anthropic" and hit:
        cost = (
            (hit / 1_000_000) * _CLAUDE_CACHE_READ_PRICE
            + (usage.get("cache_creation_input_tokens", 0) / 1_000_000) * _CLAUDE_CACHE_WRITE_PRICE
            + (out / 1_000_000) * _CLAUDE_OUTPUT_PRICE
        )
        return (
            f"tokens: in={inp:,} out={out:,} | "
            f"cache: hit={hit:,} | "
            f"~${cost:.4f}"
        )
    else:
        cache_hit_price, cache_miss_price, output_price = _get_prices()
        cost = (inp / 1_000_000) * cache_miss_price + (out / 1_000_000) * output_price
        return f"tokens: in={inp:,} out={out:,} | ~${cost:.4f}"


# ── Session-level tracking (in-memory only, resets each process) ───

_session_records: list[dict] = []


def accumulate_session(cycle_num: int, usage: dict, source: str = "normal",
                       skipped: bool = False) -> None:
    """Record a cycle's token usage for the current process session.

    Args:
        cycle_num: The cycle number.
        usage: Token usage dict (from extract_usage or log_usage).
        source: "normal", "deep_reflection", or "skipped".
        skipped: Whether this cycle was skipped (no API call).
    """
    _session_records.append({
        "cycle": cycle_num,
        "source": source,
        "usage": dict(usage),
        "skipped": skipped,
        "timestamp": datetime.now().isoformat(),
    })


def get_session_stats() -> dict:
    """Get token stats for the current process session.

    Returns a dict with cumulative totals and per-type breakdowns.
    """
    if not _session_records:
        return {
            "total_cycles": 0,
            "normal_cycles": 0,
            "deep_reflection_cycles": 0,
            "skipped_cycles": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_hit_tokens": 0,
            "total_cache_miss_tokens": 0,
            "total_cost": 0.0,
            "avg_cost_per_normal": 0.0,
            "avg_cost_per_reflection": 0.0,
        }

    provider = _detect_provider()
    cache_hit_price, cache_miss_price, output_price = _get_prices()

    normal_cycles = 0
    deep_reflection_cycles = 0
    skipped_cycles = 0
    total_input = 0
    total_output = 0
    total_cache_hit = 0
    total_cache_miss = 0
    total_cost = 0.0
    normal_cost = 0.0
    reflection_cost = 0.0

    for rec in _session_records:
        usage = rec.get("usage", {})
        source = rec.get("source", "normal")

        if rec.get("skipped"):
            skipped_cycles += 1
            continue

        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        total_input += inp
        total_output += out

        if provider == "deepseek":
            hit = usage.get("prompt_cache_hit_tokens", 0)
            miss = usage.get("prompt_cache_miss_tokens", 0)
            total_cache_hit += hit
            total_cache_miss += miss
            cost = (
                (hit / 1_000_000) * cache_hit_price
                + (miss / 1_000_000) * cache_miss_price
                + (out / 1_000_000) * output_price
            )
        else:
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_creation = usage.get("cache_creation_input_tokens", 0)
            total_cache_hit += cache_read
            total_cache_miss += cache_creation
            cost = (
                (cache_read / 1_000_000) * _CLAUDE_CACHE_READ_PRICE
                + (cache_creation / 1_000_000) * _CLAUDE_CACHE_WRITE_PRICE
                + (max(0, inp - cache_read - cache_creation) / 1_000_000) * _CLAUDE_INPUT_PRICE
                + (out / 1_000_000) * _CLAUDE_OUTPUT_PRICE
            )

        total_cost += cost

        if source == "normal":
            normal_cycles += 1
            normal_cost += cost
        elif source == "deep_reflection":
            deep_reflection_cycles += 1
            reflection_cost += cost

    total_token_cycles = normal_cycles + deep_reflection_cycles
    avg_cost_per_normal = normal_cost / normal_cycles if normal_cycles > 0 else 0.0
    avg_cost_per_reflection = reflection_cost / deep_reflection_cycles if deep_reflection_cycles > 0 else 0.0

    return {
        "total_cycles": len(_session_records),
        "normal_cycles": normal_cycles,
        "deep_reflection_cycles": deep_reflection_cycles,
        "skipped_cycles": skipped_cycles,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_hit_tokens": total_cache_hit,
        "total_cache_miss_tokens": total_cache_miss,
        "total_cost": total_cost,
        "avg_cost_per_normal": avg_cost_per_normal,
        "avg_cost_per_reflection": avg_cost_per_reflection,
        "total_token_cycles": total_token_cycles,
        "provider": provider,
    }


def format_session_display(current_usage: dict | None = None,
                           source: str = "normal") -> str:
    """Format a boxed session token/cost display for cycle-end output.

    Args:
        current_usage: Usage dict for the current cycle (may be empty for skips).
        source: "normal" or "deep_reflection".

    Returns a multi-line string with box-drawing characters.
    """
    stats = get_session_stats()
    provider = stats.get("provider", "unknown")
    cache_hit_price, cache_miss_price, output_price = _get_prices()

    # Compute current-cycle cost
    current_cost = 0.0
    if current_usage:
        hit = current_usage.get("prompt_cache_hit_tokens", 0) or current_usage.get("cache_read_input_tokens", 0)
        miss = current_usage.get("prompt_cache_miss_tokens", 0)
        out_cur = current_usage.get("output_tokens", 0)
        inp_cur = current_usage.get("input_tokens", 0)
        if provider == "deepseek" and (hit or miss):
            current_cost = (
                (hit / 1_000_000) * cache_hit_price
                + (miss / 1_000_000) * cache_miss_price
                + (out_cur / 1_000_000) * output_price
            )
        elif provider == "anthropic":
            cache_read = current_usage.get("cache_read_input_tokens", 0)
            cache_creation = current_usage.get("cache_creation_input_tokens", 0)
            current_cost = (
                (cache_read / 1_000_000) * _CLAUDE_CACHE_READ_PRICE
                + (cache_creation / 1_000_000) * _CLAUDE_CACHE_WRITE_PRICE
                + (max(0, inp_cur - cache_read - cache_creation) / 1_000_000) * _CLAUDE_INPUT_PRICE
                + (out_cur / 1_000_000) * _CLAUDE_OUTPUT_PRICE
            )
        else:
            current_cost = (inp_cur / 1_000_000) * cache_miss_price + (out_cur / 1_000_000) * output_price

    s = stats  # shorthand
    label = "DEEP REFLECTION" if source == "deep_reflection" else "CYCLE"

    lines = [
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        f"  │  Token Usage — This {label:<30}                  │",
        "  ├─────────────────────────────────────────────────────────────┤",
    ]

    if current_usage:
        hit = current_usage.get("prompt_cache_hit_tokens", 0) or current_usage.get("cache_read_input_tokens", 0)
        miss_cur = current_usage.get("prompt_cache_miss_tokens", 0)
        inp = current_usage.get("input_tokens", 0)
        out = current_usage.get("output_tokens", 0)
        cache_str = ""
        if hit or miss_cur:
            hit_rate = hit / max(hit + miss_cur, 1)
            cache_str = f"  cache: hit={hit:,} miss={miss_cur:,} ({hit_rate:.0%})"
        lines.append(f"  │  Current: in={inp:,}  out={out:,}  ${current_cost:.4f}        │")
        if cache_str:
            # Truncate/pad to fit the box width
            lines.append(f"  │  {cache_str:<59}│")

    lines.extend([
        "  ├─────────────────────────────────────────────────────────────┤",
        f"  │  Session (since process start)                             │",
        f"  │  Cycles: {s['total_cycles']} total ({s['normal_cycles']} normal + {s['deep_reflection_cycles']} reflections + {s['skipped_cycles']} skipped)                  │",
        f"  │  Tokens: {s['total_input_tokens']:,} in + {s['total_output_tokens']:,} out                              │",
        f"  │  Cost:   ${s['total_cost']:.4f}                                            │",
    ])

    if s["normal_cycles"] > 0:
        lines.append(
            f"  │  Avg / normal cycle:     ${s['avg_cost_per_normal']:.4f}                       │"
        )
    if s["deep_reflection_cycles"] > 0:
        lines.append(
            f"  │  Avg / deep reflection:  ${s['avg_cost_per_reflection']:.4f}                       │"
        )
    if s["total_cache_hit_tokens"] > 0:
        hit_rate = s["total_cache_hit_tokens"] / max(s["total_cache_hit_tokens"] + s["total_cache_miss_tokens"], 1)
        lines.append(
            f"  │  Session cache hit rate: {hit_rate:.1%}                                  │"
        )

    lines.append("  └─────────────────────────────────────────────────────────────┘")
    return "\n".join(lines)


# ── Skip counters (persisted to disk) ──────────────────────────────


def log_skip(reason: str) -> None:
    """Log a skipped L1 cycle. Persisted to disk for cross-process visibility."""
    global _skip_count, _skip_loaded
    _ensure_loaded()
    _skip_count += 1
    try:
        skip_path = _resolve_skip_path()
        os.makedirs(os.path.dirname(skip_path), exist_ok=True)
        with open(skip_path, "w", encoding="utf-8") as f:
            json.dump({
                "skip_count": _skip_count,
                "last_updated": datetime.now().isoformat(),
            }, f)
    except OSError:
        pass


def get_skip_count() -> int:
    """Get total skipped L1 cycles."""
    _ensure_loaded()
    return _skip_count


