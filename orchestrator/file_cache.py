"""File read cache (R4) — mtime-based caching for file reads.

Avoids re-reading unchanged files every cycle. Uses file modification time
(mtime) to detect changes. Cache entries expire after a configurable TTL.

Usage:
    from .file_cache import cached_read_file

    content = cached_read_file("/path/to/file.txt")
"""

import os
import time
from typing import Optional

from .config import FILE_CACHE_ENABLED, FILE_CACHE_TTL_SECONDS

# ── In-memory cache ──────────────────────────────────────────────────
# _cache: {absolute_path: {"content": str, "mtime": float, "cached_at": float}}
_cache: dict[str, dict] = {}

# Stats for diagnostics
_cache_hits = 0
_cache_misses = 0


def cached_read_file(path: str, encoding: str = "utf-8") -> Optional[str]:
    """Read a file with mtime-based caching.

    If the file hasn't been modified since the last cached read and
    the cache hasn't expired, returns the cached content. Otherwise,
    reads from disk and updates the cache.

    Args:
        path: Absolute or relative file path.
        encoding: File encoding (default utf-8).

    Returns:
        File content as string, or None if file doesn't exist or can't be read.
    """
    global _cache_hits, _cache_misses

    if not FILE_CACHE_ENABLED:
        return _read_file_direct(path, encoding)

    abs_path = os.path.abspath(path)

    # Check cache
    if abs_path in _cache:
        entry = _cache[abs_path]
        # Check TTL
        if time.time() - entry["cached_at"] < FILE_CACHE_TTL_SECONDS:
            try:
                current_mtime = os.path.getmtime(abs_path)
            except OSError:
                # File no longer exists, invalidate cache
                del _cache[abs_path]
                _cache_misses += 1
                return None

            if current_mtime == entry["mtime"]:
                _cache_hits += 1
                return entry["content"]

    # Cache miss or invalidated
    _cache_misses += 1
    content = _read_file_direct(abs_path, encoding)
    if content is not None:
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            mtime = 0.0
        _cache[abs_path] = {
            "content": content,
            "mtime": mtime,
            "cached_at": time.time(),
        }
    return content


_MAX_FILE_READ_BYTES = 50 * 1024 * 1024  # 50 MB safety cap


def _read_file_direct(path: str, encoding: str = "utf-8", max_bytes: int = _MAX_FILE_READ_BYTES) -> Optional[str]:
    """Direct file read without caching. Capped at max_bytes to prevent OOM."""
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None

    try:
        if size is not None and size > max_bytes:
            # Stream the tail portion — this is a safety fallback
            with open(path, "rb") as f:
                f.seek(size - min(size, max_bytes))
                raw = f.read(min(size, max_bytes))
            return raw.decode(encoding, errors="replace")
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def cached_list_directory(path: str) -> Optional[list[str]]:
    """List directory contents with mtime-based caching.

    Cache is invalidated when the directory's mtime changes.
    """
    global _cache_hits, _cache_misses

    if not FILE_CACHE_ENABLED:
        return _list_dir_direct(path)

    abs_path = os.path.abspath(path)
    cache_key = f"__dir__{abs_path}"

    if cache_key in _cache:
        entry = _cache[cache_key]
        if time.time() - entry["cached_at"] < FILE_CACHE_TTL_SECONDS:
            try:
                current_mtime = os.path.getmtime(abs_path)
            except OSError:
                del _cache[cache_key]
                _cache_misses += 1
                return None

            if current_mtime == entry["mtime"]:
                _cache_hits += 1
                return list(entry["content"])  # return copy

    _cache_misses += 1
    result = _list_dir_direct(abs_path)
    if result is not None:
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            mtime = 0.0
        _cache[cache_key] = {
            "content": list(result),  # store copy
            "mtime": mtime,
            "cached_at": time.time(),
        }
    return result


def _list_dir_direct(path: str) -> Optional[list[str]]:
    """Direct directory listing without caching."""
    try:
        return os.listdir(path)
    except OSError:
        return None


def invalidate_cache(path: Optional[str] = None) -> int:
    """Invalidate cache entries.

    Args:
        path: If provided, only invalidate entries matching this path.
              If None, invalidate all entries.

    Returns:
        Number of entries invalidated.
    """
    global _cache
    if path is None:
        count = len(_cache)
        _cache.clear()
        return count

    abs_path = os.path.abspath(path)
    # Invalidate exact path matches and directory listing cache
    count = 0
    keys_to_remove = []
    for key in list(_cache.keys()):
        if key == abs_path or key == f"__dir__{abs_path}":
            keys_to_remove.append(key)
            count += 1
    for key in keys_to_remove:
        del _cache[key]
    return count


def get_cache_stats() -> dict:
    """Return cache hit/miss statistics."""
    return {
        "enabled": FILE_CACHE_ENABLED,
        "entries": len(_cache),
        "hits": _cache_hits,
        "misses": _cache_misses,
        "hit_rate": round(_cache_hits / max(1, _cache_hits + _cache_misses), 3),
        "ttl_seconds": FILE_CACHE_TTL_SECONDS,
    }
