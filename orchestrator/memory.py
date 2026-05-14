"""Memory management for the Aura Agent Orchestrator.

Three-layer memory system:
- Long-term: memory/MEMORY.md (~3000 char limit) - mission, lessons, patterns
- Short-term: memory/session.md (~2000 char limit) - current context, last actions
- Review: memory/reviews/ - periodic reflection outputs

OPTIMIZATION (T0): mtime-based caching for session and long-term memory reads.
When nothing modifies these files between wakes, reads skip disk I/O entirely.
"""

import os
from datetime import datetime
from typing import Optional

from .config import MEMORY_DIR, LONG_TERM_MEMORY_MAX_CHARS, SHORT_TERM_MEMORY_MAX_CHARS

MEMORY_PATH = os.path.join(MEMORY_DIR, "MEMORY.md")
SESSION_PATH = os.path.join(MEMORY_DIR, "session.md")
REVIEWS_DIR = os.path.join(MEMORY_DIR, "reviews")

# ── T0 optimization: in-memory caches for session and long-term memory ──
# Each cache stores the last read content + its file mtime. Cache is
# invalidated when a write operation touches the corresponding file.
_session_cache: Optional[str] = None
_session_mtime: float = 0.0
_memory_cache: Optional[str] = None
_memory_mtime: float = 0.0


def load_long_term_memory() -> str:
    """Load the long-term memory file. Cached by mtime (T0 optimization)."""
    global _memory_cache, _memory_mtime

    if not os.path.exists(MEMORY_PATH):
        _memory_cache = None
        _memory_mtime = 0.0
        return "(No long-term memory yet)"

    # T0: check mtime for cache validity
    try:
        current_mtime = os.path.getmtime(MEMORY_PATH)
    except OSError:
        current_mtime = 0.0

    if _memory_cache is not None and current_mtime == _memory_mtime:
        return _memory_cache

    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    _memory_cache = content
    _memory_mtime = current_mtime
    return content


def get_memory_preview(max_chars: int = 1500) -> str:
    """Get a preview of long-term memory, truncated to max_chars.

    Used by the review engine to include memory context without
    overwhelming the review prompt.

    Args:
        max_chars: Maximum characters to return.

    Returns:
        Truncated memory preview.
    """
    content = load_long_term_memory()
    if len(content) <= max_chars:
        return content
    # Take first ~60% and last ~40% to capture both mission and recent context
    head_size = int(max_chars * 0.6)
    tail_size = max_chars - head_size
    return content[:head_size] + f"\n\n... [{len(content) - head_size - tail_size} chars omitted] ...\n\n" + content[-tail_size:]


def load_session() -> str:
    """Load the short-term session memory. Cached by mtime (T0 optimization)."""
    global _session_cache, _session_mtime

    if not os.path.exists(SESSION_PATH):
        _session_cache = None
        _session_mtime = 0.0
        return "(No session memory yet)"

    # T0: check mtime for cache validity
    try:
        current_mtime = os.path.getmtime(SESSION_PATH)
    except OSError:
        current_mtime = 0.0

    if _session_cache is not None and current_mtime == _session_mtime:
        return _session_cache

    with open(SESSION_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    _session_cache = content
    _session_mtime = current_mtime
    return content


def write_session(content: str) -> str:
    """Overwrite the short-term session memory. Updates cache (T0)."""
    global _session_cache, _session_mtime

    os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
    # Enforce character limit
    if len(content) > SHORT_TERM_MEMORY_MAX_CHARS:
        content = content[:SHORT_TERM_MEMORY_MAX_CHARS] + "\n... [truncated]"
    with open(SESSION_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    # T0: update cache so reads skip disk I/O
    _session_cache = content
    try:
        _session_mtime = os.path.getmtime(SESSION_PATH)
    except OSError:
        _session_mtime = 0.0

    return f"OK: Session memory updated ({len(content)} chars)"


def append_memory(mem_type: str, content: str) -> str:
    """Append an entry to long-term memory. Types: fact, lesson, pattern, decision.

    Invalidates the long-term memory read cache (T0) after writing.
    """
    global _memory_cache, _memory_mtime

    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)

    existing = ""
    if os.path.exists(MEMORY_PATH):
        # Use cache if available, otherwise read from disk
        if _memory_cache is not None:
            existing = _memory_cache
        else:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                existing = f.read()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    type_labels = {
        "fact": "Fact",
        "lesson": "Lesson",
        "pattern": "Pattern",
        "decision": "Decision",
    }
    label = type_labels.get(mem_type, mem_type)

    entry = f"\n## [{label}] {timestamp}\n{content}\n"

    new_content = existing + entry

    # Check if compression is needed
    if len(new_content) > LONG_TERM_MEMORY_MAX_CHARS:
        new_content = _compress(existing) + entry

    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    # T0: update cache so subsequent reads skip disk I/O
    _memory_cache = new_content
    try:
        _memory_mtime = os.path.getmtime(MEMORY_PATH)
    except OSError:
        _memory_mtime = 0.0

    return f"OK: Memory entry added as [{label}]. Total memory: {len(new_content)} chars"


def save_review(focus: str, content: str) -> str:
    """Save a review/reflection output."""
    os.makedirs(REVIEWS_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"review_{date_str}.md"
    filepath = os.path.join(REVIEWS_DIR, filename)

    full_content = f"# Review: {focus}\nDate: {datetime.now().isoformat()}\n\n{content}"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_content)

    return f"OK: Review saved to memory/reviews/{filename}"


def get_memory_size() -> int:
    """Get the current size of long-term memory in characters."""
    if not os.path.exists(MEMORY_PATH):
        return 0
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return len(f.read())


def overwrite_memory(content: str) -> str:
    """Overwrite the entire long-term memory file.
    Used by compress_memory() in review.py.
    Updates cache (T0) after writing.
    """
    global _memory_cache, _memory_mtime

    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    # T0: update cache
    _memory_cache = content
    try:
        _memory_mtime = os.path.getmtime(MEMORY_PATH)
    except OSError:
        _memory_mtime = 0.0

    return f"OK: Memory overwritten ({len(content)} chars)"


def _compress(content: str) -> str:
    """Compress long-term memory to fit within the character limit.
    Keeps the most recent entries and summarizes older ones.
    """
    if len(content) <= LONG_TERM_MEMORY_MAX_CHARS:
        return content

    # Split into sections by ## headers
    sections = content.split("\n## ")
    if len(sections) <= 1:
        return content[:LONG_TERM_MEMORY_MAX_CHARS]

    # Keep header and first section, compress middle, keep last sections
    header = sections[0]
    entries = sections[1:]

    if len(entries) <= 3:
        return content[:LONG_TERM_MEMORY_MAX_CHARS]

    # Keep first entry (often mission) and last 3 entries
    kept = [header, entries[0]] + entries[-3:]
    result = "\n## ".join(kept)

    # If still too long, truncate
    if len(result) > LONG_TERM_MEMORY_MAX_CHARS:
        result = result[:LONG_TERM_MEMORY_MAX_CHARS] + "\n\n[MEMORY COMPRESSED]"

    return result
