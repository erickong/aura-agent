"""Incremental task-file processing for Aura Agent.

Provides:
- parse_task_items(): extract checkbox/checkmark items from .md task files
- get_new_items(): return only items not yet processed (fingerprint-based dedup)
- get_pending_items(): return only unchecked (- [ ]) items
- mark_items_processed(): batch-mark items as seen
- TaskState: lightweight set-based fingerprint tracker
- CompletionMemory: persistent log of completed items
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

# ── Regex patterns ────────────────────────────────────────────────────
_CHECKBOX_RE = re.compile(r"^[-*]\s+\[([ xX])\]\s+(.*)")
_CHECKMARK_RE = re.compile(r"^✅\s*(.*)")


# ── Public API: item parsing ─────────────────────────────────────────

def parse_task_items(task_file: str) -> list[dict]:
    """Parse a .md task file and extract all task-line items.

    Recognises two formats:
      - ``- [ ] pending task``   (checkbox, not yet done)
      - ``- [x] completed task`` (checkbox, already done)
      - ``✅ completed item``    (checkmark, already done)

    Returns a list of dicts, each with:
      - line_num: 1-based line number in the file
      - text:      the task description text
      - status:    'pending' | 'completed' | 'done'
      - item_type: 'checkbox' | 'checkmark'
      - fingerprint: stable hash of (line_num, text) for dedup
    """
    path = Path(task_file)
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8")
    items = []

    for i, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()

        m_check = _CHECKBOX_RE.match(stripped)
        if m_check:
            marker = m_check.group(1)
            text = m_check.group(2).strip()
            status = "completed" if marker.lower() == "x" else "pending"
            items.append(_make_item(i, text, status, "checkbox"))
            continue

        m_cm = _CHECKMARK_RE.match(stripped)
        if m_cm:
            text = m_cm.group(1).strip() if m_cm.group(1) else ""
            items.append(_make_item(i, text, "done", "checkmark"))

    return items


def get_new_items(task_file: str, state: "TaskState") -> list[dict]:
    """Return items from *task_file* whose fingerprint is not yet in *state*."""
    items = parse_task_items(task_file)
    return [it for it in items if not state.is_processed(it["fingerprint"])]


def get_pending_items(task_file: str) -> list[dict]:
    """Return only the *pending* (``- [ ]``) items from *task_file*."""
    return [it for it in parse_task_items(task_file) if it["status"] == "pending"]


def mark_items_processed(task_file: str, state: "TaskState") -> int:
    """Mark every item currently in *task_file* as processed in *state*.

    Returns the number of items that were newly marked.
    """
    items = parse_task_items(task_file)
    count = 0
    for it in items:
        if not state.is_processed(it["fingerprint"]):
            state.mark_processed(it["fingerprint"])
            count += 1
    return count


# ── Helper ────────────────────────────────────────────────────────────

def _make_item(line_num: int, text: str, status: str, item_type: str) -> dict:
    """Build a uniform item dict with a stable fingerprint."""
    raw = f"{line_num}:{text}"
    fingerprint = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return {
        "line_num": line_num,
        "text": text,
        "status": status,
        "item_type": item_type,
        "fingerprint": fingerprint,
    }


# ═══════════════════════════════════════════════════════════════════════
# TaskState — lightweight fingerprint tracker for incremental processing
# ═══════════════════════════════════════════════════════════════════════

class TaskState:
    """Lightweight set of processed-item fingerprints persisted to disk.

    Typical usage::

        state = TaskState("projects/myproj/state/incremental_state.json")
        new = [i for i in parse_task_items("tasks/T4/task.md")
               if not state.is_processed(i["fingerprint"])]
        for item in new:
            process(item)
            state.mark_processed(item["fingerprint"])
        state.save()
    """

    def __init__(self, state_file: str | None = None):
        self._processed: set[str] = set()
        self._state_file = Path(state_file) if state_file else None
        if self._state_file is not None and self._state_file.exists():
            self.load()

    def mark_processed(self, fingerprint: str) -> None:
        """Record *fingerprint* as having been processed."""
        self._processed.add(fingerprint)

    def is_processed(self, fingerprint: str) -> bool:
        """Return True if *fingerprint* was already marked processed."""
        return fingerprint in self._processed

    def save(self) -> None:
        """Persist the processed set to the JSON file on disk."""
        if self._state_file is None:
            return
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"processed": sorted(self._processed)}
        self._state_file.write_text(json.dumps(payload, indent=2),
                                    encoding="utf-8")

    def load(self) -> None:
        """Restore the processed set from disk."""
        if self._state_file is None or not self._state_file.exists():
            return
        raw = self._state_file.read_text(encoding="utf-8").strip()
        if not raw:
            return
        data = json.loads(raw)
        self._processed = set(data.get("processed", []))

    def clear(self) -> None:
        """Remove all tracked fingerprints (in-memory only; call save() to persist)."""
        self._processed.clear()

    @property
    def processed_count(self) -> int:
        return len(self._processed)


# ═══════════════════════════════════════════════════════════════════════
# CompletionMemory — persistent completion log
# ═══════════════════════════════════════════════════════════════════════

class CompletionMemory:
    """Record of completed task items persisted to disk.

    Typical usage::

        mem = CompletionMemory("projects/myproj/memory/completions.json")
        if not mem.is_completed(item["fingerprint"]):
            result = process(item)
            mem.record_completion(item["fingerprint"], item["text"])
            mem.save()
    """

    def __init__(self, memory_file: str | None = None):
        self._completed: dict[str, dict] = {}
        self._memory_file = Path(memory_file) if memory_file else None
        if self._memory_file is not None and self._memory_file.exists():
            self.load()

    def record_completion(self, fingerprint: str, item_text: str) -> None:
        """Store a completion record with the current time."""
        self._completed[fingerprint] = {
            "text": item_text,
            "completed_at": datetime.now().isoformat(),
        }

    def is_completed(self, fingerprint: str) -> bool:
        """Return True if *fingerprint* has a completion record."""
        return fingerprint in self._completed

    def save(self) -> None:
        """Persist completion records to JSON file on disk."""
        if self._memory_file is None:
            return
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"completions": self._completed}
        self._memory_file.write_text(json.dumps(payload, indent=2),
                                     encoding="utf-8")

    def load(self) -> None:
        """Restore completion records from disk."""
        if self._memory_file is None or not self._memory_file.exists():
            return
        raw = self._memory_file.read_text(encoding="utf-8").strip()
        if not raw:
            return
        data = json.loads(raw)
        self._completed = data.get("completions", {})

    def clear(self) -> None:
        """Remove all completion records (in-memory only)."""
        self._completed.clear()

    @property
    def completion_count(self) -> int:
        return len(self._completed)
