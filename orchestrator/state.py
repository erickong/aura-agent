"""State management for the Aura Agent Orchestrator.

Manages the task tree stored in state/state.json:
- Task tree CRUD operations
- Decision logging
- Progress metrics

OPTIMIZATION (T0): mtime-based caching for load_state().
state.json is read 4+ times per wake cycle by different callers
(build_context_message, run_cycle, log_cycle, _run_phase2_eval,
render_progress). With caching, the file is read from disk at most
once per cycle as long as no external process modifies it. The cache
is invalidated on save and when file mtime changes.
"""

import copy
import difflib
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import STATE_DIR, MAX_CONCURRENT_TASKS

STATE_PATH = os.path.join(STATE_DIR, "state.json")
STATE_BAK_PATH = os.path.join(STATE_DIR, "state.json.bak")
DEFAULT_TASK_BATCH_PREFIX = "A"

# ── T0 optimization: in-memory state cache with mtime invalidation ────
# _state_cache holds the last known state dict (deep copy, never mutated by
# callers). _state_mtime tracks the file modification time at cache time.
# When load_state() is called, if the file's mtime matches, the cached copy
# is returned directly — avoiding JSON parse + disk I/O.
_state_cache: Optional[dict] = None
_state_mtime: float = 0.0


def _batch_prefix_for_index(index: int) -> str:
    """Return spreadsheet-style batch letters: A..Z, AA..AZ, BA..."""
    index = max(0, int(index))
    chars = []
    while True:
        index, remainder = divmod(index, 26)
        chars.append(chr(ord("A") + remainder))
        if index == 0:
            break
        index -= 1
    return "".join(reversed(chars))


def _batch_index_for_prefix(prefix: str) -> int:
    prefix = (prefix or DEFAULT_TASK_BATCH_PREFIX).upper()
    total = 0
    for char in prefix:
        if not ("A" <= char <= "Z"):
            return 0
        total = total * 26 + (ord(char) - ord("A") + 1)
    return max(0, total - 1)


def _task_file_hash(task_file: str) -> str:
    try:
        with open(task_file, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return ""


def _ensure_task_batch_state(state: dict) -> dict:
    batch = state.setdefault("task_batch", {})
    prefix = str(batch.get("current_prefix") or DEFAULT_TASK_BATCH_PREFIX).upper()
    index = batch.get("current_index")
    if not isinstance(index, int):
        index = _batch_index_for_prefix(prefix)
    prefix = _batch_prefix_for_index(index)
    batch["current_index"] = index
    batch["current_prefix"] = prefix
    batch.setdefault("last_task_file_hash", "")
    return batch


def _advance_task_batch_for_change(state: dict, task_file: str, changed: bool) -> bool:
    batch = _ensure_task_batch_state(state)
    current_hash = _task_file_hash(task_file)
    previous_hash = batch.get("last_task_file_hash", "")
    if current_hash and not previous_hash and not changed:
        batch["last_task_file_hash"] = current_hash
    if not changed or not current_hash:
        return False
    if previous_hash == current_hash:
        return False

    batch["current_index"] = int(batch.get("current_index", 0)) + 1
    batch["current_prefix"] = _batch_prefix_for_index(batch["current_index"])
    batch["last_task_file_hash"] = current_hash
    batch["advanced_at"] = datetime.now().isoformat()
    return True


def _current_task_batch_prefix(state: dict) -> str:
    return _ensure_task_batch_state(state)["current_prefix"]


def _iter_tasks(tasks: list):
    for task in tasks:
        yield task
        yield from _iter_tasks(task.get("children", []))


def _all_task_ids(tasks: list) -> set[str]:
    return {str(task.get("id", "")) for task in _iter_tasks(tasks)}


def _next_top_level_task_id(root: dict, prefix: str, existing_ids: set[str]) -> str:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_num = 0
    for child in root.get("children", []):
        match = pattern.fullmatch(str(child.get("id", "")))
        if match:
            max_num = max(max_num, int(match.group(1)))

    num = max_num + 1
    while f"{prefix}{num}" in existing_ids:
        num += 1
    return f"{prefix}{num}"


def _next_child_task_id(parent: dict, existing_ids: set[str]) -> str:
    parent_id = str(parent.get("id", ""))
    pattern = re.compile(rf"^{re.escape(parent_id)}\.(\d+)$")
    max_num = 0
    for child in parent.get("children", []):
        match = pattern.fullmatch(str(child.get("id", "")))
        if match:
            max_num = max(max_num, int(match.group(1)))

    num = max_num + 1
    while f"{parent_id}.{num}" in existing_ids:
        num += 1
    return f"{parent_id}.{num}"


def load_state() -> dict:
    """Load the full state from state.json.

    Uses mtime-based caching (T0 optimization): if the file hasn't been
    modified since the last cached read, returns a copy of the cached
    state — avoiding disk I/O and JSON parsing overhead.

    If the primary state file is corrupted (JSON decode error),
    attempts to recover from state.json.bak. If both fail,
    returns an empty state and logs the corruption event.
    """
    global _state_cache, _state_mtime

    if not os.path.exists(STATE_PATH):
        if os.path.exists(STATE_BAK_PATH):
            return _load_json_safe(STATE_BAK_PATH) or _empty_state()
        _state_cache = None
        _state_mtime = 0.0
        return _empty_state()

    # ── T0: Check file mtime for cache validity ────────────────────
    try:
        current_mtime = os.path.getmtime(STATE_PATH)
    except OSError:
        current_mtime = 0.0

    if _state_cache is not None and current_mtime == _state_mtime:
        # File unchanged since last read — return cached copy.
        # Use deep copy to protect the cache from caller mutations.
        return copy.deepcopy(_state_cache)

    # Cache miss — read from disk
    state = _load_json_safe(STATE_PATH)
    if state is not None:
        _state_cache = copy.deepcopy(state)
        _state_mtime = current_mtime
        return state

    # Primary file corrupted — attempt recovery from backup
    print("[RESILIENCE] state.json is corrupted (JSON decode error). Attempting recovery from state.json.bak...")
    if os.path.exists(STATE_BAK_PATH):
        recovered = _load_json_safe(STATE_BAK_PATH)
        if recovered is not None:
            print("[RESILIENCE] Successfully recovered state from state.json.bak")
            # Restore the primary file from backup
            save_state(recovered)
            return recovered
        else:
            print("[RESILIENCE] state.json.bak is also corrupted. Starting with empty state.")
    else:
        print("[RESILIENCE] No backup file found. Starting with empty state.")

    # Both files are corrupted or missing — create fresh state
    state = _empty_state()
    state["_corruption_recovered"] = True
    state["_recovery_time"] = datetime.now().isoformat()
    save_state(state)
    return state


def _load_json_safe(path: str) -> Optional[dict]:
    """Safely load a JSON file, returning None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def save_state(state: dict) -> None:
    """Save the full state to state.json, with a .bak backup copy.

    Also updates the in-memory cache (T0 optimization) so subsequent
    load_state() calls hit the cache immediately without disk I/O.
    """
    global _state_cache, _state_mtime

    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    state["updated_at"] = datetime.now().isoformat()

    # Remove internal recovery flags before saving
    state.pop("_corruption_recovered", None)
    state.pop("_recovery_time", None)

    json_text = json.dumps(state, ensure_ascii=False, indent=2)

    # Write primary state file
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        f.write(json_text)

    # Write backup copy (atomic: write to temp then rename)
    tmp_path = STATE_BAK_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json_text)
    shutil.move(tmp_path, STATE_BAK_PATH)

    # ── T0: Update in-memory cache after successful write ──────────
    _state_cache = copy.deepcopy(state)
    try:
        _state_mtime = os.path.getmtime(STATE_PATH)
    except OSError:
        _state_mtime = 0.0


def _empty_state() -> dict:
    """Create an empty initial state."""
    return {
        "mission": "",
        "project_context": {
            "final_goal": "",
            "success_criteria": "",
            "global_constraints": "",
            "execution_environment": "",
            "notes": "",
            "updated_at": "",
        },
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "total_cycles": 0,
        "task_file": "",
        "task_batch": {
            "current_index": 0,
            "current_prefix": DEFAULT_TASK_BATCH_PREFIX,
            "last_task_file_hash": "",
        },
        "tasks": [],
        "active_tasks": [],
        "decision_log": [],
    }


def parse_requirement_blocks(task_file: str) -> list[dict]:
    """Parse a task file into semantic requirement blocks.

    Section-aware aggregation:
    - '# headings' group all following content into one requirement, unless the
      section contains '- [ ]' checkboxes or '1.' numbered items (those break
      out as individual task nodes).
    - Content without headings: blank lines separate semantic blocks.
      Paragraphs, bullet lists, and table rows are aggregated within each block.
      Only '- [ ]' and '1.' markers create standalone task nodes.
    - The result is a list of coherent requirement blocks — not one per line.
    """
    path = Path(task_file)
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    heading_re = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)")
    checkbox_re = re.compile(r"^\s*[-*]\s+\[([ xX])\]\s+(.*)")
    numbered_re = re.compile(r"^\s*\d+[.)、]\s+(.*)")

    # ── Phase 1: classify each line ──────────────────────────────────
    LineKind = str  # "heading", "checkbox", "numbered", "text", "blank"

    classified: list[dict] = []
    for line_num, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("<!--"):
            classified.append({"kind": "blank", "line_num": line_num, "text": stripped})
            continue

        m = heading_re.match(stripped)
        if m:
            classified.append({
                "kind": "heading", "line_num": line_num,
                "text": m.group(2), "level": len(m.group(1)),
            })
            continue

        m = checkbox_re.match(stripped)
        if m:
            classified.append({
                "kind": "checkbox", "line_num": line_num,
                "text": m.group(2),
                "status": "completed" if m.group(1).lower() == "x" else "pending",
            })
            continue

        m = numbered_re.match(stripped)
        if m:
            classified.append({
                "kind": "numbered", "line_num": line_num,
                "text": m.group(1),
            })
            continue

        classified.append({"kind": "text", "line_num": line_num, "text": stripped})

    # ── Phase 2: group into blocks separated by blank lines ──────────
    # Also split at actionable-item boundaries (checkbox / numbered) so
    # that each task item starts its own group, even without blank lines.
    raw_blocks: list[list[dict]] = []
    current_group: list[dict] = []
    for item in classified:
        if item["kind"] == "blank":
            if current_group:
                raw_blocks.append(current_group)
                current_group = []
        elif item["kind"] in {"checkbox", "numbered"}:
            # Actionable items always start a new group
            if current_group:
                raw_blocks.append(current_group)
            current_group = [item]
        else:
            current_group.append(item)
    if current_group:
        raw_blocks.append(current_group)

    # ── Phase 3: merge heading-led sections ──────────────────────────
    # A heading block "swallows" all following text blocks until the next
    # heading of same-or-higher level or a checkbox/numbered block.
    merged: list[list[dict]] = []
    i = 0
    while i < len(raw_blocks):
        block = raw_blocks[i]
        first = block[0]
        if first["kind"] == "heading":
            hlevel = first.get("level", 1)
            j = i + 1
            while j < len(raw_blocks):
                nfirst = raw_blocks[j][0]
                if nfirst["kind"] == "heading" and nfirst.get("level", 1) <= hlevel:
                    break
                if nfirst["kind"] in {"checkbox", "numbered"}:
                    break
                # Merge this text block into the heading section
                block.extend(raw_blocks[j])
                j += 1
            merged.append(block)
            i = j
        else:
            merged.append(block)
            i += 1

    # ── Phase 4: convert merged groups to requirement blocks ─────────
    blocks: list[dict] = []
    pending_heading_text: str | None = None  # heading that lacks body, may prefix checkboxes

    # Collect all text lines from heading-free leading blocks into a preamble
    preamble_lines: list[str] = []
    preamble_start: int | None = None

    n_groups = len(merged)
    for gi, group in enumerate(merged):
        first = group[0]

        if first["kind"] == "heading":
            heading_text = first["text"]
            body_lines = [it["text"] for it in group[1:] if it["kind"] not in {"blank"}]
            has_actionable_after = (
                gi + 1 < n_groups
                and merged[gi + 1][0]["kind"] in {"checkbox", "numbered"}
            )

            if not body_lines and has_actionable_after:
                # Bare heading whose section consists only of actionable items
                # immediately after it — use as prefix, don't emit a task node.
                pending_heading_text = heading_text
                continue
            else:
                pending_heading_text = None
                full_text = heading_text
                if body_lines:
                    full_text += "\n" + "\n".join(body_lines)
                blocks.append({
                    "line_num": first["line_num"],
                    "text": full_text,
                    "type": "heading",
                    "status": "pending",
                })
            continue

        if first["kind"] == "checkbox":
            text = first["text"]
            if pending_heading_text:
                text = f"[{pending_heading_text}] {text}"
            blocks.append({
                "line_num": first["line_num"],
                "text": text,
                "type": "checkbox",
                "status": first.get("status", "pending"),
            })
            extras = [it["text"] for it in group[1:] if it["kind"] not in {"blank", "checkbox"}]
            if extras:
                blocks[-1]["text"] += "\n" + "\n".join(extras)
            continue

        if first["kind"] == "numbered":
            text = first["text"]
            if pending_heading_text:
                text = f"[{pending_heading_text}] {text}"
            blocks.append({
                "line_num": first["line_num"],
                "text": text,
                "type": "numbered",
                "status": "pending",
            })
            extras = [it["text"] for it in group[1:] if it["kind"] not in {"blank", "numbered"}]
            if extras:
                blocks[-1]["text"] += "\n" + "\n".join(extras)
            continue

        # Text block — accumulate into preamble
        pending_heading_text = None
        for it in group:
            if it["kind"] == "text":
                if preamble_start is None:
                    preamble_start = it["line_num"]
                preamble_lines.append(it["text"])

    # ── Flush preamble: if there's text before any heading/checkbox,
    #     emit ONE requirement block (not one per line) ─────────────
    if preamble_lines:
        blocks.insert(0, {
            "line_num": preamble_start or 1,
            "text": "\n".join(preamble_lines),
            "type": "paragraph",
            "status": "pending",
        })

    return [b for b in blocks if b["text"].strip()]


def init_state(mission: str, task_file: str = "") -> dict:
    """Initialize a new state with a mission."""
    state = _empty_state()
    batch = _ensure_task_batch_state(state)
    batch["last_task_file_hash"] = _task_file_hash(task_file)
    state["mission"] = mission
    state["task_file"] = task_file
    state["tasks"] = [{
        "id": "root",
        "description": mission,
        "status": "pending",
        "depth": 0,
        "created_at": datetime.now().isoformat(),
        "children": [],
    }]
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    save_state(state)
    return state


def _normalize_requirement(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_completion_directive(text: str) -> bool:
    normalized = _normalize_requirement(text)
    if not re.search(r"(上面|以上|前面|前述|之前|前面的)", normalized):
        return False
    if not re.search(r"(已经|已|都)", normalized):
        return False
    return bool(re.search(r"(完成|实现|做完|处理完|搞定|解决)", normalized))


def _is_completion_directive(text: str) -> bool:
    normalized = _normalize_requirement(text)
    if not re.search(r"(上面|以上|前面|前述|之前|前面的)", normalized):
        return False
    if not re.search(r"(已经|已|均已|都已|全部已|全都已)", normalized):
        return False
    return bool(re.search(r"(完成|实现|做完|处理完|搞定|解决)", normalized))


def _completed_by_user_directive(task: dict) -> bool:
    evidence = str(task.get("evidence", ""))
    return evidence.startswith("User stated earlier requirements were already completed")


def _completion_directive_lines(task_file: str) -> list[int]:
    return [
        block["line_num"]
        for block in parse_requirement_blocks(task_file)
        if _is_completion_directive(block["text"])
    ]


def _is_completion_directive(text: str) -> bool:
    return False


def _completion_directive_lines(task_file: str) -> list[int]:
    return []


def _requirement_similarity(left: str, right: str) -> float:
    left_norm = _normalize_requirement(left)
    right_norm = _normalize_requirement(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()


def _find_best_requirement_match(desired: dict, candidates: list[dict], used: set[int]) -> dict | None:
    desired_text = desired.get("description", "")
    desired_line = desired.get("source", {}).get("line_num")
    best: dict | None = None
    best_score = 0.0

    for candidate in candidates:
        if id(candidate) in used:
            continue
        score = _requirement_similarity(desired_text, candidate.get("description", ""))
        old_line = candidate.get("source", {}).get("line_num")
        if desired_line is not None and old_line == desired_line:
            score = max(score, 0.82)
        if score > best_score:
            best = candidate
            best_score = score

    if best_score >= 0.72:
        return best
    return None


def _is_task_file_node(task: dict, task_file: str) -> bool:
    source = task.get("source") or {}
    if os.path.normcase(source.get("task_file", "")) == os.path.normcase(task_file):
        return True
    return bool(re.fullmatch(r"[A-Z]+\d+", str(task.get("id", "")))) and task.get("depth", 1) == 1


def reconcile_task_file(
    task_file: str,
    mission: str = "",
    running_task_ids: set[str] | None = None,
    task_file_changed: bool = False,
) -> dict:
    """Record task-file metadata without making semantic planning decisions.

    Planning belongs to the orchestrator LLM during run_cycle(). This function
    only keeps the state ledger coherent: task-file path/hash, current batch,
    root metadata, stale in-progress recovery, and repair of old auto-complete
    evidence produced by earlier parser heuristics.
    """
    running_task_ids = running_task_ids or set()
    stats = {
        "kept": 0,
        "added": 0,
        "updated": 0,
        "archived": 0,
        "interrupted": 0,
        "completed_from_result": 0,
        "completed_by_user_directive": 0,
        "removed_completed": 0,
        "reopened_auto_completed": 0,
        "planning_needed": False,
        "batch": DEFAULT_TASK_BATCH_PREFIX,
        "batch_advanced": False,
    }

    state = load_state()
    if not state.get("tasks"):
        init_state(mission, task_file)
        state = load_state()

    stats["batch_advanced"] = _advance_task_batch_for_change(
        state, task_file, task_file_changed
    )
    current_batch = _current_task_batch_prefix(state)
    stats["batch"] = current_batch

    state["mission"] = mission or state.get("mission", "")
    state["task_file"] = task_file
    root = state["tasks"][0]
    root["description"] = state["mission"]
    root["status"] = "in_progress"
    now = datetime.now().isoformat()

    for task in _iter_tasks(state.get("tasks", [])):
        task.setdefault("children", [])
        if task.get("id") == "root":
            continue
        if task.get("status") == "completed" and _completed_by_user_directive(task):
            task["status"] = "pending"
            task["updated_at"] = now
            task["reason"] = "Reopened because parser-based task-file completion directives are disabled."
            task["evidence"] = "Prior completion evidence came from deprecated parser heuristics."
            stats["reopened_auto_completed"] += 1
        if task.get("status") == "in_progress" and task.get("id") not in running_task_ids:
            task["status"] = "pending"
            task["interrupted_at"] = now
            task["evidence"] = "Interrupted before completion; no running process found."
            stats["interrupted"] += 1

    root_children = root.get("children", [])
    if task_file_changed or not root_children or stats["reopened_auto_completed"] > 0:
        state["task_file_needs_planning"] = True
    stats["planning_needed"] = bool(state.get("task_file_needs_planning"))

    state["active_tasks"] = [t["id"] for t in _collect_in_progress(state["tasks"])]
    save_state(state)
    return stats


def sync_task_file_tasks(task_file: str) -> int:
    """Backward-compatible wrapper for older callers."""
    reconcile_task_file(task_file)
    return 0


def find_task(task_id: str, tasks: Optional[list] = None) -> Optional[dict]:
    """Recursively find a task node by ID."""
    if tasks is None:
        state = load_state()
        tasks = state.get("tasks", [])
    for task in tasks:
        if task["id"] == task_id:
            return task
        if "children" in task:
            found = find_task(task_id, task["children"])
            if found:
                return found
    return None


def update_task(task_id: str, new_status: str, reason: str, evidence: str) -> str:
    """Update a task node's status and log the decision."""
    state = load_state()
    task = find_task(task_id, state["tasks"])

    if task is None:
        return f"ERROR: Task {task_id} not found in tree"

    old_status = task.get("status", "unknown")
    task["status"] = new_status
    task["updated_at"] = datetime.now().isoformat()

    if new_status == "in_progress":
        task["started_at"] = datetime.now().isoformat()
    elif new_status == "completed":
        task["completed_at"] = datetime.now().isoformat()
        task["evidence"] = evidence
    elif new_status == "failed":
        task["failed_at"] = datetime.now().isoformat()

    # Update active tasks list
    active = [t["id"] for t in _collect_in_progress(state["tasks"])]
    state["active_tasks"] = active

    # Log the decision
    log_entry = {
        "time": datetime.now().isoformat(),
        "task_id": task_id,
        "old_status": old_status,
        "new_status": new_status,
        "reason": reason,
        "evidence": evidence,
    }
    state.setdefault("decision_log", []).append(log_entry)
    # Keep only last 100 decisions
    if len(state["decision_log"]) > 100:
        state["decision_log"] = state["decision_log"][-100:]

    save_state(state)
    return f"OK: Task {task_id} updated: {old_status} -> {new_status}. Reason: {reason}"


def update_project_context(
    final_goal: str = "",
    success_criteria: str = "",
    global_constraints: str = "",
    execution_environment: str = "",
    notes: str = "",
) -> str:
    """Persist LLM-extracted project-level planning context."""
    state = load_state()
    current = state.setdefault("project_context", {})
    updates = {
        "final_goal": final_goal,
        "success_criteria": success_criteria,
        "global_constraints": global_constraints,
        "execution_environment": execution_environment,
        "notes": notes,
    }
    for key, value in updates.items():
        if value is not None:
            current[key] = str(value).strip()
    current["updated_at"] = datetime.now().isoformat()
    save_state(state)
    return "OK: Project context updated."


def decompose_task(parent_task_id: str, subtasks: list[dict]) -> str:
    """Add subtasks to a parent task node."""
    state = load_state()
    _ensure_task_batch_state(state)
    parent = find_task(parent_task_id, state["tasks"])

    if parent is None:
        return f"ERROR: Parent task {parent_task_id} not found"

    new_depth = parent.get("depth", 0) + 1
    existing_ids = _all_task_ids(state.get("tasks", []))

    for st in subtasks:
        proposed_id = str(st.get("id", "")).strip()
        if parent.get("id") == "root":
            valid_prefix = _current_task_batch_prefix(state)
            valid_id = bool(re.fullmatch(rf"{re.escape(valid_prefix)}\d+", proposed_id))
            if not proposed_id or proposed_id in existing_ids or not valid_id:
                proposed_id = _next_top_level_task_id(parent, valid_prefix, existing_ids)
        else:
            parent_prefix = f"{parent_task_id}."
            if (
                not proposed_id
                or proposed_id in existing_ids
                or not proposed_id.startswith(parent_prefix)
            ):
                proposed_id = _next_child_task_id(parent, existing_ids)

        st["id"] = proposed_id
        existing_ids.add(proposed_id)
        st.setdefault("status", "pending")
        st.setdefault("depth", new_depth)
        st.setdefault("created_at", datetime.now().isoformat())
        st.setdefault("attempts", 0)
        st.setdefault("children", [])

    parent.setdefault("children", []).extend(subtasks)
    if parent.get("id") == "root":
        state["task_file_needs_planning"] = False
    save_state(state)

    subtask_ids = [st["id"] for st in subtasks]
    return f"OK: Added {len(subtasks)} subtasks ({', '.join(subtask_ids)}) to {parent_task_id}"


def count_active_tasks() -> int:
    """Count how many tasks are currently IN_PROGRESS."""
    state = load_state()
    return len(_collect_in_progress(state["tasks"]))


def can_spawn_task() -> bool:
    """Check if we can spawn another task (max 2 concurrent)."""
    return count_active_tasks() < MAX_CONCURRENT_TASKS


def get_task_tree_summary() -> str:
    """Get a text summary of the task tree for the orchestrator's context."""
    state = load_state()
    return _render_tree(state["tasks"])


def log_cycle() -> int:
    """Increment the cycle counter and return the new count."""
    state = load_state()
    state["total_cycles"] = state.get("total_cycles", 0) + 1
    save_state(state)
    return state["total_cycles"]


def _collect_in_progress(tasks: list) -> list[dict]:
    """Recursively collect all tasks with status 'in_progress'."""
    result = []
    for task in tasks:
        if task.get("status") == "in_progress" and task.get("id") != "root":
            result.append(task)
        if "children" in task:
            result.extend(_collect_in_progress(task["children"]))
    return result


def _render_tree(tasks: list, indent: int = 0) -> str:
    """Render the task tree as indented text."""
    lines = []
    status_icons = {
        "pending": "⏳",
        "in_progress": "🔄",
        "blocked": "🚫",
        "completed": "✅",
        "failed": "❌",
        "archived": "📦",
        "killed": "💀",
    }
    for task in tasks:
        icon = status_icons.get(task.get("status", "pending"), "❓")
        line = f"{'  ' * indent}{icon} [{task['id']}] {task.get('description', '')[:80]} ({task.get('status', 'pending')})"
        lines.append(line)
        if "children" in task and task["children"]:
            lines.append(_render_tree(task["children"], indent + 1))
    return "\n".join(lines)
