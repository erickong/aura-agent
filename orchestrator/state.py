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

# ── T0 optimization: in-memory state cache with mtime invalidation ────
# _state_cache holds the last known state dict (deep copy, never mutated by
# callers). _state_mtime tracks the file modification time at cache time.
# When load_state() is called, if the file's mtime matches, the cached copy
# is returned directly — avoiding JSON parse + disk I/O.
_state_cache: Optional[dict] = None
_state_mtime: float = 0.0


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
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "total_cycles": 0,
        "task_file": "",
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


def _task_nodes_from_file(task_file: str) -> list[dict]:
    blocks = parse_requirement_blocks(task_file)
    nodes = []
    for idx, block in enumerate(blocks, start=1):
        if _is_completion_directive(block["text"]):
            continue
        nodes.append({
            "id": f"T{len(nodes) + 1}",
            "description": block["text"],
            "status": block.get("status", "pending"),
            "depth": 1,
            "created_at": datetime.now().isoformat(),
            "children": [],
            "source": {
                "task_file": task_file,
                "line_num": block.get("line_num"),
                "type": block.get("type"),
            },
            "acceptance_criteria": "Implement this requirement and provide verifiable evidence.",
        })
    return nodes


def init_state(mission: str, task_file: str = "") -> dict:
    """Initialize a new state with a mission."""
    state = _empty_state()
    state["mission"] = mission
    state["task_file"] = task_file
    state["tasks"] = [{
        "id": "root",
        "description": mission,
        "status": "pending",
        "depth": 0,
        "created_at": datetime.now().isoformat(),
        "children": _task_nodes_from_file(task_file) if task_file else [],
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


def _completion_directive_lines(task_file: str) -> list[int]:
    return [
        block["line_num"]
        for block in parse_requirement_blocks(task_file)
        if _is_completion_directive(block["text"])
    ]


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
    return bool(re.fullmatch(r"T\d+", str(task.get("id", "")))) and task.get("depth", 1) == 1


def reconcile_task_file(
    task_file: str,
    mission: str = "",
    running_task_ids: set[str] | None = None,
) -> dict:
    """Reconcile the current state with the latest task file.

    Startup flow:
    1. Parse the user's requirement file.
    2. Preserve existing matching tasks and their evidence/status.
    3. Add new requirements from the file.
    4. Archive requirements removed from the file.
    5. Mark in-progress tasks with no running process as pending so they can
       be resumed by a new Layer 2 worker.
    """
    running_task_ids = running_task_ids or set()
    desired_nodes = _task_nodes_from_file(task_file)
    stats = {
        "kept": 0,
        "added": 0,
        "updated": 0,
        "archived": 0,
        "interrupted": 0,
        "completed_from_result": 0,
        "completed_by_user_directive": 0,
    }

    state = load_state()
    if not state.get("tasks"):
        init_state(mission, task_file)
        state = load_state()

    state["mission"] = mission or state.get("mission", "")
    state["task_file"] = task_file
    root = state["tasks"][0]
    root["description"] = state["mission"]
    root["status"] = "in_progress" if desired_nodes else root.get("status", "pending")

    old_children = root.get("children", [])
    by_text: dict[str, dict] = {}
    by_id: dict[str, dict] = {}
    task_file_nodes: list[dict] = []
    other_nodes: list[dict] = []

    for child in old_children:
        if _is_task_file_node(child, task_file):
            task_file_nodes.append(child)
            by_text.setdefault(_normalize_requirement(child.get("description", "")), child)
            by_id.setdefault(child.get("id", ""), child)
        else:
            other_nodes.append(child)

    reconciled: list[dict] = []
    used_old_ids: set[int] = set()
    now = datetime.now().isoformat()

    for desired in desired_nodes:
        normalized = _normalize_requirement(desired["description"])
        old = by_text.get(normalized)
        if old is not None and id(old) in used_old_ids:
            old = None
        if old is None:
            old = _find_best_requirement_match(desired, task_file_nodes, used_old_ids)
        if old is None:
            id_match = by_id.get(desired["id"])
            if id_match is not None and id(id_match) not in used_old_ids:
                old = id_match

        if old is None:
            node = desired
            stats["added"] += 1
        else:
            used_old_ids.add(id(old))
            node = old
            old_normalized = _normalize_requirement(node.get("description", ""))
            node["id"] = desired["id"]
            node["description"] = desired["description"]
            node["depth"] = desired["depth"]
            node["source"] = desired["source"]
            node.setdefault("children", [])
            node.setdefault("acceptance_criteria", desired["acceptance_criteria"])
            if old_normalized == normalized:
                stats["kept"] += 1
            else:
                stats["updated"] += 1
                if node.get("status") == "completed":
                    node["status"] = "pending"
                    node["updated_at"] = now
                    node["reason"] = "Requirement text changed in task file; reopening."

        task_dir = Path(STATE_DIR).parent / "workspace" / "tasks" / node["id"]
        result_path = task_dir / "result.md"
        output_jsonl = task_dir / "output.jsonl"
        output_txt = task_dir / "output.txt"

        if node.get("status") == "in_progress" and node["id"] not in running_task_ids:
            if result_path.exists():
                node["status"] = "completed"
                node["completed_at"] = now
                node["evidence"] = str(result_path)
                stats["completed_from_result"] += 1
            else:
                node["status"] = "pending"
                node["interrupted_at"] = now
                evidence = []
                if output_jsonl.exists():
                    evidence.append(str(output_jsonl))
                if output_txt.exists():
                    evidence.append(str(output_txt))
                node["evidence"] = "; ".join(evidence) if evidence else "Interrupted before completion."
                stats["interrupted"] += 1

        reconciled.append(node)

    for old in task_file_nodes:
        if id(old) in used_old_ids:
            continue
        old["status"] = "archived"
        old["archived_at"] = now
        old["reason"] = "Requirement no longer appears in task file."
        reconciled.append(old)
        stats["archived"] += 1

    root["children"] = reconciled + other_nodes

    directive_lines = _completion_directive_lines(task_file)
    if directive_lines:
        last_line = max(directive_lines)
        for child in root["children"]:
            source = child.get("source") or {}
            line_num = source.get("line_num")
            if not isinstance(line_num, int) or line_num >= last_line:
                continue
            if child.get("status") in {"completed", "archived"}:
                continue
            child["status"] = "completed"
            child["completed_at"] = now
            child["evidence"] = f"User stated earlier requirements were already completed in {task_file}:{last_line}"
            stats["completed_by_user_directive"] += 1

    state["active_tasks"] = [t["id"] for t in _collect_in_progress(state["tasks"])]
    save_state(state)
    return stats


def sync_task_file_tasks(task_file: str) -> int:
    """Backward-compatible wrapper for older callers."""
    return reconcile_task_file(task_file).get("added", 0)


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


def decompose_task(parent_task_id: str, subtasks: list[dict]) -> str:
    """Add subtasks to a parent task node."""
    state = load_state()
    parent = find_task(parent_task_id, state["tasks"])

    if parent is None:
        return f"ERROR: Parent task {parent_task_id} not found"

    new_depth = parent.get("depth", 0) + 1

    for st in subtasks:
        st.setdefault("status", "pending")
        st.setdefault("depth", new_depth)
        st.setdefault("created_at", datetime.now().isoformat())
        st.setdefault("attempts", 0)
        st.setdefault("children", [])

    parent.setdefault("children", []).extend(subtasks)
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
