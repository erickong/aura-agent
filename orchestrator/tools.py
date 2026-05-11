"""Tool definitions and implementations for the Aura Agent Orchestrator.

Tools are defined in Anthropic's tool schema format and implemented as callable
functions. The orchestrator calls these via Claude API tool_use blocks.
"""

import os
import json
import re
import urllib.request
import urllib.error
from datetime import datetime
from typing import Any

from . import process_mgr
from . import state as state_mgr
from . import memory as memory_mgr
from .config import get_workspace_dir
from .file_cache import cached_read_file, cached_list_directory, invalidate_cache

MIN_EXECUTABLE_TASK_DEPTH = 2

# ── Tool Definitions (Anthropic schema) ──────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Use mode='tail' for logs/outputs, 'head' for configs, or specify line ranges. Default max_chars is 2000 (not 8000).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read, relative to project root."
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return. Default 2000.",
                    "default": 2000
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "head", "tail", "full"],
                    "description": "Read mode: auto (head+tail), head, tail, full.",
                    "default": "auto"
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-indexed start line for range reads. 0 = from start.",
                    "default": 0
                },
                "end_line": {
                    "type": "integer",
                    "description": "1-indexed end line (inclusive). 0 = to end.",
                    "default": 0
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Use this to create task definitions for Layer 2 workers or update project files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write, relative to project root."
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file."
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_directory",
        "description": "List contents of a directory. Use this to explore current-task workspace, check task outputs, or discover available files. Other .aura task directories are isolated except their memory files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list, relative to project root."
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "web_fetch",
        "description": "Fetch content from a URL and extract readable text. Use this to research documentation, check APIs, or gather information from the web. Returns plain text stripped of HTML.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to fetch (https://...)."
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return. Default 5000.",
                    "default": 5000
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "spawn_task",
        "description": "Start a new Layer 2 worker (Claude Code CLI) to execute a concrete leaf task. Maximum 2 concurrent workers. Only spawn tasks at ROOT -> category -> task level or deeper; top-level ROOT children are planning categories, not executable tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Unique executable task ID from the current task tree, e.g. 'A1.1' or 'B2.3'. Do not spawn root or top-level category IDs like 'A1'."
                },
                "description": {
                    "type": "string",
                    "description": "Detailed task description. Include: goal, acceptance criteria (verifiable outputs), context, and constraints."
                },
                "budget_minutes": {
                    "type": "integer",
                    "description": "Time budget in minutes. Default 30.",
                    "default": 30
                }
            },
            "required": ["task_id", "description"]
        }
    },
    {
        "name": "kill_task",
        "description": "Kill a running Layer 2 worker process. Use this when a task is stuck, unproductive, or no longer needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "ID of the task to kill."
                }
            },
            "required": ["task_id"]
        }
    },
    {
        "name": "list_running_tasks",
        "description": "List all currently running Layer 2 worker processes with their status.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "update_task_tree",
        "description": "Update the status of a task node in the task tree. Records the decision and evidence for traceability.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "ID of the task node to update."
                },
                "new_status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "blocked", "completed", "failed", "archived", "killed"],
                    "description": "New status for the task."
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason for the status change."
                },
                "evidence": {
                    "type": "string",
                    "description": "Verifiable evidence supporting this change (e.g. file path, test result)."
                }
            },
            "required": ["task_id", "new_status", "reason", "evidence"]
        }
    },
    {
        "name": "decompose_task",
        "description": "Break a task into smaller subtasks. Use parent_task_id='root' only to create broad planning categories. Then decompose each category into concrete third-level tasks (for example A1 -> A1.1) before spawning workers. Each concrete task should have clear, verifiable acceptance criteria.",
        "input_schema": {
            "type": "object",
            "properties": {
                "parent_task_id": {
                    "type": "string",
                    "description": "ID of the parent task to decompose."
                },
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Unique subtask ID. Children should keep the parent prefix, e.g. A1 -> A1.1, B2 -> B2.1."},
                            "description": {"type": "string", "description": "What this subtask does."},
                            "acceptance_criteria": {"type": "string", "description": "How to verify completion (verifiable outputs)."}
                        },
                        "required": ["id", "description", "acceptance_criteria"]
                    },
                    "description": "List of subtasks."
                }
            },
            "required": ["parent_task_id", "subtasks"]
        }
    },
    {
        "name": "update_project_context",
        "description": "Persist project-level context extracted from task.md: final goal, success criteria, global constraints, and execution environment. Use this during planning before spawning work, especially when task.md mentions commands, API keys, env vars, working directories, models, or required tools.",
        "input_schema": {
            "type": "object",
            "properties": {
                "final_goal": {
                    "type": "string",
                    "description": "The ultimate goal the whole project must achieve."
                },
                "success_criteria": {
                    "type": "string",
                    "description": "How to know the final goal is truly achieved; include measurable thresholds and acceptance tests."
                },
                "global_constraints": {
                    "type": "string",
                    "description": "Project-wide constraints, architectural requirements, forbidden approaches, or persistent assumptions."
                },
                "execution_environment": {
                    "type": "string",
                    "description": "Commands, env vars, API key names/usages, working directories, models, runtimes, or setup steps needed by workers."
                },
                "notes": {
                    "type": "string",
                    "description": "Other durable planning context that future cycles and workers must preserve."
                }
            },
            "required": ["final_goal"]
        }
    },
    {
        "name": "write_memory",
        "description": "Write an important fact, lesson, or pattern to long-term memory (MEMORY.md). Use sparingly - only for insights that will be valuable in future cycles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mem_type": {
                    "type": "string",
                    "enum": ["fact", "lesson", "pattern", "decision"],
                    "description": "Type of memory."
                },
                "content": {
                    "type": "string",
                    "description": "The memory content. Be concise. Include evidence source when applicable."
                }
            },
            "required": ["mem_type", "content"]
        }
    },
    {
        "name": "no_op",
        "description": "No operation needed this cycle. The system will sleep and wake up again later. Use this when there's nothing to do or everything is proceeding as expected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why no action is needed."
                },
                "next_check_focus": {
                    "type": "string",
                    "description": "What to focus on in the next wake cycle."
                }
            },
            "required": ["reason"]
        }
    }
]


# ── Tool Implementations ─────────────────────────────────────────────

def _resolve_path(path: str) -> str:
    """Resolve a project-relative path to an absolute path."""
    from .config import DATA_DIR, PROJECT_ROOT

    normalized = os.path.normpath(path or ".")
    parts = normalized.split(os.sep)
    if os.altsep:
        parts = normalized.replace(os.altsep, os.sep).split(os.sep)

    if len(parts) >= 2 and parts[0] == ".aura" and parts[1] in {
        "state",
        "memory",
        "workspace",
        "summaries",
        "projects",
        "cache",
    }:
        return os.path.normpath(os.path.join(DATA_DIR, *parts[1:]))

    if os.path.isabs(normalized):
        return normalized
    return os.path.normpath(os.path.join(PROJECT_ROOT, normalized))


def _other_aura_task_path_error(path: str, operation: str) -> str | None:
    """Return an error if a tool request targets another task file's Aura data.

    Other task directories may expose memory for lessons, but their state,
    workspace, progress, summaries, and metadata are not current-task evidence.
    """
    from .config import DATA_DIR, PROJECT_ROOT

    normalized = os.path.normpath(path or ".")
    if os.path.isabs(normalized):
        try:
            rel = os.path.relpath(normalized, PROJECT_ROOT)
        except ValueError:
            return None
    else:
        rel = normalized

    rel = rel.replace(os.altsep or os.sep, os.sep)
    parts = [part for part in rel.split(os.sep) if part and part != "."]
    if len(parts) < 2 or parts[0] != ".aura":
        return None

    legacy_current_aliases = {
        "state",
        "memory",
        "workspace",
        "summaries",
        "projects",
        "cache",
    }
    if parts[1] in legacy_current_aliases:
        return None

    current_data_name = os.path.basename(os.path.normpath(DATA_DIR))
    if parts[1] == current_data_name:
        return None

    # Allow reading/listing other task memory as lesson material only. Writes
    # should go through current-task memory/project context.
    if operation in {"read", "list"} and len(parts) >= 3 and parts[2] == "memory":
        return None

    return (
        f"ERROR: Refusing to {operation} another task file's Aura data: {path}. "
        "During planning, use the current Aura data directory as source of "
        "truth. You may read .aura/<other-task>/memory/... only for lessons."
    )


# ── Safety cap: max bytes to read for non-tail modes ──────────────────
_MAX_FILE_READ_BYTES = 50 * 1024 * 1024  # 50 MB


def _read_file_tail_from_disk(abs_path: str, max_chars: int) -> tuple[str | None, int, str]:
    """Read only the tail of a file without loading everything into memory.

    Returns (text, total_size_bytes, error_reason) — text is None on error.
    """
    try:
        total_size = os.path.getsize(abs_path)
    except OSError as e:
        return None, 0, str(e)

    if total_size == 0:
        return "", 0, ""

    # Read last N bytes (4x max_chars for multi-byte encoding headroom)
    read_size = min(total_size, max(8192, max_chars * 4))
    try:
        with open(abs_path, "rb") as f:
            f.seek(total_size - read_size)
            raw = f.read(read_size)
    except OSError as e:
        return None, 0, str(e)

    text = raw.decode("utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text, total_size, ""


def _read_file_head_from_disk(abs_path: str, max_chars: int) -> tuple[str | None, int, str]:
    """Read only the head of a file without loading everything into memory."""
    try:
        total_size = os.path.getsize(abs_path)
    except OSError as e:
        return None, 0, str(e)

    if total_size == 0:
        return "", 0, ""

    # Read first N bytes (4x max_chars for encoding headroom)
    read_size = min(total_size, max(8192, max_chars * 4))
    try:
        with open(abs_path, "rb") as f:
            raw = f.read(read_size)
    except OSError as e:
        return None, 0, str(e)

    text = raw.decode("utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars]
    return text, total_size, ""


def _check_output_jsonl_guard(abs_path: str) -> str | None:
    """Intercept reads of output.jsonl in task workspace directories.

    The orchestrator should read result.md for task outcomes and trust
    Phase 2 progress signals for active workers. Raw output.jsonl is
    for worker-level debugging, not orchestrator decision-making.

    Returns an advisory message string if the read should be blocked,
    or None if the read is allowed to proceed.
    """
    workspace_dir = os.path.normpath(get_workspace_dir())
    tasks_dir = os.path.normpath(os.path.join(workspace_dir, "tasks"))
    norm_path = os.path.normpath(abs_path)

    # Only intercept files directly under a task workspace directory
    if not norm_path.startswith(tasks_dir + os.sep):
        return None

    # Must be output.jsonl or output.txt
    fname = os.path.basename(norm_path)
    if fname not in {"output.jsonl", "output.txt"}:
        return None

    # Extract task_id from path: <tasks_dir>/<task_id>/output.jsonl
    rel = os.path.relpath(norm_path, tasks_dir)
    parts = rel.split(os.sep)
    if len(parts) < 2:
        return None
    task_id = parts[0]

    # Check if result.md exists for this task
    result_path = os.path.join(tasks_dir, task_id, "result.md")
    if os.path.exists(result_path):
        try:
            result_size = os.path.getsize(result_path)
        except OSError:
            result_size = 0
        return (
            f"BLOCKED: This task has result.md ({result_size} bytes). "
            f"Read result.md instead — it contains the definitive outcome "
            f"and is 300x smaller than output.jsonl. "
            f"Path: .aura/workspace/tasks/{task_id}/result.md"
        )

    # No result.md — worker is still running. Trust Phase 2 signals.
    return (
        f"BLOCKED: No result.md yet for {task_id}. This worker's progress "
        f"is tracked by Phase 2 signals (output_delta, content_changed, "
        f"active_score, cpu_percent) in the workspace digest. "
        f"Only read output.jsonl if the digest shows **STUCK** or **LOOPING**."
    )


def impl_read_file(path: str, max_chars: int = 2000, mode: str = "auto",
                   start_line: int = 0, end_line: int = 0) -> str:
    """Read a file and return its contents.

    Tail and head modes stream only the needed portion from disk.
    Full and auto modes use the mtime cache but cap reads at 50 MB.

    Args:
        path: File path relative to project root.
        max_chars: Maximum characters to return. Default 2000.
        mode: 'auto' (head+tail for large files), 'head', 'tail', 'full'.
        start_line: If > 0, start reading from this 1-indexed line.
        end_line: If > 0, stop reading at this 1-indexed line (inclusive).
    """
    error = _other_aura_task_path_error(path, "read")
    if error:
        return error
    abs_path = _resolve_path(path)

    # ── Guard: intercept output.jsonl reads for task workspaces ─────
    # The orchestrator LLM should read result.md for outcomes and trust
    # Phase 2 signals for progress. Raw output.jsonl is 300x larger and
    # rarely contains information the LLM needs for decision-making.
    _guard = _check_output_jsonl_guard(abs_path)
    if _guard:
        return _guard

    # ── tail mode: stream from disk, no full-file load ──────────────
    if mode == "tail" and start_line <= 0 and end_line <= 0:
        text, total_size, err = _read_file_tail_from_disk(abs_path, max_chars)
        if text is None:
            return f"ERROR reading file: {path} ({err})"
        if not text:
            return f"File: {path} (empty)"
        return (
            f"File: {path} ({total_size} chars, tail {max_chars}):\n\n"
            + (f"... [first {total_size - max_chars} chars omitted] ...\n"
               if total_size > max_chars else "")
            + text
        )

    # ── head mode: stream from disk, no full-file load ─────────────
    if mode == "head" and start_line <= 0 and end_line <= 0:
        text, total_size, err = _read_file_head_from_disk(abs_path, max_chars)
        if text is None:
            return f"ERROR reading file: {path} ({err})"
        if not text:
            return f"File: {path} (empty)"
        if total_size <= max_chars:
            return f"File: {path} ({total_size} chars, full):\n\n{text}"
        return (
            f"File: {path} ({total_size} chars, head {max_chars}):\n\n"
            + text
            + f"\n... [{total_size - max_chars} more chars]"
        )

    # ── All other modes: cached read with size cap ─────────────────
    content = cached_read_file(abs_path)
    if content is None:
        return f"ERROR: File not found: {path}"

    total_chars = len(content)

    # Line-range mode (needs full content for line indexing)
    if start_line > 0 or end_line > 0:
        if start_line <= 0:
            start_line = 1
        lines = content.splitlines(True)
        if start_line > len(lines):
            return f"ERROR: start_line {start_line} exceeds file length ({len(lines)} lines)"
        if end_line == 0:
            end_line = len(lines)
        selected = "".join(lines[start_line - 1:end_line])
        if len(selected) > max_chars:
            selected = selected[:max_chars] + (
                f"\n... [truncated {len(selected) - max_chars} chars from line range]"
            )
        return (
            f"File: {path} ({total_chars} chars, {len(lines)} lines, "
            f"lines {start_line}-{min(end_line, len(lines))}):\n\n{selected}"
        )

    # Mode "full" — return the complete file (capped for safety)
    if mode == "full":
        if len(content) > max_chars:
            return (
                f"File: {path} ({total_chars} chars) — full read requested but "
                f"capped at {max_chars} chars. Use a higher max_chars or "
                f"line-range mode for the rest.\n\n"
                + content[:max_chars]
                + f"\n... [{total_chars - max_chars} more chars]"
            )
        return f"File: {path} ({total_chars} chars, full):\n\n{content}"

    # Small file → return full
    if len(content) <= max_chars:
        return content

    # Default: auto mode — show head + tail for large files
    head_size = max_chars // 2
    tail_size = max_chars - head_size
    omitted = total_chars - max_chars
    return (
        f"File: {path} ({total_chars} chars, head+tail {max_chars}):\n\n"
        + content[:head_size]
        + f"\n... [{omitted} chars omitted] ...\n"
        + content[-tail_size:]
    )


def impl_write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    error = _other_aura_task_path_error(path, "write")
    if error:
        return error
    abs_path = _resolve_path(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        invalidate_cache(abs_path)
        return f"OK: Written {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR writing file: {e}"


def impl_list_directory(path: str) -> str:
    """List directory contents (mtime-cached)."""
    error = _other_aura_task_path_error(path, "list")
    if error:
        return error
    abs_path = _resolve_path(path)
    if not os.path.exists(abs_path):
        return f"ERROR: Directory not found: {path}"
    if not os.path.isdir(abs_path):
        return f"ERROR: Not a directory: {path}"
    try:
        entries = cached_list_directory(abs_path)
        if entries is None:
            return f"ERROR: Cannot read directory: {path}"
        items = []
        for entry in sorted(entries):
            entry_path = os.path.join(abs_path, entry)
            suffix = "/" if os.path.isdir(entry_path) else ""
            size = ""
            if os.path.isfile(entry_path):
                size = f" ({os.path.getsize(entry_path)} bytes)"
            items.append(f"  {entry}{suffix}{size}")
        if not items:
            return f"Directory is empty: {path}"
        return f"Contents of {path}:\n" + "\n".join(items)
    except Exception as e:
        return f"ERROR listing directory: {e}"


def _find_task_context(task_id: str, tasks: list[dict], parent: dict | None = None,
                       path: list[dict] | None = None) -> tuple[dict | None, dict | None, list[dict]]:
    """Return the task, its parent, and the root-to-task path."""
    path = path or []
    for task in tasks:
        current_path = [*path, task]
        if task.get("id") == task_id:
            return task, parent, current_path
        found, found_parent, found_path = _find_task_context(
            task_id, task.get("children", []), task, current_path
        )
        if found:
            return found, found_parent, found_path
    return None, None, []


def _short_task_text(text: str, limit: int = 260) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _latest_decision_for(state: dict, task_id: str) -> dict | None:
    for decision in reversed(state.get("decision_log", [])):
        if decision.get("task_id") == task_id:
            return decision
    return None


def _read_task_result_preview(task_id: str, max_chars: int = 900) -> str:
    result_path = os.path.join(get_workspace_dir(), "tasks", task_id, "result.md")
    if not os.path.exists(result_path):
        return ""
    try:
        with open(result_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)
    except OSError:
        return ""
    content = content.strip()
    if len(content) > max_chars:
        content = content[:max_chars].rstrip() + "\n... [truncated]"
    return content


def _format_task_hierarchy_context(task_id: str, state: dict) -> str:
    task, parent, path = _find_task_context(task_id, state.get("tasks", []))
    if not task:
        return "(task context unavailable)"

    path_text = " -> ".join(
        f"{node.get('id')}: {_short_task_text(node.get('description', ''), 90)}"
        for node in path
    )
    parent_text = "(none)"
    if parent:
        parent_text = (
            f"{parent.get('id')}: "
            f"{_short_task_text(parent.get('description', ''), 220)}"
        )

    return "\n".join([
        f"- Path: {path_text}",
        f"- Parent category: {parent_text}",
        (
            "- Executable task level: this worker is assigned at "
            "ROOT -> category -> task level or deeper."
        ),
    ])


def _format_sibling_context(task_id: str, state: dict) -> str:
    task, parent, _path = _find_task_context(task_id, state.get("tasks", []))
    if not task:
        return "(task context unavailable)"
    if not parent:
        return "This task has no parent category in the current tree."

    peers = parent.get("children", [])
    if not peers:
        return "No peer tasks are recorded under this parent category."

    lines = [
        (
            "Use these peer outcomes when planning so you do not repeat failed "
            "attempts and can build on useful evidence."
        )
    ]
    for peer in peers[:12]:
        peer_id = str(peer.get("id", "?"))
        label = " (current task)" if peer_id == task_id else ""
        lines.append(
            f"- {peer_id}{label} [{peer.get('status', 'pending')}]: "
            f"{_short_task_text(peer.get('description', ''), 240)}"
        )
        acceptance = peer.get("acceptance_criteria")
        if acceptance:
            lines.append(f"  Acceptance: {_short_task_text(acceptance, 220)}")

        decision = _latest_decision_for(state, peer_id)
        if decision:
            lines.append(
                "  Latest decision: {old} -> {new}; reason: {reason}; evidence: {evidence}".format(
                    old=decision.get("old_status", "?"),
                    new=decision.get("new_status", "?"),
                    reason=_short_task_text(decision.get("reason", ""), 180),
                    evidence=_short_task_text(decision.get("evidence", ""), 180),
                )
            )
        elif peer.get("evidence") or peer.get("reason"):
            lines.append(
                "  Recorded outcome: reason={reason}; evidence={evidence}".format(
                    reason=_short_task_text(peer.get("reason", ""), 180),
                    evidence=_short_task_text(peer.get("evidence", ""), 180),
                )
            )
        else:
            lines.append("  Recorded outcome: not executed yet.")

        result_preview = _read_task_result_preview(peer_id)
        if result_preview:
            lines.append("  result.md preview:")
            lines.append("  ```markdown")
            lines.extend(f"  {line}" for line in result_preview.splitlines())
            lines.append("  ```")

    if len(peers) > 12:
        lines.append(f"- ... {len(peers) - 12} more peer task(s) omitted.")
    return "\n".join(lines)


def impl_spawn_task(task_id: str, description: str, budget_minutes: int = 30) -> str:
    """Spawn a Layer 2 Claude Code worker."""
    from .config import DEFAULT_MAX_TURNS, MAX_CONCURRENT_TASKS

    state = state_mgr.load_state()
    task = state_mgr.find_task(task_id, state.get("tasks", []))
    if task is None:
        return (
            f"ERROR: Task {task_id} not found in the current task tree. "
            "Use only task IDs shown in the current state snapshot."
        )
    if int(task.get("depth", 0)) < MIN_EXECUTABLE_TASK_DEPTH:
        return (
            f"ERROR: Task {task_id} is a planning/category node at depth "
            f"{task.get('depth', 0)}. Workers may only be spawned for concrete "
            "tasks at ROOT -> category -> task level or deeper, such as A1.1. "
            f"First decompose {task_id} into child tasks."
        )
    if task.get("status") in {"completed", "archived"}:
        return f"ERROR: Task {task_id} is {task.get('status')} and cannot be spawned."

    project_context = state.get("project_context", {}) or {}
    project_context_text = "\n".join([
        f"- Final goal: {project_context.get('final_goal') or '(not set)'}",
        f"- Success criteria: {project_context.get('success_criteria') or '(not set)'}",
        f"- Global constraints: {project_context.get('global_constraints') or '(not set)'}",
        f"- Execution environment: {project_context.get('execution_environment') or '(not set)'}",
        f"- Notes: {project_context.get('notes') or '(not set)'}",
    ])
    hierarchy_context_text = _format_task_hierarchy_context(task_id, state)
    sibling_context_text = _format_sibling_context(task_id, state)

    # ── Guard: check actual running worker count ─────────────────────
    running = [w for w in process_mgr.list_all() if w.get("running")]
    if len(running) >= MAX_CONCURRENT_TASKS:
        running_ids = [w["task_id"] for w in running]
        return (
            f"ERROR: Already at max concurrent workers ({MAX_CONCURRENT_TASKS}). "
            f"Running: {running_ids}. Kill a stuck task first or wait for one to complete."
        )

    ws_dir = get_workspace_dir()
    task_dir = os.path.join(ws_dir, "tasks", task_id)
    os.makedirs(task_dir, exist_ok=True)

    # Write task definition
    # IMPORTANT: Always overwrite task.md with the CORRECT description.
    # Previous bugs occurred when stale task.md files (from failed prior
    # attempts) were left in place, causing workers to execute wrong tasks.
    # By always writing fresh content here, we ensure the worker always
    # receives the orchestrator's INTENDED task, not leftover garbage.
    task_md_path = os.path.join(task_dir, "task.md")
    task_content = f"""# Task {task_id}

{description}

## Project Context
{project_context_text}

Project Context is background and constraints only. Your assigned work is the
specific Task {task_id} above. Do not attempt the whole final goal unless this
task explicitly asks you to do so.

## Task Hierarchy
{hierarchy_context_text}

Tasks are intentionally hierarchical: ROOT is the final goal, ROOT children
are broad planning categories, and concrete worker tasks live under those
categories. Treat sibling task outcomes as shared local context for this
category.

## Sibling Tasks Context
{sibling_context_text}

Do not inspect other task-file data directories under `.aura` for state,
progress, workspace outputs, summaries, caches, or task metadata. Other task
directories' memory files may be used only as transferable lessons, not as
current-task evidence.

## Constraints
- Budget: {budget_minutes} minutes
- Max turns: {DEFAULT_MAX_TURNS}
- Current working directory: {task_dir}
- This current directory is the task output directory. Put all outputs here.

## Process Safety Rule
Do not use `taskkill /IM python.exe`, `taskkill /F /IM python.exe`, `pkill python`,
or `killall python`. Only terminate subprocesses started by the current task itself.

## Hard Resource Policy
{process_mgr.resource_policy_text()}

## Progress Tracking (output.jsonl)
- The orchestrator monitors your output.jsonl for progress detection.
- This file is automatically written by the CLI with --output-format=stream-json.
- For long-running tasks (computation, downloads, training), the orchestrator uses THREE signals:
  1. File size growth (new lines written to output.jsonl)
  2. Content change (hash of last 40 lines — different lines = progress)
  3. CPU usage (> 0.5% = process is actively computing, NOT stuck)
- If you have a long computation without log output, write a status line to stderr
  or make a small tool call (e.g. read_file on a small file) to generate log activity.
- The orchestrator will NOT kill you just because output is slow — it checks CPU
  to distinguish "computing" from "dead zombie".

## Output Requirements
- When done, write a brief result summary to: result.md in the current directory.
- List all created files and what they contain.
"""
    with open(task_md_path, "w", encoding="utf-8") as f:
        f.write(task_content)

    spawn_result = process_mgr.spawn(task_id, task_dir, task_md_path, budget_minutes)
    if not spawn_result.startswith("OK:"):
        return spawn_result

    # ── Auto-sync task tree: mark as in_progress ───────────────────
    # The LLM no longer needs to call update_task_tree separately;
    # spawning a worker automatically transitions the task status.
    try:
        state_mgr.update_task(
            task_id, "in_progress",
            f"Worker spawned (PID in process registry, budget={budget_minutes}min)",
            f"process_mgr.spawn({task_id})"
        )
    except Exception as e:
        spawn_result += f"\n(Warning: task tree status not updated: {e})"

    return spawn_result


def impl_kill_task(task_id: str) -> str:
    """Kill a running Layer 2 worker and auto-update task tree status."""
    result = process_mgr.kill(task_id)
    try:
        state_mgr.update_task(task_id, "killed", "Worker killed by orchestrator", "process_mgr.kill")
    except Exception as e:
        result += f"\n(Warning: task tree status not updated: {e})"
    return result


def impl_list_running_tasks() -> str:
    """List all running Layer 2 workers."""
    tasks = process_mgr.list_all()
    if not tasks:
        return "No running tasks."
    lines = [f"Running tasks ({len(tasks)}):"]
    for t in tasks:
        gpu = "n/a" if t.get("gpu_memory_mb") is None else f"{t['gpu_memory_mb']}MB"
        gpu_avg = "n/a" if t.get("avg_gpu_memory_percent") is None else f"{t['avg_gpu_memory_percent']}%"
        gpu_util = "n/a" if t.get("avg_gpu_util_percent") is None else f"{t['avg_gpu_util_percent']}%"
        violation = f" | Resource violation: {t['resource_violation']}" if t.get("resource_violation") else ""
        lines.append(f"  {t['task_id']} | PID: {t['pid']} | "
                      f"Running: {t['running']} | "
                      f"Elapsed: {t['elapsed_minutes']}min | "
                      f"CPU avg/current: {t.get('avg_cpu_percent', 0)}%/{t.get('cpu_percent', 0)}% | "
                      f"RAM avg/current: {t.get('avg_memory_percent', 0)}%/{t.get('memory_percent', 0)}% | "
                      f"GPU mem avg/current: {gpu_avg}/{gpu} | "
                      f"GPU util avg: {gpu_util} | "
                      f"Output size: {t['output_size']} bytes"
                      f"{violation}")
    return "\n".join(lines)


def impl_update_task_tree(task_id: str, new_status: str, reason: str, evidence: str) -> str:
    """Update a task node in the task tree."""
    from .config import MAX_CONCURRENT_TASKS

    # ── Guard: prevent marking too many tasks as in_progress ────────
    if new_status == "in_progress":
        currently_active = state_mgr.count_active_tasks()
        # Check if THIS task is already in_progress (re-entry / idempotent)
        state = state_mgr.load_state()
        existing = state_mgr.find_task(task_id, state["tasks"])
        already_in_progress = existing and existing.get("status") == "in_progress"
        if existing and not already_in_progress and int(existing.get("depth", 0)) < MIN_EXECUTABLE_TASK_DEPTH:
            return (
                f"ERROR: Cannot mark {task_id} as in_progress because it is a "
                "planning/category node. Decompose it into ROOT -> category -> "
                "task level first, then spawn or start a concrete child such as A1.1."
            )
        if not already_in_progress and currently_active >= MAX_CONCURRENT_TASKS:
            return (
                f"ERROR: Cannot mark {task_id} as in_progress — already at max "
                f"concurrent tasks ({MAX_CONCURRENT_TASKS}). "
                f"Use spawn_task instead (which auto-marks the status), "
                f"or complete/kill an existing active task first."
            )

    return state_mgr.update_task(task_id, new_status, reason, evidence)


def impl_decompose_task(parent_task_id: str, subtasks: list[dict]) -> str:
    """Decompose a task into subtasks."""
    return state_mgr.decompose_task(parent_task_id, subtasks)


def impl_update_project_context(
    final_goal: str = "",
    success_criteria: str = "",
    global_constraints: str = "",
    execution_environment: str = "",
    notes: str = "",
) -> str:
    """Update project-level context extracted by the orchestrator LLM."""
    return state_mgr.update_project_context(
        final_goal=final_goal,
        success_criteria=success_criteria,
        global_constraints=global_constraints,
        execution_environment=execution_environment,
        notes=notes,
    )


def impl_write_memory(mem_type: str, content: str) -> str:
    """Write to long-term memory."""
    return memory_mgr.append_memory(mem_type, content)


def impl_web_fetch(url: str, max_chars: int = 5000) -> str:
    """Fetch a URL and return plain text content."""
    if not url.startswith(("http://", "https://")):
        return "ERROR: URL must start with http:// or https://"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 AuraAgent/1.0",
                "Accept": "text/html,text/plain,*/*",
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()

            # Try to decode
            charset = "utf-8"
            for part in content_type.split(";"):
                part = part.strip()
                if part.lower().startswith("charset="):
                    charset = part.split("=", 1)[1].strip()
                    break

            try:
                html = raw.decode(charset)
            except (UnicodeDecodeError, LookupError):
                html = raw.decode("utf-8", errors="replace")

        # Strip HTML tags
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&quot;", '"', text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Remove excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [TRUNCATED]"

        return f"Fetched {url} ({len(text)} chars of text):\n\n{text}"

    except urllib.error.HTTPError as e:
        return f"ERROR: HTTP {e.code} {e.reason} for {url}"
    except urllib.error.URLError as e:
        return f"ERROR: Failed to connect to {url}: {e.reason}"
    except Exception as e:
        return f"ERROR fetching {url}: {e}"


def impl_no_op(reason: str, next_check_focus: str = "") -> str:
    """No operation - just log."""
    focus_msg = f" Next check focus: {next_check_focus}" if next_check_focus else ""
    return f"Sleeping. Reason: {reason}.{focus_msg}"


# Tool dispatch table
TOOL_IMPLS = {
    "read_file": impl_read_file,
    "write_file": impl_write_file,
    "list_directory": impl_list_directory,
    "web_fetch": impl_web_fetch,
    "spawn_task": impl_spawn_task,
    "kill_task": impl_kill_task,
    "list_running_tasks": impl_list_running_tasks,
    "update_task_tree": impl_update_task_tree,
    "decompose_task": impl_decompose_task,
    "update_project_context": impl_update_project_context,
    "write_memory": impl_write_memory,
    "no_op": impl_no_op,
}


def execute_tool(name: str, params: dict[str, Any]) -> str:
    """Execute a tool by name with given parameters. Returns result string."""
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return f"ERROR: Unknown tool: {name}"
    try:
        return impl(**params)
    except Exception as e:
        return f"ERROR executing tool '{name}': {e}"
