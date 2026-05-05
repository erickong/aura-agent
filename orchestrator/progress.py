"""Progress report renderer for progress.md.

Renders the decision hub file that serves both human readability and
the orchestrator's own decision-making context.

OPTIMIZATION (T0): on-demand rendering. Uses a content hash of the task tree
and decision log to detect meaningful changes. If only the cycle counter or
timestamp changed without any structural state mutation, re-rendering is
skipped — saving disk I/O and Jinja2 template processing.
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from .config import STATE_DIR, PROJECT_ROOT
from . import state as state_mgr
from . import memory as memory_mgr

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
TEMPLATE_PATH = os.path.join(TEMPLATES_DIR, "progress.template.md")
PROGRESS_PATH = os.path.join(STATE_DIR, "progress.md")

# ── T0 optimization: on-demand rendering cache ─────────────────────────
# _last_progress_hash: content hash of the task tree + decision log from
# the last render. If unchanged, render is skipped.
# _last_template_mtime: template mtime at last render. If changed, re-render.
_last_progress_hash: str = ""
_last_template_mtime: float = 0.0


def _state_content_hash(state: dict) -> str:
    """Compute a hash of the meaningful state parts (task tree + decisions).

    Excludes volatile fields like cycle counter and updated_at that change
    every wake regardless of whether anything meaningful happened.
    """
    parts = json.dumps(
        [state.get("tasks", []), state.get("decision_log", [])[-20:]],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def render_progress() -> str:
    """Render the full 进展.md from current state and return the content.

    Skip re-rendering (T0 optimization) if neither the state content
    (task tree + decisions) nor the template file has changed since
    the last render.
    """
    global _last_progress_hash, _last_template_mtime

    state = state_mgr.load_state()

    # ── T0: Check if rendering is actually needed ─────────────────────
    current_hash = _state_content_hash(state)
    template_mtime = os.path.getmtime(TEMPLATE_PATH) if os.path.exists(TEMPLATE_PATH) else 0.0

    if (current_hash == _last_progress_hash and
            template_mtime == _last_template_mtime and
            os.path.exists(PROGRESS_PATH)):
        # Nothing meaningful changed — re-read the existing file content
        # rather than re-rendering, but DON'T write (avoids disk I/O).
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return f.read()

    # Update cache before rendering
    _last_progress_hash = current_hash
    _last_template_mtime = template_mtime

    # ── Render ────────────────────────────────────────────────────
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    template = env.get_template("progress.template.md")

    tasks = state.get("tasks", [])
    active_tasks = state_mgr._collect_in_progress(tasks)
    decision_log = state.get("decision_log", [])
    last_decisions = decision_log[-20:] if decision_log else []

    # Calculate indicators
    total_completed = _count_by_status(tasks, "completed")
    total_failed = _count_by_status(tasks, "failed")
    total_blocked = _count_by_status(tasks, "blocked")

    # Calculate total time
    created_at = state.get("created_at", "")
    total_hours = 0
    if created_at:
        try:
            created_dt = datetime.fromisoformat(created_at)
            delta = datetime.now() - created_dt
            total_hours = round(delta.total_seconds() / 3600, 1)
        except (ValueError, TypeError):
            pass

    content = template.render(
        mission=state.get("mission", "未设定"),
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_cycles=state.get("total_cycles", 0),
        total_hours=total_hours,
        total_completed=total_completed,
        total_failed=total_failed,
        total_blocked=total_blocked,
        active_tasks=active_tasks,
        task_tree=_render_task_tree_flat(tasks),
        last_decisions=last_decisions,
        replan_count=state.get("replan_count", 0),
        active_count=len(active_tasks),
    )

    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    return content


def _count_by_status(tasks: list, status: str) -> int:
    """Recursively count tasks with a given status."""
    count = 0
    for task in tasks:
        if task.get("status") == status:
            count += 1
        if "children" in task and task["children"]:
            count += _count_by_status(task["children"], status)
    return count


def _render_task_tree_flat(tasks: list, depth: int = 0) -> list[dict]:
    """Render task tree as a flat list for template rendering."""
    result = []
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
        item = {
            "depth": depth,
            "id": task.get("id", "?"),
            "description": task.get("description", "")[:100],
            "status": task.get("status", "pending"),
            "icon": status_icons.get(task.get("status", "pending"), "❓"),
            "indent": "  " * depth + ("└─" if depth > 0 else ""),
            "has_children": bool(task.get("children")),
        }
        result.append(item)
        if "children" in task and task["children"]:
            result.extend(_render_task_tree_flat(task["children"], depth + 1))
    return result
