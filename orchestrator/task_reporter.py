"""Task completion summary reporter (R3).

Generates a structured summary report when a Layer 2 task completes,
capturing: completion status, key implementation details, files changed,
and lessons learned. Reports are saved to summaries/ for historical reference.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import TASK_SUMMARY_DIR, TASK_SUMMARY_ENABLED, get_workspace_dir

FINAL_REPORT_NAME = "final_report.md"


def generate_task_summary(task_id: str, status: str, reason: str = "",
                          evidence: str = "", extra: Optional[dict] = None) -> str:
    """Generate a completion summary report for a finished task.

    Args:
        task_id: The task ID (e.g. 'A1.1', 'A1.2', 'B2.1').
        status: Final status ('completed', 'failed', 'killed', 'archived').
        reason: Why the task ended.
        evidence: Verifiable evidence (file paths, test results).
        extra: Optional dict with additional details like:
               - files_created: list of file paths
               - files_modified: list of file paths
               - key_decisions: list of strings
               - lessons: list of strings
               - duration_minutes: float

    Returns:
        Path to the saved summary report, or empty string if disabled.
    """
    if not TASK_SUMMARY_ENABLED:
        return ""

    extra = extra or {}

    # Gather evidence from task workspace
    task_dir = os.path.join(get_workspace_dir(), "tasks", task_id)
    files_created = extra.get("files_created", [])
    files_modified = extra.get("files_modified", [])

    # Auto-discover artifacts if not provided
    if not files_created and os.path.isdir(task_dir):
        known = {"output.jsonl", "error.log", "task.md", "process.json", "result.md"}
        for fname in sorted(os.listdir(task_dir)):
            if fname not in known and os.path.isfile(os.path.join(task_dir, fname)):
                files_created.append(os.path.join("workspace/tasks", task_id, fname))

    # Read result.md if exists
    result_text = ""
    result_path = os.path.join(task_dir, "result.md")
    if os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result_text = f.read()[:3000]
        except Exception:
            pass

    # Read output size
    output_size = 0
    output_path = os.path.join(task_dir, "output.jsonl")
    if os.path.exists(output_path):
        output_size = os.path.getsize(output_path)

    # Read error log
    error_text = ""
    error_path = os.path.join(task_dir, "error.log")
    if os.path.exists(error_path):
        try:
            error_size = os.path.getsize(error_path)
            if error_size > 0:
                with open(error_path, "r", encoding="utf-8", errors="replace") as f:
                    error_text = f.read()[:1000]
        except Exception:
            pass

    # Build summary
    status_icon = {"completed": "✅", "failed": "❌", "killed": "💀", "archived": "📦"}.get(status, "❓")

    lines = []
    lines.append(f"# Task Summary: {task_id} {status_icon}")
    lines.append(f"")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Status**: {status}")
    lines.append(f"")
    lines.append(f"## Completion Details")
    lines.append(f"")
    lines.append(f"- **Reason**: {reason or 'N/A'}")
    lines.append(f"- **Evidence**: {evidence or 'N/A'}")
    lines.append(f"- **Output size**: {output_size:,} bytes")
    lines.append(f"")

    duration = extra.get("duration_minutes")
    if duration is not None:
        lines.append(f"- **Duration**: {duration:.1f} minutes")
    budget = extra.get("budget_minutes")
    if budget is not None:
        lines.append(f"- **Budget**: {budget} minutes")

    lines.append(f"")

    # Files created
    if files_created:
        lines.append(f"## Files Created ({len(files_created)})")
        lines.append(f"")
        for fp in files_created:
            lines.append(f"- `{fp}`")
        lines.append(f"")

    # Files modified
    if files_modified:
        lines.append(f"## Files Modified ({len(files_modified)})")
        lines.append(f"")
        for fp in files_modified:
            lines.append(f"- `{fp}`")
        lines.append(f"")

    # Key decisions
    key_decisions = extra.get("key_decisions", [])
    if key_decisions:
        lines.append(f"## Key Implementation Details")
        lines.append(f"")
        for i, kd in enumerate(key_decisions, 1):
            lines.append(f"{i}. {kd}")
        lines.append(f"")

    # Lessons
    lessons = extra.get("lessons", [])
    if lessons:
        lines.append(f"## Lessons Learned")
        lines.append(f"")
        for lesson in lessons:
            lines.append(f"- {lesson}")
        lines.append(f"")

    # Result excerpt
    if result_text:
        lines.append(f"## Result.md Excerpt")
        lines.append(f"")
        lines.append(f"```")
        lines.append(result_text[:2000])
        if len(result_text) > 2000:
            lines.append(f"... (truncated, full result: {len(result_text)} chars)")
        lines.append(f"```")
        lines.append(f"")

    # Errors
    if error_text:
        lines.append(f"## Error Log")
        lines.append(f"")
        lines.append(f"```")
        lines.append(error_text)
        lines.append(f"```")
        lines.append(f"")

    lines.append(f"---")
    lines.append(f"*Auto-generated by Aura Task Reporter (R3)*")

    content = "\n".join(lines)

    # Save to summaries directory
    os.makedirs(TASK_SUMMARY_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(TASK_SUMMARY_DIR, f"{task_id}_{timestamp}.md")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)

    return summary_path


def generate_final_report(state: dict) -> str:
    """Generate a rolling project-level completion report.

    The report is updated whenever progress.md is rendered. It is intentionally
    rolling rather than one-shot: if the user completes batch A, then edits the
    task file and creates batch B, the report shows both the completed batch and
    the new outstanding work in one place.
    """
    if not TASK_SUMMARY_ENABLED:
        return ""

    os.makedirs(TASK_SUMMARY_DIR, exist_ok=True)
    report_path = os.path.join(TASK_SUMMARY_DIR, FINAL_REPORT_NAME)

    tasks = _flatten_tasks(state.get("tasks", []))
    non_root = [task for task in tasks if task.get("id") != "root"]
    current_tasks = [
        task for task in non_root
        if task.get("status") not in {"archived", "killed"}
    ]
    open_tasks = [
        task for task in current_tasks
        if task.get("status") != "completed"
    ]
    completed_tasks = [
        task for task in current_tasks
        if task.get("status") == "completed"
    ]
    archived_tasks = [
        task for task in non_root
        if task.get("status") in {"archived", "killed"}
    ]

    project_done = bool(current_tasks) and not open_tasks
    status_label = "complete" if project_done else "in_progress"
    project_context = state.get("project_context", {}) or {}

    lines = [
        "# Final Report",
        "",
        f"**Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Project status**: {status_label}",
        f"**Task file**: {state.get('task_file') or '(not recorded)'}",
        "",
        "## Mission",
        "",
        state.get("mission") or "(not set)",
        "",
        "## Project Context",
        "",
        f"- Final goal: {project_context.get('final_goal') or '(not set)'}",
        f"- Success criteria: {project_context.get('success_criteria') or '(not set)'}",
        f"- Global constraints: {project_context.get('global_constraints') or '(not set)'}",
        f"- Execution environment: {project_context.get('execution_environment') or '(not set)'}",
        "",
        "## Current Snapshot",
        "",
        "| Item | Count |",
        "|---|---:|",
        f"| Current completed tasks | {len(completed_tasks)} |",
        f"| Current open tasks | {len(open_tasks)} |",
        f"| Archived or removed tasks | {len(archived_tasks)} |",
        f"| Wake cycles | {state.get('total_cycles', 0)} |",
        "",
    ]

    if project_done:
        lines.extend([
            "## Completion Status",
            "",
            "All current non-archived tasks are completed.",
            "",
        ])
    else:
        lines.extend([
            "## Completion Status",
            "",
            "Current requirements are not fully complete yet.",
            "",
        ])

    lines.extend(_render_batches(non_root))
    lines.extend(_render_decision_timeline(state.get("decision_log", [])))

    lines.extend([
        "---",
        "*Auto-generated by Aura Task Reporter.*",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def _flatten_tasks(tasks: list, depth: int = 0) -> list[dict]:
    flattened = []
    for task in tasks:
        item = dict(task)
        item["_depth"] = depth
        flattened.append(item)
        flattened.extend(_flatten_tasks(task.get("children", []), depth + 1))
    return flattened


def _batch_key(task_id: str) -> str:
    prefix = ""
    for char in str(task_id):
        if char.isalpha():
            prefix += char
        else:
            break
    return prefix or "other"


def _short(text: str, limit: int = 120) -> str:
    clean = " ".join(str(text or "").replace("|", "\\|").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _render_task_rows(tasks: list[dict]) -> list[str]:
    rows = ["| Task | Status | Summary | Evidence |", "|---|---|---|---|"]
    for task in tasks:
        rows.append(
            "| {task_id} | {status} | {summary} | {evidence} |".format(
                task_id=task.get("id", "?"),
                status=task.get("status", "pending"),
                summary=_short(task.get("description", "")),
                evidence=_short(task.get("evidence", ""), 80),
            )
        )
    return rows


def _render_batches(tasks: list[dict]) -> list[str]:
    if not tasks:
        return ["## Requirement Batches", "", "No planned tasks yet.", ""]

    batches: dict[str, list[dict]] = {}
    for task in tasks:
        batches.setdefault(_batch_key(task.get("id", "")), []).append(task)

    lines = ["## Requirement Batches", ""]
    for batch in sorted(batches):
        batch_tasks = sorted(
            batches[batch],
            key=lambda item: (item.get("_depth", 0), str(item.get("id", ""))),
        )
        current = [
            task for task in batch_tasks
            if task.get("status") not in {"archived", "killed"}
        ]
        completed = [task for task in current if task.get("status") == "completed"]
        open_tasks = [task for task in current if task.get("status") != "completed"]
        removed = [
            task for task in batch_tasks
            if task.get("status") in {"archived", "killed"}
        ]
        batch_done = bool(current) and not open_tasks

        lines.extend([
            f"### Batch {batch}",
            "",
            f"Status: {'complete' if batch_done else 'in_progress'}",
            f"Current completion: {len(completed)}/{len(current)}",
            "",
        ])

        if completed:
            lines.extend(["Completed:", ""])
            lines.extend(_render_task_rows(completed))
            lines.append("")

        if open_tasks:
            lines.extend(["Open:", ""])
            lines.extend(_render_task_rows(open_tasks))
            lines.append("")

        if removed:
            lines.extend(["Archived or removed:", ""])
            lines.extend(_render_task_rows(removed))
            lines.append("")

    return lines


def _render_decision_timeline(decision_log: list[dict]) -> list[str]:
    relevant = [
        entry for entry in decision_log
        if entry.get("new_status") in {"completed", "archived", "killed", "failed"}
    ][-30:]
    lines = ["## Recent Completion Timeline", ""]
    if not relevant:
        lines.extend(["No completion events recorded yet.", ""])
        return lines

    lines.extend(["| Time | Task | Change | Reason |", "|---|---|---|---|"])
    for entry in relevant:
        lines.append(
            "| {time} | {task} | {old} -> {new} | {reason} |".format(
                time=str(entry.get("time", ""))[:19],
                task=entry.get("task_id", "?"),
                old=entry.get("old_status", "?"),
                new=entry.get("new_status", "?"),
                reason=_short(entry.get("reason", ""), 80),
            )
        )
    lines.append("")
    return lines


def list_summaries() -> list[dict]:
    """List all task summaries, sorted by most recent first."""
    if not os.path.isdir(TASK_SUMMARY_DIR):
        return []

    summaries = []
    for fname in sorted(os.listdir(TASK_SUMMARY_DIR), reverse=True):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(TASK_SUMMARY_DIR, fname)
        stat = os.stat(fpath)
        # Parse task_id and timestamp from filename: A1_20260504_220000.md
        parts = fname.replace(".md", "").split("_", 1)
        task_id = parts[0] if parts else "?"
        summaries.append({
            "filename": fname,
            "path": fpath,
            "task_id": task_id,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })

    return summaries
