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


def generate_task_summary(task_id: str, status: str, reason: str = "",
                          evidence: str = "", extra: Optional[dict] = None) -> str:
    """Generate a completion summary report for a finished task.

    Args:
        task_id: The task ID (e.g. 'T1', 'T2.1').
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
        # Parse task_id and timestamp from filename: T1_20260504_220000.md
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
