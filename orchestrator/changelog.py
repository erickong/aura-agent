"""Track task.md change history and avoid repeated work.

Core idea:
- Each task.md has a corresponding changelog file under the project directory.
- The changelog records detected content hashes and processing state.
- Startup compares the current hash with the last recorded hash, then only
  processes new or changed content.
- Repeated edits to task.md should not cause already completed work to run again.

Per-wake task file change detection (R7):
- Each wake checks the task file mtime first, then computes a full diff only
  when the timestamp changed.
- The diff is injected into orchestrator context to help decide whether
  replanning is needed.
- Newly added user context can be recorded to memory.

File structure:
  projects/{project_name}/changelog/
    self_upgrade.md.json      # Change record for tasks/self_upgrade.md.
    stockagent.md.json        # Change record for tasks/stockagent.md.
    self_upgrade.snapshot.md  # Last processed task-file snapshot for diffing.
"""

import difflib
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def _project_name_from_task_file(task_file: str) -> str:
    """Derive a stable project name from a task file path.

    Use the file basename without extension so tasks/self_upgrade.md and
    tasks\\self_upgrade.md map to the same project name, "self_upgrade".
    """
    normalized = task_file.replace("\\", "/")
    name = os.path.splitext(os.path.basename(normalized))[0]
    return name.replace(" ", "_").lower()


def get_changelog_dir(projects_dir: str, project_name: str) -> str:
    """Return the changelog directory path."""
    return os.path.join(projects_dir, project_name, "changelog")


def get_changelog_path(projects_dir: str, project_name: str, task_file: str) -> str:
    """Return the changelog file path.

    The task-file basename is used so slash style differences still map to
    the same changelog.
    """
    normalized = task_file.replace("\\", "/")
    basename = os.path.basename(normalized)
    if not basename.endswith(".json"):
        basename = os.path.splitext(basename)[0] + ".json"
    return os.path.join(get_changelog_dir(projects_dir, project_name), basename)


def compute_file_hash(file_path: str) -> str:
    """Compute a file SHA-256 hash."""
    path = Path(file_path)
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_changelog(changelog_path: str) -> dict:
    """Load a changelog file."""
    path = Path(changelog_path)
    if not path.exists():
        return {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "entries": [],
            "processed_hashes": {},
            "processed_items": {},
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("entries", [])
        data.setdefault("processed_hashes", {})
        data.setdefault("processed_items", {})
        return data
    except (json.JSONDecodeError, KeyError):
        return {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "entries": [],
            "processed_hashes": {},
            "processed_items": {},
        }


def save_changelog(changelog_path: str, changelog: dict) -> None:
    """Save a changelog file."""
    changelog["updated_at"] = datetime.now().isoformat()
    path = Path(changelog_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(changelog, ensure_ascii=False, indent=2), encoding="utf-8")


def get_file_change_info(task_file: str, projects_dir: str, project_name: str) -> dict:
    """Return task.md change metadata.

    Returns:
    {
        "is_new": True/False,
        "is_changed": True/False,
        "previous_hash": "...",
        "current_hash": "...",
        "last_processed_at": "...",
    }
    """
    changelog_path = get_changelog_path(projects_dir, project_name, task_file)
    changelog = load_changelog(changelog_path)
    current_hash = compute_file_hash(task_file)

    result = {
        "is_new": True,
        "is_changed": False,
        "previous_hash": "",
        "current_hash": current_hash,
        "last_processed_at": "",
        "changelog_path": changelog_path,
        "changelog": changelog,
    }

    if not current_hash:
        return result

    if changelog["processed_hashes"]:
        result["is_new"] = False
        if current_hash in changelog["processed_hashes"]:
            entry_idx = changelog["processed_hashes"][current_hash]
            if entry_idx < len(changelog["entries"]):
                entry = changelog["entries"][entry_idx]
                result["last_processed_at"] = entry.get("processed_at", "")
            result["is_changed"] = False
        else:
            result["is_changed"] = True
            if changelog["entries"]:
                result["previous_hash"] = changelog["entries"][-1].get("file_hash", "")

    return result


def mark_file_processed(task_file: str, projects_dir: str, project_name: str,
                        summary: str = "") -> dict:
    """Mark a task.md file as processed.

    Records the current file hash and processing summary in the changelog.
    """
    changelog_path = get_changelog_path(projects_dir, project_name, task_file)
    changelog = load_changelog(changelog_path)
    current_hash = compute_file_hash(task_file)

    entry = {
        "processed_at": datetime.now().isoformat(),
        "file_hash": current_hash,
        "file_path": task_file,
        "summary": summary,
        "items_count": 0,
    }

    changelog["entries"].append(entry)
    entry_idx = len(changelog["entries"]) - 1
    changelog["processed_hashes"][current_hash] = entry_idx

    save_changelog(changelog_path, changelog)

    return {
        "entry_index": entry_idx,
        "file_hash": current_hash,
    }


def mark_item_processed(task_file: str, projects_dir: str, project_name: str,
                        item_fingerprint: str, item_text: str) -> None:
    """Mark a task.md item as processed."""
    changelog_path = get_changelog_path(projects_dir, project_name, task_file)
    changelog = load_changelog(changelog_path)

    changelog["processed_items"][item_fingerprint] = {
        "text": item_text,
        "processed_at": datetime.now().isoformat(),
    }

    save_changelog(changelog_path, changelog)


def is_item_processed(task_file: str, projects_dir: str, project_name: str,
                      item_fingerprint: str) -> bool:
    """Check whether an item has already been processed."""
    changelog_path = get_changelog_path(projects_dir, project_name, task_file)
    changelog = load_changelog(changelog_path)
    return item_fingerprint in changelog.get("processed_items", {})


def get_unprocessed_items(task_file: str, projects_dir: str, project_name: str,
                          items: list[dict]) -> list[dict]:
    """Filter an items list down to items that have not been processed.

    Args:
        task_file: Task file path.
        projects_dir: Project root directory.
        project_name: Project name.
        items: Items returned by parse_task_items().

    Returns:
        Unprocessed item list.
    """
    return [
        it for it in items
        if not is_item_processed(task_file, projects_dir, project_name, it["fingerprint"])
    ]


def get_project_name_for_task(task_file: str) -> str:
    """Return the stable project name for a task file path.

    The basename without path or extension is used consistently with
    orchestrator/main.py.
    """
    return _project_name_from_task_file(task_file)


def cleanup_orphan_projects(projects_dir: str, active_task_files: list[str]) -> list[str]:
    """Return project directories whose task.md file is no longer present.

    Args:
        projects_dir: Project root directory.
        active_task_files: Currently present task.md file paths.

    Returns:
        Project names that can be cleaned.
    """
    active_names = set()
    for tf in active_task_files:
        name = get_project_name_for_task(tf)
        active_names.add(name)

    cleaned = []
    if not os.path.exists(projects_dir):
        return cleaned

    for name in os.listdir(projects_dir):
        pdir = os.path.join(projects_dir, name)
        if not os.path.isdir(pdir):
            continue
        if not os.path.exists(os.path.join(pdir, "state", "state.json")):
            continue
        if name not in active_names:
            cleaned.append(name)

    return cleaned


# R7: Per-wake task file change detection.
#
# Each wake cycle, the orchestrator checks whether the task file was modified
# using mtime first. Only when the mtime differs from the last recorded value
# do we compute a full diff against the stored snapshot.
#
# The diff is analysed with simple heuristics to answer:
#   1. Did the user add/change/remove requirements? This may need replanning.
#   2. Did the user provide new context or information? This may help progress.
#   3. Should this be recorded to memory?
#
# Flow:
#   check_task_file_on_wake()           called every cycle (fast mtime check)
#   save_task_file_snapshot()           called after processing changes
#   get_task_file_snapshot_path()       internal helper


def get_task_file_snapshot_path(projects_dir: str, project_name: str,
                                 task_file: str) -> str:
    """Path to the stored snapshot of the task file (for diff comparison)."""
    normalized = task_file.replace("\\", "/")
    basename = os.path.basename(normalized)
    snapshot_name = os.path.splitext(basename)[0] + ".snapshot.md"
    return os.path.join(get_changelog_dir(projects_dir, project_name), snapshot_name)


def check_task_file_on_wake(task_file: str, projects_dir: str,
                             project_name: str) -> dict:
    """Check whether the task file was modified since the last wake.

    Uses mtime first (cheap). Only when the mtime differs from the recorded
    value do we compute a full content diff via difflib.

    Args:
        task_file: Absolute path to the task .md file.
        projects_dir: Root projects directory.
        project_name: Project name (derived from task basename).

    Returns:
        {
            "changed": bool,
            "mtime_changed": bool,
            "content_changed": bool,
            "previous_mtime": float | None,
            "current_mtime": float | None,
            "diff_lines": list[str],
            "added_requirement_lines": list[str],
            "removed_requirement_lines": list[str],
            "added_info_lines": list[str],
            "change_summary": str,
        }
    """
    path = Path(task_file)
    result = {
        "changed": False,
        "mtime_changed": False,
        "content_changed": False,
        "previous_mtime": None,
        "current_mtime": None,
        "diff_lines": [],
        "added_requirement_lines": [],
        "removed_requirement_lines": [],
        "added_info_lines": [],
        "change_summary": "",
    }

    if not path.exists():
        return result

    current_mtime = path.stat().st_mtime
    result["current_mtime"] = current_mtime

    snapshot_path = get_task_file_snapshot_path(projects_dir, project_name, task_file)
    snapshot_file = Path(snapshot_path)

    # Fast path: compare mtime.
    previous_mtime = None
    if snapshot_file.exists():
        previous_mtime = snapshot_file.stat().st_mtime
        result["previous_mtime"] = previous_mtime

    if previous_mtime is not None and abs(current_mtime - previous_mtime) < 0.01:
        return result

    result["mtime_changed"] = True

    # Slow path: compute diff.
    current_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    previous_lines: list[str] = []
    if snapshot_file.exists():
        previous_text = snapshot_file.read_text(encoding="utf-8")
        previous_lines = previous_text.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        previous_lines,
        current_lines,
        fromfile="previous",
        tofile="current",
        lineterm="",
    ))

    if not diff:
        _touch_snapshot_mtime(snapshot_path, current_lines)
        return result

    result["changed"] = True
    result["content_changed"] = True
    result["diff_lines"] = diff

    # Classify diff hunks.
    added_req: list[str] = []
    removed_req: list[str] = []
    added_info: list[str] = []

    for line in diff:
        if not line.startswith("+") or line.startswith("+++"):
            if line.startswith("-") and not line.startswith("---"):
                stripped = line[1:].strip()
                if _is_requirement_line(stripped):
                    removed_req.append(stripped)
            continue

        content = line[1:].strip()
        if not content:
            continue

        if _is_requirement_line(content):
            added_req.append(content)
        elif _is_info_line(content):
            added_info.append(content)

    result["added_requirement_lines"] = added_req
    result["removed_requirement_lines"] = removed_req
    result["added_info_lines"] = added_info

    # Build summary.
    parts = []
    if added_req:
        parts.append(f"{len(added_req)} new/changed requirement(s)")
    if removed_req:
        parts.append(f"{len(removed_req)} removed requirement(s)")
    if added_info:
        parts.append(f"{len(added_info)} info line(s) from user")
    if not parts:
        parts.append("minor edits (no structural changes detected)")

    result["change_summary"] = "; ".join(parts)
    return result


def save_task_file_snapshot(task_file: str, projects_dir: str,
                             project_name: str) -> None:
    """Save a snapshot of the current task file for future diff comparison.

    Called after the orchestrator has processed detected changes, so the
    next wake cycle can diff against this version.
    """
    path = Path(task_file)
    if not path.exists():
        return

    snapshot_path = get_task_file_snapshot_path(projects_dir, project_name, task_file)
    snapshot_file = Path(snapshot_path)
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)

    content = path.read_text(encoding="utf-8")
    snapshot_file.write_text(content, encoding="utf-8")


def _is_requirement_line(stripped_line: str) -> bool:
    """Heuristic: does this line look like a task requirement?"""
    if not stripped_line:
        return False
    return (
        stripped_line.startswith("#") or
        stripped_line.startswith("- [ ]") or
        stripped_line.startswith("- [x]") or
        stripped_line.startswith("- [X]") or
        (stripped_line[0].isdigit() and ". " in stripped_line[:5])
    )


def _is_info_line(stripped_line: str) -> bool:
    """Heuristic: does this line look like user-provided context/info?"""
    if not stripped_line:
        return False
    return (
        not stripped_line.startswith("#") and
        not stripped_line.startswith("- [") and
        not stripped_line.startswith("```") and
        len(stripped_line) > 10
    )


def _touch_snapshot_mtime(snapshot_path: str, current_lines: list[str]) -> None:
    """Update snapshot content to match current, so mtime stays in sync.

    When the user touches the file without editing (mtime changes but content
    is identical), we still need to update the snapshot's mtime so we don't
    re-diff on every subsequent wake.
    """
    try:
        snapshot_file = Path(snapshot_path)
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file.write_text("".join(current_lines), encoding="utf-8")
    except OSError:
        pass
