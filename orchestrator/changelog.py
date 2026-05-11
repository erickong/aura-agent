"""Changelog — 追踪 task.md 文件的变更历史，避免重复执行。

核心思路：
- 每个 task.md 文件有一个对应的 changelog 文件，存储在项目目录下
- changelog 记录每次检测到的文件内容哈希和对应的处理状态
- 启动时对比当前文件哈希与上次记录的哈希，只处理新增/变更的内容
- 这样即使 task.md 被多次编辑，也不会重复执行已完成的任务

Per-wake task file change detection (R7):
- 每次唤醒时检查 task file 的 mtime，仅在时间戳变化时才做完整 diff
- diff 结果注入 orchestrator 上下文，辅助判断是否需要 replan
- 检测用户是否在任务文件中提供了新信息，自动记录到 memory

文件结构：
  projects/{project_name}/changelog/
    self_upgrade.md.json      # 对应 tasks/self_upgrade.md 的变更记录
    stockagent.md.json        # 对应 tasks/stockagent.md 的变更记录
    self_upgrade.snapshot.md  # 上次处理的 task file 快照（用于 diff）
"""

import difflib
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def _project_name_from_task_file(task_file: str) -> str:
    """从 task 文件路径派生出唯一的项目名称。

    使用 os.path.basename() 提取文件名（去掉扩展名），
    确保 tasks/self_upgrade.md 和 tasks\\self_upgrade.md
    在不同操作系统上都映射到同一个项目名 "self_upgrade"。

    与 orchestrator/main.py 中的同名函数保持完全一致。
    """
    # Normalize backslashes to forward slashes for cross-platform consistency
    normalized = task_file.replace("\\", "/")
    name = os.path.splitext(os.path.basename(normalized))[0]
    return name.replace(" ", "_").lower()


def get_changelog_dir(projects_dir: str, project_name: str) -> str:
    """获取 changelog 目录路径。"""
    return os.path.join(projects_dir, project_name, "changelog")


def get_changelog_path(projects_dir: str, project_name: str, task_file: str) -> str:
    """获取 changelog 文件路径。
    
    使用 task_file 的 basename（不含路径）作为 changelog 文件名，
    这样即使 task_file 路径格式不同（正反斜杠），也能对应到同一个 changelog。
    """
    normalized = task_file.replace("\\", "/")
    basename = os.path.basename(normalized)
    if not basename.endswith(".json"):
        basename = os.path.splitext(basename)[0] + ".json"
    return os.path.join(get_changelog_dir(projects_dir, project_name), basename)


def compute_file_hash(file_path: str) -> str:
    """计算文件的 SHA-256 哈希。"""
    path = Path(file_path)
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_changelog(changelog_path: str) -> dict:
    """加载 changelog 文件。"""
    path = Path(changelog_path)
    if not path.exists():
        return {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "entries": [],       # 历史变更记录
            "processed_hashes": {},  # {file_hash: entry_index} 已处理的文件哈希
            "processed_items": {},   # {item_fingerprint: entry_index} 已处理的 item
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # 确保必要字段存在
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
    """保存 changelog 文件。"""
    changelog["updated_at"] = datetime.now().isoformat()
    path = Path(changelog_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(changelog, ensure_ascii=False, indent=2), encoding="utf-8")


def get_file_change_info(task_file: str, projects_dir: str, project_name: str) -> dict:
    """检测 task.md 文件的变更信息。
    
    返回：
    {
        "is_new": True/False,        # 是否全新文件（从未处理过）
        "is_changed": True/False,    # 内容是否发生变化
        "previous_hash": "...",      # 上次处理的文件哈希
        "current_hash": "...",       # 当前文件哈希
        "last_processed_at": "...",  # 上次处理时间
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
    
    # 检查是否处理过这个哈希
    if changelog["processed_hashes"]:
        result["is_new"] = False
        if current_hash in changelog["processed_hashes"]:
            entry_idx = changelog["processed_hashes"][current_hash]
            if entry_idx < len(changelog["entries"]):
                entry = changelog["entries"][entry_idx]
                result["last_processed_at"] = entry.get("processed_at", "")
            result["is_changed"] = False
        else:
            # 有历史记录但哈希不同 = 文件已变更
            result["is_changed"] = True
            # 取最后一个 entry 的哈希作为 previous_hash
            if changelog["entries"]:
                result["previous_hash"] = changelog["entries"][-1].get("file_hash", "")
    
    return result


def mark_file_processed(task_file: str, projects_dir: str, project_name: str,
                        summary: str = "") -> dict:
    """标记 task.md 文件为已处理。
    
    记录当前文件哈希和处理摘要到 changelog。
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
    """标记 task.md 中的某个 item 为已处理。"""
    changelog_path = get_changelog_path(projects_dir, project_name, task_file)
    changelog = load_changelog(changelog_path)
    
    changelog["processed_items"][item_fingerprint] = {
        "text": item_text,
        "processed_at": datetime.now().isoformat(),
    }
    
    save_changelog(changelog_path, changelog)


def is_item_processed(task_file: str, projects_dir: str, project_name: str,
                      item_fingerprint: str) -> bool:
    """检查某个 item 是否已被处理过。"""
    changelog_path = get_changelog_path(projects_dir, project_name, task_file)
    changelog = load_changelog(changelog_path)
    return item_fingerprint in changelog.get("processed_items", {})


def get_unprocessed_items(task_file: str, projects_dir: str, project_name: str,
                          items: list[dict]) -> list[dict]:
    """从 items 列表中过滤出未处理的项目。
    
    Args:
        task_file: 任务文件路径
        projects_dir: 项目根目录
        project_name: 项目名称
        items: parse_task_items() 返回的 item 列表
    
    Returns:
        未处理的 item 列表
    """
    return [
        it for it in items
        if not is_item_processed(task_file, projects_dir, project_name, it["fingerprint"])
    ]


def get_project_name_for_task(task_file: str) -> str:
    """根据 task 文件路径获取唯一的项目名称。
    
    统一规则：只取文件名（不含路径和扩展名），
    这样 tasks/self_upgrade.md 和 tasks\\self_upgrade.md 都映射到 self_upgrade。
    
    与 orchestrator/main.py 中的 _project_name_from_task_file 使用相同实现。
    """
    return _project_name_from_task_file(task_file)


def cleanup_orphan_projects(projects_dir: str, active_task_files: list[str]) -> list[str]:
    """清理已无对应 task.md 文件的项目目录。

    Args:
        projects_dir: 项目根目录
        active_task_files: 当前存在的 task.md 文件路径列表

    Returns:
        被清理的项目名称列表
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
        # 检查是否有 state.json（是有效的项目）
        if not os.path.exists(os.path.join(pdir, "state", "state.json")):
            continue
        if name not in active_names:
            cleaned.append(name)

    return cleaned


# ═══════════════════════════════════════════════════════════════════════════════
# R7 — Per-wake task file change detection
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each wake cycle, the orchestrator checks whether the task file was modified
# (using mtime first — cheap).  Only when the mtime differs from the last
# recorded value do we compute a full diff against the stored snapshot.
#
# The diff is analysed with simple heuristics to answer:
#   1. Did the user add/change/remove requirements?  → may need replan
#   2. Did the user provide new context or information? → may help progress
#   3. Should this be recorded to memory?
#
# Flow:
#   check_task_file_on_wake()           ← called every cycle (fast: mtime check)
#   save_task_file_snapshot()           ← called after processing changes
#   get_task_file_snapshot_path()       ← internal helper


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

    Uses mtime first (cheap).  Only when the mtime differs from the recorded
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
            "diff_lines": list[str],        # unified diff lines (empty if no change)
            "added_requirement_lines": list[str],
            "removed_requirement_lines": list[str],
            "added_info_lines": list[str],  # non-task-markup additions (user context)
            "change_summary": str,          # one-line human-readable summary
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

    # ── Fast path: compare mtime ──────────────────────────────────────
    previous_mtime = None
    if snapshot_file.exists():
        previous_mtime = snapshot_file.stat().st_mtime
        result["previous_mtime"] = previous_mtime

    if previous_mtime is not None and abs(current_mtime - previous_mtime) < 0.01:
        # mtime unchanged — skip expensive diff
        return result

    result["mtime_changed"] = True

    # ── Slow path: compute diff ───────────────────────────────────────
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
        # mtime changed but content is identical (e.g. touched without edit)
        # Update the snapshot mtime to avoid re-diffing next cycle.
        _touch_snapshot_mtime(snapshot_path, current_lines)
        return result

    result["changed"] = True
    result["content_changed"] = True
    result["diff_lines"] = diff

    # ── Classify diff hunks ───────────────────────────────────────────
    added_req: list[str] = []
    removed_req: list[str] = []
    added_info: list[str] = []

    for line in diff:
        if not line.startswith("+") or line.startswith("+++"):
            # Track removals separately
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

    # ── Build summary ─────────────────────────────────────────────────
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
    # Any non-empty, non-requirement line that looks like a sentence or note.
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
