"""Process manager for Layer 2 (Claude Code CLI) workers.

Handles spawning, killing, and monitoring Claude Code processes on Windows.
"""

import os
import sys
import json
import time
import signal
import subprocess
from datetime import datetime
from typing import Optional

import psutil

from .config import (
    DEFAULT_MAX_TURNS,
    AURA_LAYER2_BACKEND,
    AURA_DEEPSEEK_API_KEY,
    AURA_DSCODE_MAX_TURNS,
    AURA_DSCODE_MODEL,
    AURA_DSCODE_BASE_URL,
    WAKEUP_FILE,
    get_workspace_dir,
)

_active_processes: dict[str, dict] = {}
_dead_record_cache: dict[str, float] = {}


def signal_wakeup(reason: str = "") -> str:
    """Signal the main loop to wake before the fixed sleep interval ends."""
    try:
        os.makedirs(os.path.dirname(WAKEUP_FILE), exist_ok=True)
        with open(WAKEUP_FILE, "w", encoding="utf-8") as f:
            f.write(reason or "wakeup")
        return f"OK: Wakeup signaled ({WAKEUP_FILE})"
    except Exception as e:
        return f"ERROR signaling wakeup: {e}"


def _process_record_path(task_dir: str) -> str:
    return os.path.join(task_dir, "process.json")


def _write_process_record(entry: dict) -> None:
    record_path = _process_record_path(entry["task_dir"])
    record = {
        "pid": entry["pid"],
        "task_id": entry["task_id"],
        "task_dir": entry["task_dir"],
        "started_at": entry["started_at"].isoformat()
        if hasattr(entry["started_at"], "isoformat") else str(entry["started_at"]),
        "budget_minutes": entry["budget_minutes"],
        "output_path": entry["output_path"],
    }
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    _dead_record_cache.pop(record_path, None)


def _load_process_records() -> None:
    tasks_dir = os.path.join(get_workspace_dir(), "tasks")
    if not os.path.isdir(tasks_dir):
        return

    for task_id in os.listdir(tasks_dir):
        task_dir = os.path.join(tasks_dir, task_id)
        record_path = _process_record_path(task_dir)
        if task_id in _active_processes or not os.path.exists(record_path):
            continue
        try:
            record_mtime = os.path.getmtime(record_path)
        except OSError:
            continue
        if _dead_record_cache.get(record_path) == record_mtime:
            continue
        try:
            with open(record_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            pid = int(record["pid"])
            if not _is_alive(pid):
                _dead_record_cache[record_path] = record_mtime
                continue
            started_at = datetime.fromisoformat(record.get("started_at", ""))
            _active_processes[task_id] = {
                "pid": pid,
                "task_id": task_id,
                "task_dir": record.get("task_dir", task_dir),
                "started_at": started_at,
                "budget_minutes": int(record.get("budget_minutes", 0)),
                "process": None,
                "output_path": record.get("output_path", os.path.join(task_dir, "output.jsonl")),
            }
        except Exception:
            continue


def spawn(task_id: str, task_dir: str, task_md_path: str, budget_minutes: int) -> str:
    """Spawn a Layer 2 worker process for a task.

    Dispatches to claude_code or ds_code backend based on
    AURA_LAYER2_BACKEND config (default: claude_code).
    """
    if AURA_LAYER2_BACKEND == "ds_code":
        return _spawn_dscode(task_id, task_dir, task_md_path, budget_minutes)
    else:
        return _spawn_claude(task_id, task_dir, task_md_path, budget_minutes)


def _spawn_claude(task_id: str, task_dir: str, task_md_path: str, budget_minutes: int) -> str:
    """Spawn a Claude Code CLI process for a task.

    Args:
        task_id: Unique task identifier.
        task_dir: Working directory for the task.
        task_md_path: Path to the task definition markdown file.
        budget_minutes: Time budget for the task.
    """
    if task_id in _active_processes:
        existing = _active_processes[task_id]
        if _is_alive(existing["pid"]):
            return f"ERROR: Task {task_id} is already running (PID: {existing['pid']})"

    # Build Claude Code command
    # Using -p for one-shot mode with the task file as context
    # --verbose is REQUIRED when using --output-format=stream-json
    cmd = [
        "claude",
        "-p",
        f"@task.md",
        "--max-turns", str(DEFAULT_MAX_TURNS),
        "--output-format", "stream-json",
        "--verbose",
    ]

    output_path = os.path.join(task_dir, "output.jsonl")
    error_path = os.path.join(task_dir, "error.log")

    try:
        with open(output_path, "w", encoding="utf-8") as out_f, \
             open(error_path, "w", encoding="utf-8") as err_f:

            # Use preexec_fn=os.setsid on Unix for process group
            if sys.platform == "win32":
                proc = subprocess.Popen(
                    cmd,
                    cwd=task_dir,
                    stdout=out_f,
                    stderr=err_f,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                proc = subprocess.Popen(
                    cmd,
                    cwd=task_dir,
                    stdout=out_f,
                    stderr=err_f,
                    preexec_fn=os.setsid,
                )

        _active_processes[task_id] = {
            "pid": proc.pid,
            "task_id": task_id,
            "task_dir": task_dir,
            "started_at": datetime.now(),
            "budget_minutes": budget_minutes,
            "process": proc,
            "output_path": output_path,
        }
        _write_process_record(_active_processes[task_id])

        return (f"OK: Task {task_id} started (PID: {proc.pid}, backend: claude_code). "
                f"Output: {task_dir}/output.jsonl "
                f"Budget: {budget_minutes}min")

    except Exception as e:
        return f"ERROR spawning task {task_id}: {e}"


def _spawn_dscode(task_id: str, task_dir: str, task_md_path: str, budget_minutes: int) -> str:
    """Spawn a ds-code CLI process for a task.

    Uses 'ds-code run' one-shot mode, passing the task markdown via stdin.
    """
    if task_id in _active_processes:
        existing = _active_processes[task_id]
        if _is_alive(existing["pid"]):
            return f"ERROR: Task {task_id} is already running (PID: {existing['pid']})"

    cmd = [
        "ds-code", "run",
        "--workspace", task_dir,
        "--no-color",
    ]

    if AURA_DSCODE_MODEL:
        cmd.extend(["--model", AURA_DSCODE_MODEL])

    # Build environment so ds-code can find DEEPSEEK_API_KEY
    env = os.environ.copy()
    if AURA_DEEPSEEK_API_KEY:
        env["DEEPSEEK_API_KEY"] = AURA_DEEPSEEK_API_KEY
    elif not env.get("DEEPSEEK_API_KEY"):
        # Fallback: reuse ANTHROPIC_API_KEY for the same DeepSeek endpoint
        fallback = os.environ.get("ANTHROPIC_API_KEY", "")
        if fallback:
            env["DEEPSEEK_API_KEY"] = fallback
    if AURA_DSCODE_BASE_URL:
        env["DEEPSEEK_BASE_URL"] = AURA_DSCODE_BASE_URL
    env["DS_CODE_MAX_TURNS"] = str(AURA_DSCODE_MAX_TURNS)

    output_path = os.path.join(task_dir, "output.txt")
    error_path = os.path.join(task_dir, "error.log")

    try:
        with open(task_md_path, "r", encoding="utf-8") as task_f, \
             open(output_path, "w", encoding="utf-8") as out_f, \
             open(error_path, "w", encoding="utf-8") as err_f:

            if sys.platform == "win32":
                proc = subprocess.Popen(
                    cmd,
                    cwd=task_dir,
                    stdin=task_f,
                    stdout=out_f,
                    stderr=err_f,
                    env=env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                proc = subprocess.Popen(
                    cmd,
                    cwd=task_dir,
                    stdin=task_f,
                    stdout=out_f,
                    stderr=err_f,
                    env=env,
                    preexec_fn=os.setsid,
                )

        _active_processes[task_id] = {
            "pid": proc.pid,
            "task_id": task_id,
            "task_dir": task_dir,
            "started_at": datetime.now(),
            "budget_minutes": budget_minutes,
            "process": proc,
            "output_path": output_path,
        }
        _write_process_record(_active_processes[task_id])

        return (f"OK: Task {task_id} started (PID: {proc.pid}, backend: ds_code). "
                f"Output: {task_dir}/output.txt "
                f"Budget: {budget_minutes}min")

    except FileNotFoundError:
        return ("ERROR spawning task {task_id}: ds-code command not found. "
                "Install with: cd ds_code && pip install -e .")
    except Exception as e:
        return f"ERROR spawning task {task_id}: {e}"


def kill(task_id: str) -> str:
    """Kill a running task process and all its children."""
    if task_id not in _active_processes:
        return f"ERROR: Task {task_id} is not tracked. Known tasks: {list(_active_processes.keys())}"

    entry = _active_processes[task_id]
    pid = entry["pid"]

    try:
        proc = psutil.Process(pid)
        children = proc.children(recursive=True)

        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, check=False, timeout=10)
        else:
            for child in children:
                child.terminate()
            proc.terminate()
            time.sleep(2)
            if proc.is_running():
                proc.kill()

        # Wait for process to actually exit
        try:
            proc.wait(timeout=5)
        except psutil.TimeoutExpired:
            pass

        _active_processes[task_id]["killed_at"] = datetime.now()
        _active_processes[task_id]["running"] = False
        _write_process_record(_active_processes[task_id])

        return f"OK: Killed task {task_id} (PID: {pid}, children: {len(children)})"

    except psutil.NoSuchProcess:
        _active_processes[task_id]["running"] = False
        _write_process_record(_active_processes[task_id])
        return f"WARN: Task {task_id} (PID: {pid}) was already dead"
    except subprocess.TimeoutExpired:
        _active_processes[task_id]["running"] = _is_alive(pid)
        return f"ERROR killing task {task_id}: taskkill timed out for PID {pid}"
    except Exception as e:
        return f"ERROR killing task {task_id}: {e}"


def list_all() -> list[dict]:
    """List all tracked processes with status and health metrics."""
    _load_process_records()
    result = []
    for task_id, entry in _active_processes.items():
        running = _is_alive(entry["pid"])
        entry["running"] = running
        elapsed = (datetime.now() - entry["started_at"]).total_seconds() / 60

        output_path = entry.get("output_path", os.path.join(entry["task_dir"], "output.jsonl"))
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        # ── Process health metrics ──────────────────────────────────
        cpu_percent = 0.0
        memory_mb = 0.0
        if running:
            try:
                proc = psutil.Process(entry["pid"])
                # cpu_percent needs interval for first meaningful reading
                cpu_percent = proc.cpu_percent(interval=0.1)
                mem_info = proc.memory_info()
                memory_mb = mem_info.rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                cpu_percent = 0.0
                memory_mb = 0.0

        result.append({
            "task_id": task_id,
            "pid": entry["pid"],
            "running": running,
            "elapsed_minutes": round(elapsed, 1),
            "budget_minutes": entry["budget_minutes"],
            "output_size": output_size,
            "cpu_percent": round(cpu_percent, 1),
            "memory_mb": round(memory_mb, 1),
        })

    return result


def get_task_status(task_id: str) -> Optional[dict]:
    """Get status of a specific task."""
    tasks = list_all()
    for t in tasks:
        if t["task_id"] == task_id:
            return t
    return None


def get_output_tail(task_id: str, lines: int = 50) -> str:
    """Get the last N lines of a task's output."""
    if task_id not in _active_processes:
        return f"ERROR: Unknown task: {task_id}"

    output_path = _active_processes[task_id].get(
        "output_path", os.path.join(_active_processes[task_id]["task_dir"], "output.jsonl"))
    if not os.path.exists(output_path):
        return "(No output yet)"

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return "".join(recent)
    except Exception as e:
        return f"ERROR reading output: {e}"


def is_alive(task_id: str) -> bool:
    """Check if a task's process is still running."""
    if task_id not in _active_processes:
        return False
    return _is_alive(_active_processes[task_id]["pid"])


def _is_alive(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
