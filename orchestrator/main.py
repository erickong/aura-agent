#!/usr/bin/env python3
"""Aura Agent - Main entry point.

Multi-project support - only one project runs at a time, but you can switch.
Root memory/, state/, workspace/ always reflect the active project.
Powered by DeepSeek v4-pro via Anthropic-compatible API.

Usage:
    aura start --task-file=tasks/my_mission.md
    aura status
    aura progress
    aura projects
    aura history

Data files (memory, state, workspace) are stored under a task-specific
./.aura/<task-file-name>-<path-hash>/ directory by default.
Override with: aura --data-dir=/path/to/dir start --task-file=...
"""

import argparse
import hashlib
import json
import os
import sys
import time
import signal
import re
import shutil
from datetime import datetime
from pathlib import Path

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.getcwd())
os.environ.setdefault("AURA_PROJECT_ROOT", PROJECT_ROOT)
sys.path.insert(0, CODE_ROOT)


# -- Early --data-dir parsing (before config import) --
_data_dir = None
_args_for_data = sys.argv[1:]
for i, arg in enumerate(_args_for_data):
    if arg in ("--data-dir",) and i + 1 < len(_args_for_data):
        _data_dir = _args_for_data[i + 1]
        break
    elif arg.startswith("--data-dir="):
        _data_dir = arg.split("=", 1)[1]
        break

if _data_dir:
    os.environ["AURA_DATA_DIR"] = os.path.expanduser(_data_dir)


def _early_resolve_task_file(task_file: str) -> str:
    expanded = os.path.expanduser(task_file)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(PROJECT_ROOT, expanded))


def _task_data_slug(task_file: str) -> str:
    resolved = os.path.abspath(os.path.normcase(task_file))
    stem = os.path.splitext(os.path.basename(task_file))[0] or "task"
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem).strip(" ._")
    stem = re.sub(r"\s+", "_", stem) or "task"
    suffix = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{suffix}"


def _default_aura_base_dir() -> str:
    return os.path.abspath(os.path.join(PROJECT_ROOT, ".aura"))


def _active_task_data_marker(base_dir: str) -> str:
    return os.path.join(base_dir, ".active_task_data_dir")


def _task_index_path(base_dir: str) -> str:
    return os.path.join(base_dir, "task_index.json")


def _task_data_dir_for(task_file: str, base_dir: str | None = None) -> str:
    base = os.path.abspath(os.path.expanduser(base_dir or _default_aura_base_dir()))
    return os.path.join(base, _task_data_slug(task_file))


def _detect_task_arg(argv: list[str]) -> str | None:
    known_commands = {
        "start", "restart", "status", "progress", "projects", "history",
        "changelog", "cleanup", "wake", "setup", "summaries", "cache-stats",
        "changelog-overview", "clean-workspaces",
    }
    option_takes_value = {"--config", "-c", "--data-dir", "--task-file"}
    command = None
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            if command in {"start", "restart"} and argv[i - 1] == "--task-file":
                return arg
            skip_next = False
            continue
        if arg in option_takes_value:
            skip_next = True
            continue
        if arg.startswith("--task-file="):
            return arg.split("=", 1)[1]
        if arg.startswith("-"):
            continue
        if command in {"start", "restart"}:
            return arg
        if arg in known_commands:
            command = arg
            continue
        return arg
    return None


def _select_task_data_dir_before_import() -> None:
    if _data_dir:
        return

    base_dir = _default_aura_base_dir()
    task_arg = _detect_task_arg(sys.argv[1:])
    if task_arg:
        task_file = _early_resolve_task_file(task_arg)
        os.environ["AURA_DATA_DIR"] = _task_data_dir_for(task_file, base_dir)
        return

    marker = _active_task_data_marker(base_dir)
    if os.path.exists(marker):
        try:
            active_dir = Path(marker).read_text(encoding="utf-8").strip()
            if active_dir:
                os.environ["AURA_DATA_DIR"] = active_dir
        except OSError:
            pass


def _record_task_data_dir(task_file: str, data_dir: str) -> None:
    if _data_dir:
        return

    base_dir = _default_aura_base_dir()
    task_file_abs = os.path.abspath(task_file)
    data_dir_abs = os.path.abspath(data_dir)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(data_dir_abs, exist_ok=True)

    metadata = {
        "task_file": task_file_abs,
        "task_file_norm": os.path.normcase(task_file_abs),
        "data_dir": data_dir_abs,
        "updated_at": datetime.now().isoformat(),
    }

    metadata_path = os.path.join(data_dir_abs, "task_file.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            metadata["created_at"] = existing.get("created_at") or metadata["updated_at"]
        except (OSError, json.JSONDecodeError):
            metadata["created_at"] = metadata["updated_at"]
    else:
        metadata["created_at"] = metadata["updated_at"]

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    index_path = _task_index_path(base_dir)
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError):
        index = {"tasks": {}}
    index.setdefault("tasks", {})[metadata["task_file_norm"]] = metadata
    index["updated_at"] = metadata["updated_at"]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    Path(_active_task_data_marker(base_dir)).write_text(data_dir_abs, encoding="utf-8")


_select_task_data_dir_before_import()


# -- Early config loading (before config import) --
GLOBAL_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".aura", "config.env")


def _load_dotenv(env_path: str) -> None:
    """Manually parse a .env file and set os.environ (no python-dotenv needed)."""
    if not os.path.exists(env_path):
        print(f"[WARN] Config file not found: {env_path}")
        sys.exit(1)

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Parse KEY=VALUE (handle optional quotes)
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            os.environ[key] = value


def _load_dotenv_if_exists(env_path: str) -> bool:
    if not os.path.exists(env_path):
        return False
    _load_dotenv(env_path)
    return True


# Check for --config / -c early (before config module is imported).
# Also removes these args from sys.argv so argparse doesn't choke on
# --config appearing after the subcommand (e.g. "start --config=.env ...")
_config_path = None
_new_argv = [sys.argv[0]]
_skip_next = False
for i, arg in enumerate(sys.argv[1:], 1):
    if _skip_next:
        _skip_next = False
        continue
    if arg in ("--config", "-c") and i + 1 < len(sys.argv):
        _config_path = sys.argv[i + 1]
        _skip_next = True
        continue
    elif arg.startswith("--config="):
        _config_path = arg.split("=", 1)[1]
        continue
    elif arg.startswith("-c="):
        _config_path = arg.split("=", 1)[1]
        continue
    _new_argv.append(arg)
sys.argv = _new_argv

if _config_path:
    # Load global defaults first, then allow explicit --config to override.
    _load_dotenv_if_exists(GLOBAL_CONFIG_PATH)
    _load_dotenv(_config_path)
    print(f"[CONFIG] Loaded environment from: {_config_path}")
else:
    _load_dotenv_if_exists(GLOBAL_CONFIG_PATH)

_select_task_data_dir_before_import()


from orchestrator.config import (
    CYCLE_INTERVAL_SECONDS,
    DEEP_REVIEW_INTERVAL_CYCLES,
    LLM_DEAD_THROTTLE_SECONDS,
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    ANTHROPIC_MODEL,
    ANTHROPIC_MAX_TOKENS,
    AURA_LAYER2_BACKEND,
    DATA_DIR,
    DEFAULT_MAX_TURNS,
    DEFAULT_TASK_BUDGET_MINUTES,
    FILE_CACHE_ENABLED,
    FILE_CACHE_TTL_SECONDS,
    MAX_CONCURRENT_TASKS,
    MEMORY_DIR,
    STATE_DIR,
    PROJECTS_DIR,
    PROJECT_ROOT as CFG_PROJECT_ROOT,
    REVIEW_NUDGE_INTERVAL,
    TASK_SUMMARY_ENABLED,
    WAKEUP_FILE,
    WORKER_CUDA_VISIBLE_DEVICES,
    WORKER_MAX_CPU_PERCENT,
    WORKER_MAX_GPU_MEMORY_GB,
    WORKER_MAX_GPU_MEMORY_PERCENT,
    WORKER_MAX_GPU_UTIL_PERCENT,
    WORKER_MAX_SYSTEM_MEMORY_GB,
    WORKER_MAX_SYSTEM_MEMORY_PERCENT,
    WORKER_MIN_GPU_MEMORY_FREE_GB,
    WORKER_MIN_SYSTEM_MEMORY_FREE_GB,
    WORKER_RESOURCE_AVG_WINDOW_SECONDS,
    WORKER_RESOURCE_GUARD_ENABLED,
    WORKER_RESOURCE_POLL_SECONDS,
    WORKER_RESOURCE_VIOLATION_STRIKES,
    get_workspace_dir,
)
from orchestrator import state as state_mgr
from orchestrator import memory as memory_mgr
from orchestrator import progress as progress_mgr
from orchestrator import process_mgr
from orchestrator.agent import run_cycle
from orchestrator.changelog import (
    get_file_change_info,
    mark_file_processed,
    get_project_name_for_task,
    cleanup_orphan_projects,
    check_task_file_on_wake,
    save_task_file_snapshot,
)
from orchestrator.agent_patches import apply_patches, get_startup_banner
from orchestrator.cli_extensions import register_commands
from orchestrator.task_reporter import generate_task_summary

# -- Resilient review import --
_review_available = False
_review_import_error = None
try:
    from orchestrator.review import review_cycle
    _review_available = True
except ImportError as e:
    _review_import_error = str(e)

    def review_cycle(force=False):
        return {"review_text": "", "saved_path": "", "recommendations": [], "error": str(e)}

# -- Apply R1 agent patches (system prompt + Layer 2 backend display) --
patch_results = apply_patches()

_running = True
_shutdown_requested = False

ACTIVE_PROJECT_FILE = os.path.join(STATE_DIR, ".active_project")

# -- Cycle tracking --
_consecutive_api_errors = 0
_llm_dead = False


class ShutdownRequested(KeyboardInterrupt):
    """Raised from the signal handler to abort the current API/tool cycle."""


def _kill_running_workers(prefix: str = "[SHUTDOWN]") -> None:
    """Kill all tracked Layer 2 workers that are still running."""
    running = [worker for worker in process_mgr.list_all() if worker.get("running")]
    if not running:
        print(f"{prefix} No Layer 2 workers running.")
        return

    print(f"{prefix} Killing {len(running)} Layer 2 worker(s)...")
    for worker in running:
        try:
            result = process_mgr.kill(worker["task_id"])
            print(f"  {result}")
        except Exception as kill_err:
            print(f"  [WARN] Failed to kill {worker['task_id']}: {kill_err}")


def _clear_wakeup_signal() -> None:
    try:
        if os.path.exists(WAKEUP_FILE):
            os.remove(WAKEUP_FILE)
    except OSError as err:
        print(f"  [WARN] Could not clear wakeup signal: {err}")


def _has_wakeup_signal() -> bool:
    return os.path.exists(WAKEUP_FILE)


def _sleep_until_next_wake(interval: int) -> None:
    """Sleep in short intervals and wake early on worker/external events."""
    global _running

    print(f"[Sleep] {interval}s until next wake... (touch {WAKEUP_FILE} to wake now)")
    remaining = interval
    while remaining > 0 and _running:
        try:
            time.sleep(min(remaining, 5))
            remaining -= 5
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Ctrl+C detected.")
            _running = False
            break

        if _has_wakeup_signal():
            print("[Watchdog] External wakeup signal detected.")
            _clear_wakeup_signal()
            break

        tracked = process_mgr.list_all()
        for worker in tracked:
            entry = process_mgr._active_processes.get(worker["task_id"], {})
            if entry.get("killed_at"):
                continue
            if not worker["running"]:
                print(f"[Watchdog] Worker {worker['task_id']} stopped; waking early.")
                remaining = 0
                break


def _project_name_from_cwd() -> str:
    # Derive project name from the BASENAME of the task file (without extension).
    # Using basename ensures that tasks/self_upgrade.md and tasks\self_upgrade.md
    # both map to the SAME project name "self_upgrade", preventing project
    # duplication caused by path separator differences between Windows and Linux.
    #
    # If two task files in different directories share the same filename
    # (e.g. tasks/T1/task.md and tasks/T2/task.md), they will map to the same
    # project "task". This is intentional - the task file's content (tracked via
    # changelog) differentiates them, not the directory path.
    name = os.path.basename(os.path.abspath(CFG_PROJECT_ROOT))
    return name.replace(" ", "_").lower()


def _resolve_task_file(task_file: str) -> str:
    """Resolve an arbitrary user-provided task file path."""
    expanded = os.path.expanduser(task_file)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(CFG_PROJECT_ROOT, expanded))


def _get_active_project() -> str | None:
    if os.path.exists(ACTIVE_PROJECT_FILE):
        with open(ACTIVE_PROJECT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def _set_active_project(name: str) -> None:
    os.makedirs(os.path.dirname(ACTIVE_PROJECT_FILE), exist_ok=True)
    with open(ACTIVE_PROJECT_FILE, "w", encoding="utf-8") as f:
        f.write(name)


def _project_dir(name: str) -> str:
    return os.path.join(PROJECTS_DIR, name)


def _project_exists(name: str) -> bool:
    pdir = _project_dir(name)
    return os.path.exists(os.path.join(pdir, "state", "state.json"))


def _save_project(name: str) -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(get_workspace_dir(), exist_ok=True)
    try:
        from orchestrator.changelog import get_changelog_dir
        os.makedirs(get_changelog_dir(PROJECTS_DIR, name), exist_ok=True)
    except Exception as e:
        print(f"  [WARN] changelog directory check skipped: {e}")
    print(f"  [SAVE] Project '{name}' checkpointed in {DATA_DIR}.")
    return

    pdir = _project_dir(name)
    # Only mirror memory/ and state/ to the project archive.
    # Workspace lives natively under projects/{name}/workspace/ - no copying needed.
    for src, dst in [
        (MEMORY_DIR, os.path.join(pdir, "memory")),
        (STATE_DIR, os.path.join(pdir, "state")),
    ]:
        if os.path.exists(src):
            if os.path.exists(dst):
                _rmtree_force(dst)
            shutil.copytree(src, dst)
    # Also save changelog (lives directly in project dir, not mirrored)
    try:
        from orchestrator.changelog import get_changelog_dir
        changelog_src = get_changelog_dir(PROJECTS_DIR, name)
        changelog_dst = os.path.join(pdir, "changelog")
        if os.path.isdir(changelog_src):
            if os.path.exists(changelog_dst):
                _rmtree_force(changelog_dst)
            shutil.copytree(changelog_src, changelog_dst)
    except Exception as e:
        print(f"  [WARN] changelog save skipped: {e}")
    # Remove .active_project from the project's state directory - it is
    # a global indicator, not per-project state.  Leaving it would bake a
    # stale value into the saved project, causing wrong active-project
    # detection the next time the project is restored.
    stale_active = os.path.join(pdir, "state", ".active_project")
    if os.path.exists(stale_active):
        os.remove(stale_active)
    print(f"  [SAVE] Project '{name}' saved.")


def _restore_project(name: str) -> None:
    pdir = _project_dir(name)
    # Only restore memory/ and state/ mirrors.
    # Workspace lives natively under projects/{name}/workspace/ - no copying needed.
    for src, dst in [
        (os.path.join(pdir, "memory"), MEMORY_DIR),
        (os.path.join(pdir, "state"), STATE_DIR),
    ]:
        if os.path.exists(dst):
            _rmtree_force(dst)
        if os.path.exists(src):
            shutil.copytree(src, dst)
        else:
            os.makedirs(dst, exist_ok=True)
    # Also restore changelog
    try:
        from orchestrator.changelog import get_changelog_dir
        changelog_src = os.path.join(pdir, "changelog")
        changelog_dst = get_changelog_dir(PROJECTS_DIR, name)
        if os.path.isdir(changelog_src):
            if os.path.exists(changelog_dst):
                _rmtree_force(changelog_dst)
            shutil.copytree(changelog_src, changelog_dst)
        else:
            os.makedirs(changelog_dst, exist_ok=True)
    except Exception as e:
        print(f"  [WARN] changelog restore skipped: {e}")
    # Remove any .active_project that was restored from the project archive.
    # The global active-project indicator must not be overwritten by a stale
    # value baked into a previously-saved project.
    stale_active = os.path.join(STATE_DIR, ".active_project")
    if os.path.exists(stale_active):
        os.remove(stale_active)
    print(f"  [RESTORE] Project '{name}' restored.")


def _create_new_project(name: str, task_file: str, mission: str) -> None:
    for d in [MEMORY_DIR, STATE_DIR, get_workspace_dir()]:
        os.makedirs(d, exist_ok=True)
    # Create changelog directory
    from orchestrator.changelog import get_changelog_dir
    os.makedirs(get_changelog_dir(PROJECTS_DIR, name), exist_ok=True)
    state_mgr.init_state(mission, task_file)
    memory_mgr.append_memory(
        "decision",
        f"New task started\nProject: {name}\nMission: {mission}\nTask file: {task_file}"
    )
    progress_mgr.render_progress()
    print(f"  [INIT] Project '{name}' initialized under {DATA_DIR}.")


def _rmtree_force(path: str) -> None:
    """Remove a directory tree, handling Windows read-only files."""
    def _on_rm_error(func, p, exc_info):
        os.chmod(p, 0o666)
        func(p)
    shutil.rmtree(path, onerror=_on_rm_error)


def _render_progress_safely(reason: str = "") -> None:
    try:
        progress_mgr.render_progress()
    except Exception as err:
        suffix = f" after {reason}" if reason else ""
        print(f"  [WARN] Could not render progress{suffix}: {err}")


def _format_bool(value: bool) -> str:
    return "on" if value else "off"


def _print_effective_config() -> None:
    """Print startup configuration so defaults are visible at launch."""
    key_state = "set" if ANTHROPIC_API_KEY else "missing"
    cuda_devices = WORKER_CUDA_VISIBLE_DEVICES or "(all visible)"
    print()
    print("  Effective Configuration")
    print("  -----------------------")
    print(f"  Layer 1 model: {ANTHROPIC_MODEL}")
    print(f"  Layer 1 base URL: {ANTHROPIC_BASE_URL}")
    print(f"  API key: {key_state}")
    print(f"  Max tokens: {ANTHROPIC_MAX_TOKENS}")
    print(f"  Layer 2 backend: {AURA_LAYER2_BACKEND}")
    print(f"  Max concurrent workers: {MAX_CONCURRENT_TASKS}")
    print(f"  Default task budget: {DEFAULT_TASK_BUDGET_MINUTES} min")
    print(f"  Max worker turns: {DEFAULT_MAX_TURNS}")
    print(f"  Wake interval: {CYCLE_INTERVAL_SECONDS}s")
    print(f"  Deep review interval: every {DEEP_REVIEW_INTERVAL_CYCLES} cycles")
    print(f"  Light review interval: every {REVIEW_NUDGE_INTERVAL} cycles")
    print(f"  LLM dead throttle: {LLM_DEAD_THROTTLE_SECONDS}s")
    print(f"  File cache: {_format_bool(FILE_CACHE_ENABLED)} (TTL {FILE_CACHE_TTL_SECONDS}s)")
    print(f"  Task summaries: {_format_bool(TASK_SUMMARY_ENABLED)}")
    print(f"  Resource guard: {_format_bool(WORKER_RESOURCE_GUARD_ENABLED)}")
    print(f"  Resource poll interval: {WORKER_RESOURCE_POLL_SECONDS}s")
    print(f"  Resource average window: {WORKER_RESOURCE_AVG_WINDOW_SECONDS}s")
    print(f"  Resource violation strikes: {WORKER_RESOURCE_VIOLATION_STRIKES}")
    print(f"  Worker CPU limit: {WORKER_MAX_CPU_PERCENT}%")
    print(f"  Worker system memory limit: {WORKER_MAX_SYSTEM_MEMORY_PERCENT}%")
    print(f"  Worker GPU utilization limit: {WORKER_MAX_GPU_UTIL_PERCENT}%")
    print(f"  Worker GPU memory limit: {WORKER_MAX_GPU_MEMORY_PERCENT}%")
    print(f"  Absolute system memory limit: {WORKER_MAX_SYSTEM_MEMORY_GB}GB (0 disables)")
    print(f"  Minimum free system memory: {WORKER_MIN_SYSTEM_MEMORY_FREE_GB}GB (0 disables)")
    print(f"  Absolute GPU memory limit: {WORKER_MAX_GPU_MEMORY_GB}GB (0 disables)")
    print(f"  Minimum free GPU memory: {WORKER_MIN_GPU_MEMORY_FREE_GB}GB (0 disables)")
    print(f"  CUDA_VISIBLE_DEVICES for workers: {cuda_devices}")


def _handle_resource_guard_stop(task_id: str, entry: dict) -> bool:
    """Handle a worker killed by the resource guard.

    Policy:
    1. First resource kill: return the original task to pending for one safer retry.
    2. Second resource kill: block the original task and spawn a fix subtask.
    3. Later resource kills: mark failed with concrete resource evidence.
    """
    violation = entry.get("resource_violation")
    if not violation or entry.get("resource_status_recorded"):
        return False

    state = state_mgr.load_state()
    task = state_mgr.find_task(task_id, state.get("tasks", []))
    prior_kills = int((task or {}).get("resource_guard_kills", 0))
    kill_count = prior_kills + 1

    if kill_count == 1:
        print(f"\n  [RESOURCE] Worker {task_id} was killed by resource guard: {violation}")
        print("             Returning task to pending for one smaller/safer retry.")
        state_mgr.update_task(
            task_id, "pending",
            "Resource guard killed worker; retry once with reduced resource settings",
            f".aura/workspace/tasks/{task_id}/resource_guard.log: {violation}"
        )
        state = state_mgr.load_state()
        task = state_mgr.find_task(task_id, state.get("tasks", []))
        if task is not None:
            task["resource_guard_kills"] = kill_count
            task["resource_retry_required"] = True
            task["last_resource_violation"] = violation
            task["resource_retry_guidance"] = (
                "Next attempt must lower resource use before running: reduce batch size, "
                "epochs, model size, dataloader workers, parallelism, or disable offload."
            )
        state["task_file_needs_planning"] = True
        state_mgr.save_state(state)
    elif kill_count == 2:
        print(f"\n  [RESOURCE] Worker {task_id} exceeded resource limits again: {violation}")
        print("             Blocking original task and creating a parameter-fix subtask.")
        state_mgr.update_task(
            task_id, "blocked",
            "Resource guard killed worker twice; needs a smaller execution plan",
            f".aura/workspace/tasks/{task_id}/resource_guard.log: {violation}"
        )
        state = state_mgr.load_state()
        task = state_mgr.find_task(task_id, state.get("tasks", []))
        if task is not None:
            task["resource_guard_kills"] = kill_count
            task["last_resource_violation"] = violation
        state["task_file_needs_planning"] = True
        state_mgr.save_state(state)

        fix_id = f"{task_id}.fix{kill_count}"
        fix_description = (
            f"Resource-fix task for {task_id}. The original worker was killed twice by "
            f"Aura's resource guard. Latest violation: {violation}\n\n"
            "Inspect the original task workspace, task.md, output logs, error logs, and "
            "resource_guard.log. Produce a safer execution plan and, if applicable, patch "
            "commands/scripts/configs to fit within the resource policy. Prefer lowering "
            "batch size, epochs, dataloader workers, parallelism, model size, or disabling "
            "offload. If the task is impossible on this hardware, write clear evidence and "
            "state the smallest feasible alternative. Write result.md with concrete next steps."
        )
        decompose_result = state_mgr.decompose_task(task_id, [{
            "id": fix_id,
            "description": fix_description,
            "acceptance_criteria": (
                "result.md explains the resource violation, changed parameters or commands, "
                "and whether the original task can be retried safely."
            ),
        }])
        print(f"             {decompose_result}")
        actual_fix_id = fix_id
        state = state_mgr.load_state()
        parent = state_mgr.find_task(task_id, state.get("tasks", []))
        if parent and parent.get("children"):
            actual_fix_id = parent["children"][-1].get("id", fix_id)
        try:
            from orchestrator.tools import impl_spawn_task
            spawn_result = impl_spawn_task(actual_fix_id, fix_description, budget_minutes=20)
            print(f"             Fix worker: {spawn_result}")
        except Exception as err:
            print(f"             [WARN] Could not spawn resource-fix worker: {err}")
    else:
        print(f"\n  [RESOURCE] Worker {task_id} exceeded resource limits after fix attempts: {violation}")
        state_mgr.update_task(
            task_id, "failed",
            "Resource limits exceeded repeatedly after retry/fix attempts; likely infeasible as specified",
            f".aura/workspace/tasks/{task_id}/resource_guard.log: {violation}"
        )
        state = state_mgr.load_state()
        task = state_mgr.find_task(task_id, state.get("tasks", []))
        if task is not None:
            task["resource_guard_kills"] = kill_count
            task["last_resource_violation"] = violation
        state_mgr.save_state(state)

    entry["resource_status_recorded"] = True
    return True


def cmd_start(args):
    """Start the orchestrator main loop."""
    print(get_startup_banner())
    global _running, _shutdown_requested, _consecutive_api_errors, _llm_dead

    task_file = args.task_file
    task_file_path = _resolve_task_file(task_file)

    if not os.path.exists(task_file_path):
        print(f"[ERROR] Task file not found: {task_file_path}")
        sys.exit(1)

    _record_task_data_dir(task_file_path, DATA_DIR)

    mission = _extract_mission(task_file_path)
    if not mission:
        print(f"[ERROR] Could not extract mission from task file.")
        sys.exit(1)

    # Use a stable project name based on the project root basename.
    project_name = _project_name_from_cwd()
    active = _get_active_project()

    # Changelog: detect task file changes.
    change_info = get_file_change_info(task_file_path, PROJECTS_DIR, project_name)
    if change_info["is_new"]:
        print(f"\n  [CHANGELOG] New task file; first start")
    elif change_info["is_changed"]:
        print(f"\n  [CHANGELOG] Task file changed; previous hash: {change_info['previous_hash'][:12]}...")
        print(f"              Current hash: {change_info['current_hash'][:12]}...")
    else:
        print(f"\n  [CHANGELOG] Task file unchanged; continuing previous progress")
        if change_info["last_processed_at"]:
            print(f"              Last processed: {change_info['last_processed_at'][:19]}")

    print(f"\n{'='*60}")
    print(f"  Aura Agent - {project_name}")
    print(f"  [AURA] Layer 2 Backend: {AURA_LAYER2_BACKEND}")
    print(f"{'='*60}")
    print(f"  Project: {project_name}")
    print(f"  Project root: {CFG_PROJECT_ROOT}")
    print(f"  Aura dir: {DATA_DIR}")
    print(f"  Task data mapping: {os.path.join(DATA_DIR, 'task_file.json')}")
    print(f"  Task file: {task_file_path}")
    _print_effective_config()
    active = project_name

    # Project switching logic.
    # Core rule: project_name is determined consistently before switching.
    # It should not vary with path separator differences.
    if active and active != project_name:
        print(f"\n[SWITCH] Changing from '{active}' to '{project_name}'")
        _save_project(active)
        if _project_exists(project_name):
            _restore_project(project_name)
        else:
            _create_new_project(project_name, task_file_path, mission)
    elif active and active == project_name:
        state = state_mgr.load_state()
        if state.get("total_cycles", 0) > 0 or state.get("tasks"):
            print(f"\n[CONTINUE] Resuming (Cycle #{state.get('total_cycles', 0)})")
            # If the task file changed, record it in the changelog.
            if change_info["is_changed"] or change_info["is_new"]:
                mark_file_processed(task_file_path, PROJECTS_DIR, project_name,
                                    summary=f"Cycle #{state.get('total_cycles', 0)} resumed; task file change detected")
        else:
            _create_new_project(project_name, task_file_path, mission)
    elif _project_exists(project_name):
        _restore_project(project_name)
    else:
        if active:
            _save_project(active)
        _create_new_project(project_name, task_file_path, mission)

    # Mark the task file as processed in the changelog.
    running_task_ids = {
        worker["task_id"]
        for worker in process_mgr.list_all()
        if worker.get("running")
    }
    reconcile_stats = state_mgr.reconcile_task_file(
        task_file_path,
        mission=mission,
        running_task_ids=running_task_ids,
        task_file_changed=change_info["is_changed"],
    )
    print(
        "  [TASKS] Reconciled task file: "
        f"batch={reconcile_stats.get('batch')}, "
        f"batch_advanced={reconcile_stats.get('batch_advanced')}, "
        f"kept={reconcile_stats['kept']}, "
        f"added={reconcile_stats['added']}, "
        f"updated={reconcile_stats['updated']}, "
        f"archived={reconcile_stats['archived']}, "
        f"removed_completed={reconcile_stats.get('removed_completed', 0)}, "
        f"reopened_auto_completed={reconcile_stats.get('reopened_auto_completed', 0)}, "
        f"planning_needed={reconcile_stats.get('planning_needed', False)}, "
        f"interrupted={reconcile_stats['interrupted']}, "
        f"completed_from_result={reconcile_stats['completed_from_result']}, "
        f"completed_by_user_directive={reconcile_stats['completed_by_user_directive']}"
    )
    _render_progress_safely("task file reconcile")

    mark_file_processed(task_file_path, PROJECTS_DIR, project_name,
                        summary=f"Started task: {mission[:60]}")

    print(f"  Mission: {mission[:120]}")
    _set_active_project(project_name)

    interval = CYCLE_INTERVAL_SECONDS
    print(f"\n{'='*60}")
    print(f"  Main loop - wake every {interval}s ({interval // 60} min)")
    print(f"  Deep review every {DEEP_REVIEW_INTERVAL_CYCLES} cycles (~{DEEP_REVIEW_INTERVAL_CYCLES * interval // 3600}h)")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    def signal_handler(sig, frame):
        global _running, _shutdown_requested
        if _shutdown_requested:
            print("\n[SHUTDOWN] Ctrl+C received again; forcing exit.")
            raise KeyboardInterrupt

        _shutdown_requested = True
        _running = False
        print("\n[SHUTDOWN] Ctrl+C detected; aborting current API/tool cycle...")
        _kill_running_workers()
        raise ShutdownRequested

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    cycle_count = 0
    # R7: Save the initial task-file snapshot for later wake-time diffs.
    save_task_file_snapshot(task_file_path, PROJECTS_DIR, project_name)

    while _running:
        try:
            worker_status_changed = False

            # R7: Check whether the task file changed on each wake.
            wake_change = check_task_file_on_wake(
                task_file_path, PROJECTS_DIR, project_name
            )
            if wake_change["changed"]:
                print(f"\n  [R7] Task file changed: {wake_change['change_summary']}")
                if wake_change["diff_lines"]:
                    diff_preview = wake_change["diff_lines"][:8]
                    for dl in diff_preview:
                        print(f"       {dl[:120]}")
                running_task_ids = {
                    worker["task_id"]
                    for worker in process_mgr.list_all()
                    if worker.get("running")
                }
                reconcile_stats = state_mgr.reconcile_task_file(
                    task_file_path,
                    mission=mission,
                    running_task_ids=running_task_ids,
                    task_file_changed=True,
                )
                print(
                    "  [TASKS] Reconciled changed task file: "
                    f"batch={reconcile_stats.get('batch')}, "
                    f"batch_advanced={reconcile_stats.get('batch_advanced')}, "
                    f"kept={reconcile_stats['kept']}, "
                    f"added={reconcile_stats['added']}, "
                    f"updated={reconcile_stats['updated']}, "
                    f"archived={reconcile_stats['archived']}, "
                    f"removed_completed={reconcile_stats.get('removed_completed', 0)}, "
                    f"reopened_auto_completed={reconcile_stats.get('reopened_auto_completed', 0)}, "
                    f"planning_needed={reconcile_stats.get('planning_needed', False)}"
                )
                _render_progress_safely("changed task file reconcile")

            result = run_cycle(wake_change=wake_change)
            cycle_count += 1
            actual_cycle = result.get("cycle", cycle_count)

            # -- API error tracking --
            if result.get("error"):
                _consecutive_api_errors += 1
                if _consecutive_api_errors >= 3 and not _llm_dead:
                    _llm_dead = True
                    print(f"\n{'!'*60}")
                    print(f"  [BRAIN DEAD] LLM API failed {_consecutive_api_errors} times in a row")
                    print("  The orchestrator model is not responding. Please check:")
                    print("  1. API key validity")
                    print("  2. Account balance or quota")
                    print("  3. DeepSeek service availability")
                    print("  Aura will keep retrying, but manual intervention may be needed.")
                    print(f"{'!'*60}\n")
            elif _llm_dead:
                # LLM recovered!
                print("\n  [RECOVERED] LLM API recovered.")
                _llm_dead = False
                _consecutive_api_errors = 0
            else:
                _consecutive_api_errors = 0

            # -- Layer 2 crash detection --
            tracked = process_mgr.list_all()
            for worker in tracked:
                if not worker["running"]:
                    task_id = worker["task_id"]
                    entry = process_mgr._active_processes.get(task_id, {})
                    if entry.get("killed_at"):
                        if _handle_resource_guard_stop(task_id, entry):
                            worker_status_changed = True
                        continue

                    elapsed = worker["elapsed_minutes"]
                    output_size = worker["output_size"]

                    # If worker produced substantial output, it likely completed successfully
                    if output_size > 0:
                        print(f"\n  [DONE] Worker {task_id} finished (PID {worker['pid']}). "
                              f"Output: {output_size} bytes. Marking completed.")
                        try:
                            state_mgr.update_task(task_id, "completed",
                                f"Worker finished. Output: {output_size} bytes.",
                                f".aura/workspace/tasks/{task_id}/output.jsonl ({output_size} bytes)")
                            worker_status_changed = True
                            try:
                                generate_task_summary(task_id, "completed",
                                    f"Worker finished. Output: {output_size} bytes.",
                                    f".aura/workspace/tasks/{task_id}/output.jsonl ({output_size} bytes)")
                            except Exception:
                                pass
                        except Exception as state_err:
                            print(f"    [WARN] Could not update task: {state_err}")
                    else:
                        # No output = genuine failure
                        print(f"\n  [CRASH] Worker {task_id} (PID {worker['pid']}) died with NO output.")
                        try:
                            state_mgr.update_task(task_id, "failed",
                                f"Worker died after {elapsed}min with no output.", "(no output)")
                            worker_status_changed = True
                            try:
                                generate_task_summary(task_id, "failed",
                                    f"Worker died after {elapsed}min with no output.", "(no output)")
                            except Exception:
                                pass
                        except Exception as state_err:
                            print(f"    [WARN] Could not update task: {state_err}")

                    process_mgr._active_processes[task_id]["killed_at"] = datetime.now().isoformat()
                    process_mgr._active_processes[task_id]["running"] = False

            # -- Hourly deep review --
            if worker_status_changed:
                _render_progress_safely("worker status update")

            if actual_cycle % DEEP_REVIEW_INTERVAL_CYCLES == 0:
                print(f"\n  {'-'*50}")
                print(f"  [DEEP REVIEW] Scheduled deep review (Cycle #{actual_cycle})")
                print(f"  {'-'*50}")

                if _review_available:
                    try:
                        review_result = review_cycle(force=True)
                        if review_result.get("error"):
                            print(f"  Review error: {review_result['error']}")
                        elif review_result.get("recommendations"):
                            print(f"  Recommendations:")
                            for r in review_result["recommendations"]:
                                print(f"    - {r}")
                    except Exception as review_err:
                        print(f"  Review engine failed: {review_err}")
                else:
                    print(f"  (Review engine not loaded: {_review_import_error})")

            # -- Periodic light review (every REVIEW_NUDGE_INTERVAL) --
            elif _review_available and actual_cycle % REVIEW_NUDGE_INTERVAL == 0:
                try:
                    review_cycle(force=False)
                except Exception:
                    pass  # Silent fail for light reviews

            # -- Status line --
            status_parts = [f"Cycle #{cycle_count}"]
            if result.get("tool_calls", 0) > 0:
                status_parts.append(f"{result['tool_calls']} tool calls")
            if result.get("activity_mode"):
                status_parts.append(f"mode: {result['activity_mode']}")
            if _llm_dead:
                status_parts.append("BRAIN DEAD")
            print(f"  [{' | '.join(status_parts)}]")

            # R7: If the task file changed, save a new snapshot for next diff.
            if wake_change.get("mtime_changed"):
                save_task_file_snapshot(task_file_path, PROJECTS_DIR, project_name)

        except ShutdownRequested:
            print("\n[INTERRUPT] Current cycle interrupted.")
            _running = False
            break
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Ctrl+C detected.")
            _running = False
            _kill_running_workers()
            break
        except Exception as e:
            print(f"\n[ERROR] Unexpected crash in cycle: {e}")
            _render_progress_safely("unexpected cycle crash")
            import traceback
            traceback.print_exc()

        if not _running:
            break

        try:
            _save_project(project_name)
        except Exception as save_err:
            print(f"  [WARN] Project save failed (will retry next cycle): {save_err}")

        if worker_status_changed:
            print("[Watchdog] Worker status changed; starting next cycle immediately.")
            continue

        _sleep_until_next_wake(interval)

    _kill_running_workers()
    print(f"\n[SHUTDOWN] {project_name} saved. {cycle_count} cycles completed.")
    try:
        _save_project(project_name)
    except Exception as save_err:
        print(f"  [WARN] Final project save failed: {save_err}")


def _clear_task_data_dir(task_file_path: str) -> None:
    target = os.path.abspath(DATA_DIR)
    if target in {os.path.abspath(os.sep), os.path.abspath(CFG_PROJECT_ROOT)}:
        raise RuntimeError(f"Refusing to clear unsafe data directory: {target}")

    metadata_path = os.path.join(target, "task_file.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            recorded = os.path.normcase(os.path.abspath(metadata.get("task_file", "")))
            requested = os.path.normcase(os.path.abspath(task_file_path))
            if recorded and recorded != requested:
                raise RuntimeError(
                    "Refusing to clear data directory because its task_file.json "
                    f"points to {metadata.get('task_file')}, not {task_file_path}"
                )
        except json.JSONDecodeError as err:
            raise RuntimeError(f"Refusing to clear data directory with invalid metadata: {err}") from err

    os.makedirs(target, exist_ok=True)
    for name in os.listdir(target):
        path = os.path.join(target, name)
        if os.path.isdir(path):
            _rmtree_force(path)
        else:
            os.remove(path)


def cmd_restart(args):
    """Clear the task-specific Aura data directory, then start fresh."""
    task_file_path = _resolve_task_file(args.task_file)
    if not os.path.exists(task_file_path):
        print(f"[ERROR] Task file not found: {task_file_path}")
        sys.exit(1)

    print(f"[RESTART] Clearing Aura data for task file: {task_file_path}")
    print(f"[RESTART] Data directory: {DATA_DIR}")
    _kill_running_workers(prefix="[RESTART]")
    _clear_task_data_dir(task_file_path)
    _record_task_data_dir(task_file_path, DATA_DIR)
    print("[RESTART] Data cleared. Starting fresh run.")
    cmd_start(args)


def cmd_status():
    active = _get_active_project()
    if not active:
        print("No active project. Start one with: aura start --task-file=...")
        return
    state = state_mgr.load_state()
    print(f"\n  Project: {active}")
    print(f"  Mission: {state.get('mission', '?')[:120]}")
    print(f"  Cycles: {state.get('total_cycles', 0)}")
    print(f"  Active tasks: {len(state.get('active_tasks', []))}")
    print(f"\n  Task Tree:")
    print(state_mgr.get_task_tree_summary())


def cmd_progress():
    progress_mgr.render_progress()
    print(f"Progress report written to {os.path.join(STATE_DIR, 'progress.md')}")


def cmd_projects():
    base_dir = _default_aura_base_dir()
    index_path = _task_index_path(base_dir)
    active_data_dir = None
    marker = _active_task_data_marker(base_dir)
    if os.path.exists(marker):
        try:
            active_data_dir = Path(marker).read_text(encoding="utf-8").strip()
        except OSError:
            active_data_dir = None

    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            print("\n  Task-file data directories:")
            for item in sorted(index.get("tasks", {}).values(), key=lambda x: x.get("task_file", "")):
                marker_text = " ACTIVE" if os.path.abspath(item.get("data_dir", "")) == os.path.abspath(active_data_dir or "") else ""
                print(f"    {item.get('task_file', '?')}")
                print(f"      -> {item.get('data_dir', '?')}{marker_text}")
        except (OSError, json.JSONDecodeError) as err:
            print(f"Could not read task index: {err}")

    if not os.path.exists(PROJECTS_DIR):
        if not os.path.exists(index_path):
            print("No projects yet.")
        return
    active = _get_active_project()
    print(f"\n  Saved projects:")
    for name in sorted(os.listdir(PROJECTS_DIR)):
        pdir = os.path.join(PROJECTS_DIR, name)
        if not os.path.isdir(pdir):
            continue
        sf = os.path.join(pdir, "state", "state.json")
        if os.path.exists(sf):
            with open(sf, "r", encoding="utf-8") as f:
                s = json.load(f)
            marker = " -> ACTIVE" if name == active else ""
            print(f"    {name}: {s.get('mission', '?')[:80]} ({s.get('total_cycles', 0)} cycles){marker}")


def cmd_history():
    state = state_mgr.load_state()
    decisions = state.get("decision_log", [])
    if not decisions:
        print("No decisions recorded yet.")
        return
    print(f"\n  Decision History ({len(decisions)} entries):")
    for d in decisions[-30:]:
        print(f"    [{d['time'][:19]}] {d['task_id']}: "
              f"{d.get('old_status', '?')} -> {d.get('new_status', '?')} - {d.get('reason', '')[:60]}")


def cmd_changelog():
    """Show the active project's task-file changelog."""
    from orchestrator.changelog import load_changelog, get_changelog_path

    active = _get_active_project()
    if not active:
        print("No active project.")
        return

    state = state_mgr.load_state()
    task_file = state.get("task_file", "")
    if not task_file:
        print("No task file recorded in state.")
        return

    task_file_path = os.path.join(CFG_PROJECT_ROOT, task_file)
    changelog_path = get_changelog_path(PROJECTS_DIR, active, task_file_path)
    changelog = load_changelog(changelog_path)

    print(f"\n  Changelog for: {task_file}")
    print(f"  Project: {active}")
    print(f"  Entries: {len(changelog.get('entries', []))}")
    print(f"  Processed items: {len(changelog.get('processed_items', {}))}")
    print()

    for i, entry in enumerate(changelog.get("entries", [])):
        print(f"  [{i}] {entry.get('processed_at', '?')[:19]}")
        print(f"      Hash: {entry.get('file_hash', '')[:16]}...")
        print(f"      Summary: {entry.get('summary', '')[:80]}")
        print()

    print(f"  Processed items ({len(changelog.get('processed_items', {}))}):")
    for fp, item in list(changelog.get("processed_items", {}).items())[:10]:
        print(f"    {fp[:12]}... -> {item.get('text', '')[:60]}")
        print(f"      at {item.get('processed_at', '')[:19]}")


def _collect_active_task_files() -> list[str]:
    tasks_dir = os.path.join(CFG_PROJECT_ROOT, "tasks")
    active_task_files = []
    if os.path.exists(tasks_dir):
        for f in os.listdir(tasks_dir):
            if f.endswith(".md"):
                active_task_files.append(os.path.join(tasks_dir, f))
    return active_task_files


def cmd_cleanup():
    """Show orphan projects that no longer have a matching task file."""
    from orchestrator.changelog import cleanup_orphan_projects

    orphans = cleanup_orphan_projects(PROJECTS_DIR, _collect_active_task_files())
    if not orphans:
        print("\n  No orphan projects to clean.")
        return

    print(f"\n  Found {len(orphans)} orphan project(s) without a matching task.md file:")
    for name in orphans:
        pdir = os.path.join(PROJECTS_DIR, name)
        size = 0
        for root, dirs, files in os.walk(pdir):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    size += os.path.getsize(fp)
                except OSError:
                    pass
        print(f"    {name}/ ({size/1024:.0f} KB)")

    print("\n  Use `aura cleanup --force` to delete these projects.")
    print("  Or delete manually: rmdir /s projects/<name>")


def cmd_cleanup_force():
    """Delete orphan projects that no longer have a matching task file."""
    from orchestrator.changelog import cleanup_orphan_projects

    orphans = cleanup_orphan_projects(PROJECTS_DIR, _collect_active_task_files())
    if not orphans:
        print("\n  No orphan projects to clean.")
        return

    for name in orphans:
        pdir = os.path.join(PROJECTS_DIR, name)
        print(f"  Deleting: {name}/")
        _rmtree_force(pdir)

    print(f"\n  Cleaned {len(orphans)} orphan project(s).")


def _extract_mission(task_file: str) -> str:
    """Extract the mission description from a .md task file.

    Priority:
    1. Extract from a line starting with '# ' (existing behavior).
    2. Fall back to the first non-empty, non-comment line.
    3. Return "" if the file has no usable content.

    Args:
        task_file: Path to the .md task file.

    Returns:
        The extracted mission string, or "" if nothing usable is found.
    """
    path = Path(task_file)

    if not path.exists():
        return ""

    content = path.read_text(encoding="utf-8")

    fallback_lines = []
    # Pattern to detect HTML comment lines in markdown.
    comment_pattern = re.compile(r"^\s*<!--")

    for line in content.splitlines():
        stripped = line.strip()

        # Skip empty lines.
        if not stripped:
            continue

        # Skip HTML comment lines (e.g. <!-- comment -->).
        if comment_pattern.match(stripped):
            continue

        # Priority 1: extract from '# Title' heading.
        if stripped.startswith("# "):
            return stripped[2:].strip()

        fallback_lines.append(stripped)

    return "\n".join(fallback_lines).strip()


def main():
    known_commands = {
        "start", "restart", "status", "progress", "projects", "history", "changelog",
        "cleanup", "wake", "setup", "summaries", "cache-stats",
        "changelog-overview", "clean-workspaces",
    }
    option_takes_value = {"--config", "-c", "--data-dir"}
    skip_next = False
    for idx, arg in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if arg in option_takes_value:
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        if arg not in known_commands:
            sys.argv[idx:idx + 1] = ["start", "--task-file", arg]
        break
    parser = argparse.ArgumentParser(description="Aura Agent - Autonomous Task Orchestrator")
    parser.add_argument("--config", "-c", help="Path to .env config file", default=None)
    parser.add_argument("--data-dir", default=None,
                        help="Data directory for process files (memory, state, workspace). "
                             "Default: ./.aura/<task-file-name>-<path-hash>/ for task-file commands.")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("start", help="Start the orchestrator loop")
    p.add_argument("--task-file", required=True, help="Path to task .md file")

    p_restart = sub.add_parser("restart", help="Clear this task file's Aura data and start fresh")
    p_restart.add_argument("task_file_pos", nargs="?", help="Path to task .md file")
    p_restart.add_argument("--task-file", dest="task_file_opt", help="Path to task .md file")

    sub.add_parser("status", help="Show current project status")
    sub.add_parser("progress", help="Generate progress report")
    sub.add_parser("projects", help="List saved projects")
    sub.add_parser("history", help="Show decision history")
    sub.add_parser("changelog", help="Show task-file changelog for the active project")
    p_cleanup = sub.add_parser("cleanup", help="Clean orphan projects without a matching task.md file")
    p_cleanup.add_argument("--force", action="store_true",
                           help="Delete orphan projects without confirmation")

    register_commands(sub)

    args = parser.parse_args()
    if args.command == "restart":
        args.task_file = args.task_file_opt or args.task_file_pos
        if not args.task_file:
            parser.error("restart requires a task file, e.g. aura restart tasks/task.md")

    # Apply --data-dir if passed via argparse (overrides early parse for edge cases)
    if args.data_dir:
        os.environ["AURA_DATA_DIR"] = os.path.expanduser(args.data_dir)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "start": cmd_start,
        "restart": cmd_restart,
        "status": cmd_status,
        "progress": cmd_progress,
        "projects": cmd_projects,
        "history": cmd_history,
        "changelog": cmd_changelog,
        "cleanup": cmd_cleanup,
    }
    cmd = cmds.get(args.command)
    if cmd:
        if args.command in {"start", "restart"}:
            cmd(args)
        elif args.command == "cleanup":
            if args.force:
                cmd_cleanup_force()
            else:
                cmd_cleanup()
        else:
            cmd()
    elif hasattr(args, "func"):
        args.func(args, None)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
