"""Process manager for Layer 2 (Claude Code CLI) workers.

Handles spawning, killing, and monitoring Claude Code processes on Windows.
"""

import os
import sys
import json
import time
import signal
import math
import re
import shutil
import threading
import subprocess
from datetime import datetime
from typing import Optional

import psutil

from .config import (
    DEFAULT_MAX_TURNS,
    WORKER_RESOURCE_GUARD_ENABLED,
    WORKER_RESOURCE_POLL_SECONDS,
    WORKER_RESOURCE_AVG_WINDOW_SECONDS,
    WORKER_RESOURCE_VIOLATION_STRIKES,
    WORKER_MAX_CPU_PERCENT,
    WORKER_MAX_SYSTEM_MEMORY_PERCENT,
    WORKER_MAX_GPU_UTIL_PERCENT,
    WORKER_MAX_GPU_MEMORY_PERCENT,
    WORKER_MAX_SYSTEM_MEMORY_GB,
    WORKER_MIN_SYSTEM_MEMORY_FREE_GB,
    WORKER_MAX_GPU_MEMORY_GB,
    WORKER_MIN_GPU_MEMORY_FREE_GB,
    WORKER_CUDA_VISIBLE_DEVICES,
    AURA_LAYER2_BACKEND,
    AURA_CLAUDE_BIN,
    AURA_DEEPSEEK_API_KEY,
    AURA_DSCODE_MAX_TURNS,
    AURA_DSCODE_MODEL,
    AURA_DSCODE_BASE_URL,
    WAKEUP_FILE,
    get_workspace_dir,
)

_active_processes: dict[str, dict] = {}
_dead_record_cache: dict[str, float] = {}
_process_lock = threading.RLock()
_resource_monitor_thread: threading.Thread | None = None
_resource_monitor_stop = threading.Event()

_BYTES_PER_GB = 1024 ** 3
_PROCESS_START_TOLERANCE_SECONDS = 30


def _gb_to_mb(value: float) -> float:
    return value * 1024


def _resource_limits() -> dict:
    return {
        "enabled": WORKER_RESOURCE_GUARD_ENABLED,
        "poll_seconds": WORKER_RESOURCE_POLL_SECONDS,
        "avg_window_seconds": WORKER_RESOURCE_AVG_WINDOW_SECONDS,
        "violation_strikes": WORKER_RESOURCE_VIOLATION_STRIKES,
        "max_cpu_percent": WORKER_MAX_CPU_PERCENT,
        "max_system_memory_percent": WORKER_MAX_SYSTEM_MEMORY_PERCENT,
        "max_gpu_util_percent": WORKER_MAX_GPU_UTIL_PERCENT,
        "max_gpu_memory_percent": WORKER_MAX_GPU_MEMORY_PERCENT,
        "max_system_memory_gb": WORKER_MAX_SYSTEM_MEMORY_GB,
        "min_system_memory_free_gb": WORKER_MIN_SYSTEM_MEMORY_FREE_GB,
        "max_gpu_memory_gb": WORKER_MAX_GPU_MEMORY_GB,
        "min_gpu_memory_free_gb": WORKER_MIN_GPU_MEMORY_FREE_GB,
        "cuda_visible_devices": WORKER_CUDA_VISIBLE_DEVICES,
    }


def _has_hard_resource_limits(limits: dict | None = None) -> bool:
    limits = limits or _resource_limits()
    return bool(
        limits.get("enabled")
        and (
            limits.get("max_cpu_percent", 0) > 0
            or limits.get("max_system_memory_percent", 0) > 0
            or limits.get("max_gpu_util_percent", 0) > 0
            or limits.get("max_gpu_memory_percent", 0) > 0
            or limits.get("max_system_memory_gb", 0) > 0
            or limits.get("min_system_memory_free_gb", 0) > 0
            or limits.get("max_gpu_memory_gb", 0) > 0
            or limits.get("min_gpu_memory_free_gb", 0) > 0
        )
    )


def resource_policy_text() -> str:
    """Return a human-readable resource policy for generated task.md files."""
    limits = _resource_limits()
    if not limits["enabled"]:
        return "- Resource guard: disabled"

    lines = [
        f"- Resource guard: enabled; poll interval {limits['poll_seconds']}s; "
        f"rolling average window {limits['avg_window_seconds']}s"
    ]
    if limits["max_cpu_percent"] > 0:
        lines.append(f"- CPU rolling-average ceiling per worker process tree: {limits['max_cpu_percent']}% of total system CPU")
    if limits["max_system_memory_percent"] > 0:
        lines.append(f"- Host memory ceiling per worker process tree: {limits['max_system_memory_percent']}% of total system RAM")
    if limits["max_gpu_util_percent"] > 0:
        lines.append(f"- GPU utilization rolling-average ceiling per worker process tree: {limits['max_gpu_util_percent']}% when per-process GPU utilization is available")
    if limits["max_gpu_memory_percent"] > 0:
        lines.append(f"- GPU memory ceiling per worker process tree: {limits['max_gpu_memory_percent']}% of visible GPU memory")
    if limits["max_gpu_memory_gb"] > 0:
        lines.append(f"- Optional absolute GPU memory ceiling per worker: {limits['max_gpu_memory_gb']}GB")
    if limits["min_gpu_memory_free_gb"] > 0:
        lines.append(f"- GPU free-memory reserve before spawning: {limits['min_gpu_memory_free_gb']}GB")
    if limits["max_system_memory_gb"] > 0:
        lines.append(f"- Host memory hard ceiling per worker process tree: {limits['max_system_memory_gb']}GB RSS")
    if limits["min_system_memory_free_gb"] > 0:
        lines.append(f"- System memory reserve: keep at least {limits['min_system_memory_free_gb']}GB free")
    if limits["cuda_visible_devices"]:
        lines.append(f"- CUDA_VISIBLE_DEVICES: {limits['cuda_visible_devices']}")
    if len(lines) == 1:
        lines.append("- No numeric resource ceilings are configured.")
    lines.append("- These limits apply to this worker's own process tree, not unrelated programs on the machine.")
    lines.append("- Sustained rolling-average violations will cause Aura to terminate this worker and report the exact reason.")
    lines.append("- If terminated once, the original task gets one safer retry. If terminated twice, Aura creates a resource-fix subtask before continuing.")
    lines.append("- Do not enable CPU/NVMe/model offload unless the task explicitly requires it.")
    return "\n".join(lines)


def signal_wakeup(reason: str = "") -> str:
    """Signal the main loop to wake before the fixed sleep interval ends."""
    try:
        os.makedirs(os.path.dirname(WAKEUP_FILE), exist_ok=True)
        with open(WAKEUP_FILE, "w", encoding="utf-8") as f:
            f.write(reason or "wakeup")
        return f"OK: Wakeup signaled ({WAKEUP_FILE})"
    except Exception as e:
        return f"ERROR signaling wakeup: {e}"


def _worker_env(base_env: dict | None = None) -> dict:
    env = (base_env or os.environ).copy()
    limits = _resource_limits()
    if limits["cuda_visible_devices"]:
        env["CUDA_VISIBLE_DEVICES"] = limits["cuda_visible_devices"]

    # These do not hard-cap memory, but they reduce common framework
    # preallocation behavior so the watchdog has room to intervene.
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    env["AURA_WORKER_RESOURCE_GUARD"] = "1" if limits["enabled"] else "0"
    env["AURA_WORKER_MAX_CPU_PERCENT"] = str(limits["max_cpu_percent"])
    env["AURA_WORKER_MAX_SYSTEM_MEMORY_PERCENT"] = str(limits["max_system_memory_percent"])
    env["AURA_WORKER_MAX_GPU_UTIL_PERCENT"] = str(limits["max_gpu_util_percent"])
    env["AURA_WORKER_MAX_GPU_MEMORY_PERCENT"] = str(limits["max_gpu_memory_percent"])
    env["AURA_WORKER_MAX_SYSTEM_MEMORY_GB"] = str(limits["max_system_memory_gb"])
    env["AURA_WORKER_MIN_SYSTEM_MEMORY_FREE_GB"] = str(limits["min_system_memory_free_gb"])
    env["AURA_WORKER_MAX_GPU_MEMORY_GB"] = str(limits["max_gpu_memory_gb"])
    env["AURA_WORKER_MIN_GPU_MEMORY_FREE_GB"] = str(limits["min_gpu_memory_free_gb"])
    env["AURA_WORKER_FORBID_OFFLOAD"] = "1"
    return env


def _visible_gpu_indexes() -> set[int] | None:
    value = WORKER_CUDA_VISIBLE_DEVICES.strip()
    if not value:
        return None
    indexes: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            indexes.add(int(part))
    return indexes or None


def _run_nvidia_smi(args: list[str]) -> subprocess.CompletedProcess | None:
    executable = shutil.which("nvidia-smi") or "nvidia-smi"
    try:
        return subprocess.run(
            [executable, *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _query_gpu_free_memory_mb() -> dict[int, float] | None:
    proc = _run_nvidia_smi([
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if proc is None or proc.returncode != 0:
        return None

    visible = _visible_gpu_indexes()
    result: dict[int, float] = {}
    for line in proc.stdout.splitlines():
        if not line.strip() or "," not in line:
            continue
        index_text, free_text = [part.strip() for part in line.split(",", 1)]
        try:
            index = int(index_text)
            free_mb = float(free_text)
        except ValueError:
            continue
        if visible is None or index in visible:
            result[index] = free_mb
    return result


def _query_visible_gpu_total_memory_mb() -> float | None:
    proc = _run_nvidia_smi([
        "--query-gpu=index,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if proc is None or proc.returncode != 0:
        return None

    visible = _visible_gpu_indexes()
    total = 0.0
    for line in proc.stdout.splitlines():
        if not line.strip() or "," not in line:
            continue
        index_text, total_text = [part.strip() for part in line.split(",", 1)]
        try:
            index = int(index_text)
            total_mb = float(total_text)
        except ValueError:
            continue
        if visible is None or index in visible:
            total += total_mb
    return total or None


def _query_gpu_memory_by_pid_mb(pids: set[int]) -> float | None:
    if not pids:
        return 0.0

    proc = _run_nvidia_smi([
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if proc is None or proc.returncode != 0:
        return None

    total = 0.0
    for line in proc.stdout.splitlines():
        if not line.strip() or "," not in line:
            continue
        pid_text, memory_text = [part.strip() for part in line.split(",", 1)]
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid not in pids:
            continue
        digits = "".join(ch for ch in memory_text if ch.isdigit() or ch == ".")
        if digits:
            total += float(digits)
    return total


def _query_gpu_util_by_pid_percent(pids: set[int]) -> float | None:
    if not pids:
        return 0.0

    # nvidia-smi pmon reports per-process SM utilization on many NVIDIA
    # driver modes. It is best-effort; when unavailable, utilization checks
    # are skipped while GPU memory checks still work.
    proc = _run_nvidia_smi(["pmon", "-c", "1", "-s", "u"])
    if proc is None or proc.returncode != 0:
        return None

    total = 0.0
    found = False
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = re.split(r"\s+", stripped)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if pid not in pids:
            continue
        try:
            sm_util = float(parts[3])
        except ValueError:
            continue
        if sm_util >= 0:
            total += sm_util
            found = True
    return min(total, 100.0) if found else None


def _resource_preflight() -> tuple[bool, str]:
    limits = _resource_limits()
    if not limits["enabled"]:
        return True, "Resource guard disabled."

    min_free_gb = limits["min_system_memory_free_gb"]
    if min_free_gb > 0:
        available_gb = psutil.virtual_memory().available / _BYTES_PER_GB
        if available_gb < min_free_gb:
            return (
                False,
                f"System memory reserve would be violated: "
                f"{available_gb:.1f}GB free < {min_free_gb:.1f}GB required.",
            )

    min_gpu_free_gb = limits["min_gpu_memory_free_gb"]
    if min_gpu_free_gb > 0:
        gpu_free = _query_gpu_free_memory_mb()
        if gpu_free is None:
            return False, "Cannot verify GPU free-memory reserve because nvidia-smi is unavailable."
        if not gpu_free:
            return False, "No visible NVIDIA GPU found for GPU free-memory reserve check."
        best_free_gb = max(gpu_free.values()) / 1024
        if best_free_gb < min_gpu_free_gb:
            return (
                False,
                f"GPU memory reserve would be violated: best visible GPU has "
                f"{best_free_gb:.1f}GB free < {min_gpu_free_gb:.1f}GB required.",
            )

    return True, "Resource preflight passed."


def _process_tree(pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _process_tree_metrics(pid: int, sample_cpu: bool = False) -> dict:
    procs = _process_tree(pid)
    raw_cpu = 0.0
    rss_bytes = 0
    live_pids: set[int] = set()

    for proc in procs:
        try:
            with proc.oneshot():
                live_pids.add(proc.pid)
                if sample_cpu:
                    raw_cpu += proc.cpu_percent(interval=0.02)
                else:
                    raw_cpu += proc.cpu_percent(interval=None)
                rss_bytes += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    cpu_count = psutil.cpu_count(logical=True) or 1
    system_cpu_percent = raw_cpu / cpu_count
    gpu_memory_mb = _query_gpu_memory_by_pid_mb(live_pids)
    total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
    gpu_total_memory_mb = _query_visible_gpu_total_memory_mb()
    gpu_memory_percent = None
    if gpu_memory_mb is not None and gpu_total_memory_mb:
        gpu_memory_percent = (gpu_memory_mb / gpu_total_memory_mb) * 100
    gpu_util_percent = _query_gpu_util_by_pid_percent(live_pids)
    return {
        "cpu_percent": system_cpu_percent,
        "process_cpu_percent": raw_cpu,
        "memory_mb": rss_bytes / (1024 * 1024),
        "memory_percent": (rss_bytes / (1024 * 1024)) / total_memory_mb * 100,
        "gpu_memory_mb": gpu_memory_mb,
        "gpu_memory_percent": gpu_memory_percent,
        "gpu_util_percent": gpu_util_percent,
        "child_count": max(0, len(live_pids) - 1),
        "pids": sorted(live_pids),
    }


def _apply_cpu_affinity(pid: int) -> list[int] | None:
    limit = WORKER_MAX_CPU_PERCENT
    cpu_count = psutil.cpu_count(logical=True) or 1
    if limit <= 0 or limit >= 100 or cpu_count <= 1:
        return None

    allowed_count = max(1, math.floor(cpu_count * (limit / 100.0)))
    cpus = list(range(allowed_count))
    try:
        proc = psutil.Process(pid)
        proc.cpu_affinity(cpus)
        return cpus
    except (AttributeError, psutil.Error, ValueError):
        return None


def _record_resource_sample(entry: dict, metrics: dict) -> dict:
    now = time.time()
    limits = entry.get("resource_limits") or _resource_limits()
    window = max(1, int(limits.get("avg_window_seconds", WORKER_RESOURCE_AVG_WINDOW_SECONDS)))
    samples = entry.setdefault("resource_samples", [])
    samples.append({"time": now, **metrics})
    cutoff = now - window
    entry["resource_samples"] = [sample for sample in samples if sample.get("time", now) >= cutoff]
    return _aggregate_resource_samples(entry["resource_samples"])


def _average_present(samples: list[dict], key: str) -> float | None:
    values = [sample[key] for sample in samples if sample.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _max_present(samples: list[dict], key: str) -> float | None:
    values = [sample[key] for sample in samples if sample.get(key) is not None]
    if not values:
        return None
    return max(values)


def _aggregate_resource_samples(samples: list[dict]) -> dict:
    return {
        "sample_count": len(samples),
        "avg_cpu_percent": _average_present(samples, "cpu_percent") or 0.0,
        "peak_cpu_percent": _max_present(samples, "cpu_percent") or 0.0,
        "avg_memory_percent": _average_present(samples, "memory_percent") or 0.0,
        "peak_memory_percent": _max_present(samples, "memory_percent") or 0.0,
        "avg_memory_mb": _average_present(samples, "memory_mb") or 0.0,
        "peak_memory_mb": _max_present(samples, "memory_mb") or 0.0,
        "avg_gpu_util_percent": _average_present(samples, "gpu_util_percent"),
        "peak_gpu_util_percent": _max_present(samples, "gpu_util_percent"),
        "avg_gpu_memory_percent": _average_present(samples, "gpu_memory_percent"),
        "peak_gpu_memory_percent": _max_present(samples, "gpu_memory_percent"),
        "avg_gpu_memory_mb": _average_present(samples, "gpu_memory_mb"),
        "peak_gpu_memory_mb": _max_present(samples, "gpu_memory_mb"),
    }


def _evaluate_resource_violation(entry: dict, metrics: dict) -> str | None:
    limits = entry.get("resource_limits") or _resource_limits()
    if not _has_hard_resource_limits(limits):
        return None

    aggregate = _record_resource_sample(entry, metrics)
    min_samples = max(2, min(3, limits.get("violation_strikes", WORKER_RESOURCE_VIOLATION_STRIKES)))
    candidates: list[tuple[str, str]] = []
    if aggregate["sample_count"] < min_samples:
        return None

    if limits["max_cpu_percent"] > 0 and aggregate["avg_cpu_percent"] > limits["max_cpu_percent"]:
        candidates.append((
            "cpu",
            f"own worker CPU rolling avg {aggregate['avg_cpu_percent']:.1f}% "
            f"(peak {aggregate['peak_cpu_percent']:.1f}%) > {limits['max_cpu_percent']:.1f}% total-system limit",
        ))
    if limits["max_system_memory_percent"] > 0 and aggregate["avg_memory_percent"] > limits["max_system_memory_percent"]:
        candidates.append((
            "memory_percent",
            f"own worker host memory rolling avg {aggregate['avg_memory_percent']:.1f}% "
            f"(peak {aggregate['peak_memory_percent']:.1f}%) > {limits['max_system_memory_percent']:.1f}% RAM limit",
        ))
    if limits["max_system_memory_gb"] > 0:
        memory_gb = aggregate["avg_memory_mb"] / 1024
        if memory_gb > limits["max_system_memory_gb"]:
            candidates.append((
                "memory",
                f"own worker host RSS rolling avg {memory_gb:.1f}GB > {limits['max_system_memory_gb']:.1f}GB limit",
            ))
    if limits["min_system_memory_free_gb"] > 0 and metrics["memory_mb"] > 128:
        free_gb = psutil.virtual_memory().available / _BYTES_PER_GB
        if free_gb < limits["min_system_memory_free_gb"]:
            candidates.append((
                "system_free_memory",
                f"system memory free {free_gb:.1f}GB < {limits['min_system_memory_free_gb']:.1f}GB reserve while worker is active",
            ))
    if (
        limits["max_gpu_util_percent"] > 0
        and aggregate["avg_gpu_util_percent"] is not None
        and aggregate["avg_gpu_util_percent"] > limits["max_gpu_util_percent"]
    ):
        candidates.append((
            "gpu_util",
            f"own worker GPU utilization rolling avg {aggregate['avg_gpu_util_percent']:.1f}% "
            f"(peak {aggregate['peak_gpu_util_percent']:.1f}%) > {limits['max_gpu_util_percent']:.1f}% limit",
        ))
    if (
        limits["max_gpu_memory_percent"] > 0
        and aggregate["avg_gpu_memory_percent"] is not None
        and aggregate["avg_gpu_memory_percent"] > limits["max_gpu_memory_percent"]
    ):
        candidates.append((
            "gpu_memory_percent",
            f"own worker GPU memory rolling avg {aggregate['avg_gpu_memory_percent']:.1f}% "
            f"(peak {aggregate['peak_gpu_memory_percent']:.1f}%) > {limits['max_gpu_memory_percent']:.1f}% limit",
        ))
    if limits["max_gpu_memory_gb"] > 0 and aggregate["avg_gpu_memory_mb"] is not None:
        gpu_gb = aggregate["avg_gpu_memory_mb"] / 1024
        if gpu_gb > limits["max_gpu_memory_gb"]:
            candidates.append((
                "gpu_memory",
                f"own worker GPU memory rolling avg {gpu_gb:.1f}GB > {limits['max_gpu_memory_gb']:.1f}GB limit",
            ))
    if limits["min_gpu_memory_free_gb"] > 0 and metrics.get("gpu_memory_mb"):
        gpu_free = _query_gpu_free_memory_mb()
        if gpu_free:
            best_free_gb = max(gpu_free.values()) / 1024
            if best_free_gb < limits["min_gpu_memory_free_gb"]:
                candidates.append((
                    "gpu_free_memory",
                    f"best visible GPU free {best_free_gb:.1f}GB < {limits['min_gpu_memory_free_gb']:.1f}GB reserve while worker is using GPU",
                ))

    if not candidates:
        entry["resource_strikes"] = {}
        return None

    strikes = entry.setdefault("resource_strikes", {})
    key, reason = candidates[0]
    strikes[key] = strikes.get(key, 0) + 1
    if strikes[key] >= max(1, limits.get("violation_strikes", WORKER_RESOURCE_VIOLATION_STRIKES)):
        return reason
    return None


def _mark_resource_violation_and_kill(task_id: str, reason: str) -> None:
    with _process_lock:
        entry = _active_processes.get(task_id)
        if not entry or entry.get("resource_violation"):
            return
        entry["resource_violation"] = reason
        try:
            guard_log = os.path.join(entry["task_dir"], "resource_guard.log")
            with open(guard_log, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] Resource guard kill: {reason}\n")
                f.write("Aura will wake the orchestrator to replan a smaller/safe retry when possible.\n")
        except Exception:
            pass
        _write_process_record(entry)

    print(f"[ResourceGuard] Killing {task_id}: {reason}")
    kill(task_id)
    with _process_lock:
        entry = _active_processes.get(task_id)
        if entry:
            entry["resource_violation"] = reason
            _write_process_record(entry)
    signal_wakeup(f"resource guard killed {task_id}: {reason}")


def _resource_monitor_loop() -> None:
    poll = max(2, WORKER_RESOURCE_POLL_SECONDS)
    while not _resource_monitor_stop.wait(poll):
        with _process_lock:
            items = list(_active_processes.items())
        for task_id, entry in items:
            if entry.get("killed_at") or not _entry_process_is_alive(entry):
                continue
            metrics = _process_tree_metrics(entry["pid"], sample_cpu=True)
            aggregate = _aggregate_resource_samples(entry.get("resource_samples", []))
            entry["resource_metrics"] = {
                "cpu_percent": round(metrics["cpu_percent"], 1),
                "avg_cpu_percent": round(aggregate["avg_cpu_percent"], 1),
                "memory_mb": round(metrics["memory_mb"], 1),
                "memory_percent": round(metrics["memory_percent"], 1),
                "avg_memory_percent": round(aggregate["avg_memory_percent"], 1),
                "gpu_memory_mb": None if metrics["gpu_memory_mb"] is None else round(metrics["gpu_memory_mb"], 1),
                "gpu_memory_percent": None if metrics["gpu_memory_percent"] is None else round(metrics["gpu_memory_percent"], 1),
                "avg_gpu_memory_percent": None if aggregate["avg_gpu_memory_percent"] is None else round(aggregate["avg_gpu_memory_percent"], 1),
                "gpu_util_percent": None if metrics["gpu_util_percent"] is None else round(metrics["gpu_util_percent"], 1),
                "avg_gpu_util_percent": None if aggregate["avg_gpu_util_percent"] is None else round(aggregate["avg_gpu_util_percent"], 1),
            }
            reason = _evaluate_resource_violation(entry, metrics)
            if reason:
                _mark_resource_violation_and_kill(task_id, reason)


def _ensure_resource_monitor() -> None:
    global _resource_monitor_thread
    if not WORKER_RESOURCE_GUARD_ENABLED:
        return
    if _resource_monitor_thread and _resource_monitor_thread.is_alive():
        return
    _resource_monitor_stop.clear()
    _resource_monitor_thread = threading.Thread(
        target=_resource_monitor_loop,
        name="aura-resource-guard",
        daemon=True,
    )
    _resource_monitor_thread.start()


def _process_record_path(task_dir: str) -> str:
    return os.path.join(task_dir, "process.json")


def _write_process_record(entry: dict) -> None:
    record_path = _process_record_path(entry["task_dir"])
    killed_at = entry.get("killed_at")
    if hasattr(killed_at, "isoformat"):
        killed_at = killed_at.isoformat()
    record = {
        "pid": entry["pid"],
        "task_id": entry["task_id"],
        "task_dir": entry["task_dir"],
        "started_at": entry["started_at"].isoformat()
        if hasattr(entry["started_at"], "isoformat") else str(entry["started_at"]),
        "budget_minutes": entry["budget_minutes"],
        "output_path": entry["output_path"],
        "killed_at": killed_at,
        "resource_limits": entry.get("resource_limits", _resource_limits()),
        "resource_violation": entry.get("resource_violation"),
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
            if record.get("killed_at") or not _record_process_is_alive(record):
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
                "killed_at": record.get("killed_at"),
                "resource_limits": record.get("resource_limits") or _resource_limits(),
                "resource_violation": record.get("resource_violation"),
            }
        except Exception:
            continue


def _resolve_executable(name: str, configured: str = "") -> str | None:
    """Resolve an executable path cross-platform.

    On Windows, Python subprocess may fail with bare npm shim names like
    'claude' even when PowerShell can run them, because the actual executable
    is often a .cmd shim. Prefer .cmd/.bat when resolving by PATH.
    """
    if configured:
        path = os.path.expandvars(os.path.expanduser(configured.strip()))
        return path if path else None

    if sys.platform == "win32":
        candidates = [
            f"{name}.cmd",
            f"{name}.CMD",
            f"{name}.bat",
            f"{name}.BAT",
            f"{name}.exe",
            name,
        ]
    else:
        candidates = [name]

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    return None


def _wrap_windows_script_cmd(exe: str, args: list[str]) -> list[str]:
    """Run .cmd/.bat through cmd.exe for maximum Windows compatibility."""
    if sys.platform == "win32" and exe.lower().endswith((".cmd", ".bat")):
        return ["cmd.exe", "/d", "/c", exe, *args]
    return [exe, *args]


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
        if _entry_process_is_alive(existing):
            return f"ERROR: Task {task_id} is already running (PID: {existing['pid']})"

    ok, preflight = _resource_preflight()
    if not ok:
        return f"ERROR: Resource preflight failed for task {task_id}: {preflight}"

    # Resolve claude executable cross-platform
    claude_bin = _resolve_executable("claude", AURA_CLAUDE_BIN)
    if not claude_bin:
        return (
            "ERROR: Claude Code CLI not found in PATH. "
            "Install it with `npm install -g @anthropic-ai/claude-code`, "
            "or set AURA_CLAUDE_BIN to the executable path."
        )

    claude_args = [
        "-p",
        "@task.md",
        "--max-turns", str(DEFAULT_MAX_TURNS),
        "--output-format", "stream-json",
        "--verbose",
    ]

    cmd = _wrap_windows_script_cmd(claude_bin, claude_args)

    output_path = os.path.join(task_dir, "output.jsonl")
    error_path = os.path.join(task_dir, "error.log")
    env = _worker_env()

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
                    env=env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                proc = subprocess.Popen(
                    cmd,
                    cwd=task_dir,
                    stdout=out_f,
                    stderr=err_f,
                    env=env,
                    preexec_fn=os.setsid,
                )

        cpu_affinity = _apply_cpu_affinity(proc.pid)
        _active_processes[task_id] = {
            "pid": proc.pid,
            "task_id": task_id,
            "task_dir": task_dir,
            "started_at": datetime.now(),
            "budget_minutes": budget_minutes,
            "process": proc,
            "output_path": output_path,
            "resource_limits": _resource_limits(),
            "cpu_affinity": cpu_affinity,
        }
        _write_process_record(_active_processes[task_id])
        _ensure_resource_monitor()

        affinity_note = f" CPU affinity: {cpu_affinity}." if cpu_affinity else ""
        return (f"OK: Task {task_id} started (PID: {proc.pid}, backend: claude_code). "
                f"Output: {task_dir}/output.jsonl "
                f"Budget: {budget_minutes}min.{affinity_note} {preflight}")

    except FileNotFoundError as e:
        return (
            "ERROR: Failed to start Claude Code CLI.\n"
            f"Resolved claude_bin: {claude_bin!r}\n"
            f"Command: {cmd!r}\n"
            f"cwd: {task_dir}\n"
            f"PATH: {os.environ.get('PATH', '')}\n"
            f"Details: {e}"
        )
    except Exception as e:
        return (
            "ERROR: Failed to spawn Claude Code worker.\n"
            f"Resolved claude_bin: {claude_bin!r}\n"
            f"Command: {cmd!r}\n"
            f"cwd: {task_dir}\n"
            f"Details: {type(e).__name__}: {e}"
        )


def _spawn_dscode(task_id: str, task_dir: str, task_md_path: str, budget_minutes: int) -> str:
    """Spawn a ds-code CLI process for a task.

    Uses 'ds-code run' one-shot mode, passing the task markdown via stdin.
    """
    if task_id in _active_processes:
        existing = _active_processes[task_id]
        if _entry_process_is_alive(existing):
            return f"ERROR: Task {task_id} is already running (PID: {existing['pid']})"

    cmd = [
        "ds-code", "run",
        "--workspace", task_dir,
        "--no-color",
    ]

    if AURA_DSCODE_MODEL:
        cmd.extend(["--model", AURA_DSCODE_MODEL])

    ok, preflight = _resource_preflight()
    if not ok:
        return f"ERROR: Resource preflight failed for task {task_id}: {preflight}"

    # Build environment so ds-code can find DEEPSEEK_API_KEY
    env = _worker_env()
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

        cpu_affinity = _apply_cpu_affinity(proc.pid)
        _active_processes[task_id] = {
            "pid": proc.pid,
            "task_id": task_id,
            "task_dir": task_dir,
            "started_at": datetime.now(),
            "budget_minutes": budget_minutes,
            "process": proc,
            "output_path": output_path,
            "resource_limits": _resource_limits(),
            "cpu_affinity": cpu_affinity,
        }
        _write_process_record(_active_processes[task_id])
        _ensure_resource_monitor()

        affinity_note = f" CPU affinity: {cpu_affinity}." if cpu_affinity else ""
        return (f"OK: Task {task_id} started (PID: {proc.pid}, backend: ds_code). "
                f"Output: {task_dir}/output.txt "
                f"Budget: {budget_minutes}min.{affinity_note} {preflight}")

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
        if not _entry_process_is_alive(entry):
            _active_processes[task_id]["running"] = False
            _active_processes[task_id]["killed_at"] = datetime.now()
            _write_process_record(_active_processes[task_id])
            return f"WARN: Task {task_id} (PID: {pid}) was already dead or stale"

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
        _active_processes[task_id]["running"] = _entry_process_is_alive(_active_processes[task_id])
        return f"ERROR killing task {task_id}: taskkill timed out for PID {pid}"
    except Exception as e:
        return f"ERROR killing task {task_id}: {e}"


def list_all() -> list[dict]:
    """List all tracked processes with status and health metrics."""
    _load_process_records()
    _ensure_resource_monitor()
    result = []
    with _process_lock:
        items = list(_active_processes.items())
    for task_id, entry in items:
        running = _entry_process_is_alive(entry)
        entry["running"] = running
        elapsed = (datetime.now() - entry["started_at"]).total_seconds() / 60

        output_path = entry.get("output_path", os.path.join(entry["task_dir"], "output.jsonl"))
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

        # ── Process health metrics ──────────────────────────────────
        cpu_percent = 0.0
        memory_mb = 0.0
        memory_percent = 0.0
        gpu_memory_mb = None
        gpu_memory_percent = None
        gpu_util_percent = None
        child_count = 0
        if running:
            metrics = _process_tree_metrics(entry["pid"], sample_cpu=True)
            cpu_percent = metrics["cpu_percent"]
            memory_mb = metrics["memory_mb"]
            memory_percent = metrics["memory_percent"]
            gpu_memory_mb = metrics["gpu_memory_mb"]
            gpu_memory_percent = metrics["gpu_memory_percent"]
            gpu_util_percent = metrics["gpu_util_percent"]
            child_count = metrics["child_count"]
        aggregate = _aggregate_resource_samples(entry.get("resource_samples", []))

        result.append({
            "task_id": task_id,
            "pid": entry["pid"],
            "running": running,
            "elapsed_minutes": round(elapsed, 1),
            "budget_minutes": entry["budget_minutes"],
            "output_size": output_size,
            "cpu_percent": round(cpu_percent, 1),
            "avg_cpu_percent": round(aggregate["avg_cpu_percent"], 1),
            "memory_mb": round(memory_mb, 1),
            "memory_percent": round(memory_percent, 1),
            "avg_memory_percent": round(aggregate["avg_memory_percent"], 1),
            "gpu_memory_mb": None if gpu_memory_mb is None else round(gpu_memory_mb, 1),
            "gpu_memory_percent": None if gpu_memory_percent is None else round(gpu_memory_percent, 1),
            "avg_gpu_memory_percent": None if aggregate["avg_gpu_memory_percent"] is None else round(aggregate["avg_gpu_memory_percent"], 1),
            "gpu_util_percent": None if gpu_util_percent is None else round(gpu_util_percent, 1),
            "avg_gpu_util_percent": None if aggregate["avg_gpu_util_percent"] is None else round(aggregate["avg_gpu_util_percent"], 1),
            "child_count": child_count,
            "resource_violation": entry.get("resource_violation"),
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
    return _entry_process_is_alive(_active_processes[task_id])


def _is_alive(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _parse_started_at(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _pid_matches_started_at(pid: int, started_at) -> bool:
    """Guard against OS PID reuse when restoring process records.

    PIDs can be recycled after a worker exits. A stale process.json may point at
    a completely unrelated process that happens to have the same PID later.
    """
    expected = _parse_started_at(started_at)
    if expected is None:
        return False
    try:
        proc = psutil.Process(pid)
        actual = datetime.fromtimestamp(proc.create_time())
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError):
        return False
    delta = abs((actual - expected).total_seconds())
    return delta <= _PROCESS_START_TOLERANCE_SECONDS


def _record_process_is_alive(record: dict) -> bool:
    pid = int(record["pid"])
    return _is_alive(pid) and _pid_matches_started_at(pid, record.get("started_at"))


def _entry_process_is_alive(entry: dict) -> bool:
    if entry.get("killed_at"):
        return False
    pid = int(entry["pid"])
    process = entry.get("process")
    if process is not None:
        try:
            if process.poll() is not None:
                return False
        except Exception:
            pass
    return _is_alive(pid) and _pid_matches_started_at(pid, entry.get("started_at"))
