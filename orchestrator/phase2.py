"""Phase 2 — Intelligent Decision Upgrade for Aura Agent.

Provides:
  P2.2 — Progress evaluation (evaluate_progress) with content-aware analysis
  P2.3 — Global replanning trigger (check_replan_needed)
  P2.4 — Depth vs breadth decision matrix (decision_matrix)
  P2.1 helper — Activity mode detection (get_activity_mode)
"""

import hashlib
import os
import json
from datetime import datetime, timezone
from typing import Optional

from .config import (
    get_workspace_dir,
    STUCK_THRESHOLD_CYCLES,
)

# ── Content hash cache (module-level, persists across evaluate_progress calls) ──
# Keyed by task_id. Tracks both the last hash and the last N lines so we can
# detect looping even when the hash changes (same lines cycling).
_last_tail_state: dict[str, dict] = {}


def _read_tail_lines(output_path: str, n: int = 40) -> list[str]:
    """Read the last N lines of output.jsonl efficiently."""
    if not os.path.exists(output_path):
        return []
    try:
        with open(output_path, "r", encoding="utf-8", errors="replace") as f:
            # Simple tail: read all lines if file is small, otherwise seek back
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return []
            # Read last ~8KB or whole file, whichever is smaller
            chunk_start = max(0, size - 8192)
            f.seek(chunk_start)
            # Skip partial first line if we seeked mid-file
            if chunk_start > 0:
                f.readline()
            lines = f.readlines()
            return lines[-n:] if len(lines) > n else lines
    except OSError:
        return []


def _analyze_output_tail(task_dir: str) -> dict:
    """Read the last ~50 lines of output.jsonl and extract simple signals.

    Returns:
        dict with:
          - tail_hash: hash of last 40 lines (for change detection)
          - tail_line_count: how many lines were read
          - has_recent_activity: any line with a timestamp from the last 10 min
          - unique_tool_names: set of distinct tool names seen in tail
          - is_looping: True if the same tool+args pair appears 8+ times
    """
    output_path = os.path.join(task_dir, "output.jsonl")
    lines = _read_tail_lines(output_path, n=50)

    result = {
        "tail_hash": "",
        "tail_line_count": len(lines),
        "has_recent_activity": False,
        "unique_tool_names": [],
        "is_looping": False,
    }

    if not lines:
        return result

    # Content hash of last 40 lines
    tail_text = "".join(lines[-40:])
    result["tail_hash"] = hashlib.sha256(tail_text.encode("utf-8", errors="replace")).hexdigest()

    # Parse lines for tool names and timestamps
    tool_call_counts: dict[str, int] = {}
    now = datetime.now()

    for line in lines:
        try:
            obj = json.loads(line.strip())
        except json.JSONDecodeError:
            continue

        # Extract tool name from stream-json format
        # Claude Code format: {"type":"tool_use","name":"read_file",...}
        # or {"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash",...}]}}
        tool_name = None
        if obj.get("type") == "tool_use":
            tool_name = obj.get("name", "")
        elif obj.get("type") == "tool_result":
            tool_name = "(tool_result)"
        elif obj.get("type") == "assistant":
            content = obj.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_name = block.get("name", "")
                        break
        elif obj.get("type") == "user":
            tool_name = "(user_msg)"

        if tool_name:
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

        # Check for recent timestamp
        ts_str = obj.get("timestamp") or obj.get("ts") or obj.get("created_at", "")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if (now - ts.replace(tzinfo=None)).total_seconds() < 600:
                    result["has_recent_activity"] = True
            except (ValueError, TypeError):
                pass

    result["unique_tool_names"] = list(tool_call_counts.keys())

    # Detect looping: same tool name appearing 8+ times
    for name, count in tool_call_counts.items():
        if count >= 8:
            result["is_looping"] = True
            break

    return result


def _detect_looping(task_id: str, tail_hash: str, lines: list[str]) -> bool:
    """Detect if the worker is writing a tight loop of identical output.

    Compares current tail hash with the previous cycle's hash. If the
    hashes match AND the file size hasn't changed, it's a loop.
    """
    prev = _last_tail_state.get(task_id, {})
    prev_hash = prev.get("tail_hash", "")
    # Update cache
    _last_tail_state[task_id] = {"tail_hash": tail_hash}
    if not prev_hash:
        return False
    return tail_hash == prev_hash


def evaluate_progress(
    task_id: str,
    previous_output_size: int = 0,
    previous_content_hash: str = "",
    process_cpu: float = 0.0,
) -> dict:
    """Evaluate progress of a Layer 2 task with multi-signal analysis.

    Uses three independent signals:
      1. File size delta (bytes written since last cycle)
      2. Content hash delta (did the last N lines change?)
      3. Process CPU (is the process actively computing?)

    These three signals together can distinguish:
      - Genuine progress (size grows, content changes, CPU > 0)
      - Long computation (size static, content static, CPU > 0) → NOT stuck
      - Dead zombie (size static, content static, CPU == 0) → stuck
      - Looping (size grows, content cycling, CPU > 0) → stuck variant

    Args:
        task_id: The task ID to evaluate.
        previous_output_size: Size of output.jsonl from the previous cycle.
        previous_content_hash: Hash of last 40 lines from previous cycle.
        process_cpu: CPU usage percentage from psutil (0.0 if unknown).

    Returns:
        dict with progress signals and analysis.
    """
    task_dir = os.path.join(get_workspace_dir(), "tasks", task_id)
    result = {
        "active_score": 0.0,
        "has_output": False,
        "output_size": 0,
        "output_delta": 0,
        "content_hash": "",
        "content_changed": False,
        "is_stuck": False,
        "stuck_cycles": 0,
        "artifacts": [],
        "error_log_size": 0,
        "is_looping": False,
        "tail_analysis": {},
    }

    if not os.path.isdir(task_dir):
        return result

    # ── Signal 1: File size ──────────────────────────────────────────
    output_path = os.path.join(task_dir, "output.jsonl")
    if os.path.exists(output_path):
        current_size = os.path.getsize(output_path)
        result["output_size"] = current_size
        result["output_delta"] = current_size - previous_output_size
        if current_size > 0:
            result["has_output"] = True

    # ── Signal 2: Content analysis (tail hash + loop detection) ──────
    tail = _analyze_output_tail(task_dir)
    result["tail_analysis"] = tail
    result["content_hash"] = tail["tail_hash"]

    if previous_content_hash and tail["tail_hash"]:
        result["content_changed"] = (tail["tail_hash"] != previous_content_hash)

    result["is_looping"] = tail.get("is_looping", False)

    # Also check cross-cycle hash stagnation
    if _detect_looping(task_id, tail["tail_hash"], []):
        result["is_looping"] = True

    # ── Signal 3: Error log ──────────────────────────────────────────
    error_path = os.path.join(task_dir, "error.log")
    if os.path.exists(error_path):
        result["error_log_size"] = os.path.getsize(error_path)

    # ── Artifacts (non-output files) ─────────────────────────────────
    known_files = {"output.jsonl", "error.log", "task.md", "process.json"}
    try:
        for fname in os.listdir(task_dir):
            if fname not in known_files:
                fpath = os.path.join(task_dir, fname)
                if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
                    result["artifacts"].append(fname)
    except OSError:
        pass

    # ── Composite active_score ───────────────────────────────────────
    score = 0.0
    if result["has_output"]:
        score += 0.2
    if result["output_delta"] > 0:
        growth_score = min(result["output_delta"] / 10000.0, 1.0) * 0.3
        score += growth_score
    if result["content_changed"]:
        score += 0.2  # New content is a strong signal
    if result["artifacts"]:
        score += 0.15 * min(len(result["artifacts"]), 3) / 3.0
    if process_cpu > 5.0:
        score += 0.15  # Process actively computing
    elif process_cpu > 0.1:
        score += 0.05
    if result["error_log_size"] == 0:
        score += 0.05
    else:
        score -= 0.15
    if result["is_looping"]:
        score -= 0.3  # Strong penalty for detected loops

    result["active_score"] = max(0.0, min(1.0, score))

    # ── Stuck detection: multi-signal ────────────────────────────────
    # A task is stuck ONLY when ALL signals agree:
    #   - No file size growth
    #   - No content change (tail hash same as last cycle)
    #   - No new artifacts
    #   - CPU is near zero (process idle, not computing)
    #
    # If CPU > 0 but no output changes, the worker is likely doing a long
    # computation — NOT stuck, just slow to report.
    size_stagnant = (result["output_delta"] == 0 and result["output_size"] == previous_output_size)
    content_stagnant = not result["content_changed"]
    no_artifacts = not result["artifacts"]
    cpu_idle = process_cpu < 0.5  # Less than 0.5% CPU = effectively idle

    if size_stagnant and content_stagnant and no_artifacts and cpu_idle:
        result["is_stuck"] = True

    return result


def check_replan_needed(
    consecutive_no_progress_cycles: int,
    total_elapsed_hours: float,
    has_any_output: bool,
    has_new_requirements: bool = False,
) -> dict:
    """Check whether a global replan should be triggered.

    Triggers when:
      - 6+ consecutive cycles with no verifiable output, OR
      - 4+ hours elapsed with zero overall progress, OR
      - User modified the task file with new/changed requirements (R7).

    Args:
        consecutive_no_progress_cycles: Cycles since last meaningful output.
        total_elapsed_hours: Total hours since mission started.
        has_any_output: Whether ANY output has been produced across all tasks.
        has_new_requirements: True if wake-change detected new/changed
            requirements in the task file (R7 per-wake check).

    Returns:
        dict with:
          - replan_requested (bool): Whether replanning should trigger.
          - trigger_reason (str): Human-readable reason if triggered.
          - urgency (float 0.0-1.0): How urgent the replan is.
    """
    result = {
        "replan_requested": False,
        "trigger_reason": "",
        "urgency": 0.0,
    }

    # Trigger 1: N cycles with no output
    NO_OUTPUT_THRESHOLD = 6
    if consecutive_no_progress_cycles >= NO_OUTPUT_THRESHOLD:
        result["replan_requested"] = True
        result["trigger_reason"] = (
            f"{consecutive_no_progress_cycles} consecutive cycles with no verifiable output "
            f"(threshold: {NO_OUTPUT_THRESHOLD})"
        )
        result["urgency"] = min(1.0, consecutive_no_progress_cycles / NO_OUTPUT_THRESHOLD)

    # Trigger 2: 4+ hours with zero progress
    NO_PROGRESS_HOURS = 4.0
    if not has_any_output and total_elapsed_hours >= NO_PROGRESS_HOURS:
        result["replan_requested"] = True
        if result["trigger_reason"]:
            result["trigger_reason"] += "; "
        result["trigger_reason"] += (
            f"{total_elapsed_hours:.1f} hours elapsed with zero progress "
            f"(threshold: {NO_PROGRESS_HOURS}h)"
        )
        result["urgency"] = max(result["urgency"], min(1.0, total_elapsed_hours / NO_PROGRESS_HOURS))

    # Trigger 3 (R7): User modified task file with new/changed requirements
    if has_new_requirements:
        result["replan_requested"] = True
        if result["trigger_reason"]:
            result["trigger_reason"] += "; "
        result["trigger_reason"] += (
            "Task file modified with new/changed requirements — user wants different or additional work"
        )
        result["urgency"] = max(result["urgency"], 0.85)

    return result


def decision_matrix(
    progress: dict,
    task_age_minutes: float,
    budget_remaining_minutes: float,
) -> dict:
    """Make a structured decision about what to do with a task.

    Uses multi-signal progress data (content hash, CPU, tail analysis)
    to make smarter kill/continue/decompose decisions.

    Args:
        progress: Result from evaluate_progress().
        task_age_minutes: How long the task has been running.
        budget_remaining_minutes: Remaining time budget.

    Returns:
        dict with:
          - action (str): One of "continue_deeper", "switch_breadth",
            "kill", "decompose", "replan".
          - confidence (float 0.0-1.0): Confidence in this recommendation.
          - reasoning (str): Explanation of the decision.
    """
    active_score = progress.get("active_score", 0.0)
    has_output = progress.get("has_output", False)
    is_stuck = progress.get("is_stuck", False)
    is_looping = progress.get("is_looping", False)
    output_delta = progress.get("output_delta", 0)
    content_changed = progress.get("content_changed", False)
    artifacts = progress.get("artifacts", [])
    error_log_size = progress.get("error_log_size", 0)
    tail = progress.get("tail_analysis", {})

    # ── Decision logic ──────────────────────────────────────────────

    # High activity → continue
    if active_score >= 0.5 and has_output and not is_stuck:
        return {
            "action": "continue_deeper",
            "confidence": min(1.0, active_score),
            "reasoning": (
                f"Active score {active_score:.2f}, output growing "
                f"(+{output_delta} bytes), {len(artifacts)} artifacts."
            ),
        }

    # Over budget + no output at all → kill
    if budget_remaining_minutes <= 0 and not has_output:
        return {
            "action": "kill",
            "confidence": 0.9,
            "reasoning": (
                f"Budget exhausted ({task_age_minutes:.0f}min), no output."
            ),
        }

    # Stuck with errors + no output → kill
    if is_stuck and error_log_size > 0 and not has_output:
        return {
            "action": "kill",
            "confidence": 0.85,
            "reasoning": (
                f"Stuck with errors ({error_log_size} bytes in error.log) "
                f"and no output. Likely config or environment issue."
            ),
        }

    # Looping with high repetition → kill (even if file grows)
    if is_looping and not artifacts and active_score < 0.3:
        return {
            "action": "kill",
            "confidence": 0.8,
            "reasoning": (
                f"Detected output loop (same tool repeated {tail.get('unique_tool_names', [])}). "
                f"Worker is cycling without making real progress."
            ),
        }

    # Stuck but had prior output → decompose
    if is_stuck and has_output and active_score < 0.3:
        return {
            "action": "decompose",
            "confidence": 0.7,
            "reasoning": (
                f"Produced output but now stuck (score {active_score:.2f}). "
                f"Decomposing for fresh approach."
            ),
        }

    # Stuck with no output, but budget remains → switch approach
    if is_stuck and not has_output and budget_remaining_minutes > 5:
        return {
            "action": "switch_breadth",
            "confidence": 0.65,
            "reasoning": (
                f"Stuck with no output, {budget_remaining_minutes:.0f}min remaining. "
                f"Trying different approach."
            ),
        }

    # Low activity → switch breadth
    if active_score < 0.2 and task_age_minutes > 5:
        return {
            "action": "switch_breadth",
            "confidence": 0.55,
            "reasoning": (
                f"Low activity (score {active_score:.2f}) after {task_age_minutes:.0f}min."
            ),
        }

    # Moderate activity, continuing
    if active_score >= 0.2 and has_output:
        return {
            "action": "continue_deeper",
            "confidence": 0.5,
            "reasoning": (
                f"Moderate activity (score {active_score:.2f}) with output."
            ),
        }

    # Default: insufficient data → continue with low confidence
    return {
        "action": "continue_deeper",
        "confidence": 0.3,
        "reasoning": "Insufficient data for strong decision.",
    }


def get_activity_mode(progress_results: list[dict]) -> str:
    """Determine the overall activity mode based on all active tasks.

    Used for adaptive wake interval selection (P2.1).

    Args:
        progress_results: List of evaluate_progress() results for all active tasks.

    Returns:
        One of "active", "calm", "idle".
    """
    if not progress_results:
        return "idle"

    scores = [p.get("active_score", 0.0) for p in progress_results]
    any_output = any(p.get("has_output", False) for p in progress_results)
    any_delta = any(p.get("output_delta", 0) > 0 for p in progress_results)
    any_content_change = any(p.get("content_changed", False) for p in progress_results)
    avg_score = sum(scores) / len(scores) if scores else 0.0

    if any_delta or any_content_change or avg_score >= 0.4:
        return "active"
    elif any_output or avg_score >= 0.1:
        return "calm"
    else:
        return "idle"
