"""Core decision engine for the Aura Agent Orchestrator.

Each wake cycle:
1. Load all context (memories, state, progress)
2. Call Claude API with tools
3. Execute tool calls in a loop until Claude is done
4. Update session memory and progress report
5. Return a cycle summary

RESILIENCE: All Phase-N code is loaded safely with fallbacks. If any upgrade
module fails to import or crashes at runtime, the orchestrator degrades
gracefully to basic mode and can spawn Layer 2 workers to self-heal.
"""

import hashlib
import json
import os
import time
import sys
from datetime import datetime
from typing import Any

import anthropic

from .config import (
    AURA_API_KEY,
    AURA_API_BASE_URL,
    AURA_API_MODEL,
    AURA_API_MAX_TOKENS,
    API_RETRY_COUNT,
    API_RETRY_BASE_DELAY,
    API_TIMEOUT_SECONDS,
    STUCK_THRESHOLD_CYCLES,
    get_workspace_dir,
    MEMORY_DIR,
    STATE_DIR,
    AURA_EXPLICIT_PROMPT_CACHE,
    AURA_SKIP_HEALTHY_CYCLES,
    AURA_MAX_SKIPPED_CYCLES,
    MAX_TOOL_CALLS_PER_CYCLE,
    MAX_CONCURRENT_TASKS,
    TOOL_CALL_BUDGET_NORMAL,
    TOOL_CALL_BUDGET_DIAGNOSTIC,
    TOOL_CALL_BUDGET_PLANNING,
)
from .tools import TOOL_DEFINITIONS, execute_tool, get_active_tool_definitions
from . import state as state_mgr
from . import memory as memory_mgr
from . import progress as progress_mgr
from . import process_mgr
from .changelog import (
    get_file_change_info,
    get_project_name_for_task,
)
from .token_tracker import log_usage, log_skip, format_cycle_stats, extract_usage

# ── RESILIENT IMPORTS: Phase modules with safe fallbacks ─────────────
# Each upgrade module is loaded in a try/except. If loading fails,
# dummy functions are used instead, allowing the orchestrator to keep
# running and spawn Layer 2 workers to fix the broken code.

_phase_modules_ok = True
_phase_load_errors = []

try:
    from .phase2 import (
        evaluate_progress,
        decision_matrix,
        check_replan_needed,
        get_activity_mode,
    )
except Exception as e:
    _phase_modules_ok = False
    _phase_load_errors.append(f"phase2: {e}")
    print(f"[RESILIENCE] phase2.py failed to load: {e}. Using fallbacks.")

    def evaluate_progress(task_id, previous_output_size=0):
        return {"active_score": 0.0, "has_output": False, "output_size": 0,
                "output_delta": 0, "is_stuck": False, "stuck_cycles": 0,
                "artifacts": [], "error_log_size": 0}

    def decision_matrix(progress, task_age_minutes, budget_remaining_minutes):
        return {"action": "continue_deeper", "confidence": 0.0,
                "reasoning": "Phase 2 unavailable — using basic fallback"}

    def check_replan_needed(consecutive_no_progress_cycles, total_elapsed_hours, has_any_output):
        return {"replan_requested": False, "trigger_reason": "", "urgency": 0.0}

    def get_activity_mode(progress_results):
        return "active"


# ── Phase 3: Resilient review import ─────────────────────────────────
_review_available = False
try:
    from .review import review_cycle
    _review_available = True
except Exception as e:
    _phase_load_errors.append(f"review: {e}")
    print(f"[RESILIENCE] review.py failed to load: {e}. Review features disabled.")

    def review_cycle(force=False):
        return {"review_text": "", "saved_path": "", "recommendations": [], "error": str(e)}


_ORCHESTRATOR_SYSTEM_PROMPT = """You are Aura Agent's Global Orchestrator — the top-level controller of a two-layer autonomous agent system.

## Your Identity
You are NOT a chatbot. You are a goal-completion engine. Your only purpose is to achieve the assigned mission. You wake up periodically, assess the situation, make decisions, and go back to sleep while Layer 2 workers execute your commands.

## Your Operating Philosophy

In an uncertain, complex world, you approach truth through evidence and reasoning,
understand causality through systems thinking, constrain capability with human values,
and correct yourself through continuous feedback.

Specifically:
1. Don't attribute outcomes to a single cause — seek multi-variable, multi-level, multi-feedback explanations
2. Don't pursue a one-shot ultimate answer — continuously reduce error rates
3. Every conclusion must be testable, falsifiable, and have clear assumption boundaries
4. Distinguish fact, inference, opinion, and speculation — be honest about uncertainty
5. A viewpoint's value lies in how it changes decisions, not how sophisticated it sounds
6. Continuously verify, continuously correct, continuously act — don't believe in grand narratives
7. Humbly seek truth, systematically think, cautiously act

## Your Decision Framework

Each wake cycle, follow this structure:

### 1. SITUATION ASSESSMENT
- First use the current state snapshot supplied in the user message.
- Aura data is task-file scoped. If you must read files, use the explicit Aura data directory shown in the context or the provided workspace snapshot. Do not assume legacy `.aura/state` or `.aura/workspace` paths are authoritative.
- For active tasks, prefer the included Phase 2 signals and workspace/output summaries before making extra tool calls.
- If Planning needed is true, update the task tree from the task file before no_op or spawning work.
- If there are no active/running/pending tasks, no task-file change, and Planning needed is false, call no_op without extra file reads.

### 2. PROGRESS EVALUATION
For each active task, evaluate:
- Has any verifiable output been produced? (files created/modified, data collected, code that runs)
- How long has it been running vs. its budget?
- Is it making meaningful activity or is it stuck in a loop?
- Does the output actually contribute to the mission?

**Output reading policy** (ENFORCED — read_file tool blocks violations):
- Do not attempt to read output.jsonl or output.txt; the tool blocks these files in all task states.
- For active tasks, the Phase 2 monitoring line plus process.json is sufficient. If a registered subprocess log is needed, use the `log_path` recorded in process.json.
- Do not list_directory or read config files for completed tasks (has_result_md=true) unless the current cycle needs that specific evidence for a state-changing decision (e.g. replan, spawn a follow-up task).

Additional hard output policy:
- Never read output.jsonl or output.txt. The tool will block it.
- For process state, read only the task's **process.json**. If it lists a live `managed_subprocesses` entry, the task is still processing even if result.md exists.
- If a task has `has_result_md: true`, read **result.md** only as a completion claim, then verify process.json before any status change.

### 3. DECISION
Based on evaluation, choose one or more:
- **Continue deeper**: Task is making progress, let it keep going
- **Switch direction (breadth)**: Try a different approach or parallel branch
- **Kill task**: Task is stuck, looping, or producing nothing useful
- **Decompose further**: Break a stuck task into smaller, clearer pieces
- **Trigger replanning**: Global reassessment — all current approaches may be wrong
- **Do nothing**: Everything is fine, check again next cycle

### 4. EXECUTION
Use your tools to implement your decisions. Don't just think — act.

### 5. RECORD
Update the task tree, write important learnings to memory. Keep the progress report current.

## Task Hierarchy Rules

- The task tree must be hierarchical. Do not flatten concrete work directly under ROOT.
- ROOT represents the final goal. ROOT children such as A1, A2, B1 are broad planning categories or major steps, not worker-executable tasks.
- Concrete worker tasks must live at ROOT -> category -> task level or deeper, for example A1.1, A1.2, B2.1. Even if a category has only one concrete task, create that third-level node.
- Before spawning work, decompose the relevant top-level category into third-level tasks. Use fourth-level tasks only when a third-level task truly needs another split.
- When creating or spawning a third-level task, include sibling context: what sibling tasks tried, their evidence, and whether they succeeded, failed, were killed, or remain pending.

## Rules

- Maximum 2 concurrent Layer 2 tasks at any time. spawn_task automatically marks the task as in_progress — you do NOT need to call update_task_tree afterwards. update_task_tree will reject in_progress if already at the 2-task limit, preventing phantom in_progress tasks that have no worker.
- If a task returns to pending after the resource guard killed it once, retry it only with a smaller/safer plan: reduce batch size, epochs, model size, data subset, workers, or disable offload.
- If a task is blocked because the resource guard killed it twice, prefer the generated resource-fix subtask. Use its result to continue the original goal if feasible. If the requirement is plainly impossible on the available hardware, record that with concrete evidence instead of retrying forever.
- The code does not parse the user's task file into tasks. You own semantic planning: create, update, and archive task-tree nodes from the task-file content using tools.
- During planning, first identify and persist project context with update_project_context: final goal, success criteria, global constraints, and execution environment such as commands, env vars/API key usage, models, and working directories.
- During planning, do not inspect other task-file data directories under `.aura` for current state, progress, workspace outputs, summaries, caches, or task metadata. You may read other task directories' memory files only as lessons, not as evidence for the current task; record any borrowed lesson explicitly in current project context or memory.
- New top-level categories use the current batch prefix shown in context, such as A1, A2, ...; after the task file changes, newly added top-level requirements use the next prefix such as B1, B2, ... Existing A categories keep their IDs and children of A tasks must continue as A1.1, A1.2, etc.
- Preserve and build on existing subtasks, evidence, and completed work. Do not re-plan from scratch just because the task file is broad or edited.
- Treat completed tasks and result.md evidence as coverage for matching requirements; avoid repeating completed work unless the requirement text materially changed.
- Do not mark tasks completed from task-file wording alone. Completion requires verifiable evidence, worker artifacts, or an explicit user request.
- If there is free Layer 2 capacity and multiple independent pending requirements exist, start work on up to 2 of them instead of focusing only on the first item.
- If a task runs 12+ cycles with NO verifiable output → kill it or trigger replanning
- If the entire project has no effective progress for several hours → comprehensive replanning
- NEVER trust a task's self-report — check actual output evidence before changing status
- When uncertain, gather the smallest specific missing evidence first
- Write genuinely important lessons to long-term memory — don't spam it
- Every status change must have a reason AND evidence
- The mission does not end until the goal is achieved
- NEVER mention result.md in task descriptions or deliverables. result.md is an infrastructure file produced automatically when the task truly completes or fails. Asking workers to create it encourages premature termination with subprocesses still running.
- When a stopped worker has result.md, you must verify it before marking the task complete. Read result.md and check: (a) does the content show the assigned task is truly DONE, not just "launched"? (b) are there still live subprocesses (check list_running_tasks, workspace output tail, or resource signals)? If uncertain, leave the task in its current state and wait for the next cycle for more evidence.

## Current Context

The user message contains the current state snapshot, memory preview, progress preview, and task workspace summaries. Read it carefully, then use tools only for missing evidence or actions.

Remember: You are the decider. Wake up, assess, decide, act, record, sleep."""


def _format_wake_change_info(task_file: str, wake_change: dict) -> str:
    """Format per-wake task file change info for the orchestrator context.

    Includes:
    - Change summary
    - Diff preview (trimmed to avoid blowing up context)
    - Actionable hints: should we replan? Is there new user info?
    """
    summary = wake_change.get("change_summary", "unknown changes")
    diff_lines = wake_change.get("diff_lines", [])
    added_req = wake_change.get("added_requirement_lines", [])
    removed_req = wake_change.get("removed_requirement_lines", [])
    added_info = wake_change.get("added_info_lines", [])

    # ── Diff preview (max 30 lines) ────────────────────────────────
    diff_preview = "\n".join(diff_lines[:30])
    if len(diff_lines) > 30:
        diff_preview += f"\n... ({len(diff_lines) - 30} more diff lines)"

    # ── Actionable hints ───────────────────────────────────────────
    hints = []
    if added_req:
        hints.append(
            f"**可能需要重新规划**: 检测到 {len(added_req)} 个新增/变更的任务需求。"
            f"请读取任务文件，判断是否需要调整任务树（decompose、新增 T 节点、调整优先级）。"
        )
    if removed_req:
        hints.append(
            f"**需求已移除**: {len(removed_req)} 个任务项被删除。"
            f"请检查是否有对应的 active task 需要标记为 obsolete。"
        )
    if added_info:
        hints.append(
            f"**用户提供了新信息**: {len(added_info)} 行新的上下文信息。"
            f"这些信息可能有助于当前进展——请读取并判断是否应记录到长期记忆 (write_memory)。"
        )
    if not hints:
        hints.append("变更较小，可能无需调整计划。但仍建议读取任务文件确认。")

    hints_text = "\n".join(f"  - {h}" for h in hints)

    return (
        f"\n\n### ⚠️ 任务文件在本周期被修改\n"
        f"**变更摘要**: {summary}\n"
        f"**文件**: {task_file}\n\n"
        f"**Diff 预览**:\n```diff\n{diff_preview}\n```\n\n"
        f"**行动建议**:\n{hints_text}\n"
    )


def _read_text_preview(path: str, max_chars: int = 4000) -> str:
    """Read a bounded text preview for the orchestrator context."""
    if not os.path.exists(path):
        return "(missing)"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        return f"(unreadable: {e})"

    if len(content) <= max_chars:
        return content

    head = max_chars // 2
    tail = max_chars - head
    omitted = len(content) - max_chars
    return (
        content[:head]
        + f"\n... [truncated {omitted} chars] ...\n"
        + content[-tail:]
    )


def _read_tail_preview(path: str, max_lines: int = 30, max_chars: int = 3000) -> str:
    """Read the tail of a text output file without loading huge logs."""
    if not os.path.exists(path):
        return "(missing)"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_chars * 4))
            if f.tell() > 0:
                f.readline()
            lines = f.readlines()[-max_lines:]
    except OSError as e:
        return f"(unreadable: {e})"

    text = "".join(lines)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text or "(empty)"


def _collect_status_counts(tasks: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        status = task.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
        for key, value in _collect_status_counts(task.get("children", [])).items():
            counts[key] = counts.get(key, 0) + value
    return counts


def _format_phase2_summary(p2_result: dict | None) -> str:
    if not p2_result:
        return "(not available)"

    lines = [
        f"Activity mode: {p2_result.get('activity_mode', '?')}",
        f"Replan requested: {p2_result.get('replan_requested', False)}",
    ]
    if p2_result.get("replan_reason"):
        lines.append(f"Replan reason: {p2_result['replan_reason']}")

    progress_results = p2_result.get("progress_results", [])
    if not progress_results:
        lines.append("Task progress signals: (none)")
        return "\n".join(lines)

    lines.append("Task progress signals:")
    for item in progress_results:
        artifacts = item.get("artifacts") or []
        lines.append(
            "- {task_id}: score={score:.2f}, output={output} bytes, "
            "delta={delta}, content_changed={changed}, stuck={stuck}, "
            "looping={looping}, artifacts={artifacts}".format(
                task_id=item.get("task_id", "?"),
                score=float(item.get("active_score", 0.0)),
                output=item.get("output_size", 0),
                delta=item.get("output_delta", 0),
                changed=item.get("content_changed", False),
                stuck=item.get("is_stuck", False),
                looping=item.get("is_looping", False),
                artifacts=", ".join(artifacts[:5]) if artifacts else "(none)",
            )
        )
    return "\n".join(lines)


def _pending_deep_review(state: dict) -> dict:
    review = state.get("pending_deep_review") or {}
    if not isinstance(review, dict) or review.get("consumed"):
        return {}
    return review


def _format_pending_deep_review(state: dict) -> str:
    review = _pending_deep_review(state)
    if not review:
        return "(none)"

    recommendations = review.get("recommendations") or []
    lines = [
        "A deep reflection ran just before this normal cycle. Treat these as advisory inputs, not commands; decide whether to act with tools now.",
        f"- saved_path: {review.get('saved_path') or '(not saved)'}",
        f"- created_at: {review.get('created_at') or '(unknown)'}",
    ]
    if recommendations:
        lines.append("- recommendations:")
        for item in recommendations[:5]:
            lines.append(f"  - {item}")

    excerpt = str(review.get("excerpt") or "").strip()
    if excerpt:
        if len(excerpt) > 1800:
            excerpt = excerpt[:1800].rstrip() + "\n... [truncated]"
        lines.extend(["", "review excerpt:", "```markdown", excerpt, "```"])
    return "\n".join(lines)


def _consume_pending_deep_review() -> None:
    state = state_mgr.load_state()
    review = state.get("pending_deep_review")
    if isinstance(review, dict) and not review.get("consumed"):
        review["consumed"] = True
        review["consumed_at"] = datetime.now().isoformat()
        state_mgr.save_state(state)


def _task_ids_for_workspace_snapshot(active_tasks: list, last_decisions: list) -> list[str]:
    ids: list[str] = []
    for task_id in active_tasks:
        if task_id and task_id not in ids:
            ids.append(task_id)
    for decision in reversed(last_decisions):
        task_id = decision.get("task_id")
        if task_id and task_id not in ids:
            ids.append(task_id)
        if len(ids) >= 8:
            break
    return ids


def _format_workspace_snapshot(task_ids: list[str], active_tasks: list) -> str:
    if not task_ids:
        return "(no active or recent task workspaces)"

    tasks_root = os.path.join(get_workspace_dir(), "tasks")
    chunks: list[str] = []
    for task_id in task_ids:
        task_dir = os.path.join(tasks_root, task_id)
        if not os.path.isdir(task_dir):
            chunks.append(f"### {task_id}\n(missing workspace)")
            continue

        entries = []
        try:
            for name in sorted(os.listdir(task_dir)):
                path = os.path.join(task_dir, name)
                try:
                    size = os.path.getsize(path) if os.path.isfile(path) else 0
                    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%m-%d %H:%M")
                except OSError:
                    size = 0
                    mtime = "unknown"
                suffix = "/" if os.path.isdir(path) else ""
                entries.append(f"- {name}{suffix} ({size} bytes, {mtime})")
        except OSError as e:
            entries.append(f"- (cannot list: {e})")

        if len(entries) > 16:
            entries = entries[:16] + [f"- ... ({len(entries) - 16} more entries)"]

        chunk = [f"### {task_id}", *entries]

        result_path = os.path.join(task_dir, "result.md")
        if os.path.exists(result_path):
            chunk.append("\nresult.md preview:\n```text")
            chunk.append(_read_text_preview(result_path, max_chars=1400))
            chunk.append("```")

        if task_id in active_tasks:
            chunk.append("\nactive output tail: hidden; use process.json and registered subprocess log paths")

        chunks.append("\n".join(chunk))

    return "\n\n".join(chunks)


def _format_workspace_digest(task_ids: list[str], active_tasks: list, p2_result: dict | None = None) -> str:
    """Lightweight structured workspace summary — only raw tail for exceptions.

    Output.jsonl data is hidden for healthy workers and completed tasks to
    prevent the LLM from wasting tokens on raw worker logs. The LLM should
    read result.md for outcomes and trust Phase 2 signals for progress.

    Layout per task state:
      - has_result_md=True:  status + artifacts + result.md preview (no output.jsonl fields)
      - healthy running:     Phase 2 monitoring summary line (no raw size/hash fields)
      - stuck/looping/idle:  full signals + raw output tail for diagnosis
    """
    if not task_ids:
        return "(no active or recent task workspaces)"

    # Build a lookup from Phase 2 progress results
    progress_by_id: dict[str, dict] = {}
    if p2_result:
        for pr in p2_result.get("progress_results", []):
            progress_by_id[pr.get("task_id", "")] = pr

    tasks_root = os.path.join(get_workspace_dir(), "tasks")
    chunks: list[str] = []
    for task_id in task_ids:
        task_dir = os.path.join(tasks_root, task_id)
        if not os.path.isdir(task_dir):
            chunks.append(f"### {task_id}\n(missing workspace)")
            continue

        pr = progress_by_id.get(task_id, {})
        is_active = task_id in active_tasks
        is_stuck = pr.get("is_stuck", False)
        is_looping = pr.get("is_looping", False)
        has_no_delta = pr.get("output_delta", -1) == 0
        cpu = pr.get("cpu_percent", 0.0)
        output_delta = pr.get("output_delta", 0)
        content_changed = pr.get("content_changed", False)
        active_score = pr.get("active_score", 0.0)

        # ── result.md check (filesystem, always authoritative) ──────
        result_path = os.path.join(task_dir, "result.md")
        has_result = os.path.exists(result_path)
        process_status = process_mgr.get_process_record_status(task_id)
        live_subprocesses = process_status.get("live_subprocesses", [])

        # ── Determine task health category ──────────────────────────
        is_exception = is_stuck or is_looping or (has_no_delta and is_active and cpu < 0.5 and not has_result)

        lines = [f"### {task_id}"]

        # ── Status line ─────────────────────────────────────────────
        if has_result and live_subprocesses:
            lines.append("- status: completion_pending (result.md exists, registered subprocess still running)")
        elif has_result:
            lines.append("- status: result_ready (result.md exists, requires verification)")
        elif is_active:
            lines.append(f"- status: running")
        else:
            lines.append(f"- status: recent (no worker running)")

        if process_status.get("exists"):
            lines.append("- process_json: present")
            for child in live_subprocesses[:3]:
                age = child.get("log_age_seconds")
                age_text = "unknown" if age is None else f"{age:.0f}s"
                lines.append(
                    f"- registered_subprocess: pid={child.get('pid')} "
                    f"kind={child.get('kind', 'subprocess')} "
                    f"log={child.get('log_path') or '(none)'} "
                    f"log_age={age_text}"
                )

        # ── Phase 2 monitoring (only for exception states without result.md) ──
        # Healthy workers: hide output.jsonl details to reduce LLM temptation.
        # Completed tasks: output.jsonl is irrelevant, result.md is the source.
        if has_result:
            # Completed task — output.jsonl fields are noise. Only show
            # artifacts and the result.md preview.
            lines.append(f"- artifacts: {', '.join(pr.get('artifacts', [])[:5]) or '(none)'}")
            if pr.get("error_log_size", 0) > 0:
                lines.append(f"- error_log_size: {pr['error_log_size']} (errors present)")
            lines.append(f"- has_result_md: True")
            if live_subprocesses:
                lines.append("- completion_gate: blocked by live registered subprocess")
            else:
                lines.append("- completion_gate: open; L1 must still verify acceptance criteria")

            lines.append("\nresult.md preview:\n```text")
            lines.append(_read_text_preview(result_path, max_chars=1000))
            lines.append("```")

        elif is_exception:
            # Exception state — show full signals for diagnosis
            lines.append(f"- output_size: {pr.get('output_size', 0)}")
            lines.append(f"- output_delta: {output_delta}")
            lines.append(f"- tail_hash_changed: {content_changed}")
            lines.append(f"- active_score: {active_score:.2f}")
            lines.append(f"- cpu_percent: {cpu:.1f}%")
            lines.append(f"- artifacts: {', '.join(pr.get('artifacts', [])[:5]) or '(none)'}")
            lines.append(f"- new_artifacts: {', '.join(pr.get('new_artifacts', [])[:5]) or '(none)'}")
            lines.append(f"- error_log_size: {pr.get('error_log_size', 0)}")
            lines.append(f"- has_result_md: False")
            if is_stuck:
                lines.append("- **STUCK**")
            if is_looping:
                lines.append("- **LOOPING**")

            lines.append("- raw_worker_output_tail: hidden; read process.json and registered log_path instead")

        else:
            # Healthy active worker without result.md — condensed Phase 2 summary
            delta_str = f"+{output_delta}" if output_delta > 0 else str(output_delta)
            lines.append(
                f"- Phase 2 monitoring: delta={delta_str}, content_changed={content_changed}, "
                f"score={active_score:.2f}, cpu={cpu:.1f}% — healthy, do NOT read output.jsonl"
            )
            lines.append(f"- artifacts: {', '.join(pr.get('artifacts', [])[:5]) or '(none)'}")
            lines.append(f"- new_artifacts: {', '.join(pr.get('new_artifacts', [])[:5]) or '(none)'}")
            if pr.get("error_log_size", 0) > 0:
                lines.append(f"- error_log_size: {pr['error_log_size']} (errors present)")
            lines.append(f"- has_result_md: False")

        chunks.append("\n".join(lines))

    return "\n\n".join(chunks)


def _get_task_file_block(state: dict, wake_change: dict | None) -> tuple[str, bool]:
    """Return task file content block and whether planning is needed.

    When planning IS needed or task file changed: return full text (up to 5000 chars).
    When planning is NOT needed: return hash + summary only.
    This keeps the stable prefix cacheable when nothing changed.
    """
    task_file = state.get("task_file", "")
    if not task_file:
        return "(no task file recorded)", False

    from .config import PROJECT_ROOT, PROJECTS_DIR
    task_file_path = os.path.join(PROJECT_ROOT, task_file)
    if not os.path.exists(task_file_path):
        return f"(task file not found: {task_file})", False

    root_children = state.get("tasks", [{}])[0].get("children", []) if state.get("tasks") else []
    task_file_changed = bool(wake_change and wake_change.get("changed"))
    task_file_needs_planning = bool(state.get("task_file_needs_planning"))
    planning_needed = (not root_children) or task_file_changed or task_file_needs_planning

    if planning_needed or task_file_changed:
        # Full content for planning/changes
        content = _read_text_preview(task_file_path, max_chars=5000)
        return content, planning_needed

    # Stable summary — hash ensures cache can be keyed on content identity
    try:
        with open(task_file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        file_hash = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]
        first_lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("<!--")][:3]
        summary = "\n".join(first_lines) if first_lines else "(empty)"
    except OSError:
        file_hash = "unreadable"
        summary = "(unreadable)"

    block = (
        f"Task file hash: {file_hash}\n"
        f"Task file summary (first lines):\n{summary}"
    )
    return block, planning_needed


def build_context_message(
    wake_change: dict | None = None,
    p2_result: dict | None = None,
) -> str:
    """Build the context message that describes the current state to the orchestrator.

    OPTIMIZATION: Returns a string with two logical sections:
      1. Stable prefix (mission, project context, task file, durable rules)
      2. Dynamic delta (cycle, active tasks, Phase 2, workspace digest, wake change)

    These are returned as a single string for backward compatibility.
    The stable content is positioned first so DeepSeek's automatic KV-cache
    can reuse it across cycles.

    Args:
        wake_change: Result from changelog.check_task_file_on_wake(), or None.
        p2_result: Phase 2 evaluation result, or None.
    """
    state = state_mgr.load_state()
    task_tree_structure = state_mgr.get_task_tree_structure()
    task_tree_status = state_mgr.get_task_tree_status()
    active_tasks = state.get("active_tasks", [])
    decision_log = state.get("decision_log", [])
    last_decisions = decision_log[-5:] if decision_log else []
    status_counts = _collect_status_counts(state.get("tasks", []))
    pending_count = status_counts.get("pending", 0)
    root_children = state.get("tasks", [{}])[0].get("children", []) if state.get("tasks") else []
    task_file_changed = bool(wake_change and wake_change.get("changed"))
    task_file_needs_planning = bool(state.get("task_file_needs_planning"))
    planning_needed = (not root_children) or task_file_changed or task_file_needs_planning

    # ── Stable prefix section ──────────────────────────────────────
    project_context = state.get("project_context", {}) or {}
    project_context_text = (
        f"Final goal: {project_context.get('final_goal') or '(not set)'}\n"
        f"Success criteria: {project_context.get('success_criteria') or '(not set)'}\n"
        f"Global constraints: {project_context.get('global_constraints') or '(not set)'}\n"
        f"Execution environment: {project_context.get('execution_environment') or '(not set)'}\n"
        f"Notes: {project_context.get('notes') or '(not set)'}\n"
        f"Updated: {project_context.get('updated_at') or '(never)'}"
    )

    task_file = state.get("task_file", "")
    task_file_block, _pf = _get_task_file_block(state, wake_change)

    # ── Changelog info (dynamic — only appears when file changed) ──
    changelog_info = ""
    if task_file and task_file_changed:
        from .config import PROJECT_ROOT, PROJECTS_DIR
        task_file_path = os.path.join(PROJECT_ROOT, task_file)
        project_name = get_project_name_for_task(task_file)
        if wake_change and wake_change.get("changed"):
            changelog_info = _format_wake_change_info(task_file, wake_change)
        else:
            change_info = get_file_change_info(task_file_path, PROJECTS_DIR, project_name)
            if change_info["is_changed"]:
                changelog_info = (
                    f"\n\n### ⚠️ 任务文件已变更\n"
                    f"检测到 {task_file} 内容已更新！\n"
                    f"上次处理哈希: {change_info['previous_hash'][:12]}...\n"
                    f"当前哈希: {change_info['current_hash'][:12]}...\n"
                    f"请读取任务文件内容，识别新增/变更的任务项，并更新任务树。\n"
                )

    # ── Dynamic delta section ─────────────────────────────────────
    running_info = ""
    running_tasks = [w for w in process_mgr.list_all() if w.get("running")]
    if running_tasks:
        for rt in running_tasks:
            running_info += (f"\n- {rt['task_id']} | PID: {rt['pid']} | "
                             f"Running: {rt['running']} | "
                             f"Elapsed: {rt['elapsed_minutes']}min | "
                             f"Budget: {rt['budget_minutes']}min | "
                             f"Monitor: {rt.get('monitor_path') or '(none)'} | "
                             f"Monitor size: {rt['output_size']} bytes")

    last_decision_str = ""
    for d in last_decisions:
        last_decision_str += f"\n  [{d['time'][:19]}] {d['task_id']}: {d['old_status']} → {d['new_status']} — {d['reason']}"

    # ── Session memory: shrink to 500 chars max ──────────────────
    session_preview = _read_text_preview(
        os.path.join(MEMORY_DIR, "session.md"),
        max_chars=500,
    )

    # ── Workspace: digest by default ─────────────────────────────
    workspace_snapshot = _format_workspace_digest(
        _task_ids_for_workspace_snapshot(active_tasks, last_decisions),
        active_tasks,
        p2_result,
    )

    phase2_summary = _format_phase2_summary(p2_result)
    deep_review_summary = _format_pending_deep_review(state)

    # ── Assemble: stable prefix first, then dynamic delta ─────────
    context = f"""## Stable Project Context

### Mission
{state.get('mission', 'NOT SET')}

### Project Context
{project_context_text}

### Aura Data Directory
{os.path.dirname(STATE_DIR)}

### Task File
Path: {task_file or '(none)'}
Current task batch prefix for new top-level categories: {state.get('task_batch', {}).get('current_prefix', 'A')}

```markdown
{task_file_block}
```

### Task Tree (stable structure)
{task_tree_structure}

### Durable Rules
- Use the dynamic cycle delta below for current decisions.
- Do not re-read files already summarized unless evidence is missing for a state-changing decision.
- If Planning needed is True, use the task file content above.
- Never spawn root or top-level category nodes (A1/B1). Workers go at level 3+ (A1.1).
- Max {MAX_CONCURRENT_TASKS} concurrent workers.
- Tool call budget: ~{TOOL_CALL_BUDGET_NORMAL} for normal cycles, ~{TOOL_CALL_BUDGET_PLANNING} for diagnostic/planning. Use fewer when possible — the budget is guidance, not a target.
- Every status change needs reason AND evidence.

---

## Dynamic Cycle Delta

### Cycle
Cycle #{state.get('total_cycles', 0)} | Created: {state.get('created_at', 'unknown')}
Changed this wake: {task_file_changed}
Planning needed: {planning_needed}

### Active Tasks
{active_tasks if active_tasks else '(none)'}

### Task Status Counts
{status_counts}

### Task Tree Status (dynamic)
{task_tree_status}

### Running Processes
{running_info if running_info else '(none)'}

### Last Decisions (most recent 5)
{last_decision_str if last_decision_str else '(none)'}

### Phase 2 Progress Signals
{phase2_summary}

### Pending Deep Reflection
{deep_review_summary}

### Session Memory (current focus)
```text
{session_preview}
```

### Workspace Digest (active + recent tasks)
{workspace_snapshot}
{changelog_info}

---

Now assess the situation and decide what to do.
- Use the snapshot above as the default source of truth. Do not re-read progress.md, session.md, state.json, or task directories unless the snapshot is missing evidence needed for a state-changing decision.
- Code does not parse task.md into semantic tasks. If Planning needed is True, read the task file content above, identify the final goal, success criteria, global constraints, and execution environment, then call update_project_context before decompose_task/update_task_tree/spawn_task.
- When planning, do not read other `.aura/<task-data-dir>/state`, `workspace`, `progress`, `summaries`, `cache`, or task metadata as current evidence. Other `.aura/<task-data-dir>/memory/...` files may be used only for transferable lessons, and any borrowed lesson should be noted with its scope.
- For a first plan, call decompose_task with parent_task_id="root" to create broad top-level categories, then decompose each category into concrete third-level tasks such as A1.1 before spawning work. For a changed task file, add only genuinely new/changed categories or concrete children as appropriate, and explicitly archive obsolete non-completed tasks when appropriate. Do not mark tasks completed from task.md wording alone; completed requires verifiable evidence or an explicit user request.
- Never spawn root or top-level category nodes such as A1/B1. Every worker-executable task must be at ROOT -> category -> task level or deeper, even when a category has only one concrete child.
- When planning or spawning a concrete task, preserve sibling awareness: summarize sibling tasks under the same parent, their attempted approach, evidence, and success/failure/pending outcome.
- Carry Project Context into every new task description, especially commands, env vars/API key usage, working directories, model/runtime choices, and success criteria.
- If there are active tasks or running processes, evaluate them from the Phase 2 signals and workspace snapshot first; read only the specific missing file if needed.
- If there are no active tasks, no running processes, no pending tasks (pending={pending_count}), no task-file change, and Planning needed is False, use no_op without extra file reads.
- Otherwise take action (spawn, kill, update, decompose, write_memory, or no_op) and record evidence for any status change."""

    return context


def _is_freshly_progressing(p: dict) -> bool:
    """Check if a worker is making fresh progress this cycle (not just resting on old artifacts)."""
    return (
        p.get("output_delta", 0) > 0
        or p.get("content_changed", False)
        or p.get("cpu_percent", 0.0) > 0.5
        or bool(p.get("new_artifacts", []))
    )


def _check_root_finalized(state: dict) -> bool:
    """Return True if the root task is in a terminal state."""
    tasks = state.get("tasks", [])
    if not tasks:
        return False
    root = tasks[0]
    return root.get("status") in {"completed", "failed", "archived"}


def should_skip_l1_cycle(
    state: dict | None = None,
    wake_change: dict | None = None,
    p2_result: dict | None = None,
) -> tuple[bool, str]:
    """Deterministic gate: decide if we can skip the L1 LLM call this cycle.

    Returns (skip: bool, reason: str).

    Skip only when ALL of these hold:
    - Task file unchanged
    - Planning not needed
    - No replan requested
    - No stuck/looping workers
    - Workers are freshly progressing (output_delta > 0, content_changed, or CPU active)
    - No free capacity for pending work
    - OR: nothing running AND nothing pending AND root is finalized

    This can eliminate 40-70% of L1 calls during healthy worker runs.
    """
    if not AURA_SKIP_HEALTHY_CYCLES:
        return False, "skip gate disabled"

    if state is None:
        state = state_mgr.load_state()

    # Task file changed → must call L1
    if wake_change and wake_change.get("changed"):
        return False, "task file changed"

    # Planning needed → must call L1
    if state.get("task_file_needs_planning"):
        return False, "planning needed"

    if p2_result is None:
        return False, "no Phase 2 result"

    # Replan requested → must call L1
    if p2_result.get("replan_requested"):
        return False, "replan requested"

    progress_results = p2_result.get("progress_results", [])

    # Stuck or looping → must call L1
    if _pending_deep_review(state):
        return False, "pending deep reflection recommendations"

    if any(p.get("is_stuck") or p.get("is_looping") for p in progress_results):
        return False, "worker stuck or looping"

    # Count pending leaf tasks and running workers
    all_tasks = state.get("tasks", [])
    pending = _count_pending_leaf_tasks(all_tasks)
    running_workers = [w for w in process_mgr.list_all() if w.get("running")]

    # Free capacity and pending work → call L1 to spawn
    if pending > 0 and len(running_workers) < MAX_CONCURRENT_TASKS:
        return False, "free worker capacity and pending work"

    # Workers running → skip only if ALL are freshly progressing
    if running_workers and progress_results:
        # Build lookup by task_id for progress results
        pr_by_id = {p.get("task_id", ""): p for p in progress_results}
        all_fresh = True
        for w in running_workers:
            pr = pr_by_id.get(w["task_id"], {})
            if not _is_freshly_progressing(pr):
                all_fresh = False
                break
        if all_fresh:
            return True, "all workers freshly progressing"

    # Nothing running and nothing pending → skip ONLY if root is finalized
    if not running_workers and pending == 0 and not state.get("task_file_needs_planning"):
        if not _check_root_finalized(state):
            return False, "finalization needed — root not in terminal state"
        return True, "nothing pending or running; root finalized"

    return False, "needs L1 decision"


def _count_pending_leaf_tasks(tasks: list) -> int:
    """Count pending leaf tasks (tasks with no children that are pending)."""
    count = 0
    for task in tasks:
        children = task.get("children", [])
        if children:
            count += _count_pending_leaf_tasks(children)
        elif task.get("status") == "pending":
            count += 1
    return count

# ── Phase 2 state (module-level, persists across cycles) ────────────
_safe_mode: bool = False
_consecutive_no_progress: int = 0
_consecutive_crashes: int = 0
MAX_CONSECUTIVE_CRASHES: int = 3
_previous_output_sizes: dict[str, int] = {}
_previous_content_hashes: dict[str, str] = {}
_stuck_cycle_counters: dict[str, int] = {}

# ── Phase 3: Review nudge tracking ───────────────────────────────────
_consecutive_skips: int = 0

_STATE_CHANGING_TOOLS = {
    "spawn_task",
    "kill_task",
    "update_task_tree",
    "decompose_task",
    "update_project_context",
}


def _render_progress_safely(reason: str = "") -> None:
    try:
        progress_mgr.render_progress()
    except Exception as err:
        suffix = f" after {reason}" if reason else ""
        print(f"  [WARN] Could not render progress{suffix}: {err}")

# ── T0 optimization: session write dedup ──────────────────────────────
# session.md is written every cycle via _update_session(). Since the
# content structure is mostly static (only cycle_num and timestamp
# change), we track the last written content to skip writes when
# the content is identical (e.g. consecutive cycles with same tool count).
_last_session_content: str = ""


def _run_phase2_eval(active_tasks: list, wake_change: dict | None = None) -> dict:
    """Run Phase 2 pre-cycle evaluation with full crash protection.

    Any error in Phase 2 code is caught here — the orchestrator NEVER
    crashes because of upgrade module bugs. Falls back to basic mode.

    Args:
        active_tasks: Currently active task IDs.
        wake_change: R7 per-wake task file change info, or None.
    """
    global _consecutive_no_progress, _consecutive_crashes, _safe_mode

    if _safe_mode:
        return {
            "activity_mode": "active",
            "replan_requested": False,
            "replan_reason": "",
            "progress_results": [],
            "phase2_ok": False,
        }

    try:
        # Get process health metrics for all running workers
        worker_health: dict[str, dict] = {}
        for w in process_mgr.list_all():
            worker_health[w["task_id"]] = {
                "cpu": w.get("cpu_percent", 0.0),
                "memory_mb": w.get("memory_mb", 0.0),
                "monitor_path": w.get("monitor_path"),
            }

        progress_results = []
        for task_id in active_tasks:
            prev_size = _previous_output_sizes.get(task_id, 0)
            prev_hash = _previous_content_hashes.get(task_id, "")
            cpu = worker_health.get(task_id, {}).get("cpu", 0.0)
            monitor_path = worker_health.get(task_id, {}).get("monitor_path")

            result = evaluate_progress(task_id, prev_size, prev_hash, cpu, monitor_path=monitor_path)
            _previous_output_sizes[task_id] = result["output_size"]
            _previous_content_hashes[task_id] = result.get("content_hash", "")
            progress_results.append({"task_id": task_id, **result})

            if result["is_stuck"]:
                _stuck_cycle_counters[task_id] = _stuck_cycle_counters.get(task_id, 0) + 1
                if _stuck_cycle_counters[task_id] >= STUCK_THRESHOLD_CYCLES:
                    tail = result.get("tail_analysis", {})
                    print(f"  [P2] Task {task_id}: STUCK for {_stuck_cycle_counters[task_id]} cycles "
                          f"(cpu={cpu:.1f}%, hash_changed={result.get('content_changed')}, "
                          f"looping={result.get('is_looping')})")
            else:
                _stuck_cycle_counters[task_id] = 0

        any_progress = any(
            p["output_delta"] > 0
            or p.get("content_changed")
            or p.get("new_artifacts")
            or p.get("cpu_percent", 0.0) > 0.5
            for p in progress_results
        )
        if any_progress:
            _consecutive_no_progress = 0
        else:
            _consecutive_no_progress += 1

        state = state_mgr.load_state()
        created_at = state.get("created_at", datetime.now().isoformat())
        try:
            created_dt = datetime.fromisoformat(created_at)
            elapsed_hours = (datetime.now() - created_dt).total_seconds() / 3600.0
        except (ValueError, TypeError):
            elapsed_hours = 0.0

        has_any_output_ever = any(
            pr["has_output"] for pr in progress_results
        ) or any(
            os.path.exists(os.path.join(get_workspace_dir(), "tasks", t, "result.md"))
            for t in active_tasks
        )

        # R7: 检测用户是否在 task file 中添加了新的需求
        has_new_requirements = (
            wake_change is not None
            and wake_change.get("changed")
            and (
                len(wake_change.get("added_requirement_lines", [])) > 0
                or len(wake_change.get("removed_requirement_lines", [])) > 0
            )
        )

        replan_check = check_replan_needed(
            consecutive_no_progress_cycles=_consecutive_no_progress,
            total_elapsed_hours=elapsed_hours,
            has_any_output=has_any_output_ever,
            has_new_requirements=has_new_requirements,
        )

        activity_mode = get_activity_mode(progress_results)

        if replan_check["replan_requested"]:
            print(f"  [P2] REPLAN TRIGGERED: {replan_check['trigger_reason']}")

        print(f"  [P2] Activity: {activity_mode} | No-progress cycles: {_consecutive_no_progress}")

        # Reset crash counter on successful Phase 2 execution
        _consecutive_crashes = 0

        return {
            "activity_mode": activity_mode,
            "replan_requested": replan_check["replan_requested"],
            "replan_reason": replan_check["trigger_reason"],
            "progress_results": progress_results,
            "phase2_ok": True,
        }

    except Exception as e:
        _consecutive_crashes += 1
        print(f"  [RESILIENCE] Phase 2 eval failed: {e}")
        print(f"  [RESILIENCE] Consecutive Phase 2 crashes: {_consecutive_crashes}/{MAX_CONSECUTIVE_CRASHES}")

        if _consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
            _safe_mode = True
            print(f"  [RESILIENCE] ENTERING SAFE MODE — Phase 2 disabled.")
            print(f"  [RESILIENCE] Will attempt self-heal by spawning fixer worker.")

        return {
            "activity_mode": "active",
            "replan_requested": _consecutive_crashes >= MAX_CONSECUTIVE_CRASHES,
            "replan_reason": f"Phase 2 crashed {_consecutive_crashes} times consecutively — code bug suspected" if _consecutive_crashes >= MAX_CONSECUTIVE_CRASHES else "",
            "progress_results": [],
            "phase2_ok": False,
        }


def run_cycle(wake_change: dict | None = None) -> dict:
    """Execute one full wake cycle. Returns a summary dict.

    Args:
        wake_change: Result from changelog.check_task_file_on_wake(), or None.
                     Contains diff info when the task file was modified.

    CRASH RESILIENCE: The orchestrator NEVER crashes permanently.
    - Phase 2 code is wrapped in try/except with fallbacks
    - After N consecutive Phase 2 crashes, safe mode activates
    - In safe mode, the orchestrator can still spawn fixer workers
    - Safe mode resets when Phase 2 is fixed (import succeeds)
    """
    global _safe_mode, _consecutive_skips

    cycle_start = time.time()
    cycle_num = state_mgr.log_cycle()

    print(f"\n{'='*60}")
    print(f"  Aura Agent — Cycle #{cycle_num}")
    if _safe_mode:
        print(f"  *** SAFE MODE — Phase 2+ features disabled ***")
    if _phase_load_errors:
        for err in _phase_load_errors:
            print(f"  Load error: {err}")
    print(f"  Wake at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # ── Phase 2: Pre-cycle evaluation (with full crash protection) ──
    state = state_mgr.load_state()
    active_tasks = state.get("active_tasks", [])
    p2_result = _run_phase2_eval(active_tasks, wake_change=wake_change)

    # ── Deterministic skip gate ────────────────────────────────────
    global _consecutive_skips
    forced_l1_reason = ""
    skip, skip_reason = should_skip_l1_cycle(state, wake_change, p2_result)
    if skip and not _safe_mode:
        _consecutive_skips += 1
        if _consecutive_skips > AURA_MAX_SKIPPED_CYCLES:
            skip = False
            skip_reason = f"max consecutive skips ({AURA_MAX_SKIPPED_CYCLES}) reached"
            forced_l1_reason = skip_reason
            _consecutive_skips = 0
        else:
            log_skip(skip_reason)
            print(f"  [Skip] Skipping L1 call: {skip_reason} (consecutive skips: {_consecutive_skips})")
            _update_session(cycle_num, 0, p2_result)
            elapsed = time.time() - cycle_start
            return {
                "cycle": cycle_num,
                "tool_calls": 0,
                "api_calls": 0,
                "elapsed": round(elapsed, 2),
                "error": False,
                "activity_mode": p2_result["activity_mode"],
                "replan_requested": p2_result["replan_requested"],
                "review_requested": False,
                "safe_mode": _safe_mode,
                "token_usage": {},
                "skipped": True,
                "skip_reason": skip_reason,
            }
    else:
        _consecutive_skips = 0

    # ── Build context and call API ──────────────────────────────────
    context_msg = build_context_message(wake_change=wake_change, p2_result=p2_result)

    # If in safe mode, prepend a self-healing instruction
    if _safe_mode:
        context_msg = (
            "*** SAFE MODE ACTIVE — Your upgrade modules have bugs. ***\n"
            "The Phase 2 code is broken and has been disabled.\n"
            "Your priority: spawn a Layer 2 worker to fix the broken "
            f"code files. Load errors: {'; '.join(_phase_load_errors) if _phase_load_errors else 'runtime crash'}.\n"
            "After fixing, safe mode will auto-reset when imports succeed.\n\n"
        ) + context_msg

    client = anthropic.Anthropic(
        base_url=AURA_API_BASE_URL,
        api_key=AURA_API_KEY,
        auth_token=AURA_API_KEY,
    )

    messages = [{"role": "user", "content": context_msg}]

    # Apply explicit cache_control for Anthropic native prompt caching.
    # Split into stable prefix (cacheable) and dynamic delta to maximise
    # cache reuse across cycles. Only enabled for Anthropic native API
    # because DeepSeek may not support the cache_control field.
    if AURA_EXPLICIT_PROMPT_CACHE and "api.anthropic.com" in AURA_API_BASE_URL:
        parts = context_msg.split("\n---\n", 1)
        if len(parts) == 2:
            stable_prefix, dynamic_delta = parts
            messages[0]["content"] = [
                {
                    "type": "text",
                    "text": stable_prefix + "\n---\n",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": dynamic_delta,
                },
            ]
        else:
            messages[0]["content"] = [
                {
                    "type": "text",
                    "text": context_msg,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

    tool_call_count = 0
    api_call_count = 0
    # Accumulate usage across tool-call loops within this cycle
    cycle_usage: dict[str, int] = {}

    try:
        while True:
            # Hard cap: prevent catastrophic $1+ cycles where the model
            # loops on polling calls (list_directory / read_file / no_op)
            # indefinitely without producing a text decision.
            if api_call_count >= MAX_TOOL_CALLS_PER_CYCLE:
                print(f"\n  [LIMIT] Hit {MAX_TOOL_CALLS_PER_CYCLE} API calls — forcing cycle end.")
                elapsed = time.time() - cycle_start
                _render_progress_safely("forced cycle end")
                _update_session(cycle_num, tool_call_count, p2_result)
                return {
                    "cycle": cycle_num,
                    "tool_calls": tool_call_count,
                    "api_calls": api_call_count,
                    "elapsed": round(elapsed, 2),
                    "error": False,
                    "activity_mode": p2_result["activity_mode"],
                    "replan_requested": p2_result["replan_requested"],
                    "review_requested": False,
                    "safe_mode": _safe_mode,
                    "token_usage": cycle_usage,
                    "forced_l1_reason": f"max API calls ({MAX_TOOL_CALLS_PER_CYCLE}) reached",
                }

            response = _call_api_with_retry(
                client,
                system=_ORCHESTRATOR_SYSTEM_PROMPT,
                messages=messages,
                tools=get_active_tool_definitions(),
            )
            api_call_count += 1

            # ── Log token usage ─────────────────────────────────
            usage = log_usage("run_cycle", response, extra={"cycle": cycle_num})
            for k, v in usage.items():
                cycle_usage[k] = cycle_usage.get(k, 0) + v

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                final_text = "".join(
                    b.text for b in response.content if b.type == "text"
                )
                print(f"\n[Orchestrator] Decision complete: {final_text[:200]}...")
                if cycle_usage:
                    print(f"  [Tokens] {format_cycle_stats(cycle_usage)}")

                _render_progress_safely("cycle completion")

                # Update session memory
                _update_session(cycle_num, tool_call_count, p2_result)
                _consume_pending_deep_review()

                review_requested = False
                # Reflection is scheduled by main.py. Keeping the review
                # trigger there avoids double-running reviews in the same
                # cycle while preserving this counter for context nudges.

                elapsed = time.time() - cycle_start
                return {
                    "cycle": cycle_num,
                    "tool_calls": tool_call_count,
                    "api_calls": api_call_count,
                    "elapsed": round(elapsed, 2),
                    "error": False,
                    "activity_mode": p2_result["activity_mode"],
                    "replan_requested": p2_result["replan_requested"],
                    "review_requested": review_requested,
                    "safe_mode": _safe_mode,
                    "token_usage": cycle_usage,
                    "forced_l1_reason": forced_l1_reason,
                }

            # Execute ALL tool calls, then loop back to API
            # Preserve ALL content blocks (thinking, text, tool_use) for DeepSeek compatibility
            messages.append({
                "role": "assistant",
                "content": [
                    b.model_dump() if hasattr(b, 'model_dump') else b
                    for b in response.content
                ],
            })

            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_input = tool_use.input if isinstance(tool_use.input, dict) else {}
                print(f"  [Tool] {tool_name}({json.dumps(tool_input, ensure_ascii=False)[:120]})")

                result_str = execute_tool(tool_name, tool_input)
                tool_call_count += 1
                if tool_name in _STATE_CHANGING_TOOLS:
                    _render_progress_safely(tool_name)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_str,
                })

            messages.append({
                "role": "user",
                "content": tool_results,
            })

    except Exception as e:
        print(f"[ERROR] Cycle API call failed: {e}")
        import traceback
        traceback.print_exc()

        _render_progress_safely("cycle error")

        elapsed = time.time() - cycle_start
        return {
            "cycle": cycle_num,
            "tool_calls": tool_call_count,
            "api_calls": api_call_count,
            "elapsed": round(elapsed, 2),
            "error": True,
            "error_message": str(e),
            "activity_mode": p2_result["activity_mode"],
            "review_requested": False,
            "safe_mode": _safe_mode,
            "token_usage": cycle_usage,
            "forced_l1_reason": forced_l1_reason,
        }


def _update_session(cycle_num: int, tool_count: int, p2_result: dict) -> None:
    """Update short-term session memory.

    T0 optimization: only writes to disk if the content actually changed
    from the previous write. session.md is rewritten every cycle, but
    the meaningful content (activity mode, tool count, safe mode) rarely
    changes. Avoiding unnecessary writes saves disk I/O.
    """
    global _last_session_content

    content = (
        f"# Session Memory\n"
        f"## Last Cycle: #{cycle_num} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Tools called: {tool_count}\n"
        f"Activity mode: {p2_result.get('activity_mode', '?')}\n"
        f"Safe mode: {p2_result.get('phase2_ok', True) == False}\n"
        f"\n"
        f"### Current Focus\n"
        f"- Check for changes in output files since last cycle\n"
        f"- Evaluate whether active tasks are producing verifiable results\n"
        f"\n"
    )

    if content == _last_session_content:
        return  # Nothing changed — skip write

    _last_session_content = content
    memory_mgr.write_session(content)


# ── API timeout (seconds) — from config file only ──────────────────────
_API_TIMEOUT = API_TIMEOUT_SECONDS


def _call_api_with_retry(
    client: anthropic.Anthropic,
    system: str,
    messages: list,
    tools: list,
) -> Any:
    """Call the Claude API with timeout, logging, and exponential backoff."""
    last_error = None
    for attempt in range(API_RETRY_COUNT):
        try:
            t0 = time.time()
            print(f"  [API] Calling {AURA_API_MODEL}...")
            result = client.messages.create(
                model=AURA_API_MODEL,
                max_tokens=AURA_API_MAX_TOKENS,
                system=system,
                messages=messages,
                tools=tools,
                timeout=_API_TIMEOUT,
            )
            elapsed = time.time() - t0
            print(f"  [API] Done in {elapsed:.1f}s")
            return result
        except Exception as e:
            last_error = e
            elapsed = time.time() - t0
            print(f"  [API] Failed in {elapsed:.1f}s: {e}")
            if attempt < API_RETRY_COUNT - 1:
                delay = API_RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  [Retry] Attempt {attempt + 1} failed. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  [Retry] All {API_RETRY_COUNT} attempts failed.")
                raise last_error
