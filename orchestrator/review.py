"""Review Engine for Aura Agent Orchestrator.

Provides independent reflection/review capabilities:
  - review_cycle(): Periodic deep review of strategy and progress
  - extract_skill(): Extract reusable patterns from completed tasks
  - compress_memory(): Compress long-term memory when exceeding limits

Phase 3 — Reflection & Evolution.
"""

import os
import json
import time
from datetime import datetime
from typing import Optional

import anthropic

from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    ANTHROPIC_MODEL,
    ANTHROPIC_MAX_TOKENS,
    API_RETRY_COUNT,
    API_RETRY_BASE_DELAY,
    LONG_TERM_MEMORY_MAX_CHARS,
    MEMORY_DIR,
    SKILLS_DIR,
    STATE_DIR,
    get_workspace_dir,
    STUCK_THRESHOLD_CYCLES,
)
from . import state as state_mgr
from . import memory as memory_mgr
from .token_tracker import log_usage, format_cycle_stats, extract_usage


_REVIEW_SYSTEM_PROMPT = """You are Aura Agent's Reflection Engine — an independent reviewer that analyzes the agent's own performance.

## Your Role
You are NOT the decision-maker. You are the evaluator. Your job is to step back and ask: "Is the current approach working? What could be improved? What patterns have emerged?"

## Review Framework

Analyze the provided context and answer:

### 1. Progress Assessment
- Is the mission making meaningful progress? (Evidence-based)
- What percentage of tasks have verifiable output?
- Are there any stuck tasks? For how long?

### 2. Strategy Evaluation
- Is the current approach (depth-first, breadth-first, decomposition) appropriate?
- Are there alternative strategies that haven't been tried?
- Is the task decomposition at the right granularity?

### 3. Pattern Recognition
- What patterns have emerged from successful tasks?
- What patterns have emerged from failed tasks?
- Are there recurring obstacles?

### 4. Recommendations
- Concrete, actionable changes to improve progress
- Specific tasks that should be killed, decomposed, or retried
- Strategy adjustments

## Output Format

Produce a concise review in markdown with these sections. Be specific. Cite evidence. Every recommendation must connect to observable facts.

Keep the output under 2000 characters — focus on the most impactful observations."""


def build_review_context() -> str:
    """Build context for the review engine."""
    state = state_mgr.load_state()
    task_tree = state_mgr.get_task_tree_summary()
    active_tasks = state.get("active_tasks", [])
    decision_log = state.get("decision_log", [])
    recent_decisions = decision_log[-20:]

    # Count task statuses
    status_counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0, "blocked": 0, "killed": 0}
    all_tasks = state.get("tasks", [])
    _count_statuses(all_tasks, status_counts)

    # Read long-term memory summary
    memory_preview = memory_mgr.get_memory_preview(1500)

    context = f"""## Review Context

### Mission
{state.get('mission', 'NOT SET')}

### Overall Stats
- Total cycles: {state.get('total_cycles', 0)}
- Active tasks: {len(active_tasks)}
- Task status distribution: {json.dumps(status_counts)}
- Total decisions logged: {len(decision_log)}

### Task Tree
{task_tree}

### Recent Decisions (last 20)
"""
    for d in recent_decisions:
        context += f"- [{d['time'][:19]}] {d['task_id']}: {d['old_status']} → {d['new_status']} — {d['reason']}\n"

    context += f"""
### Long-term Memory Preview
{memory_preview}

### Active Tasks Status
"""
    for task_id in active_tasks:
        task_dir = os.path.join(get_workspace_dir(), "tasks", task_id)
        if os.path.isdir(task_dir):
            output_path = os.path.join(task_dir, "output.jsonl")
            result_path = os.path.join(task_dir, "result.md")
            error_path = os.path.join(task_dir, "error.log")
            output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            has_result = os.path.exists(result_path)
            error_size = os.path.getsize(error_path) if os.path.exists(error_path) else 0
            context += f"- {task_id}: output={output_size}B, result={'yes' if has_result else 'no'}, errors={error_size}B\n"
        else:
            context += f"- {task_id}: no task directory found\n"

    return context


def review_cycle(force: bool = False) -> dict:
    """Run an independent review cycle.

    Makes a separate Claude API call with a review-specific prompt to
    evaluate the agent's own performance and strategy.

    Args:
        force: If True, run review regardless of nudge interval.

    Returns:
        dict with:
          - review_text (str): The review report in markdown.
          - saved_path (str): Path where the review was saved.
          - recommendations (list[str]): Key recommendations extracted.
    """
    print(f"\n  [Review] Starting reflection cycle...")

    context = build_review_context()

    client = anthropic.Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=ANTHROPIC_API_KEY,
    )

    messages = [{"role": "user", "content": context}]

    _API_TIMEOUT = int(os.environ.get("AURA_API_TIMEOUT", "300"))

    review_text = ""
    review_usage = {}
    for attempt in range(API_RETRY_COUNT):
        try:
            t0 = time.time()
            print(f"  [Review] Calling {ANTHROPIC_MODEL} (attempt {attempt + 1}/{API_RETRY_COUNT}, timeout {_API_TIMEOUT}s)...")
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=min(ANTHROPIC_MAX_TOKENS, 2048),
                system=_REVIEW_SYSTEM_PROMPT,
                messages=messages,
                timeout=_API_TIMEOUT,
            )
            elapsed = time.time() - t0
            print(f"  [Review] Done in {elapsed:.1f}s")
            review_text = "".join(
                block.text for block in response.content
                if block.type == "text"
            )
            review_usage = extract_usage(response)
            log_usage("review_cycle", response)
            break
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [Review] Failed in {elapsed:.1f}s: {e}")
            if attempt < API_RETRY_COUNT - 1:
                delay = API_RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  [Review] Retry {attempt + 1} failed. Waiting {delay}s...")
                time.sleep(delay)
            else:
                print(f"  [Review] All {API_RETRY_COUNT} attempts failed.")
                return {
                    "review_text": f"ERROR: Review failed after {API_RETRY_COUNT} attempts: {e}",
                    "saved_path": "",
                    "recommendations": [],
                    "error": str(e),
                    "token_usage": {},
                }

    # Save review to memory directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    review_filename = f"review_{timestamp}.md"
    reviews_dir = os.path.join(MEMORY_DIR, "reviews")
    review_path = os.path.join(reviews_dir, review_filename)

    os.makedirs(reviews_dir, exist_ok=True)
    with open(review_path, "w", encoding="utf-8") as f:
        f.write(f"# Reflection Review — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(review_text)
        f.write(f"\n\n---\n*Generated by Aura Agent Review Engine (Phase 3)*\n")

    # Extract key recommendations
    recommendations = _extract_recommendations(review_text)

    # Log to long-term memory
    memory_mgr.append_memory(
        "decision",
        f"反思审查完成 ({timestamp})\n"
        f"关键建议: {'; '.join(recommendations[:3]) if recommendations else '无特别建议'}"
    )

    print(f"  [Review] Saved to {review_path}")
    print(f"  [Review] {len(recommendations)} recommendations extracted")

    return {
        "review_text": review_text,
        "saved_path": review_path,
        "recommendations": recommendations,
        "error": None,
        "token_usage": review_usage,
    }


def extract_skill(task_id: str) -> Optional[str]:
    """Extract reusable skill patterns from a completed task.

    Analyzes the task's result.md and output to identify:
    - Methods and approaches that worked
    - Tool usage patterns
    - Decision heuristics
    - Common pitfalls to avoid

    Args:
        task_id: ID of a completed task.

    Returns:
        Skill name if skill was extracted and saved, None otherwise.
    """
    task_dir = os.path.join(get_workspace_dir(), "tasks", task_id)
    result_path = os.path.join(task_dir, "result.md")

    if not os.path.exists(result_path):
        print(f"  [Skill] No result.md for task {task_id}")
        return None

    with open(result_path, "r", encoding="utf-8") as f:
        result_text = f.read()

    # Use a simple heuristic to extract skill patterns
    # In a full implementation, this would use Claude API
    skill_name = _derive_skill_name(task_id, result_text)
    skill_content = _build_skill_content(task_id, result_text)

    os.makedirs(SKILLS_DIR, exist_ok=True)
    skill_path = os.path.join(SKILLS_DIR, f"{skill_name}.md")

    with open(skill_path, "w", encoding="utf-8") as f:
        f.write(skill_content)

    print(f"  [Skill] Extracted '{skill_name}' from task {task_id} → skills/{skill_name}.md")
    return skill_name


def compress_memory(force: bool = False) -> dict:
    """Compress long-term memory when it exceeds the character limit.

    Uses a Claude API call to distill the memory into a structured summary
    while preserving the most critical information.

    Args:
        force: If True, compress regardless of size limit.

    Returns:
        dict with compression results.
    """
    memory_path = os.path.join(MEMORY_DIR, "MEMORY.md")
    if not os.path.exists(memory_path):
        return {"compressed": False, "reason": "No memory file exists"}

    with open(memory_path, "r", encoding="utf-8") as f:
        current_memory = f.read()

    if not force and len(current_memory) <= LONG_TERM_MEMORY_MAX_CHARS:
        return {"compressed": False, "reason": f"Under limit ({len(current_memory)}/{LONG_TERM_MEMORY_MAX_CHARS})"}

    print(f"  [Memory] Compressing: {len(current_memory)} → target {LONG_TERM_MEMORY_MAX_CHARS}")

    # Use Claude API to compress
    client = anthropic.Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=ANTHROPIC_API_KEY,
    )

    compress_prompt = f"""You are a memory compression engine. Condense the following long-term memory into a structured summary.

## Rules
- Preserve ALL mission-critical information
- Preserve verified lessons and patterns
- Preserve key decisions and their outcomes
- Remove redundant or outdated information
- Use concise bullet points
- Target length: under {LONG_TERM_MEMORY_MAX_CHARS} characters

## Current Memory ({len(current_memory)} chars)

{current_memory}

## Output
Produce only the compressed memory (no preamble, no explanation)."""

    compressed = ""
    for attempt in range(API_RETRY_COUNT):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=min(ANTHROPIC_MAX_TOKENS, 4096),
                system="You are a memory compression engine. Output ONLY the compressed memory text.",
                messages=[{"role": "user", "content": compress_prompt}],
            )
            compressed = "".join(
                block.text for block in response.content
                if block.type == "text"
            )
            log_usage("compress_memory", response)
            break
        except Exception as e:
            if attempt < API_RETRY_COUNT - 1:
                time.sleep(API_RETRY_BASE_DELAY * (2 ** attempt))
            else:
                return {"compressed": False, "error": str(e)}

    if compressed:
        with open(memory_path, "w", encoding="utf-8") as f:
            f.write(compressed)
        print(f"  [Memory] Compressed: {len(current_memory)} → {len(compressed)} chars")
        return {
            "compressed": True,
            "before_chars": len(current_memory),
            "after_chars": len(compressed),
        }

    return {"compressed": False, "reason": "Compression produced no output"}


# ── Helpers ──────────────────────────────────────────────────────────


def _count_statuses(tasks: list, counts: dict) -> None:
    """Recursively count task statuses."""
    for task in tasks:
        status = task.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
        if "children" in task:
            _count_statuses(task["children"], counts)


def _extract_recommendations(text: str) -> list[str]:
    """Extract recommendations from review text."""
    recs = []
    in_rec_section = False
    for line in text.split("\n"):
        line = line.strip()
        if (
            ("recommend" in line.lower() or "建议" in line or "改进" in line)
            and ("##" in line or "###" in line)
        ):
            in_rec_section = True
            continue
        if in_rec_section and line.startswith(("- ", "* ", "1. ", "2. ", "3. ")):
            rec = line.lstrip("-* 0123456789. ")
            if len(rec) > 10:
                recs.append(rec)
        if in_rec_section and line.startswith("##") and "recommend" not in line.lower():
            in_rec_section = False
    return recs


def _derive_skill_name(task_id: str, result_text: str) -> str:
    """Derive a skill name from task ID and result."""
    # Simple heuristic: use task_id as base, clean it up
    name = task_id.lower().replace(" ", "_").replace(".", "_")
    # Check for keywords in result
    keywords = ["debug", "fix", "implement", "refactor", "test", "deploy", "config"]
    for kw in keywords:
        if kw in result_text.lower()[:500]:
            name = f"{kw}_{name}"
            break
    return name[:50]


def _build_skill_content(task_id: str, result_text: str) -> str:
    """Build skill markdown content from task result.

    Uses heuristic extraction first (free). Only calls Claude API for
    high-value candidates where the heuristic found promising signals.
    """
    patterns_text = _extract_patterns_heuristic(result_text)
    if _is_high_value_skill_candidate(result_text, patterns_text):
        claude_result = _extract_patterns_via_claude(task_id, result_text)
        if claude_result:
            patterns_text = claude_result

    return f"""# Skill: {task_id}

*Extracted from completed task {task_id}*

## Source Task
- Task ID: {task_id}
- Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Patterns

{patterns_text}

## Full Result
See .aura/workspace/tasks/{task_id}/result.md for complete details.
"""


def _extract_patterns_via_claude(task_id: str, result_text: str) -> str:
    """Use Claude API to extract reusable patterns from a task result.

    Identifies: methods used, tools employed, key decisions, pitfalls,
    and patterns that can be reused in future tasks.
    """
    if len(result_text) < 50:
        return ""

    skill_prompt = f"""Analyze the following completed task result and extract reusable patterns.

## Task Result ({task_id})

{result_text[:6000]}

## Extraction Instructions

Identify and summarize:
1. **Methods & Approaches**: What specific methods or approaches were used? What worked?
2. **Tools Used**: What tools were employed and how effectively?
3. **Key Decisions**: What decisions were made and what was their impact?
4. **Pitfalls & Lessons**: What went wrong? What should future tasks avoid?
5. **Reusable Pattern**: What is the core repeatable pattern from this task?

Output ONLY the extracted patterns in concise markdown bullet points. Be specific and actionable."""

    try:
        client = anthropic.Anthropic(
            base_url=ANTHROPIC_BASE_URL,
            api_key=ANTHROPIC_API_KEY,
        )
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=min(ANTHROPIC_MAX_TOKENS, 1024),
            system="You are a pattern extraction engine. Output ONLY the extracted patterns as markdown bullet points.",
            messages=[{"role": "user", "content": skill_prompt}],
        )
        extracted = "".join(
            block.text for block in response.content
            if block.type == "text"
        )
        log_usage("extract_skill", response, extra={"task_id": task_id})
        return extracted.strip()
    except Exception as e:
        print(f"  [Skill] Claude extraction failed for {task_id}: {e}")
        return ""


def _is_high_value_skill_candidate(result_text: str, heuristic_text: str) -> bool:
    """Decide whether a task result justifies a Claude API call for skill extraction.

    Returns True when the heuristic found multiple pattern signals AND the
    result text is substantial enough that LLM extraction would add value.
    """
    if len(result_text) < 200:
        return False
    if len(heuristic_text) < 80:
        return False
    # Count distinct pattern signals
    signal_keywords = ["pattern", "lesson", "approach", "method", "tool", "decision", "fix", "bug"]
    signal_count = sum(1 for kw in signal_keywords if kw in heuristic_text.lower())
    return signal_count >= 2


def _extract_patterns_heuristic(result_text: str) -> str:
    """Heuristic extraction of patterns from task result (no API call)."""
    lines = result_text.split("\n")
    summary_lines = []
    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in ["##", "pattern", "lesson", "approach", "method", "tool", "decision", "fix", "bug"]):
            summary_lines.append(line)

    if not summary_lines:
        return result_text[:500]

    return "\n".join(summary_lines)
