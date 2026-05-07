# Project Progress Report — {{ updated_at }}

## 1. Long-term Mission

> {{ mission }}

## 2. Project Status Overview

## Project Context

- **Final goal**: {{ project_context.get('final_goal') or '(not set)' }}
- **Success criteria**: {{ project_context.get('success_criteria') or '(not set)' }}
- **Global constraints**: {{ project_context.get('global_constraints') or '(not set)' }}
- **Execution environment**: {{ project_context.get('execution_environment') or '(not set)' }}

| Indicator | Value |
|------|------|
| Total Elapsed | {{ total_hours }} hours |
| Wake Cycles | {{ total_cycles }} |
| Completed Tasks | {{ total_completed }} |
| Active Tasks | {{ active_count }} |
| Blocked Tasks | {{ total_blocked }} |
| Failed Tasks | {{ total_failed }} |
| Replans | {{ replan_count }} |

## 3. Task Tree Overview

{% for task in task_tree %}
{% if task.strike %}
{{ task.indent }}{{ task.icon }} ~~[{{ task.id }}] {{ task.description }} - {{ task.status }}~~
{% else %}
{{ task.indent }}{{ task.icon }} [{{ task.id }}] {{ task.description }} — {{ task.status }}
{% endif %}
{% endfor %}

## 4. Active Task Details

{% if active_tasks %}
{% for task in active_tasks %}
### {{ task.id }} — {{ task.description | truncate(80) }}
- **Status**: {{ task.status }}
- **Depth**: {{ task.get('depth', 0) }}
- **Attempts**: {{ task.get('attempts', 0) }}
- **Started**: {{ task.get('started_at', 'unknown') }}
{% if task.get('evidence') %}
- **Evidence**: {{ task.evidence }}
{% endif %}
{% endfor %}
{% else %}
*No active tasks.*
{% endif %}

## 5. Recent Decision Log

| Time | Task | State Change | Reason |
|------|------|----------|------|
{% for d in last_decisions %}
| {{ d.time[:19] }} | {{ d.task_id }} | {{ d.old_status }} → {{ d.new_status }} | {{ d.reason[:60] }} |
{% endfor %}

## 6. Global Assessment

*(Updated by the Orchestrator after each wake cycle)*

---
*Auto-generated at {{ updated_at }}*
