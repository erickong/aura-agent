# Aura Agent Architecture

Aura Agent is a persistent, evidence-driven task orchestrator. It is built for long-running goals where a single chat session is not enough.

After installation, Aura can be launched from any project directory. The current working directory becomes the project root, and runtime data is stored in that directory's `.aura/` unless `--data-dir` is provided.

Configuration is user-level by default. `aura setup` writes `~/.aura/config.env` (`%USERPROFILE%\.aura\config.env` on Windows), and Aura loads it automatically. A per-run `--config` file can override those defaults.

## Goals

- Convert a Markdown goal into a durable task tree.
- Run bounded Layer 2 workers on concrete subtasks.
- Track real artifacts instead of trusting worker self-reports.
- Keep a decision log with reasons and evidence.
- Wake periodically, inspect progress, adjust strategy, and sleep again.
- Recover from corrupted state through backups and safe-mode fallbacks.

## System Model

```text
Task file
  -> startup reconciliation
  -> state/task tree
  -> Layer 1 orchestrator cycle
  -> Layer 2 worker process
  -> artifacts/logs/result.md
  -> progress evaluation
  -> state update + memory + review
```

## Layer 1: Orchestrator

The orchestrator is implemented in `orchestrator/agent.py` and driven by `orchestrator/main.py`.

Each cycle:

1. Increment cycle counter.
2. Load state and active tasks.
3. Run Phase 2 progress evaluation.
4. Build a compact context snapshot:
   - task tree
   - active/running tasks
   - recent decisions
   - session memory preview
   - progress report preview
   - active/recent workspace summaries
   - Phase 2 progress signals
5. Call the configured Anthropic-compatible model.
6. Execute tool calls.
7. Render progress and update session memory.
8. Return to the main loop for checkpointing and sleep.

The prompt intentionally discourages repeated broad file reads. The model should use the context snapshot first and read only specific missing evidence.

## Layer 2: Workers

Layer 2 workers are spawned through `orchestrator/process_mgr.py`.

Supported backends:

- `claude`: runs `claude -p @task.md --output-format stream-json`
- `ds_code`: runs `ds-code run --workspace <task_dir>`
- `opencode`: runs `opencode run --file task.md --format json`

Each worker receives a generated `task.md` and works inside `.aura/workspace/tasks/<task-id>/`.

Expected outputs:

- `output.jsonl` or `output.txt`
- `error.log`
- `result.md`
- any task-specific artifacts

The default maximum concurrency is 2 workers.

Worker resource limits are enforced in `process_mgr.py`, not by trusting task
text. The resource guard can:

- block spawning when configured system/GPU free-memory reserves are not met
- inject `CUDA_VISIBLE_DEVICES` and framework-friendly allocation environment
  variables into worker processes
- apply CPU affinity when a total-system CPU ceiling is configured and the
  platform supports affinity
- poll only the Aura worker process tree for CPU, RSS, NVIDIA GPU memory, and
  best-effort per-process GPU utilization
- use rolling averages over the recent window so short peaks do not trigger
  unnecessary kills
- kill workers after repeated sustained resource violations and wake the
  orchestrator
- return the original task to `pending` once for a smaller retry; if it exceeds
  limits again, mark it `blocked` and create/spawn a resource-fix subtask

Key controls are `AURA_MAX_CONCURRENT_TASKS`,
`AURA_WORKER_MAX_CPU_PERCENT`, `AURA_WORKER_MAX_SYSTEM_MEMORY_PERCENT`,
`AURA_WORKER_MAX_GPU_UTIL_PERCENT`, `AURA_WORKER_MAX_GPU_MEMORY_PERCENT`,
`AURA_WORKER_RESOURCE_AVG_WINDOW_SECONDS`,
`AURA_WORKER_RESOURCE_VIOLATION_STRIKES`, and
`AURA_WORKER_CUDA_VISIBLE_DEVICES`. Optional absolute GB reserve/ceiling
controls are still available for machines that need them.

## State

State lives under `.aura/state/`.

- `state.json`: mission, task tree, active task IDs, decision log
- `state.json.bak`: backup written with every save
- `progress.md`: human-readable rendered progress report
- `.active_project`: current project marker

`state.py` uses mtime-based caching and restores from `state.json.bak` if the primary JSON is corrupted.

## Memory

Runtime memory lives under `.aura/memory/`.

- `MEMORY.md`: long-term lessons and important facts
- `session.md`: short-term cycle context
- `reviews/`: periodic review outputs

These files are runtime data and should not be committed.

## Progress Evaluation

`phase2.py` evaluates active workers through multiple signals:

- output file size
- output size delta
- tail hash / content change
- repeated tool names in recent output
- process CPU usage
- error log size
- task artifacts

A task is considered stuck only when multiple signals agree: no output growth, no content change, no new artifacts, and low CPU.

## Tool Surface

Layer 1 can call tools defined in `orchestrator/tools.py`:

- `read_file`
- `write_file`
- `list_directory`
- `web_fetch`
- `spawn_task`
- `kill_task`
- `list_running_tasks`
- `update_task_tree`
- `decompose_task`
- `write_memory`
- `no_op`

Status-changing tools require reasons and evidence.

## Task File Changes

`changelog.py` tracks task-file hashes and per-wake diffs. When a task file changes while Aura is running, the diff is injected into the next orchestrator context so the task tree can be updated without restarting.

## Review

`review.py` runs periodic reflection. It inspects state, progress, memory, and recent decisions, then writes review reports under `.aura/memory/reviews/`.

## CLI

Core commands:

- `aura start --task-file PATH`
- `aura status`
- `aura progress`
- `aura projects`
- `aura history`
- `aura changelog`
- `aura cleanup`

Extension commands:

- `aura wake`
- `aura setup`
- `aura summaries`
- `aura cache-stats`
- `aura changelog-overview`
- `aura clean-workspaces`

## Comparison

Aura is inspired by Hermes-style reflection loops and OpenClaw / 小龙虾-style task ledgers, but optimizes for a narrower target:

- fewer workers
- clearer state
- stricter evidence requirements
- lower default cost
- simpler local filesystem persistence
- safer long-running execution

It is best used when auditability and steady project completion matter more than maximum parallel exploration.
