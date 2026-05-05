# Aura Agent

Aura Agent is a two-layer autonomous task orchestrator. You give it a long-running goal in a Markdown file; it keeps waking up, inspecting evidence, launching bounded worker tasks, recording decisions, and adjusting direction until the goal is genuinely handled.

It is designed for people who want an agent that can keep working across many cycles without losing the task ledger, forgetting why a decision was made, or trusting a worker's self-report without checking artifacts.

## Why Use Aura

Most coding agents are excellent inside a single session. Aura is aimed at a different problem: multi-hour or multi-day goal execution.

Aura gives you:

- A persistent task tree in `.aura/state/state.json`
- Evidence-based status changes with decision history
- Up to two Layer 2 workers running in parallel
- Worker monitoring by process health, output size, output tail hash, artifacts, and error logs
- Periodic reflection and review
- Task-file change detection while the orchestrator is already running
- External wake-up with `aura wake`
- Local project checkpoints under `.aura/`
- Support for both `claude_code` and `ds_code` worker backends

## How It Works

Aura uses two layers:

```text
User goal.md
    |
    v
Layer 1: Orchestrator
  - reads the task tree, memory, progress report, and worker signals
  - decides whether to spawn, kill, continue, decompose, replan, or no-op
  - records decisions and evidence
    |
    v
Layer 2: Workers
  - execute specific task.md files in isolated workspace directories
  - produce result.md, code, data, reports, logs, or other verifiable artifacts
```

The orchestrator does not consider a task complete just because a worker says so. It checks actual files and process evidence before changing state.

## Compared With Hermes And OpenClaw / 小龙虾

Aura borrows useful ideas from Hermes and OpenClaw, but has a different center of gravity.

| Dimension | Hermes | OpenClaw / 小龙虾 | Aura Agent |
|---|---|---|---|
| Main idea | Self-evolving loop | Many parallel "lobster" workers and task ledger | Evidence-driven goal completion with conservative orchestration |
| Concurrency | Higher parallelism | Very high parallelism | Max 2 workers by default, quality first |
| Decision style | Reflection-heavy | Broad exploration | Progress signals + evidence + explicit decision log |
| Cost control | Depends on setup | Can become expensive with many workers | Small worker count, cached file reads, compact state snapshots |
| Failure handling | Self-improvement oriented | Broad retry / branch exploration | State backups, safe mode, worker health checks, external wake-up |
| Best fit | Research into self-evolving agents | Wide search and many candidate branches | Long-running engineering tasks where traceability matters |

Aura's advantage is not "more workers". It is that every cycle has a durable state, every status change needs evidence, and the orchestrator is allowed to wait, inspect, kill, replan, or continue based on actual output.

Use Aura when you want:

- A project-level autonomous agent rather than a single chat session
- Long-running work with periodic wake/sleep cycles
- A searchable audit trail of what changed and why
- Controlled parallelism instead of a swarm that is hard to supervise
- Local, inspectable state files that can be backed up or reviewed manually

## Current Capabilities

- Task-file reconciliation on startup
- Per-wake task-file change detection
- Persistent project state and automatic `.bak` backup for `state.json`
- Layer 2 worker spawning and killing
- Worker output tracking through `output.jsonl` or `output.txt`
- Progress report rendering to `.aura/state/progress.md`
- Long-term and short-term memory files
- Reflection review engine
- Task completion summaries
- File read and directory listing caches
- External wake-up signal
- Interactive setup wizard

## Installation

Requirements:

- Python 3.10+
- An Anthropic-compatible API endpoint for Layer 1, defaulting to DeepSeek's Anthropic-compatible endpoint
- One Layer 2 backend:
  - `claude_code`: Claude Code CLI available as `claude`
  - `ds_code`: `ds-code` CLI available as `ds-code`

Install locally:

```bash
git clone <your-repo-url>
cd aura-agent
python -m pip install -e .
```

Create the global config:

```bash
aura setup
```

By default, `aura setup` writes to `~/.aura/config.env` on Linux/macOS or `%USERPROFILE%\.aura\config.env` on Windows. Aura loads this global config automatically from any project directory.

You can also create a one-off config at a custom path:

```bash
aura setup --output C:\path\to\project.env
```

An explicit `--config` file overrides the global config for that run:

```bash
aura --config C:\path\to\project.env start --task-file tasks/example_mission.md
```

## Quick Start

After installation, `aura` can be run from any project directory, similar to Claude Code. The directory where you run the command becomes Aura's project root. Runtime state is written under that directory's `.aura/` unless you pass `--data-dir`.

Create a task file:

```markdown
# Build a small FastAPI service

Create a minimal FastAPI app with one `/health` endpoint, tests, and a README section explaining how to run it.
```

Run Aura:

```bash
aura start --task-file tasks/example_mission.md
```

Useful commands:

```bash
aura status
aura progress
aura history
aura projects
aura wake
aura summaries
aura cache-stats
```

On Windows you can also use:

```bat
start.bat start --task-file tasks/example_mission.md
```

## Configuration

Global config is loaded from `~/.aura/config.env` (`%USERPROFILE%\.aura\config.env` on Windows). The default runtime data directory is `.aura/` under the project where you start Aura. Override runtime data with `--data-dir`, and override config with `--config`.

Important environment variables:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | API key for the Layer 1 orchestrator model |
| `ANTHROPIC_BASE_URL` | `https://api.deepseek.com/anthropic` | Anthropic-compatible API endpoint |
| `ANTHROPIC_MODEL` | `deepseek-v4-pro[1m]` | Layer 1 model |
| `AURA_LAYER2_BACKEND` | `claude_code` | `claude_code` or `ds_code` |
| `AURA_CYCLE_INTERVAL` | `300` | Wake interval in seconds |
| `AURA_DEEP_REVIEW_INTERVAL` | `12` | Review interval in cycles |
| `AURA_MAX_TOKENS` | `4096` | Max output tokens per Layer 1 API call |
| `AURA_TASK_BUDGET` | `30` | Default Layer 2 worker budget in minutes |
| `AURA_MAX_TURNS` | `50` | Max turns for Claude Code workers |
| `AURA_DSCODE_MODEL` | `deepseek-v4-pro` | ds-code model when using `AURA_LAYER2_BACKEND=ds_code` |
| `AURA_FILE_CACHE` | `1` | Enable mtime-based file cache |

## CLI Reference

Core commands:

| Command | Purpose |
|---|---|
| `aura start --task-file PATH` | Start the orchestrator loop |
| `aura status` | Show active project and task tree |
| `aura progress` | Render the progress report |
| `aura projects` | List saved projects |
| `aura history` | Show recent decision history |
| `aura changelog` | Show task-file changelog for the active project |
| `aura cleanup` | Clean orphan project records |

Extension commands:

| Command | Purpose |
|---|---|
| `aura wake` | Wake a sleeping orchestrator early |
| `aura setup` | Create the global config interactively |
| `aura summaries` | List task completion summary reports |
| `aura cache-stats` | Show in-memory file cache stats |
| `aura changelog-overview` | Generate changelog overview across projects |
| `aura clean-workspaces --dry-run` | Preview task workspace cleanup |
| `aura clean-workspaces --no-dry-run --age 7` | Delete old task workspaces |

## Runtime Data

Aura writes runtime data to `.aura/` in the current project directory by default:

```text
.aura/
  memory/
    MEMORY.md
    session.md
    reviews/
  state/
    state.json
    state.json.bak
    progress.md
    .active_project
  workspace/
    tasks/<task-id>/
      task.md
      output.jsonl or output.txt
      error.log
      result.md
  projects/
    <project-name>/
```

Do not commit `.aura/` to Git. It is local execution state.

## Repository Layout

```text
orchestrator/
  main.py              CLI and main loop
  agent.py             Layer 1 decision engine
  process_mgr.py       Worker process management
  phase2.py            Progress evaluation and replan signals
  review.py            Reflection review engine
  state.py             Task tree and decision log
  memory.py            Long-term and session memory
  progress.py          Progress report renderer
  tools.py             Tool definitions used by Layer 1
  changelog.py         Task-file change tracking
  task_reporter.py     Completion summary generation
  file_cache.py        mtime-based file/directory cache
  templates/
    progress.template.md
tasks/
  example_mission.md
```

## Safety Notes

- Global config lives outside the project by default at `~/.aura/config.env` or `%USERPROFILE%\.aura\config.env`.
- `.env` is ignored by Git. Keep real keys out of tracked files if you choose to create a project-local override manually.
- `.env.example` must contain placeholders only.
- `.aura/`, `memory/`, `state/`, `workspace/`, `projects/`, and build outputs are ignored.
- Worker output can contain private code, prompts, logs, or API errors. Treat `.aura/workspace/` as sensitive.

## Limitations

- Aura is an orchestrator, not a guarantee of success. Bad goals still need human judgment.
- The default design favors two high-quality workers over large swarms.
- Layer 2 backend quality depends on your configured CLI and model.
- The project is young; review generated changes before merging them into important repositories.
