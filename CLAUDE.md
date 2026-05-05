# CLAUDE.md

This repository contains Aura Agent, a two-layer autonomous task orchestrator.

## Project Rules

- Treat `.aura/`, `memory/`, `state/`, `workspace/`, `projects/`, and `archive/` as runtime data. They should not be committed.
- Never commit real API keys. `.env.example` must contain placeholders only.
- Preserve the evidence-first design: task status changes should cite real files, logs, process state, or test results.
- Keep Layer 1 orchestration logic conservative. Do not skip wake-up, task-file change detection, worker completion reconciliation, or state backups unless the replacement behavior is verified.
- Prefer small, focused changes. The agent is meant to be durable across long runs.

## Important Files

- `orchestrator/main.py`: CLI and main wake/sleep loop
- `orchestrator/agent.py`: Layer 1 decision prompt, context building, API/tool loop
- `orchestrator/process_mgr.py`: Layer 2 worker spawning, killing, and monitoring
- `orchestrator/state.py`: persistent task tree and decision log
- `orchestrator/phase2.py`: progress and stuck-task signals
- `orchestrator/review.py`: periodic reflection review
- `orchestrator/tools.py`: tools exposed to the Layer 1 model

## Verification

Before publishing or opening a PR, run:

```bash
python -m compileall orchestrator
python -m orchestrator.main --help
```

