"""CLI Extensions for Aura Agent (R2, R5, R6).

Provides additional CLI commands:
  aura wake        - External wake-up signal (R2)
  aura setup       - Interactive configuration (R5)
  aura changelog-overview - View changelog overview (R6)
  aura clean-workspaces   - Clean old task workspaces (R6)
  aura summaries   - List task completion reports (R3)
  aura cache-stats - Show file read cache statistics (R4)
"""

import argparse
import os
import sys
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def get_global_config_path() -> str:
    """Return the default user-level Aura config path."""
    return os.path.join(os.path.expanduser("~"), ".aura", "config.env")


def cmd_wake(args, config_ns):
    """R2: Signal the main loop to wake up immediately."""
    from orchestrator.config import WAKEUP_FILE
    from orchestrator.process_mgr import signal_wakeup
    result = signal_wakeup("CLI wake command")
    print(result)
    if "ERROR" not in result:
        print(f"Wakeup signal written to: {WAKEUP_FILE}")
        print("The orchestrator will wake on its next poll cycle.")


def cmd_setup(args, config_ns):
    """R5: Interactive first-time configuration wizard."""
    print("=" * 60)
    print("  Aura Agent - First Time Setup")
    print("=" * 60)
    print()
    print("This wizard will help you configure your Aura Agent.")
    print("Press Enter to accept defaults, or type a new value.")
    print()

    env_path = os.path.abspath(os.path.expanduser(args.output or get_global_config_path()))
    if os.path.exists(env_path) and not args.force:
        print(f"[WARN] {env_path} already exists.")
        print("Use --force to overwrite or --output for different path.")
        return

    config = {}

    default_key = os.environ.get("ANTHROPIC_API_KEY", "")
    masked = default_key[:8] + "***" if len(default_key) > 8 else "(not set)"
    val = input(f"API Key [{masked}]: ").strip()
    config["ANTHROPIC_API_KEY"] = val or default_key

    default_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    val = input(f"API Base URL [{default_url}]: ").strip()
    config["ANTHROPIC_BASE_URL"] = val or default_url

    default_model = os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]")
    val = input(f"Model [{default_model}]: ").strip()
    config["ANTHROPIC_MODEL"] = val or default_model

    default_tokens = os.environ.get("AURA_MAX_TOKENS", "4096")
    val = input(f"Max Tokens [{default_tokens}]: ").strip()
    config["AURA_MAX_TOKENS"] = val or default_tokens

    default_cycle = os.environ.get("AURA_CYCLE_INTERVAL", "300")
    val = input(f"Wake Interval (seconds) [{default_cycle}]: ").strip()
    config["AURA_CYCLE_INTERVAL"] = val or default_cycle

    default_backend = os.environ.get("AURA_LAYER2_BACKEND", "claude_code")
    val = input(f"Layer 2 Backend (claude_code/ds_code) [{default_backend}]: ").strip().lower()
    if val not in ("claude_code", "ds_code", ""):
        print("[WARN] Invalid backend. Using default: claude_code")
        val = ""
    config["AURA_LAYER2_BACKEND"] = val or default_backend

    default_dscode_model = os.environ.get("AURA_DSCODE_MODEL", "deepseek-v4-pro")
    val = input(f"ds-code Model [{default_dscode_model}]: ").strip()
    config["AURA_DSCODE_MODEL"] = val or default_dscode_model

    default_budget = os.environ.get("AURA_TASK_BUDGET", "30")
    val = input(f"Default Task Budget (min) [{default_budget}]: ").strip()
    config["AURA_TASK_BUDGET"] = val or default_budget

    default_turns = os.environ.get("AURA_MAX_TURNS", "50")
    val = input(f"Max Turns per Task [{default_turns}]: ").strip()
    config["AURA_MAX_TURNS"] = val or default_turns

    default_cache = os.environ.get("AURA_FILE_CACHE", "1")
    val = input(f"File Read Cache (1=on, 0=off) [{default_cache}]: ").strip()
    config["AURA_FILE_CACHE"] = val or default_cache

    lines = ["# Aura Agent Configuration",
             f"# Generated: {datetime.now().isoformat()}", ""]
    for key, value in config.items():
        if value:
            lines.append(f"{key}={value}")
    lines.append("")

    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print()
    print(f"[OK] Configuration written to: {env_path}")
    print("This is the default global config path. You can now run Aura from any project directory:")
    print("  aura start --task-file=tasks/task.md")
    print(f"To override it for one run, pass: aura --config={env_path} start --task-file=tasks/task.md")


def cmd_changelog(args, config_ns):
    """R6: Display changelog overview for all projects."""
    from orchestrator.config import PROJECTS_DIR, CHANGELOG_OVERVIEW_PATH

    if not os.path.isdir(PROJECTS_DIR):
        print("No projects found.")
        return

    lines = ["# Changelog Overview",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]

    for project_name in sorted(os.listdir(PROJECTS_DIR)):
        project_dir = os.path.join(PROJECTS_DIR, project_name)
        changelog_dir = os.path.join(project_dir, "changelog")
        if not os.path.isdir(changelog_dir):
            continue

        lines.append(f"## Project: {project_name}")
        lines.append("")

        for cl_file in sorted(os.listdir(changelog_dir)):
            if not cl_file.endswith(".json"):
                continue
            cl_path = os.path.join(changelog_dir, cl_file)
            try:
                with open(cl_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries = data.get("entries", [])
                processed = data.get("processed_items", {})
                lines.append(f"### {cl_file}")
                lines.append(f"- Entries: {len(entries)}")
                lines.append(f"- Processed items: {len(processed)}")
                if entries:
                    last = entries[-1]
                    lines.append(f"- Last: {last.get('processed_at', 'N/A')[:19]}")
                lines.append("")
            except Exception as e:
                lines.append(f"- Error: {e}")
                lines.append("")

    content = "\n".join(lines)
    print(content)
    os.makedirs(os.path.dirname(CHANGELOG_OVERVIEW_PATH), exist_ok=True)
    with open(CHANGELOG_OVERVIEW_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved to: {CHANGELOG_OVERVIEW_PATH}")


def cmd_cleanup(args, config_ns):
    """R6: Clean up old task workspace directories."""
    from orchestrator.config import get_workspace_dir, TASK_CLEANUP_AGE_DAYS

    tasks_dir = os.path.join(get_workspace_dir(), "tasks")
    if not os.path.isdir(tasks_dir):
        print("No task workspace found.")
        return

    age_days = args.age or TASK_CLEANUP_AGE_DAYS
    cutoff = datetime.now() - timedelta(days=age_days)
    cleaned = 0
    total_size = 0

    print(f"Cleaning task workspaces older than {age_days} days...")

    for task_id in sorted(os.listdir(tasks_dir)):
        task_dir = os.path.join(tasks_dir, task_id)
        if not os.path.isdir(task_dir):
            continue
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(task_dir))
        except OSError:
            continue
        if mtime > cutoff and not args.all:
            continue
        size = 0
        for dirpath, dirnames, filenames in os.walk(task_dir):
            for fn in filenames:
                try:
                    size += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
        if args.dry_run:
            print(f"  [DRY RUN] {task_id} ({(datetime.now()-mtime).days}d, {size:,}B)")
        else:
            try:
                shutil.rmtree(task_dir)
                print(f"  [OK] {task_id} ({size:,}B)")
                cleaned += 1
                total_size += size
            except OSError as e:
                print(f"  [ERROR] {task_id}: {e}")

    if args.dry_run:
        print(f"\nDry run. Use --no-dry-run to delete.")
    else:
        print(f"\nCleaned {cleaned} tasks, freed {total_size:,} bytes.")


def cmd_summaries(args, config_ns):
    """R3: List task completion summaries."""
    from orchestrator.task_reporter import list_summaries
    summaries = list_summaries()
    if not summaries:
        print("No task summaries found.")
        return
    print(f"{'Task':<8} {'Generated':<22} {'Size':>8}")
    print("-" * 42)
    for s in summaries:
        ts = s["modified"][:19]
        size_kb = s["size"] / 1024
        print(f"{s['task_id']:<8} {ts:<22} {size_kb:>7.1f}K")
    print(f"\n{len(summaries)} summaries total.")


def cmd_cache_stats(args, config_ns):
    """R4: Show file read cache statistics."""
    from orchestrator.file_cache import get_cache_stats
    stats = get_cache_stats()
    print("File Read Cache Statistics")
    print("-" * 30)
    for key, value in stats.items():
        print(f"  {key}: {value}")


def register_commands(subparsers):
    """Register all new CLI subcommands on an existing subparsers object."""

    p_wake = subparsers.add_parser("wake",
        help="Signal the orchestrator to wake up immediately (R2)")
    p_wake.set_defaults(func=cmd_wake)

    p_setup = subparsers.add_parser("setup",
        help="Interactive first-time configuration wizard (R5)")
    p_setup.add_argument("--output", "-o",
        help=f"Output config path (default: {get_global_config_path()})")
    p_setup.add_argument("--force", "-f", action="store_true",
        help="Overwrite existing config")
    p_setup.set_defaults(func=cmd_setup)

    p_cl = subparsers.add_parser("changelog-overview",
        help="View changelog overview for all projects (R6)")
    p_cl.set_defaults(func=cmd_changelog)

    p_clean = subparsers.add_parser("clean-workspaces",
        help="Clean old task workspace directories (R6)")
    p_clean.add_argument("--age", type=int, help="Age threshold in days")
    p_clean.add_argument("--all", action="store_true",
        help="Clean all tasks regardless of age")
    p_clean.add_argument("--dry-run", action="store_true", default=True,
        help="Preview only (default)")
    p_clean.add_argument("--no-dry-run", action="store_false", dest="dry_run",
        help="Actually delete files")
    p_clean.set_defaults(func=cmd_cleanup)

    p_sum = subparsers.add_parser("summaries",
        help="List task completion summary reports (R3)")
    p_sum.set_defaults(func=cmd_summaries)

    p_cache = subparsers.add_parser("cache-stats",
        help="Show file read cache statistics (R4)")
    p_cache.set_defaults(func=cmd_cache_stats)
