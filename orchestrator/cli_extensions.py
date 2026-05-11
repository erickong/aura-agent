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
import time
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


_KNOWN_BASE_URLS = [
    ("DeepSeek (Anthropic-compatible)", "https://api.deepseek.com/anthropic",
     "deepseek-v4-pro[1m]", "recommended"),
    ("Anthropic (Official)", "https://api.anthropic.com",
     "claude-sonnet-4-6-20250514", ""),
    ("Custom BaseUrl", "", "", ""),
]


def _detect_provider_from_url(url: str) -> str:
    """Detect API provider type from the base URL. Called once at setup time."""
    if not url:
        return "unknown"
    url_lower = url.lower()
    if "deepseek" in url_lower:
        return "deepseek"
    if "anthropic" in url_lower:
        return "anthropic"
    if "openai" in url_lower:
        return "openai"
    return "anthropic"  # default: assume Anthropic-compatible


def _test_api_connectivity(base_url: str, api_key: str, model: str) -> tuple[bool, str]:
    """Test whether the configured API is reachable with a minimal call."""
    if not api_key or not base_url:
        return False, "No API key or base URL configured"

    masked = api_key[:8] + "****" + api_key[-4:] if len(api_key) > 12 else "****"
    try:
        import anthropic
        client = anthropic.Anthropic(base_url=base_url, api_key=api_key, auth_token=api_key)
        t0 = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=4,
            system="Reply with just the word OK.",
            messages=[{"role": "user", "content": "Say OK"}],
            timeout=30,
        )
        elapsed = time.time() - t0
        text = "".join(b.text for b in response.content if b.type == "text")
        if "OK" in text:
            return True, f"OK ({elapsed:.1f}s, key={masked})"
        return True, f"Connected ({elapsed:.1f}s, key={masked})"
    except Exception as e:
        return False, f"key={masked} — {e}"


def _parse_env_file(path: str) -> dict[str, str]:
    """Parse a .env file and return a dict of key→value (no os.environ side effects)."""
    config: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            config[key] = value
    return config


def cmd_setup(args, config_ns):
    """R5: Interactive first-time configuration wizard."""
    print("=" * 60)
    print("  Aura Agent - First Time Setup")
    print("=" * 60)
    print()
    print("This wizard will help you configure your Aura Agent.")
    print("Press Enter to accept defaults.")
    print()

    env_path = os.path.abspath(os.path.expanduser(args.output or get_global_config_path()))
    if os.path.exists(env_path) and not args.force:
        print(f"[WARN] {env_path} already exists.")
        print("Use --force to overwrite or --output for different path.")
        return

    # Read existing config file (if any) for defaults — never from os.environ
    existing_config: dict[str, str] = {}
    if os.path.exists(env_path):
        try:
            existing_config = _parse_env_file(env_path)
        except (OSError, Exception):
            pass

    config: dict[str, str] = {}

    # ── API Key (one key for all providers) ──────────────────────────
    default_key = existing_config.get("AURA_API_KEY", "")
    masked = default_key[:8] + "***" if len(default_key) > 8 else "(not set)"
    print("─ API Key ──────────────────────────────────────────────────")
    val = input(f"  API Key [{masked}]: ").strip()
    config["AURA_API_KEY"] = val or default_key
    print()

    # ── Base URL selection ───────────────────────────────────────────
    print("─ API Provider ─────────────────────────────────────────────")
    print("  Select the API provider and base URL:")
    default_url = existing_config.get("AURA_API_BASE_URL", "https://api.deepseek.com/anthropic")
    default_idx = 0
    for i, (label, url, model, tag) in enumerate(_KNOWN_BASE_URLS):
        tag_str = f" ({tag})" if tag else ""
        marker = " →" if url == default_url else "  "
        print(f"  {i + 1}. {label}{tag_str}")
        if url:
            print(f"     {marker} {url}")
        else:
            print(f"     {marker} (enter custom URL)")
        if url == default_url:
            default_idx = i
    print()
    val = input(f"  Choose provider [1-{len(_KNOWN_BASE_URLS)}] (default: {default_idx + 1}): ").strip()

    try:
        choice = int(val) - 1 if val else default_idx
        if 0 <= choice < len(_KNOWN_BASE_URLS):
            _, selected_url, default_model, _ = _KNOWN_BASE_URLS[choice]
        else:
            choice = default_idx
            _, selected_url, default_model, _ = _KNOWN_BASE_URLS[default_idx]
    except (ValueError, IndexError):
        choice = default_idx
        _, selected_url, default_model, _ = _KNOWN_BASE_URLS[default_idx]

    if choice == len(_KNOWN_BASE_URLS) - 1 and not selected_url:
        val = input("  Custom base URL: ").strip()
        config["AURA_API_BASE_URL"] = val or ""
    else:
        val = input(f"  Base URL [{selected_url}]: ").strip()
        config["AURA_API_BASE_URL"] = val or selected_url

    # ── Detect and persist provider type ──────────────────────────────
    final_url = config["AURA_API_BASE_URL"]
    provider = _detect_provider_from_url(final_url)
    config["AURA_API_PROVIDER"] = provider
    print(f"  → Detected provider: {provider}")
    print()

    # ── Model ────────────────────────────────────────────────────────
    current_model = existing_config.get("AURA_API_MODEL", default_model or "deepseek-v4-pro[1m]")
    val = input(f"  Model [{current_model}]: ").strip()
    config["AURA_API_MODEL"] = val or current_model
    print()

    default_tokens = existing_config.get("AURA_API_MAX_TOKENS", "4096")
    val = input(f"Max Tokens [{default_tokens}]: ").strip()
    config["AURA_API_MAX_TOKENS"] = val or default_tokens

    default_cycle = existing_config.get("AURA_CYCLE_INTERVAL", "300")
    val = input(f"Wake Interval (seconds) [{default_cycle}]: ").strip()
    config["AURA_CYCLE_INTERVAL"] = val or default_cycle

    default_backend = existing_config.get("AURA_LAYER2_BACKEND", "claude")
    val = input(f"Layer 2 Backend (claude/ds_code) [{default_backend}]: ").strip().lower()
    if val not in ("claude", "ds_code", ""):
        print("[WARN] Invalid backend. Using default: claude")
        val = ""
    config["AURA_LAYER2_BACKEND"] = val or default_backend

    default_dscode_model = existing_config.get("AURA_DSCODE_MODEL", "deepseek-v4-pro")
    val = input(f"ds-code Model [{default_dscode_model}]: ").strip()
    config["AURA_DSCODE_MODEL"] = val or default_dscode_model

    default_budget = existing_config.get("AURA_TASK_BUDGET", "30")
    val = input(f"Default Task Budget (min) [{default_budget}]: ").strip()
    config["AURA_TASK_BUDGET"] = val or default_budget

    default_turns = existing_config.get("AURA_MAX_TURNS", "50")
    val = input(f"Max Turns per Task [{default_turns}]: ").strip()
    config["AURA_MAX_TURNS"] = val or default_turns

    default_cache = existing_config.get("AURA_FILE_CACHE", "1")
    val = input(f"File Read Cache (1=on, 0=off) [{default_cache}]: ").strip()
    config["AURA_FILE_CACHE"] = val or default_cache

    print()
    print("--- Token Pricing (USD per 1M tokens) ---")
    default_hit = existing_config.get("AURA_TOKEN_PRICE_CACHE_HIT", "0.145")
    val = input(f"Cache Hit Price [$/{default_hit}M]: ").strip()
    config["AURA_TOKEN_PRICE_CACHE_HIT"] = val or default_hit

    default_miss = existing_config.get("AURA_TOKEN_PRICE_CACHE_MISS", "1.74")
    val = input(f"Cache Miss (Input) Price [$/{default_miss}M]: ").strip()
    config["AURA_TOKEN_PRICE_CACHE_MISS"] = val or default_miss

    default_out = existing_config.get("AURA_TOKEN_PRICE_OUTPUT", "1.74")
    val = input(f"Output Price [$/{default_out}M]: ").strip()
    config["AURA_TOKEN_PRICE_OUTPUT"] = val or default_out

    print()
    print("--- Tool Call Budget (prompt guidance, not hard limits) ---")
    default_normal = existing_config.get("AURA_TOOL_CALL_BUDGET_NORMAL", "12")
    val = input(f"Normal cycle max tool calls [{default_normal}]: ").strip()
    config["AURA_TOOL_CALL_BUDGET_NORMAL"] = val or default_normal

    default_diag = existing_config.get("AURA_TOOL_CALL_BUDGET_DIAGNOSTIC", "40")
    val = input(f"Diagnostic cycle max tool calls [{default_diag}]: ").strip()
    config["AURA_TOOL_CALL_BUDGET_DIAGNOSTIC"] = val or default_diag

    default_plan = existing_config.get("AURA_TOOL_CALL_BUDGET_PLANNING", "40")
    val = input(f"Planning cycle max tool calls [{default_plan}]: ").strip()
    config["AURA_TOOL_CALL_BUDGET_PLANNING"] = val or default_plan

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

    # ── Connectivity test ────────────────────────────────────────────
    api_key = config.get("AURA_API_KEY", "")
    api_url = config.get("AURA_API_BASE_URL", "")
    model = config.get("AURA_API_MODEL", "deepseek-v4-pro[1m]")
    if api_key and api_url:
        print()
        print("  Testing API connectivity...")
        ok, detail = _test_api_connectivity(api_url, api_key, model)
        if ok:
            print(f"  [OK] API connected — {detail}")
        else:
            print(f"  [WARN] API test failed — {detail}")
            print("  Config saved, but please check your key, URL, and model.")

    print()
    print("You can now run Aura from any project directory:")
    print("  aura start --task-file=tasks/task.md")
    print(f"To override config for one run: aura --config={env_path} start --task-file=tasks/task.md")


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


def cmd_token_stats(args, config_ns):
    """Show token usage and cache hit statistics."""
    from orchestrator.token_tracker import get_stats, format_stats, get_skip_count
    stats = get_stats()
    print(format_stats(stats))
    skips = get_skip_count()
    if skips > 0:
        print(f"\n  Skipped L1 cycles: {skips}")
    if stats["total_calls"] > 0:
        print(f"\n  Avg tokens/call: {stats['total_input_tokens'] // stats['total_calls']:,} in "
              f"+ {stats['total_output_tokens'] // stats['total_calls']:,} out")
        print(f"  Total estimated cost: ${stats['estimated_cost']:.4f}")
        print(f"  Estimated savings:    ${stats['estimated_cost_saved']:.4f}")


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

    p_tokens = subparsers.add_parser("token-stats",
        help="Show token usage and cache hit statistics")
    p_tokens.set_defaults(func=cmd_token_stats)
