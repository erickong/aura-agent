"""Aura Agent configuration — reads ONLY from config.env file, never from system environment variables.

Design:
  - Config is parsed directly from the .env file at import time.
  - NO os.environ.get() or similar system env reads — system env vars cannot leak in.
  - main.py can override CONFIG_FILE_PATH or set DATA_DIR_OVERRIDE / PROJECT_ROOT_OVERRIDE
    before this module is imported (by setting module attributes via importlib).
  - If the config file is missing → FileNotFoundError with setup instructions.
  - If critical keys (AURA_API_KEY) are missing → RuntimeError with setup instructions.
"""

import os
import sys as _sys


# ── Overridable by main.py before import ──────────────────────────────
# NOTE: DATA_DIR_OVERRIDE and PROJECT_ROOT_OVERRIDE are NOT declared here
# with `= None` defaults. main.py sets them on the module object via
# importlib.util.module_from_spec BEFORE exec_module() runs. If we declared
# them here, the module-level assignment would overwrite main.py's values.
# Instead, _get_override() below checks the module's __dict__ at access time.

# CONFIG_FILE_PATH is NOT declared at module level because main.py sets it
# via importlib before exec_module(). If we declared it here with a default,
# the module-level assignment would overwrite main.py's override value.
# Instead, _resolve_config_path() below uses _get_override("CONFIG_FILE_PATH").
_DEFAULT_CONFIG_PATH: str = os.path.join(os.path.expanduser("~"), ".aura", "config.env")


def _get_override(name: str):
    """Return an override value set by main.py before module execution.
    
    Uses globals() to check the module's __dict__ for attributes that
    main.py set on the module object via importlib before exec_module().
    """
    return globals().get(name)


# ── Config file parser (file only, no system env) ─────────────────────
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
            # Strip surrounding quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            config[key] = value
    return config


def _resolve_config_path() -> str:
    """Return the config file path, checking if it exists.
    
    Uses CONFIG_FILE_PATH override from main.py (set before exec_module),
    falling back to ~/.aura/config.env.
    
    Raises FileNotFoundError with setup instructions if the file doesn't exist.
    """
    path = _get_override("CONFIG_FILE_PATH") or _DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] Config file not found: {path}\n"
            f"\n"
            f"Aura loads its configuration from a .env file. The default location is:\n"
            f"  {_DEFAULT_CONFIG_PATH}\n"
            f"\n"
            f"You need to create this file first. Run the setup wizard:\n"
            f"  aura setup\n"
            f"\n"
            f"Or create the file manually with the required keys:\n"
            f"  AURA_API_KEY=your-api-key-here\n"
            f"  AURA_API_BASE_URL=https://api.deepseek.com/anthropic\n"
            f"  AURA_API_MODEL=deepseek-v4-pro\n"
        )
    return path


# ── Load config from file (once, at import time) ──────────────────────
_config_path = _resolve_config_path()
_config_data = _parse_env_file(_config_path)


def _get(key: str, default: str | None = None) -> str | None:
    """Get a config value from the parsed file ONLY.
    
    Never falls back to os.environ. If the key is not in the file and no
    default is given, returns None (callers should handle missing required keys).
    """
    return _config_data.get(key, default)


def _get_int(key: str, default: str | None = None) -> int:
    val = _get(key, default)
    if val is None or str(val).strip() == "":
        return int(default) if default is not None else 0
    return int(val)


def _get_float(key: str, default: str | None = None) -> float:
    val = _get(key, default)
    if val is None or str(val).strip() == "":
        return float(default) if default is not None else 0.0
    return float(val)


def _get_bool(key: str, default: str | None = None) -> bool:
    """Get boolean from file, supporting 0/1/true/false/yes/no/on/off."""
    val = _get(key, default)
    if val is None:
        return bool(default) if default is not None else False
    s = str(val).strip().lower()
    if s == "":
        return bool(default) if default is not None else False
    return s not in {"0", "false", "no", "off"}


# ── Base paths (overridable via module-level variables) ───────────────
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PROJECT_ROOT: main.py can set PROJECT_ROOT_OVERRIDE before import.
# If not set, falls back to AURA_PROJECT_ROOT from file, then to CWD.
_project_root_val = _get_override("PROJECT_ROOT_OVERRIDE") or _get("AURA_PROJECT_ROOT") or os.getcwd()
PROJECT_ROOT = os.path.abspath(_project_root_val)

# DATA_DIR: main.py can set DATA_DIR_OVERRIDE before import.
# If not set, falls back to AURA_DATA_DIR from file, then to PROJECT_ROOT/.aura.
_data_dir_val = _get_override("DATA_DIR_OVERRIDE") or _get("AURA_DATA_DIR") or os.path.join(PROJECT_ROOT, ".aura")
DATA_DIR = os.path.abspath(os.path.expanduser(_data_dir_val))


# ── Critical API settings (required) ──────────────────────────────────
AURA_API_KEY = _get("AURA_API_KEY")
if not AURA_API_KEY:
    raise RuntimeError(
        f"[ERROR] AURA_API_KEY is not set in config file: {_config_path}\n"
        f"\n"
        f"Your API key must be specified in the config file. Run:\n"
        f"  aura setup\n"
        f"\n"
        f"Or add the following line to {_config_path}:\n"
        f"  AURA_API_KEY=your-api-key-here\n"
    )

AURA_API_BASE_URL = _get("AURA_API_BASE_URL", "https://api.deepseek.com/anthropic")
AURA_API_MODEL = _get("AURA_API_MODEL", "deepseek-v4-pro[1m]")
AURA_API_MAX_TOKENS = _get_int("AURA_API_MAX_TOKENS", "4096")
AURA_API_PROVIDER = _get("AURA_API_PROVIDER", "")

# ── Derived paths ────────────────────────────────────────────────────
MEMORY_DIR = os.path.join(DATA_DIR, "memory")
STATE_DIR = os.path.join(DATA_DIR, "state")
WORKSPACE_DIR = os.path.join(DATA_DIR, "workspace")
WAKEUP_FILE = os.path.join(DATA_DIR, "wakeup")
SKILLS_DIR = os.path.join(CODE_ROOT, "skills")
TASKS_DIR = os.path.join(PROJECT_ROOT, "tasks")
PROJECTS_DIR = os.path.join(DATA_DIR, "projects")
TASK_SUMMARY_DIR = os.path.join(DATA_DIR, "summaries")
FILE_CACHE_DIR = os.path.join(DATA_DIR, "cache")
CHANGELOG_OVERVIEW_PATH = os.path.join(DATA_DIR, "changelog_overview.md")


def get_active_project():
    active_file = os.path.join(STATE_DIR, ".active_project")
    if os.path.exists(active_file):
        with open(active_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def get_workspace_dir():
    return WORKSPACE_DIR


# ── Cycle / timing ────────────────────────────────────────────────────
CYCLE_INTERVAL_SECONDS = _get_int("AURA_CYCLE_INTERVAL", "300")
# Deep reflection interval — independent of wake cycles. A deep reflection
# replaces a normal wake cycle when the process has been running for this
# many minutes AND the last deep reflection was at least this long ago.
DEEP_REFLECTION_INTERVAL_MINUTES = _get_int("AURA_DEEP_REFLECTION_INTERVAL", "120")
LLM_DEAD_THROTTLE_SECONDS = _get_int("AURA_LLM_DEAD_THROTTLE", "300")

MAX_CONCURRENT_TASKS = _get_int("AURA_MAX_CONCURRENT_TASKS", "2")
DEFAULT_TASK_BUDGET_MINUTES = _get_int("AURA_TASK_BUDGET", "30")
DEFAULT_MAX_TURNS = _get_int("AURA_MAX_TURNS", "50")

# ── Worker resource guard ────────────────────────────────────────────
WORKER_RESOURCE_GUARD_ENABLED = _get_bool("AURA_WORKER_RESOURCE_GUARD", "1")
WORKER_RESOURCE_POLL_SECONDS = _get_int("AURA_WORKER_RESOURCE_POLL_SECONDS", "10")
WORKER_RESOURCE_AVG_WINDOW_SECONDS = _get_int("AURA_WORKER_RESOURCE_AVG_WINDOW_SECONDS", "180")
WORKER_RESOURCE_VIOLATION_STRIKES = _get_int("AURA_WORKER_RESOURCE_VIOLATION_STRIKES", "3")
WORKER_MAX_CPU_PERCENT = _get_float("AURA_WORKER_MAX_CPU_PERCENT", "80.0")
WORKER_MAX_SYSTEM_MEMORY_PERCENT = _get_float("AURA_WORKER_MAX_SYSTEM_MEMORY_PERCENT", "80.0")
WORKER_MAX_GPU_UTIL_PERCENT = _get_float("AURA_WORKER_MAX_GPU_UTIL_PERCENT", "80.0")
WORKER_MAX_GPU_MEMORY_PERCENT = _get_float("AURA_WORKER_MAX_GPU_MEMORY_PERCENT", "80.0")
WORKER_MAX_SYSTEM_MEMORY_GB = _get_float("AURA_WORKER_MAX_SYSTEM_MEMORY_GB", "0.0")
WORKER_MIN_SYSTEM_MEMORY_FREE_GB = _get_float("AURA_WORKER_MIN_SYSTEM_MEMORY_FREE_GB", "0.0")
WORKER_MAX_GPU_MEMORY_GB = _get_float("AURA_WORKER_MAX_GPU_MEMORY_GB", "0.0")
WORKER_MIN_GPU_MEMORY_FREE_GB = _get_float("AURA_WORKER_MIN_GPU_MEMORY_FREE_GB", "0.0")
WORKER_CUDA_VISIBLE_DEVICES = (_get("AURA_WORKER_CUDA_VISIBLE_DEVICES") or "").strip()

# ── Layer 2 backend ─────────────────────────────────────────────────
AURA_LAYER2_BACKEND = _get("AURA_LAYER2_BACKEND", "claude")
AURA_CLAUDE_BIN = (_get("AURA_CLAUDE_BIN") or "").strip()
AURA_DEEPSEEK_API_KEY = _get("AURA_DEEPSEEK_API_KEY") or ""
AURA_DSCODE_MODEL = _get("AURA_DSCODE_MODEL", "deepseek-v4-pro")
AURA_DSCODE_MAX_TURNS = _get_int("AURA_DSCODE_MAX_TURNS", str(DEFAULT_MAX_TURNS))
AURA_DSCODE_BASE_URL = _get("AURA_DSCODE_BASE_URL") or ""

# ── Memory / stuck detection ─────────────────────────────────────────
LONG_TERM_MEMORY_MAX_CHARS = 3000
SHORT_TERM_MEMORY_MAX_CHARS = 2000
STUCK_THRESHOLD_CYCLES = 12
API_RETRY_COUNT = 4
API_RETRY_BASE_DELAY = 5

# ── API timeout (was in agent.py, moved here for consistency) ────────
API_TIMEOUT_SECONDS = _get_int("AURA_API_TIMEOUT", "300")

# ── File cache ───────────────────────────────────────────────────────
FILE_CACHE_ENABLED = _get("AURA_FILE_CACHE", "1") != "0"
FILE_CACHE_TTL_SECONDS = _get_int("AURA_FILE_CACHE_TTL", "60")

# ── Task summaries ──────────────────────────────────────────────────
TASK_SUMMARY_ENABLED = _get("AURA_TASK_SUMMARY", "1") != "0"
TASK_CLEANUP_AGE_DAYS = _get_int("AURA_TASK_CLEANUP_AGE", "7")

# ── Token optimization ───────────────────────────────────────────────
# Enable explicit cache_control headers (Anthropic native only).
# DeepSeek auto-caches without explicit headers; setting this on DeepSeek
# may cause errors. Default off.
AURA_EXPLICIT_PROMPT_CACHE = _get("AURA_EXPLICIT_PROMPT_CACHE", "0") == "1"

# Skip L1 LLM calls when workers are healthy and no decision is needed.
# Set to "0" to disable the skip gate (always call L1 every cycle).
AURA_SKIP_HEALTHY_CYCLES = _get("AURA_SKIP_HEALTHY_CYCLES", "1") != "0"

# Maximum consecutive skipped cycles before forcing an L1 call.
# Higher values reduce token cost when workers are progressing well.
AURA_MAX_SKIPPED_CYCLES = _get_int("AURA_MAX_SKIPPED_CYCLES", "8")

# ── Token pricing per 1M tokens (USD) ────────────────────────────────
TOKEN_PRICE_CACHE_HIT = _get_float("AURA_TOKEN_PRICE_CACHE_HIT", "0.145")
TOKEN_PRICE_CACHE_MISS = _get_float("AURA_TOKEN_PRICE_CACHE_MISS", "1.74")
TOKEN_PRICE_OUTPUT = _get_float("AURA_TOKEN_PRICE_OUTPUT", "1.74")

# ── Tool call budget hints (prompt guidance only, not hard limits) ──
TOOL_CALL_BUDGET_NORMAL = _get_int("AURA_TOOL_CALL_BUDGET_NORMAL", "12")
TOOL_CALL_BUDGET_DIAGNOSTIC = _get_int("AURA_TOOL_CALL_BUDGET_DIAGNOSTIC", "40")
TOOL_CALL_BUDGET_PLANNING = _get_int("AURA_TOOL_CALL_BUDGET_PLANNING", "40")
