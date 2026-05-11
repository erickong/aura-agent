import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float = 0.0) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.environ.get("AURA_PROJECT_ROOT", os.getcwd()))
_data_dir = os.environ.get("AURA_DATA_DIR", os.path.join(PROJECT_ROOT, ".aura"))
DATA_DIR = os.path.abspath(os.path.expanduser(_data_dir))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]")
ANTHROPIC_MAX_TOKENS = int(os.environ.get("AURA_MAX_TOKENS", "4096"))

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


CYCLE_INTERVAL_SECONDS = int(os.environ.get("AURA_CYCLE_INTERVAL", "300"))
# Deep reflection interval — independent of wake cycles. A deep reflection
# replaces a normal wake cycle when the process has been running for this
# many minutes AND the last deep reflection was at least this long ago.
DEEP_REFLECTION_INTERVAL_MINUTES = int(os.environ.get("AURA_DEEP_REFLECTION_INTERVAL", "120"))
LLM_DEAD_THROTTLE_SECONDS = int(os.environ.get("AURA_LLM_DEAD_THROTTLE", "300"))

MAX_CONCURRENT_TASKS = int(os.environ.get("AURA_MAX_CONCURRENT_TASKS", "2"))
DEFAULT_TASK_BUDGET_MINUTES = int(os.environ.get("AURA_TASK_BUDGET", "30"))
DEFAULT_MAX_TURNS = int(os.environ.get("AURA_MAX_TURNS", "50"))

WORKER_RESOURCE_GUARD_ENABLED = _env_bool("AURA_WORKER_RESOURCE_GUARD", True)
WORKER_RESOURCE_POLL_SECONDS = int(os.environ.get("AURA_WORKER_RESOURCE_POLL_SECONDS", "10"))
WORKER_RESOURCE_AVG_WINDOW_SECONDS = int(os.environ.get("AURA_WORKER_RESOURCE_AVG_WINDOW_SECONDS", "180"))
WORKER_RESOURCE_VIOLATION_STRIKES = int(os.environ.get("AURA_WORKER_RESOURCE_VIOLATION_STRIKES", "3"))
WORKER_MAX_CPU_PERCENT = _env_float("AURA_WORKER_MAX_CPU_PERCENT", 80.0)
WORKER_MAX_SYSTEM_MEMORY_PERCENT = _env_float("AURA_WORKER_MAX_SYSTEM_MEMORY_PERCENT", 80.0)
WORKER_MAX_GPU_UTIL_PERCENT = _env_float("AURA_WORKER_MAX_GPU_UTIL_PERCENT", 80.0)
WORKER_MAX_GPU_MEMORY_PERCENT = _env_float("AURA_WORKER_MAX_GPU_MEMORY_PERCENT", 80.0)
WORKER_MAX_SYSTEM_MEMORY_GB = _env_float("AURA_WORKER_MAX_SYSTEM_MEMORY_GB", 0.0)
WORKER_MIN_SYSTEM_MEMORY_FREE_GB = _env_float("AURA_WORKER_MIN_SYSTEM_MEMORY_FREE_GB", 0.0)
WORKER_MAX_GPU_MEMORY_GB = _env_float("AURA_WORKER_MAX_GPU_MEMORY_GB", 0.0)
WORKER_MIN_GPU_MEMORY_FREE_GB = _env_float("AURA_WORKER_MIN_GPU_MEMORY_FREE_GB", 0.0)
WORKER_CUDA_VISIBLE_DEVICES = os.environ.get("AURA_WORKER_CUDA_VISIBLE_DEVICES", "").strip()

AURA_LAYER2_BACKEND = os.environ.get("AURA_LAYER2_BACKEND", "claude_code")

AURA_CLAUDE_BIN = os.environ.get("AURA_CLAUDE_BIN", "").strip()

AURA_DEEPSEEK_API_KEY = os.environ.get("AURA_DEEPSEEK_API_KEY", "")
AURA_DSCODE_MODEL = os.environ.get("AURA_DSCODE_MODEL", "deepseek-v4-pro")
AURA_DSCODE_MAX_TURNS = int(os.environ.get("AURA_DSCODE_MAX_TURNS", str(DEFAULT_MAX_TURNS)))
AURA_DSCODE_BASE_URL = os.environ.get("AURA_DSCODE_BASE_URL", "")

LONG_TERM_MEMORY_MAX_CHARS = 3000
SHORT_TERM_MEMORY_MAX_CHARS = 2000

STUCK_THRESHOLD_CYCLES = 12

API_RETRY_COUNT = 4
API_RETRY_BASE_DELAY = 5

FILE_CACHE_ENABLED = os.environ.get("AURA_FILE_CACHE", "1") != "0"
FILE_CACHE_TTL_SECONDS = int(os.environ.get("AURA_FILE_CACHE_TTL", "60"))

TASK_SUMMARY_ENABLED = os.environ.get("AURA_TASK_SUMMARY", "1") != "0"

TASK_CLEANUP_AGE_DAYS = int(os.environ.get("AURA_TASK_CLEANUP_AGE", "7"))

# ── Token optimization config ───────────────────────────────────────
# Enable explicit cache_control headers (Anthropic native only).
# DeepSeek auto-caches without explicit headers; setting this on DeepSeek
# may cause errors. Default off.
AURA_EXPLICIT_PROMPT_CACHE = os.environ.get("AURA_EXPLICIT_PROMPT_CACHE", "0") == "1"

# Skip L1 LLM calls when workers are healthy and no decision is needed.
# Set to "0" to disable the skip gate (always call L1 every cycle).
AURA_SKIP_HEALTHY_CYCLES = os.environ.get("AURA_SKIP_HEALTHY_CYCLES", "1") != "0"

# Maximum consecutive skipped cycles before forcing an L1 call.
# Higher values reduce token cost when workers are progressing well.
AURA_MAX_SKIPPED_CYCLES = int(os.environ.get("AURA_MAX_SKIPPED_CYCLES", "8"))

# ── Token pricing per 1M tokens (USD) ────────────────────────────────
TOKEN_PRICE_CACHE_HIT = float(os.environ.get("AURA_TOKEN_PRICE_CACHE_HIT", "0.145"))
TOKEN_PRICE_CACHE_MISS = float(os.environ.get("AURA_TOKEN_PRICE_CACHE_MISS", "1.74"))
TOKEN_PRICE_OUTPUT = float(os.environ.get("AURA_TOKEN_PRICE_OUTPUT", "1.74"))

# ── Tool call budget hints (prompt guidance only, not hard limits) ──
TOOL_CALL_BUDGET_NORMAL = int(os.environ.get("AURA_TOOL_CALL_BUDGET_NORMAL", "12"))
TOOL_CALL_BUDGET_DIAGNOSTIC = int(os.environ.get("AURA_TOOL_CALL_BUDGET_DIAGNOSTIC", "40"))
TOOL_CALL_BUDGET_PLANNING = int(os.environ.get("AURA_TOOL_CALL_BUDGET_PLANNING", "40"))
