import os

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
DEEP_REVIEW_INTERVAL_CYCLES = int(os.environ.get("AURA_DEEP_REVIEW_INTERVAL", "12"))
LLM_DEAD_THROTTLE_SECONDS = int(os.environ.get("AURA_LLM_DEAD_THROTTLE", "300"))

MAX_CONCURRENT_TASKS = 2
DEFAULT_TASK_BUDGET_MINUTES = int(os.environ.get("AURA_TASK_BUDGET", "30"))
DEFAULT_MAX_TURNS = int(os.environ.get("AURA_MAX_TURNS", "50"))

AURA_LAYER2_BACKEND = os.environ.get("AURA_LAYER2_BACKEND", "claude_code")

AURA_DEEPSEEK_API_KEY = os.environ.get("AURA_DEEPSEEK_API_KEY", "")
AURA_DSCODE_MODEL = os.environ.get("AURA_DSCODE_MODEL", "deepseek-v4-pro")
AURA_DSCODE_MAX_TURNS = int(os.environ.get("AURA_DSCODE_MAX_TURNS", str(DEFAULT_MAX_TURNS)))
AURA_DSCODE_BASE_URL = os.environ.get("AURA_DSCODE_BASE_URL", "")

LONG_TERM_MEMORY_MAX_CHARS = 3000
SHORT_TERM_MEMORY_MAX_CHARS = 2000

STUCK_THRESHOLD_CYCLES = 12
REVIEW_NUDGE_INTERVAL = 10

API_RETRY_COUNT = 4
API_RETRY_BASE_DELAY = 5

FILE_CACHE_ENABLED = os.environ.get("AURA_FILE_CACHE", "1") != "0"
FILE_CACHE_TTL_SECONDS = int(os.environ.get("AURA_FILE_CACHE_TTL", "60"))

TASK_SUMMARY_ENABLED = os.environ.get("AURA_TASK_SUMMARY", "1") != "0"

TASK_CLEANUP_AGE_DAYS = int(os.environ.get("AURA_TASK_CLEANUP_AGE", "7"))
