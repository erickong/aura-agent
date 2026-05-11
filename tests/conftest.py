"""Shared pytest fixtures for Aura Agent tests.

Provides:
- Config setup with test data in a temp directory
- Temporary project directories
- Common mock fixtures
"""

import os
import sys
import tempfile
import importlib.util
from pathlib import Path

import pytest

# Ensure orchestrator is importable
CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))


def _create_test_config(temp_dir: Path) -> Path:
    """Create a config.env file with test settings in the given directory.

    Returns the path to the created config file.
    """
    config_path = temp_dir / "config.env"
    config_path.write_text(
        "# Test config for Aura Agent\n"
        f"AURA_DATA_DIR={temp_dir / '.aura'}\n"
        f"AURA_PROJECT_ROOT={temp_dir}\n"
        "AURA_API_KEY=test-key-1234\n"
        "AURA_API_BASE_URL=https://api.test.local\n"
        "AURA_API_MODEL=test-model\n"
        "AURA_API_MAX_TOKENS=1024\n"
        "AURA_CYCLE_INTERVAL=60\n"
        "AURA_LAYER2_BACKEND=claude\n"
        "AURA_DEEP_REFLECTION_INTERVAL=9999\n"
        "AURA_API_TIMEOUT=300\n",
        encoding="utf-8",
    )
    return config_path


@pytest.fixture(autouse=True, scope="session")
def _init_test_config():
    """Session-scoped: create a temp config and load orchestrator.config from it.

    This runs ONCE before all tests. It creates a temporary config file
    with test-safe values and uses importlib to load orchestrator.config
    from that file (not from ~/.aura/config.env or system env vars).
    """
    test_dir = Path(tempfile.mkdtemp(prefix="aura_test_config_"))
    config_file = _create_test_config(test_dir)

    # Ensure .aura subdirs exist
    (test_dir / ".aura" / "state").mkdir(parents=True, exist_ok=True)
    (test_dir / ".aura" / "memory").mkdir(parents=True, exist_ok=True)
    (test_dir / ".aura" / "workspace").mkdir(parents=True, exist_ok=True)
    (test_dir / ".aura" / "projects").mkdir(parents=True, exist_ok=True)
    (test_dir / ".aura" / "summaries").mkdir(parents=True, exist_ok=True)
    (test_dir / ".aura" / "cache").mkdir(parents=True, exist_ok=True)

    # Store for use by other fixtures
    _init_test_config.test_dir = test_dir
    _init_test_config.config_file = config_file

    # Load config module with overrides via importlib
    spec = importlib.util.find_spec("orchestrator.config")
    mod = importlib.util.module_from_spec(spec)
    mod.CONFIG_FILE_PATH = str(config_file)
    mod.PROJECT_ROOT_OVERRIDE = str(test_dir)
    sys.modules["orchestrator.config"] = mod
    spec.loader.exec_module(mod)

    yield

    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def temp_project(_init_test_config):
    """Return a temp project directory (subdir of the test config's dir).

    Tests can create task files, workspaces, etc. in this directory.
    The shared .aura/ lives under _init_test_config.test_dir.
    """
    base = Path(_init_test_config.test_dir)
    project_dir = base / "test_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "tasks").mkdir(exist_ok=True)
    # Also create tasks dir in the base dir (for tests that use AURA_PROJECT_ROOT directly)
    (base / "tasks").mkdir(exist_ok=True)
    return project_dir
