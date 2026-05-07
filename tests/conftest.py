"""Shared pytest fixtures for Aura Agent tests.

Provides:
- Environment setup with test API key
- Temporary project directories
- Common mock fixtures
"""

import os
import sys
import tempfile

import pytest

# Ensure orchestrator is importable
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_ROOT)


@pytest.fixture(autouse=True)
def setup_env():
    """Set up required environment variables for every test."""
    old_env = dict(os.environ)
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-1234")
    os.environ.setdefault("ANTHROPIC_BASE_URL", "https://api.test.local")
    os.environ.setdefault("ANTHROPIC_MODEL", "test-model")
    os.environ.setdefault("AURA_MAX_TOKENS", "1024")
    os.environ.setdefault("AURA_CYCLE_INTERVAL", "60")
    yield
    # Restore original env
    for key in list(os.environ.keys()):
        if key not in old_env:
            del os.environ[key]
    os.environ.update({k: v for k, v in old_env.items() if k not in os.environ})
