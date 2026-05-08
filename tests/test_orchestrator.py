"""Comprehensive test suite for the Aura Agent orchestrator.

Covers:
  1. Fresh start from a new requirement file (init, parse, mission extraction)
  2. Changing requirements (changelog hash detection, batch prefix advancement)
  3. Same-name requirement file at different paths (slug uniqueness, project dedup)
  4. Runtime requirement changes (per-wake diff detection, interruption recovery)
  5. Different requirement files (multi-project switching, task index)
  6. Dynamic planning for newly added tasks (decompose, batch advancement, planning flags)
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Setup: add orchestrator to path ───────────────────────────────────────
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_ROOT)

# Set required env vars before importing orchestrator modules
os.environ["ANTHROPIC_API_KEY"] = "test-key-1234"
os.environ["AURA_PROJECT_ROOT"] = os.path.join(tempfile.gettempdir(), "aura_test_project")

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory with .aura data dir."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    aura_dir = project_dir / ".aura"
    aura_dir.mkdir()
    (aura_dir / "memory").mkdir()
    (aura_dir / "state").mkdir()
    (aura_dir / "workspace").mkdir()
    (aura_dir / "projects").mkdir()
    (aura_dir / "summaries").mkdir()
    (aura_dir / "cache").mkdir()
    yield project_dir


@pytest.fixture
def task_file_simple(temp_project):
    """Create a simple task file."""
    tf = temp_project / "tasks" / "test_mission.md"
    tf.parent.mkdir(exist_ok=True)
    tf.write_text("""# Build a web app

## Phase 1: Setup
- [ ] Create project scaffold
- [ ] Install dependencies

## Phase 2: Features
- [ ] Implement login page
- [ ] Implement dashboard

## Phase 3: Polish
1. Add error handling
2. Write tests
""", encoding="utf-8")
    return tf


@pytest.fixture
def task_file_multiple_sections(temp_project):
    """Create a task file with multiple sections and mixed item types."""
    tf = temp_project / "tasks" / "complex_mission.md"
    tf.parent.mkdir(exist_ok=True)
    tf.write_text("""# Complex Multi-Phase Project

<!-- This is a comment that should be ignored -->

Preamble text that describes the overall project goals and requirements.
This should be aggregated into a single preamble block.

## Research Phase
- [x] Literature review (already done)
- [ ] Benchmark existing solutions

## Implementation
- [ ] Build core engine
- [ ] Build API layer

Detailed paragraph about the implementation approach.
This should be merged into the heading section above.

## Deployment
1. Set up CI/CD pipeline
2. Configure monitoring
3. Deploy to staging
""", encoding="utf-8")
    return tf


@pytest.fixture
def task_file_minimal(temp_project):
    """Create a minimal task file with only a heading."""
    tf = temp_project / "tasks" / "minimal_mission.md"
    tf.parent.mkdir(exist_ok=True)
    tf.write_text("# Just one heading\n", encoding="utf-8")
    return tf


@pytest.fixture
def patched_config(temp_project):
    """Patch orchestrator config paths to use a temp directory."""
    aura_dir = str(temp_project / ".aura")
    with patch.dict(os.environ, {
        "AURA_DATA_DIR": aura_dir,
        "AURA_PROJECT_ROOT": str(temp_project),
    }):
        yield aura_dir


@pytest.fixture
def reset_state_cache():
    """Reset the state module's internal caches between tests."""
    from orchestrator import state as state_mgr
    state_mgr._state_cache = None
    state_mgr._state_mtime = 0.0
    yield
    state_mgr._state_cache = None
    state_mgr._state_mtime = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: 从头启动新需求 (Fresh Start)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFreshStart:
    """Test initializing a brand-new task from a requirement file."""

    def test_init_state_creates_proper_structure(self, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr
        from orchestrator.config import STATE_DIR

        task_file = os.path.join(os.environ["AURA_PROJECT_ROOT"], "tasks", "hello.md")
        state = state_mgr.init_state("Build a web app", task_file)

        assert state["mission"] == "Build a web app"
        assert state["task_file"] == task_file
        assert state["total_cycles"] == 0
        assert len(state["tasks"]) == 1
        assert state["tasks"][0]["id"] == "root"
        assert state["tasks"][0]["status"] == "pending"
        assert state["tasks"][0]["depth"] == 0
        assert state["task_batch"]["current_prefix"] == "A"
        assert state["task_batch"]["current_index"] == 0
        assert os.path.exists(os.path.join(STATE_DIR, "state.json"))
        assert os.path.exists(os.path.join(STATE_DIR, "state.json.bak"))

    def test_init_state_records_file_hash(self, task_file_simple, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr

        state = state_mgr.init_state("Test", str(task_file_simple))
        stored_hash = state["task_batch"]["last_task_file_hash"]
        assert stored_hash  # Hash should be non-empty
        # Hash should match actual file content
        expected_hash = state_mgr._task_file_hash(str(task_file_simple))
        assert stored_hash == expected_hash

    def test_parse_requirement_blocks_simple(self, task_file_simple):
        from orchestrator.state import parse_requirement_blocks

        blocks = parse_requirement_blocks(str(task_file_simple))

        # The section-aware aggregator:
        # - Merges ## Phase 1 into the # heading's body (it appears before checkboxes)
        # - Uses ## Phase 2 and ## Phase 3 as prefixes on their actionable items
        checkboxes = [b for b in blocks if b["type"] == "checkbox"]
        numbered = [b for b in blocks if b["type"] == "numbered"]

        assert len(checkboxes) == 4
        assert len(numbered) == 2

        # Phase 1 text merged into main heading
        headings = [b for b in blocks if b["type"] == "heading"]
        assert len(headings) == 1
        assert "Build a web app" in headings[0]["text"]
        assert "Phase 1: Setup" in headings[0]["text"]

        # Phase 2/3 prefixed onto their actionable items
        assert any("Phase 2: Features" in c["text"] for c in checkboxes)
        assert any("Phase 3: Polish" in n["text"] for n in numbered)

    def test_parse_requirement_blocks_section_merging(self, task_file_multiple_sections):
        from orchestrator.state import parse_requirement_blocks

        blocks = parse_requirement_blocks(str(task_file_multiple_sections))

        # Preamble text gets merged into the main heading's body by the
        # section-aware aggregator (headings swallow following text blocks).
        headings = [b for b in blocks if b["type"] == "heading"]
        assert len(headings) == 1
        assert "Complex Multi-Phase Project" in headings[0]["text"]
        assert "Preamble text" in headings[0]["text"]

        # "Detailed paragraph..." after Implementation checkboxes becomes
        # a standalone paragraph block (not merged into any heading).
        paragraphs = [b for b in blocks if b["type"] == "paragraph"]
        assert len(paragraphs) == 1
        assert "Detailed paragraph about the implementation approach" in paragraphs[0]["text"]

        # Comments should be ignored
        all_text = " ".join(b["text"] for b in blocks)
        assert "should be ignored" not in all_text

        # Already-done checkbox
        checkboxes = [b for b in blocks if b["type"] == "checkbox"]
        done = [b for b in checkboxes if b["status"] == "completed"]
        pending = [b for b in checkboxes if b["status"] == "pending"]
        assert len(done) == 1
        assert done[0]["text"] == "Literature review (already done)"
        assert len(pending) == 3

        # Numbered items
        numbered = [b for b in blocks if b["type"] == "numbered"]
        assert len(numbered) == 3

    def test_parse_requirement_blocks_headings_without_body(self, task_file_minimal):
        from orchestrator.state import parse_requirement_blocks

        blocks = parse_requirement_blocks(str(task_file_minimal))
        headings = [b for b in blocks if b["type"] == "heading"]
        assert len(headings) == 1
        assert headings[0]["text"] == "Just one heading"

    def test_parse_requirement_blocks_empty_file(self, temp_project):
        from orchestrator.state import parse_requirement_blocks

        tf = temp_project / "empty.md"
        tf.write_text("", encoding="utf-8")
        blocks = parse_requirement_blocks(str(tf))
        assert blocks == []

    def test_extract_mission_heading(self, task_file_simple):
        from orchestrator.main import _extract_mission

        mission = _extract_mission(str(task_file_simple))
        assert mission == "Build a web app"

    def test_extract_mission_fallback(self, temp_project):
        from orchestrator.main import _extract_mission

        tf = temp_project / "no_heading.md"
        tf.write_text("This is the first line.\nSecond line here.\n", encoding="utf-8")
        mission = _extract_mission(str(tf))
        assert "This is the first line." in mission

    def test_extract_mission_skips_comments(self, temp_project):
        from orchestrator.main import _extract_mission

        tf = temp_project / "comment_first.md"
        tf.write_text("<!-- comment -->\n# Actual Title\n", encoding="utf-8")
        mission = _extract_mission(str(tf))
        assert mission == "Actual Title"

    def test_load_and_save_state_roundtrip(self, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr

        task_file = os.path.join(os.environ["AURA_PROJECT_ROOT"], "tasks", "t.md")
        state_mgr.init_state("Roundtrip test", task_file)

        loaded = state_mgr.load_state()
        assert loaded["mission"] == "Roundtrip test"
        assert loaded["task_file"] == task_file

    def test_state_backup_created_on_save(self, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr
        from orchestrator.config import STATE_DIR

        task_file = os.path.join(os.environ["AURA_PROJECT_ROOT"], "tasks", "t.md")
        state = state_mgr.init_state("Backup test", task_file)

        bak_path = os.path.join(STATE_DIR, "state.json.bak")
        assert os.path.exists(bak_path)
        with open(bak_path, "r", encoding="utf-8") as f:
            bak_data = json.load(f)
        assert bak_data["mission"] == "Backup test"

    def test_state_recovery_from_corrupted_primary(self, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr
        from orchestrator.config import STATE_DIR

        task_file = os.path.join(os.environ["AURA_PROJECT_ROOT"], "tasks", "t.md")
        state_mgr.init_state("Recovery test", task_file)

        # Corrupt primary state.json
        state_path = os.path.join(STATE_DIR, "state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            f.write("{this is not valid json")

        # Invalidate cache so we force a disk read
        state_mgr._state_cache = None
        state_mgr._state_mtime = 0.0

        loaded = state_mgr.load_state()
        # Should recover from .bak
        assert loaded["mission"] == "Recovery test"


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: 更改需求 (Changing Requirements)
# ═══════════════════════════════════════════════════════════════════════════════


class TestChangingRequirements:
    """Test detecting and handling requirement changes."""

    def test_compute_file_hash_changes_with_content(self, temp_project):
        from orchestrator.changelog import compute_file_hash

        tf = temp_project / "test.md"
        tf.write_text("# Version 1\n- [ ] Task A\n", encoding="utf-8")
        hash1 = compute_file_hash(str(tf))

        tf.write_text("# Version 2\n- [ ] Task A\n- [ ] Task B\n", encoding="utf-8")
        hash2 = compute_file_hash(str(tf))

        assert hash1 != hash2
        assert len(hash1) == 64  # SHA-256

    def test_compute_file_hash_consistency(self, temp_project):
        from orchestrator.changelog import compute_file_hash

        tf = temp_project / "test.md"
        tf.write_text("Same content\n", encoding="utf-8")
        hash1 = compute_file_hash(str(tf))
        hash2 = compute_file_hash(str(tf))
        assert hash1 == hash2

    def test_get_file_change_info_new_file(self, task_file_simple, temp_project):
        from orchestrator.changelog import get_file_change_info

        projects_dir = str(temp_project / ".aura" / "projects")
        info = get_file_change_info(str(task_file_simple), projects_dir, "test_mission")

        assert info["is_new"] is True
        assert info["is_changed"] is False
        assert info["current_hash"]

    def test_get_file_change_info_detects_modification(self, task_file_simple, temp_project):
        from orchestrator.changelog import (
            get_file_change_info,
            mark_file_processed,
        )

        projects_dir = str(temp_project / ".aura" / "projects")

        # First: mark as processed
        mark_file_processed(str(task_file_simple), projects_dir, "test_mission",
                           summary="Initial processing")

        # Then: modify the file
        task_file_simple.write_text("# Updated Content\n- [ ] New Task\n", encoding="utf-8")

        # Check: should detect change
        info = get_file_change_info(str(task_file_simple), projects_dir, "test_mission")
        assert info["is_new"] is False
        assert info["is_changed"] is True
        assert info["previous_hash"]  # Should have a previous hash

    def test_get_file_change_info_no_change(self, task_file_simple, temp_project):
        from orchestrator.changelog import (
            get_file_change_info,
            mark_file_processed,
        )

        projects_dir = str(temp_project / ".aura" / "projects")

        mark_file_processed(str(task_file_simple), projects_dir, "test_mission")
        info = get_file_change_info(str(task_file_simple), projects_dir, "test_mission")

        assert info["is_new"] is False
        assert info["is_changed"] is False

    def test_mark_file_processed_creates_changelog(self, task_file_simple, temp_project):
        from orchestrator.changelog import (
            mark_file_processed,
            load_changelog,
            get_changelog_path,
        )

        projects_dir = str(temp_project / ".aura" / "projects")
        result = mark_file_processed(str(task_file_simple), projects_dir, "test_mission",
                                     summary="Test run")

        assert result["entry_index"] == 0
        assert result["file_hash"]

        changelog_path = get_changelog_path(projects_dir, "test_mission",
                                            str(task_file_simple))
        changelog = load_changelog(changelog_path)
        assert len(changelog["entries"]) == 1
        assert changelog["entries"][0]["summary"] == "Test run"

    def test_batch_prefix_advances_on_file_change(self, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr

        task_file = os.path.join(os.environ["AURA_PROJECT_ROOT"], "tasks", "t.md")
        # Create the file
        os.makedirs(os.path.dirname(task_file), exist_ok=True)
        Path(task_file).write_text("# Initial\n", encoding="utf-8")

        state = state_mgr.init_state("Initial", task_file)
        assert state["task_batch"]["current_prefix"] == "A"
        assert state["task_batch"]["current_index"] == 0

        # Simulate a file change by advancing batch
        state = state_mgr.load_state()
        Path(task_file).write_text("# Changed\n", encoding="utf-8")
        advanced = state_mgr._advance_task_batch_for_change(state, task_file, True)
        assert advanced is True

        state_mgr.save_state(state)
        state = state_mgr.load_state()
        assert state["task_batch"]["current_prefix"] == "B"
        assert state["task_batch"]["current_index"] == 1

    def test_batch_prefix_advances_multiple_times(self, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr

        task_file = os.path.join(os.environ["AURA_PROJECT_ROOT"], "tasks", "t.md")
        os.makedirs(os.path.dirname(task_file), exist_ok=True)

        Path(task_file).write_text("# V1\n", encoding="utf-8")
        state = state_mgr.init_state("V1", task_file)
        assert state["task_batch"]["current_prefix"] == "A"

        # Change 1: A → B
        Path(task_file).write_text("# V2\n", encoding="utf-8")
        state = state_mgr.load_state()
        state_mgr._advance_task_batch_for_change(state, task_file, True)
        state_mgr.save_state(state)
        assert state["task_batch"]["current_prefix"] == "B"

        # Change 2: B → C
        Path(task_file).write_text("# V3\n", encoding="utf-8")
        state = state_mgr.load_state()
        state_mgr._advance_task_batch_for_change(state, task_file, True)
        state_mgr.save_state(state)
        assert state["task_batch"]["current_prefix"] == "C"

    def test_batch_prefix_after_z_goes_to_aa(self, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr

        task_file = os.path.join(os.environ["AURA_PROJECT_ROOT"], "tasks", "t.md")
        os.makedirs(os.path.dirname(task_file), exist_ok=True)

        Path(task_file).write_text("# V1\n", encoding="utf-8")
        state = state_mgr.init_state("V1", task_file)

        # Manually advance to Z (index 25 = Z)
        state["task_batch"]["current_index"] = 25
        state["task_batch"]["current_prefix"] = "Z"
        state_mgr.save_state(state)

        # Next change should go to AA
        Path(task_file).write_text("# V26\n", encoding="utf-8")
        state = state_mgr.load_state()
        state_mgr._advance_task_batch_for_change(state, task_file, True)
        state_mgr.save_state(state)
        assert state["task_batch"]["current_prefix"] == "AA"
        assert state["task_batch"]["current_index"] == 26

    def test_no_batch_advance_without_change(self, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr

        task_file = os.path.join(os.environ["AURA_PROJECT_ROOT"], "tasks", "t.md")
        os.makedirs(os.path.dirname(task_file), exist_ok=True)

        Path(task_file).write_text("# V1\n", encoding="utf-8")
        state = state_mgr.init_state("V1", task_file)

        # No file change
        state = state_mgr.load_state()
        advanced = state_mgr._advance_task_batch_for_change(state, task_file, False)
        assert advanced is False
        assert state["task_batch"]["current_prefix"] == "A"

    def test_reconcile_task_file_with_change_marks_planning_needed(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr

        # Init with some existing tasks
        state_mgr.init_state("Test mission", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Task 1", "acceptance_criteria": "Done"},
            {"id": "A2", "description": "Task 2", "acceptance_criteria": "Done"},
        ])

        # Actually modify the task file so hash differs
        task_file_simple.write_text(
            task_file_simple.read_text(encoding="utf-8") + "\n## New Phase\n- [ ] Extra task\n",
            encoding="utf-8",
        )

        # Simulate a file change reconciliation
        stats = state_mgr.reconcile_task_file(
            str(task_file_simple),
            mission="Updated mission",
            task_file_changed=True,
        )
        assert stats["planning_needed"] is True
        assert stats["batch_advanced"] is True
        assert stats["kept"] >= 0

    def test_reconcile_recovers_interrupted_tasks(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Task", "acceptance_criteria": "Done"},
        ])

        # Manually set A1 as in_progress (simulating a crash)
        state_mgr.update_task("A1", "in_progress", "Started", "manual")
        state = state_mgr.load_state()
        assert state_mgr.find_task("A1", state["tasks"])["status"] == "in_progress"

        # Reconcile without A1 in running_task_ids → should be marked as interrupted
        stats = state_mgr.reconcile_task_file(
            str(task_file_simple),
            mission="Test",
            running_task_ids=set(),  # A1 is NOT running
        )

        assert stats["interrupted"] == 1
        state = state_mgr.load_state()
        task = state_mgr.find_task("A1", state["tasks"])
        assert task["status"] == "pending"
        assert "Interrupted" in task.get("evidence", "")


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3: 同名需求文件不同位置 (Same-name file at different paths)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSameNameDifferentPaths:
    """Test that same-name task files at different locations are handled correctly."""

    def test_task_data_slug_differs_by_path(self):
        from orchestrator.main import _task_data_slug

        slug1 = _task_data_slug("/home/user/projectA/tasks/task.md")
        slug2 = _task_data_slug("/home/user/projectB/tasks/task.md")

        # Same stem ("task"), different hash suffix
        assert slug1.startswith("task-")
        assert slug2.startswith("task-")
        assert slug1 != slug2  # Different hashes

    def test_task_data_slug_same_for_same_path(self):
        from orchestrator.main import _task_data_slug

        slug1 = _task_data_slug("c:/projects/myapp/tasks/mission.md")
        slug2 = _task_data_slug("c:/projects/myapp/tasks/mission.md")

        assert slug1 == slug2

    def test_task_data_slug_normalizes_case_on_windows(self):
        from orchestrator.main import _task_data_slug

        slug1 = _task_data_slug("C:/Projects/MyApp/tasks/Mission.md")
        slug2 = _task_data_slug("c:/projects/myapp/tasks/mission.md")

        # os.path.normcase normalizes the path for hashing, making the
        # hash suffix identical. The stem part preserves basename case.
        # On Windows both should match because normcase lowercases the
        # full resolved path used in the hash.
        if sys.platform == "win32":
            # Same hash suffix, but stem may differ by case
            parts1 = slug1.rsplit("-", 1)
            parts2 = slug2.rsplit("-", 1)
            assert parts1[1] == parts2[1]  # hash suffix matches
        else:
            assert slug1 != slug2  # Different on case-sensitive systems

    def test_project_name_uses_basename_only(self):
        from orchestrator.changelog import get_project_name_for_task
        from orchestrator.main import _project_name_from_cwd

        # Different paths, same basename → same project name
        name1 = get_project_name_for_task("tasks/self_upgrade.md")
        name2 = get_project_name_for_task("tasks\\self_upgrade.md")
        name3 = get_project_name_for_task("/any/path/self_upgrade.md")

        assert name1 == "self_upgrade"
        assert name2 == "self_upgrade"
        assert name3 == "self_upgrade"
        assert name1 == name2 == name3

    def test_different_basename_gives_different_project(self):
        from orchestrator.changelog import get_project_name_for_task

        name1 = get_project_name_for_task("tasks/app_a.md")
        name2 = get_project_name_for_task("tasks/app_b.md")

        assert name1 != name2
        assert name1 == "app_a"
        assert name2 == "app_b"

    def test_resolve_task_file(self, temp_project):
        # _resolve_task_file uses CFG_PROJECT_ROOT from config (set via env var).
        # Patch env, reload config and main to pick up the new path.
        with patch.dict(os.environ, {"AURA_PROJECT_ROOT": str(temp_project)}):
            import importlib
            import orchestrator.config as cfg
            import orchestrator.main as main_mod

            cfg = importlib.reload(cfg)
            main_mod = importlib.reload(main_mod)

            # Relative path
            resolved = main_mod._resolve_task_file("tasks/my_task.md")
            assert resolved == os.path.normpath(str(temp_project / "tasks" / "my_task.md"))

            # Absolute path
            abs_path = str(temp_project / "tasks" / "absolute.md")
            resolved = main_mod._resolve_task_file(abs_path)
            assert resolved == os.path.normpath(abs_path)

    def test_record_task_data_dir_creates_metadata(self, temp_project, task_file_simple):
        from orchestrator.main import _record_task_data_dir, _default_aura_base_dir, _task_data_dir_for

        base_dir = _default_aura_base_dir()
        data_dir = _task_data_dir_for(str(task_file_simple), base_dir)

        _record_task_data_dir(str(task_file_simple), data_dir)

        # Check metadata file
        metadata_path = os.path.join(data_dir, "task_file.json")
        assert os.path.exists(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert "task_file" in meta
        assert "data_dir" in meta
        assert "created_at" in meta

        # Check task index
        index_path = os.path.join(base_dir, "task_index.json")
        assert os.path.exists(index_path)
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        assert len(index.get("tasks", {})) >= 1

    def test_task_data_dir_unique_per_task_file(self, temp_project):
        from orchestrator.main import _task_data_dir_for, _default_aura_base_dir

        task1 = str(temp_project / "tasks" / "project_a.md")
        task2 = str(temp_project / "tasks" / "project_b.md")

        # Create the files
        os.makedirs(os.path.dirname(task1), exist_ok=True)
        Path(task1).touch()
        Path(task2).touch()

        base_dir = _default_aura_base_dir()
        dir1 = _task_data_dir_for(task1, base_dir)
        dir2 = _task_data_dir_for(task2, base_dir)

        assert dir1 != dir2

    def test_same_filename_different_dir_different_slug(self):
        from orchestrator.main import _task_data_slug

        slug1 = _task_data_slug("/a/tasks/task.md")
        slug2 = _task_data_slug("/b/tasks/task.md")

        assert slug1 != slug2
        assert slug1.startswith("task-")
        assert slug2.startswith("task-")


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 4: 运行时更改需求 (Runtime Requirement Changes)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRuntimeChanges:
    """Test per-wake change detection and handling during active operation."""

    def test_check_task_file_on_wake_no_change(self, task_file_simple, temp_project):
        from orchestrator.changelog import (
            check_task_file_on_wake,
            save_task_file_snapshot,
        )

        projects_dir = str(temp_project / ".aura" / "projects")

        # Save initial snapshot
        save_task_file_snapshot(str(task_file_simple), projects_dir, "test_mission")

        # Check immediately — should be no change
        result = check_task_file_on_wake(str(task_file_simple), projects_dir,
                                         "test_mission")
        assert result["changed"] is False
        assert result["mtime_changed"] is False

    def test_check_task_file_on_wake_detects_content_change(
        self, task_file_simple, temp_project
    ):
        import time
        from orchestrator.changelog import (
            check_task_file_on_wake,
            save_task_file_snapshot,
        )

        projects_dir = str(temp_project / ".aura" / "projects")
        save_task_file_snapshot(str(task_file_simple), projects_dir, "test_mission")

        # Ensure mtime differs from snapshot
        time.sleep(1.5)

        # Modify the file
        original = task_file_simple.read_text(encoding="utf-8")
        task_file_simple.write_text(original + "\n- [ ] New requirement\n", encoding="utf-8")

        result = check_task_file_on_wake(str(task_file_simple), projects_dir,
                                         "test_mission")

        assert result["changed"] is True
        assert result["content_changed"] is True
        assert len(result["diff_lines"]) > 0
        # Should detect the new checkbox as a requirement
        assert len(result["added_requirement_lines"]) >= 1

    def test_check_task_file_on_wake_detects_info_lines(
        self, task_file_simple, temp_project
    ):
        import time
        from orchestrator.changelog import (
            check_task_file_on_wake,
            save_task_file_snapshot,
        )

        projects_dir = str(temp_project / ".aura" / "projects")
        save_task_file_snapshot(str(task_file_simple), projects_dir, "test_mission")

        time.sleep(1.5)

        # Add informational text (not a requirement line)
        task_file_simple.write_text(
            task_file_simple.read_text(encoding="utf-8") +
            "\nHere is some additional context from the user.\n",
            encoding="utf-8"
        )

        result = check_task_file_on_wake(str(task_file_simple), projects_dir,
                                         "test_mission")
        assert result["changed"] is True
        assert len(result["added_info_lines"]) >= 1

    def test_save_snapshot_then_recheck_no_change(self, task_file_simple, temp_project):
        import time
        from orchestrator.changelog import (
            check_task_file_on_wake,
            save_task_file_snapshot,
        )

        projects_dir = str(temp_project / ".aura" / "projects")

        # Save initial
        save_task_file_snapshot(str(task_file_simple), projects_dir, "test_mission")
        time.sleep(1.5)

        # Modify, then save new snapshot
        task_file_simple.write_text("# Changed\n", encoding="utf-8")
        result1 = check_task_file_on_wake(str(task_file_simple), projects_dir,
                                          "test_mission")
        assert result1["changed"] is True

        save_task_file_snapshot(str(task_file_simple), projects_dir, "test_mission")

        # Now should be no change (snapshot matches current content exactly)
        result2 = check_task_file_on_wake(str(task_file_simple), projects_dir,
                                          "test_mission")
        assert result2["changed"] is False

    def test_check_task_file_on_wake_removed_requirements(
        self, task_file_simple, temp_project
    ):
        import time
        from orchestrator.changelog import (
            check_task_file_on_wake,
            save_task_file_snapshot,
        )

        projects_dir = str(temp_project / ".aura" / "projects")
        save_task_file_snapshot(str(task_file_simple), projects_dir, "test_mission")
        time.sleep(1.5)

        # Remove checkbox lines
        lines = task_file_simple.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = [l for l in lines if "- [ ]" not in l]
        task_file_simple.write_text("".join(new_lines), encoding="utf-8")

        result = check_task_file_on_wake(str(task_file_simple), projects_dir,
                                         "test_mission")
        assert result["changed"] is True
        assert len(result["removed_requirement_lines"]) >= 1

    def test_check_task_file_on_wake_missing_file(self, temp_project):
        from orchestrator.changelog import check_task_file_on_wake

        result = check_task_file_on_wake(
            str(temp_project / "nonexistent.md"),
            str(temp_project / ".aura" / "projects"),
            "test"
        )
        assert result["changed"] is False

    def test_change_summary_formatting(self, task_file_simple, temp_project):
        import time
        from orchestrator.changelog import (
            check_task_file_on_wake,
            save_task_file_snapshot,
        )

        projects_dir = str(temp_project / ".aura" / "projects")
        save_task_file_snapshot(str(task_file_simple), projects_dir, "test_mission")
        time.sleep(1.5)

        task_file_simple.write_text("# New heading\n- [ ] Task 1\nMore info here\n", encoding="utf-8")

        result = check_task_file_on_wake(str(task_file_simple), projects_dir,
                                         "test_mission")
        assert result["change_summary"]
        assert len(result["change_summary"]) > 0

    def test_mtime_touch_without_content_change(self, task_file_simple, temp_project):
        """When mtime changes but content is identical, changed should be False."""
        from orchestrator.changelog import (
            check_task_file_on_wake,
            save_task_file_snapshot,
        )
        import time

        projects_dir = str(temp_project / ".aura" / "projects")
        content = task_file_simple.read_text(encoding="utf-8")
        save_task_file_snapshot(str(task_file_simple), projects_dir, "test_mission")

        # Touch the file (rewrite same content)
        time.sleep(1.5)  # Ensure mtime differs
        task_file_simple.write_text(content, encoding="utf-8")

        result = check_task_file_on_wake(str(task_file_simple), projects_dir,
                                         "test_mission")
        # mtime changed but content is identical → changed should be False
        # (the function updates snapshot mtime internally)
        assert result["content_changed"] is False

    def test_reconcile_with_wake_change_integration(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        """Integration: reconcile_task_file with wake_change flow."""
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Task A1", "acceptance_criteria": "Done"},
            {"id": "A2", "description": "Task A2", "acceptance_criteria": "Done"},
        ])

        # Actually modify the file to trigger batch advance
        task_file_simple.write_text(
            task_file_simple.read_text(encoding="utf-8") + "\n- [ ] New task\n",
            encoding="utf-8",
        )

        # Simulate runtime change reconciliation
        stats = state_mgr.reconcile_task_file(
            str(task_file_simple),
            mission="Test",
            running_task_ids=set(),
            task_file_changed=True,
        )

        assert "batch" in stats
        assert "planning_needed" in stats
        assert stats["planning_needed"] is True
        # Batch should advance since file content actually changed
        assert stats["batch_advanced"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 5: 不同的需求文件 (Different Requirement Files)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDifferentRequirementFiles:
    """Test handling of multiple different task files and project switching."""

    def test_multiple_projects_independent_state(
        self, temp_project, patched_config, reset_state_cache
    ):
        import importlib
        from orchestrator import state as state_mgr
        import orchestrator.config as cfg

        def _switch_project(data_dir):
            """Fully switch state module to a different data directory."""
            cfg.DATA_DIR = data_dir
            cfg.STATE_DIR = os.path.join(data_dir, "state")
            cfg.MEMORY_DIR = os.path.join(data_dir, "memory")
            # Update state_mgr's module-level paths (computed at import time)
            state_mgr.STATE_DIR = cfg.STATE_DIR
            state_mgr.STATE_PATH = os.path.join(cfg.STATE_DIR, "state.json")
            state_mgr.STATE_BAK_PATH = os.path.join(cfg.STATE_DIR, "state.json.bak")
            state_mgr._state_cache = None
            state_mgr._state_mtime = 0.0
            os.makedirs(cfg.STATE_DIR, exist_ok=True)

        # Project A
        state_a_dir = str(temp_project / ".aura" / "project_a")
        _switch_project(state_a_dir)
        state_mgr.init_state("Mission A", "tasks/a.md")
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "A task 1", "acceptance_criteria": "Done"},
        ])

        # Project B — different data dir
        state_b_dir = str(temp_project / ".aura" / "project_b")
        _switch_project(state_b_dir)
        state_mgr.init_state("Mission B", "tasks/b.md")
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "B task 1", "acceptance_criteria": "Done"},
        ])

        # Verify project B state
        state = state_mgr.load_state()
        assert state["mission"] == "Mission B"
        tasks = state["tasks"][0]["children"]
        assert tasks[0]["description"] == "B task 1"

        # Switch back to project A
        _switch_project(state_a_dir)
        state = state_mgr.load_state()
        assert state["mission"] == "Mission A"
        tasks = state["tasks"][0]["children"]
        assert tasks[0]["description"] == "A task 1"

    def test_active_project_marker(self, temp_project, patched_config):
        from orchestrator.main import _set_active_project, _get_active_project

        _set_active_project("project_x")
        assert _get_active_project() == "project_x"

        _set_active_project("project_y")
        assert _get_active_project() == "project_y"

    def test_task_index_records_multiple_projects(self, temp_project, task_file_simple):
        from orchestrator.main import (
            _record_task_data_dir, _task_data_dir_for,
            _default_aura_base_dir, _task_index_path,
        )

        task2 = temp_project / "tasks" / "other_mission.md"
        task2.parent.mkdir(exist_ok=True)
        task2.write_text("# Other mission\n", encoding="utf-8")

        base_dir = _default_aura_base_dir()
        dir1 = _task_data_dir_for(str(task_file_simple), base_dir)
        dir2 = _task_data_dir_for(str(task2), base_dir)

        _record_task_data_dir(str(task_file_simple), dir1)
        _record_task_data_dir(str(task2), dir2)

        index_path = _task_index_path(base_dir)
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        assert len(index.get("tasks", {})) >= 2

    def test_cleanup_orphan_projects(self, temp_project):
        from orchestrator.changelog import cleanup_orphan_projects
        import orchestrator.state as state_mgr

        projects_dir = str(temp_project / ".aura" / "projects")

        # Create two project dirs
        for name in ["active_proj", "orphan_proj"]:
            pdir = os.path.join(projects_dir, name, "state")
            os.makedirs(pdir, exist_ok=True)
            # Write a minimal state so it's recognized
            state = state_mgr._empty_state()
            state["mission"] = name
            with open(os.path.join(pdir, "state.json"), "w", encoding="utf-8") as f:
                json.dump(state, f)

        # Only "active_proj" has a corresponding task file
        active_files = [str(temp_project / "tasks" / "active_proj.md")]
        os.makedirs(os.path.dirname(active_files[0]), exist_ok=True)
        Path(active_files[0]).touch()

        orphans = cleanup_orphan_projects(projects_dir, active_files)
        assert "orphan_proj" in orphans
        assert "active_proj" not in orphans

    def test_project_name_consistent_across_backslashes(self):
        """Regression: paths with backslashes must map to same project name."""
        from orchestrator.changelog import get_project_name_for_task

        n1 = get_project_name_for_task("tasks/hello.md")
        n2 = get_project_name_for_task("tasks\\hello.md")
        n3 = get_project_name_for_task("c:\\foo\\bar\\hello.md")
        n4 = get_project_name_for_task("/mnt/data/hello.md")

        assert n1 == n2 == n3 == n4 == "hello"


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 6: 新增任务动态plan (Dynamic Planning for New Tasks)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDynamicPlanning:
    """Test dynamic task planning: decompose, batch prefix, and plan flags."""

    def test_decompose_root_creates_top_level_tasks(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))

        result = state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Setup project", "acceptance_criteria": "Scaffold exists"},
            {"id": "A2", "description": "Add tests", "acceptance_criteria": "Tests pass"},
        ])

        assert "OK" in result
        assert "A1" in result
        assert "A2" in result

        state = state_mgr.load_state()
        root = state["tasks"][0]
        assert len(root["children"]) == 2
        assert root["children"][0]["id"] == "A1"
        assert root["children"][0]["depth"] == 1
        assert root["children"][1]["id"] == "A2"

    def test_decompose_root_auto_assigns_ids(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))

        # Don't provide IDs — should auto-generate
        result = state_mgr.decompose_task("root", [
            {"id": "", "description": "Auto ID 1", "acceptance_criteria": "Done"},
            {"id": "", "description": "Auto ID 2", "acceptance_criteria": "Done"},
        ])

        assert "OK" in result
        state = state_mgr.load_state()
        root = state["tasks"][0]
        # IDs should be auto-assigned as A1, A2
        ids = [c["id"] for c in root["children"]]
        assert ids == ["A1", "A2"]

    def test_decompose_root_uses_current_batch_prefix(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))

        # Advance batch to B
        state = state_mgr.load_state()
        state["task_batch"]["current_index"] = 1
        state["task_batch"]["current_prefix"] = "B"
        state_mgr.save_state(state)

        result = state_mgr.decompose_task("root", [
            {"id": "", "description": "Batch B task", "acceptance_criteria": "Done"},
        ])
        assert "B1" in result

    def test_decompose_child_creates_dotted_ids(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Parent task", "acceptance_criteria": "Done"},
        ])

        # Decompose A1 into subtasks
        result = state_mgr.decompose_task("A1", [
            {"id": "A1.1", "description": "Child 1", "acceptance_criteria": "Done"},
            {"id": "A1.2", "description": "Child 2", "acceptance_criteria": "Done"},
        ])

        assert "A1.1" in result
        assert "A1.2" in result

        state = state_mgr.load_state()
        parent = state_mgr.find_task("A1", state["tasks"])
        assert len(parent["children"]) == 2
        assert parent["children"][0]["id"] == "A1.1"
        assert parent["children"][0]["depth"] == 2

    def test_decompose_child_auto_assigns_ids(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Parent", "acceptance_criteria": "Done"},
        ])

        result = state_mgr.decompose_task("A1", [
            {"id": "", "description": "Auto child", "acceptance_criteria": "Done"},
        ])
        assert "A1.1" in result

    def test_decompose_category_clears_planning_flag(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))

        # Set planning needed flag
        state = state_mgr.load_state()
        state["task_file_needs_planning"] = True
        state_mgr.save_state(state)

        # Decomposing root creates categories only, so planning remains needed
        # until each category has at least one concrete child task.
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Category", "acceptance_criteria": "Done"},
        ])
        state = state_mgr.load_state()
        assert state.get("task_file_needs_planning") is True

        state_mgr.decompose_task("A1", [
            {"id": "A1.1", "description": "Concrete task", "acceptance_criteria": "Done"},
        ])

        state = state_mgr.load_state()
        assert state.get("task_file_needs_planning") is False

    def test_update_task_status_flow(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Test task", "acceptance_criteria": "Done"},
        ])

        # pending → in_progress
        result = state_mgr.update_task("A1", "in_progress", "Starting work", "manual")
        assert "OK" in result

        state = state_mgr.load_state()
        task = state_mgr.find_task("A1", state["tasks"])
        assert task["status"] == "in_progress"
        assert "started_at" in task

        # in_progress → completed
        result = state_mgr.update_task("A1", "completed", "Work done",
                                        "result.md exists at workspace/tasks/A1/")
        assert "OK" in result

        state = state_mgr.load_state()
        task = state_mgr.find_task("A1", state["tasks"])
        assert task["status"] == "completed"
        assert "completed_at" in task

    def test_update_task_decision_logged(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Task", "acceptance_criteria": "Done"},
        ])
        state_mgr.update_task("A1", "completed", "Done", "evidence.txt")

        state = state_mgr.load_state()
        assert len(state["decision_log"]) >= 1
        last = state["decision_log"][-1]
        assert last["task_id"] == "A1"
        assert last["old_status"] == "pending"
        assert last["new_status"] == "completed"
        assert last["reason"] == "Done"

    def test_task_tree_summary(self, task_file_simple, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "First task", "acceptance_criteria": "Done"},
        ])

        summary = state_mgr.get_task_tree_summary()
        assert "root" in summary
        assert "A1" in summary
        assert "First task" in summary

    def test_count_active_tasks(self, task_file_simple, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "T1", "acceptance_criteria": "Done"},
            {"id": "A2", "description": "T2", "acceptance_criteria": "Done"},
        ])

        assert state_mgr.count_active_tasks() == 0

        state_mgr.update_task("A1", "in_progress", "Start", "manual")
        assert state_mgr.count_active_tasks() == 1

        state_mgr.update_task("A2", "in_progress", "Start", "manual")
        assert state_mgr.count_active_tasks() == 2

    def test_can_spawn_task_limit(self, task_file_simple, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "T1", "acceptance_criteria": "Done"},
            {"id": "A2", "description": "T2", "acceptance_criteria": "Done"},
            {"id": "A3", "description": "T3", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A1", [
            {"id": "A1.1", "description": "T1 concrete", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A2", [
            {"id": "A2.1", "description": "T2 concrete", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A3", [
            {"id": "A3.1", "description": "T3 concrete", "acceptance_criteria": "Done"},
        ])

        assert state_mgr.can_spawn_task() is True

        state_mgr.update_task("A1.1", "in_progress", "Start", "manual")
        state_mgr.update_task("A2.1", "in_progress", "Start", "manual")

        assert state_mgr.can_spawn_task() is False  # Max 2 concurrent

    def test_find_task_recursive(self, task_file_simple, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Level 1", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A1", [
            {"id": "A1.1", "description": "Level 2", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A1.1", [
            {"id": "A1.1.1", "description": "Level 3", "acceptance_criteria": "Done"},
        ])

        state = state_mgr.load_state()
        assert state_mgr.find_task("root", state["tasks"]) is not None
        assert state_mgr.find_task("A1", state["tasks"]) is not None
        assert state_mgr.find_task("A1.1", state["tasks"]) is not None
        assert state_mgr.find_task("A1.1.1", state["tasks"]) is not None
        assert state_mgr.find_task("nonexistent", state["tasks"]) is None

    def test_reconcile_sets_planning_needed_for_empty_tree(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr

        # Init state but don't decompose — no children
        state_mgr.init_state("Test", str(task_file_simple))

        stats = state_mgr.reconcile_task_file(
            str(task_file_simple),
            mission="Test",
        )
        assert stats["planning_needed"] is True

    def test_deeply_nested_decompose(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr

        state_mgr.init_state("Deep test", str(task_file_simple))

        # Build a 4-level tree
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "L1", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A1", [
            {"id": "A1.1", "description": "L2a", "acceptance_criteria": "Done"},
            {"id": "A1.2", "description": "L2b", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A1.1", [
            {"id": "A1.1.1", "description": "L3", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A1.1.1", [
            {"id": "A1.1.1.1", "description": "L4", "acceptance_criteria": "Done"},
        ])

        state = state_mgr.load_state()
        l1 = state_mgr.find_task("A1", state["tasks"])
        assert l1["depth"] == 1
        l4 = state_mgr.find_task("A1.1.1.1", state["tasks"])
        assert l4["depth"] == 4


# ═══════════════════════════════════════════════════════════════════════════════
# Tools Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTools:
    """Test tool implementations (those that don't need actual subprocess/API)."""

    def test_resolve_path_aura_prefix(self, patched_config):
        from orchestrator.tools import _resolve_path
        from orchestrator.config import DATA_DIR

        resolved = _resolve_path(".aura/state/state.json")
        assert resolved == os.path.normpath(os.path.join(DATA_DIR, "state", "state.json"))

        resolved = _resolve_path(".aura/memory/MEMORY.md")
        assert resolved == os.path.normpath(os.path.join(DATA_DIR, "memory", "MEMORY.md"))

    def test_resolve_path_relative(self, patched_config):
        from orchestrator.tools import _resolve_path
        from orchestrator.config import PROJECT_ROOT

        resolved = _resolve_path("tasks/test.md")
        assert resolved == os.path.normpath(os.path.join(PROJECT_ROOT, "tasks", "test.md"))

    def test_read_and_write_file(self, patched_config):
        from orchestrator.tools import impl_read_file, impl_write_file

        # Write
        result = impl_write_file("test_output.txt", "Hello, world!")
        assert "OK" in result

        # Read
        content = impl_read_file("test_output.txt")
        assert "Hello, world!" in content

    def test_list_directory(self, patched_config, temp_project):
        from orchestrator.tools import impl_list_directory, impl_write_file

        impl_write_file("test_dir/a.txt", "a")
        impl_write_file("test_dir/b.txt", "b")

        result = impl_list_directory("test_dir")
        assert "a.txt" in result
        assert "b.txt" in result

    def test_other_aura_task_data_is_isolated_but_memory_readable(self, patched_config):
        from orchestrator.tools import impl_list_directory, impl_read_file, impl_write_file
        from orchestrator.config import PROJECT_ROOT

        other_root = Path(PROJECT_ROOT) / ".aura" / "other-task"
        (other_root / "state").mkdir(parents=True, exist_ok=True)
        (other_root / "workspace").mkdir(parents=True, exist_ok=True)
        (other_root / "memory").mkdir(parents=True, exist_ok=True)
        (other_root / "state" / "state.json").write_text("{}", encoding="utf-8")
        (other_root / "memory" / "MEMORY.md").write_text("lesson", encoding="utf-8")

        assert "Refusing to read" in impl_read_file(".aura/other-task/state/state.json")
        assert "Refusing to list" in impl_list_directory(".aura/other-task/workspace")
        assert "lesson" in impl_read_file(".aura/other-task/memory/MEMORY.md")
        assert "MEMORY.md" in impl_list_directory(".aura/other-task/memory")
        assert "Refusing to write" in impl_write_file(
            ".aura/other-task/memory/MEMORY.md", "do not mutate other memory"
        )

    def test_no_op(self):
        from orchestrator.tools import impl_no_op

        result = impl_no_op("Everything is fine", "Check A1 progress")
        assert "Everything is fine" in result
        assert "Check A1 progress" in result

    def test_write_memory(self, patched_config):
        from orchestrator.tools import impl_write_memory

        result = impl_write_memory("fact", "This is a test fact.")
        assert "OK" in result

    def test_update_task_tree_guard_in_progress_limit(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr
        from orchestrator.tools import impl_update_task_tree

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "T1", "acceptance_criteria": "Done"},
            {"id": "A2", "description": "T2", "acceptance_criteria": "Done"},
            {"id": "A3", "description": "T3", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A1", [
            {"id": "A1.1", "description": "T1 concrete", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A2", [
            {"id": "A2.1", "description": "T2 concrete", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A3", [
            {"id": "A3.1", "description": "T3 concrete", "acceptance_criteria": "Done"},
        ])

        # Put two tasks in_progress
        state_mgr.update_task("A1.1", "in_progress", "Start", "manual")
        state_mgr.update_task("A2.1", "in_progress", "Start", "manual")

        # Try to mark A3 as in_progress via update_task_tree → should be rejected
        result = impl_update_task_tree("A3.1", "in_progress", "Trying",
                                        "should fail")
        assert "ERROR" in result
        assert "max concurrent" in result.lower()

    def test_update_task_tree_allows_reentry(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr
        from orchestrator.tools import impl_update_task_tree

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "T1", "acceptance_criteria": "Done"},
            {"id": "A2", "description": "T2", "acceptance_criteria": "Done"},
        ])

        state_mgr.update_task("A1", "in_progress", "Start", "manual")
        state_mgr.update_task("A2", "in_progress", "Start", "manual")

        # Re-marking A1 as in_progress (idempotent) should be allowed
        result = impl_update_task_tree("A1", "in_progress", "Re-entry",
                                        "idempotent")
        assert "OK" in result

    def test_kill_task_updates_status(self, task_file_simple, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr
        from orchestrator.tools import impl_kill_task
        from orchestrator import process_mgr

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "T1", "acceptance_criteria": "Done"},
        ])
        state_mgr.update_task("A1", "in_progress", "Start", "manual")

        # Kill a task that isn't actually running (process_mgr doesn't have it)
        result = impl_kill_task("A1")
        # Should say error from process_mgr but still update tree status
        assert "killed" in state_mgr.load_state()["tasks"][0]["children"][0]["status"] or \
               "ERROR" in result

    def test_spawn_task_rejects_top_level_category(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr
        from orchestrator.tools import impl_spawn_task

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Planning category", "acceptance_criteria": "Done"},
        ])

        result = impl_spawn_task("A1", "Do the work", budget_minutes=1)
        assert "ERROR" in result
        assert "planning/category" in result
        assert "A1.1" in result

    def test_spawn_task_writes_hierarchy_and_sibling_context(
        self, task_file_simple, patched_config, reset_state_cache, monkeypatch
    ):
        from orchestrator import process_mgr
        from orchestrator import state as state_mgr
        from orchestrator.config import get_workspace_dir
        from orchestrator.tools import impl_spawn_task

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Implementation category", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A1", [
            {"id": "A1.1", "description": "Current concrete task", "acceptance_criteria": "Result exists"},
            {"id": "A1.2", "description": "Sibling experiment", "acceptance_criteria": "Result exists"},
        ])
        state_mgr.update_task("A1.2", "failed", "Tried incompatible approach", "sibling failure evidence")

        sibling_dir = Path(get_workspace_dir()) / "tasks" / "A1.2"
        sibling_dir.mkdir(parents=True, exist_ok=True)
        (sibling_dir / "result.md").write_text("Sibling result summary", encoding="utf-8")

        monkeypatch.setattr(process_mgr, "list_all", lambda: [])
        monkeypatch.setattr(
            process_mgr,
            "spawn",
            lambda task_id, task_dir, task_md_path, budget_minutes: "OK: spawned",
        )

        result = impl_spawn_task("A1.1", "Execute current task", budget_minutes=1)
        assert "OK" in result

        task_md = Path(get_workspace_dir()) / "tasks" / "A1.1" / "task.md"
        content = task_md.read_text(encoding="utf-8")
        assert "Guiding Philosophy" not in content
        assert content.startswith("# Task A1.1")
        assert "## Task Hierarchy" in content
        assert "## Sibling Tasks Context" in content
        assert "A1.1 (current task)" in content
        assert "A1.2 [failed]" in content
        assert "sibling failure evidence" in content
        assert "Sibling result summary" in content

    def test_tool_dispatch(self):
        from orchestrator.tools import execute_tool

        result = execute_tool("no_op", {"reason": "test"})
        assert "test" in result

        result = execute_tool("nonexistent_tool", {})
        assert "ERROR" in result
        assert "Unknown tool" in result

    def test_web_fetch_invalid_url(self):
        from orchestrator.tools import impl_web_fetch

        result = impl_web_fetch("not-a-url", 100)
        assert "ERROR" in result

    def test_list_running_tasks_empty(self):
        from orchestrator.tools import impl_list_running_tasks

        result = impl_list_running_tasks()
        assert "No running tasks" in result

    def test_process_identity_rejects_reused_pid(self):
        from orchestrator import process_mgr

        current = process_mgr.psutil.Process(os.getpid())
        stale_started_at = datetime.fromtimestamp(current.create_time()) - timedelta(hours=1)
        entry = {
            "pid": os.getpid(),
            "task_id": "A1.1",
            "started_at": stale_started_at,
            "killed_at": None,
        }

        assert process_mgr._is_alive(os.getpid())
        assert not process_mgr._entry_process_is_alive(entry)


# ═══════════════════════════════════════════════════════════════════════════════
# Progress & Memory Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestProgressAndMemory:
    """Test progress rendering and memory operations."""

    def test_render_progress(self, task_file_simple, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr, progress as progress_mgr

        state_mgr.init_state("Test mission", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Task 1", "acceptance_criteria": "Done"},
        ])
        state_mgr.update_task("A1", "in_progress", "Working", "manual")

        content = progress_mgr.render_progress()
        assert "Test mission" in content
        assert "A1" in content

        assert os.path.exists(progress_mgr.PROGRESS_PATH)

    def test_render_progress_skips_when_unchanged(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr, progress as progress_mgr

        state_mgr.init_state("Test", str(task_file_simple))

        content1 = progress_mgr.render_progress()
        content2 = progress_mgr.render_progress()

        # Both calls with no state change should produce same content
        assert content1 == content2

    def test_render_progress_removes_legacy_root_progress(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr, progress as progress_mgr

        state_mgr.init_state("Test", str(task_file_simple))
        legacy_path = progress_mgr.LEGACY_PROGRESS_PATH
        os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
        Path(legacy_path).write_text("legacy", encoding="utf-8")

        progress_mgr.render_progress()

        assert os.path.exists(progress_mgr.PROGRESS_PATH)
        assert not os.path.exists(legacy_path)

    def test_render_progress_updates_final_report_by_batch(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr, progress as progress_mgr
        from orchestrator import task_reporter

        state_mgr.init_state("Test mission", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "First batch done", "acceptance_criteria": "Done"},
            {"id": "A2", "description": "Second A task", "acceptance_criteria": "Done"},
        ])
        state_mgr.update_task("A1", "completed", "Done", "a1/result.md")
        state_mgr.update_task("A2", "completed", "Done", "a2/result.md")

        state = state_mgr.load_state()
        state["task_batch"]["current_index"] = 1
        state["task_batch"]["current_prefix"] = "B"
        state_mgr.save_state(state)
        state_mgr.decompose_task("root", [
            {"id": "B1", "description": "New requirement still open", "acceptance_criteria": "Done"},
        ])

        progress_mgr.render_progress()

        report_path = os.path.join(task_reporter.TASK_SUMMARY_DIR, "final_report.md")
        assert os.path.exists(report_path)
        report = Path(report_path).read_text(encoding="utf-8")
        assert "Batch A" in report
        assert "Batch B" in report
        assert "A1" in report
        assert "B1" in report
        assert "Project status**: in_progress" in report

    def test_append_memory(self, patched_config):
        from orchestrator import memory as memory_mgr

        result = memory_mgr.append_memory("fact", "The sky is blue.")
        assert "OK" in result

        mem = memory_mgr.load_long_term_memory()
        assert "The sky is blue" in mem

    def test_write_and_load_session(self, patched_config):
        from orchestrator import memory as memory_mgr

        memory_mgr.write_session("# Session\ntest content")
        content = memory_mgr.load_session()
        assert "test content" in content

    def test_memory_compression(self, patched_config):
        from orchestrator import memory as memory_mgr

        # Write a large memory chunk
        big = "## Fact\n" + ("x" * 6000)
        memory_mgr.overwrite_memory(big)

        # Append more — should trigger compression
        result = memory_mgr.append_memory("lesson", "Added after compression")
        assert "OK" in result

        mem = memory_mgr.load_long_term_memory()
        assert "Added after compression" in mem


# ═══════════════════════════════════════════════════════════════════════════════
# Changelog Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestChangelogEdgeCases:
    """Edge case tests for the changelog system."""

    def test_load_changelog_missing_file(self, temp_project):
        from orchestrator.changelog import load_changelog

        cl = load_changelog(str(temp_project / "nonexistent.json"))
        assert cl["version"] == 1
        assert cl["entries"] == []
        assert cl["processed_hashes"] == {}

    def test_load_changelog_corrupted(self, temp_project):
        from orchestrator.changelog import load_changelog

        path = temp_project / "corrupted.json"
        path.write_text("{not valid json", encoding="utf-8")

        cl = load_changelog(str(path))
        assert cl["version"] == 1
        assert cl["entries"] == []

    def test_changelog_multiple_entries(self, task_file_simple, temp_project):
        from orchestrator.changelog import (
            mark_file_processed,
            load_changelog,
            get_changelog_path,
        )

        projects_dir = str(temp_project / ".aura" / "projects")

        # Process, edit, process again
        mark_file_processed(str(task_file_simple), projects_dir, "test_mission",
                           summary="First run")
        task_file_simple.write_text("# Changed\n", encoding="utf-8")
        mark_file_processed(str(task_file_simple), projects_dir, "test_mission",
                           summary="Second run")
        task_file_simple.write_text("# Changed again\n", encoding="utf-8")
        mark_file_processed(str(task_file_simple), projects_dir, "test_mission",
                           summary="Third run")

        cl_path = get_changelog_path(projects_dir, "test_mission",
                                     str(task_file_simple))
        cl = load_changelog(cl_path)
        assert len(cl["entries"]) == 3
        assert cl["entries"][0]["summary"] == "First run"
        assert cl["entries"][2]["summary"] == "Third run"


# ═══════════════════════════════════════════════════════════════════════════════
# State Caching (T0 optimization) Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateCaching:
    """Verify the T0 mtime-based caching works correctly."""

    def test_cache_hit_after_save(self, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr

        task_file = os.path.join(os.environ["AURA_PROJECT_ROOT"], "tasks", "t.md")
        os.makedirs(os.path.dirname(task_file), exist_ok=True)
        Path(task_file).write_text("# Test\n", encoding="utf-8")

        state = state_mgr.init_state("Cache test", task_file)

        # load_state should return from cache
        state_mgr._state_cache = state
        state_mgr._state_mtime = os.path.getmtime(state_mgr.STATE_PATH)

        loaded = state_mgr.load_state()
        assert loaded["mission"] == "Cache test"

    def test_cache_invalidation_after_save(self, patched_config, reset_state_cache):
        from orchestrator import state as state_mgr

        task_file = os.path.join(os.environ["AURA_PROJECT_ROOT"], "tasks", "t.md")
        os.makedirs(os.path.dirname(task_file), exist_ok=True)
        Path(task_file).write_text("# Test\n", encoding="utf-8")

        state_mgr.init_state("Before", task_file)
        state = state_mgr.load_state()
        state["mission"] = "After modification"
        state_mgr.save_state(state)

        loaded = state_mgr.load_state()
        assert loaded["mission"] == "After modification"


# ═══════════════════════════════════════════════════════════════════════════════
# Context Building Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextBuilding:
    """Test build_context_message and related formatting."""

    def test_build_context_message_basic(self, task_file_simple,
                                          patched_config, reset_state_cache):
        from orchestrator import state as state_mgr
        from orchestrator.agent import build_context_message

        state_mgr.init_state("Test Mission", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Task one", "acceptance_criteria": "Done"},
        ])

        ctx = build_context_message()
        assert "Test Mission" in ctx
        assert "A1" in ctx
        assert "Task one" in ctx
        assert "Task File" in ctx

    def test_build_context_message_with_wake_change(
        self, task_file_simple, patched_config, reset_state_cache
    ):
        from orchestrator import state as state_mgr
        from orchestrator.agent import build_context_message

        state_mgr.init_state("Test Mission", str(task_file_simple))

        wake_change = {
            "changed": True,
            "content_changed": True,
            "mtime_changed": True,
            "diff_lines": ["+## New Section", "+- [ ] New Task"],
            "added_requirement_lines": ["New Task"],
            "removed_requirement_lines": [],
            "added_info_lines": [],
            "change_summary": "1 new/changed requirement(s)",
        }

        ctx = build_context_message(wake_change=wake_change)
        assert "任务文件在本周期被修改" in ctx
        assert "New Task" in ctx

    def test_format_workspace_snapshot(self, task_file_simple,
                                        patched_config, reset_state_cache):
        from orchestrator import state as state_mgr
        from orchestrator.agent import _format_workspace_snapshot

        state_mgr.init_state("Test", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Task", "acceptance_criteria": "Done"},
        ])

        # Create workspace dir for A1
        from orchestrator.config import get_workspace_dir
        task_dir = os.path.join(get_workspace_dir(), "tasks", "A1")
        os.makedirs(task_dir, exist_ok=True)
        Path(os.path.join(task_dir, "result.md")).write_text("# Result\nDone!", encoding="utf-8")

        snap = _format_workspace_snapshot(["A1"], ["A1"])
        assert "A1" in snap
        assert "result.md" in snap


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLowTokenCycles:
    """Test local no-API cycle decisions."""

    def test_run_cycle_skips_api_when_idle(
        self, task_file_simple, patched_config, reset_state_cache, monkeypatch
    ):
        from orchestrator import agent
        from orchestrator import state as state_mgr

        state_mgr.init_state("Idle mission", str(task_file_simple))
        state = state_mgr.load_state()
        state["task_file_needs_planning"] = False
        state_mgr.save_state(state)

        def fail_api(*args, **kwargs):
            raise AssertionError("API should not be called for idle local skip")

        monkeypatch.setattr(agent, "_call_api_with_retry", fail_api)
        result = agent.run_cycle()

        assert result["skipped_api"] is True
        assert result["api_calls"] == 0
        assert result["tool_calls"] == 0

    def test_local_skip_does_not_skip_pending_executable_with_capacity(
        self, task_file_simple, patched_config, reset_state_cache, monkeypatch
    ):
        from orchestrator import agent
        from orchestrator import process_mgr
        from orchestrator import state as state_mgr

        state_mgr.init_state("Needs work", str(task_file_simple))
        state_mgr.decompose_task("root", [
            {"id": "A1", "description": "Category", "acceptance_criteria": "Done"},
        ])
        state_mgr.decompose_task("A1", [
            {"id": "A1.1", "description": "Concrete pending", "acceptance_criteria": "Done"},
        ])
        state = state_mgr.load_state()
        state["task_file_needs_planning"] = False
        state_mgr.save_state(state)
        monkeypatch.setattr(process_mgr, "list_all", lambda: [])

        skip, reason = agent._should_skip_llm_cycle(
            state_mgr.load_state(),
            {"activity_mode": "active", "replan_requested": False, "progress_results": []},
            wake_change=None,
        )

        assert skip is False
        assert "executable_pending=1" in reason


class TestPhase2:
    """Test Phase 2 progress evaluation and decision logic."""

    def test_evaluate_progress_no_output(self, patched_config):
        from orchestrator.phase2 import evaluate_progress
        from orchestrator.config import get_workspace_dir

        # Create an empty task workspace
        task_dir = os.path.join(get_workspace_dir(), "tasks", "empty_task")
        os.makedirs(task_dir, exist_ok=True)

        result = evaluate_progress("empty_task", 0, "", 0.0)
        assert result["has_output"] is False
        # Score is low but not exactly zero: no error log gives +0.05
        assert result["active_score"] < 0.1
        # No output + CPU 0 → stuck (all signals agree)
        assert result["is_stuck"] is True

    def test_evaluate_progress_with_cpu_not_stuck(self, patched_config):
        from orchestrator.phase2 import evaluate_progress
        from orchestrator.config import get_workspace_dir

        task_dir = os.path.join(get_workspace_dir(), "tasks", "computing_task")
        os.makedirs(task_dir, exist_ok=True)

        # No output file but high CPU → NOT stuck (computing)
        result = evaluate_progress("computing_task", 0, "", 10.0)
        assert result["is_stuck"] is False
        assert result["active_score"] > 0

    def test_check_replan_needed_no_progress(self):
        from orchestrator.phase2 import check_replan_needed

        result = check_replan_needed(6, 2.0, False)
        assert result["replan_requested"] is True
        assert "consecutive cycles" in result["trigger_reason"]

    def test_check_replan_needed_long_time(self):
        from orchestrator.phase2 import check_replan_needed

        result = check_replan_needed(2, 5.0, False)
        assert result["replan_requested"] is True
        assert "hours elapsed" in result["trigger_reason"]

    def test_check_replan_needed_new_requirements(self):
        from orchestrator.phase2 import check_replan_needed

        result = check_replan_needed(1, 1.0, True, has_new_requirements=True)
        assert result["replan_requested"] is True
        assert "new/changed requirements" in result["trigger_reason"]

    def test_check_replan_needed_all_fine(self):
        from orchestrator.phase2 import check_replan_needed

        result = check_replan_needed(2, 1.0, True)
        assert result["replan_requested"] is False

    def test_decision_matrix_continue(self):
        from orchestrator.phase2 import decision_matrix

        progress = {
            "active_score": 0.7,
            "has_output": True,
            "is_stuck": False,
            "is_looping": False,
            "output_delta": 5000,
            "artifacts": ["result.md"],
            "error_log_size": 0,
            "tail_analysis": {},
        }
        result = decision_matrix(progress, 5, 25)
        assert result["action"] == "continue_deeper"
        assert result["confidence"] > 0.5

    def test_decision_matrix_kill_over_budget(self):
        from orchestrator.phase2 import decision_matrix

        progress = {
            "active_score": 0.1,
            "has_output": False,
            "is_stuck": True,
            "is_looping": False,
            "output_delta": 0,
            "artifacts": [],
            "error_log_size": 0,
            "tail_analysis": {},
        }
        result = decision_matrix(progress, 35, -5)
        assert result["action"] == "kill"

    def test_get_activity_mode(self):
        from orchestrator.phase2 import get_activity_mode

        assert get_activity_mode([]) == "idle"

        active = get_activity_mode([
            {"active_score": 0.5, "has_output": True, "output_delta": 1000, "content_changed": True},
        ])
        assert active == "active"

        calm = get_activity_mode([
            {"active_score": 0.15, "has_output": True, "output_delta": 0, "content_changed": False},
        ])
        assert calm == "calm"

        idle = get_activity_mode([
            {"active_score": 0.05, "has_output": False, "output_delta": 0, "content_changed": False},
        ])
        assert idle == "idle"


# ═══════════════════════════════════════════════════════════════════════════════
# Requirement Similarity Tests (for future match-planning)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRequirementSimilarity:
    """Test the requirement similarity matching used in reconcile."""

    def test_exact_match(self):
        from orchestrator.state import _requirement_similarity

        score = _requirement_similarity("Build web app", "Build web app")
        assert score == 1.0

    def test_similar_match(self):
        from orchestrator.state import _requirement_similarity

        score = _requirement_similarity(
            "Build a web application with React",
            "Build a web application with React framework",
        )
        assert score > 0.8

    def test_different_requirements(self):
        from orchestrator.state import _requirement_similarity

        score = _requirement_similarity("Build web app", "Deploy database")
        assert score < 0.5

    def test_empty_strings(self):
        from orchestrator.state import _requirement_similarity

        assert _requirement_similarity("", "") == 0.0
        assert _requirement_similarity("Something", "") == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfig:
    """Test configuration values."""

    def test_default_values(self):
        from orchestrator import config

        assert config.MAX_CONCURRENT_TASKS == 2
        assert config.STUCK_THRESHOLD_CYCLES == 12
        assert config.API_RETRY_COUNT == 4
        assert config.CYCLE_INTERVAL_SECONDS >= 60
        assert config.WORKER_RESOURCE_GUARD_ENABLED is True
        assert config.WORKER_MAX_CPU_PERCENT == 80
        assert config.WORKER_MAX_SYSTEM_MEMORY_PERCENT == 80
        assert config.WORKER_MAX_GPU_MEMORY_PERCENT == 80
        assert config.WORKER_MAX_GPU_UTIL_PERCENT == 80

    def test_env_override(self):
        import importlib
        import orchestrator.config as cfg

        with patch.dict(os.environ, {"AURA_CYCLE_INTERVAL": "60"}):
            # Reload config to pick up env changes
            cfg = importlib.reload(cfg)
            assert cfg.CYCLE_INTERVAL_SECONDS == 60


class TestResourceGuard:
    """Test worker resource guard helpers."""

    def test_resource_violation_requires_two_strikes(self):
        from orchestrator import process_mgr

        limits = {
            "enabled": True,
            "poll_seconds": 10,
            "avg_window_seconds": 180,
            "violation_strikes": 1,
            "max_cpu_percent": 0,
            "max_system_memory_percent": 0,
            "max_gpu_util_percent": 0,
            "max_gpu_memory_percent": 0,
            "max_system_memory_gb": 1,
            "min_system_memory_free_gb": 0,
            "max_gpu_memory_gb": 0,
            "min_gpu_memory_free_gb": 0,
            "cuda_visible_devices": "",
        }
        entry = {"resource_limits": limits}
        metrics = {
            "cpu_percent": 0,
            "memory_mb": 2048,
            "memory_percent": 1,
            "gpu_memory_mb": None,
            "gpu_memory_percent": None,
            "gpu_util_percent": None,
        }

        assert process_mgr._evaluate_resource_violation(entry, metrics) is None
        reason = process_mgr._evaluate_resource_violation(entry, metrics)
        assert "host RSS" in reason

    def test_worker_env_exports_resource_policy(self):
        from orchestrator import process_mgr

        env = process_mgr._worker_env({})
        assert env["AURA_WORKER_RESOURCE_GUARD"] in {"0", "1"}
        assert env["AURA_WORKER_FORBID_OFFLOAD"] == "1"
        assert env["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
        assert "AURA_WORKER_MAX_SYSTEM_MEMORY_PERCENT" in env
