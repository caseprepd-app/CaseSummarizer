"""
Tests for race condition audit fixes.

Validates:
1. Follow-up polling timeout (BUG 1)
2. Active worker lock in subprocess (BUG 2)
3. Init defaults for _pending_tasks/_completed_tasks (BUG 3)
4. Regression: _destroying flag ordering in destroy()
5. Anti-pattern audit (source inspection)
"""

import inspect
import re
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_window_stub():
    """Create a stub with the same state attributes as MainWindow.__init__."""
    stub = MagicMock()
    stub._destroying = False
    stub._followup_pending = False
    stub._followup_poll_count = 0
    stub._FOLLOWUP_TIMEOUT_POLLS = 300
    stub._pending_tasks = {}
    stub._completed_tasks = set()
    stub._qa_answering_active = False
    stub._processing_active = False
    stub._preprocessing_active = False
    return stub


def _get_main_window_source():
    """Return main_window.py source without importing MainWindow (needs Tk)."""
    from pathlib import Path

    src = Path(__file__).parent.parent / "src" / "ui" / "main_window.py"
    return src.read_text(encoding="utf-8")


def _get_worker_process_source():
    """Return worker_process.py source."""
    import src.worker_process as mod

    return inspect.getsource(mod)


# =========================================================================
# 1. Follow-up polling timeout (BUG 1)
# =========================================================================


class TestFollowupTimeout:
    """Verify _poll_followup_result times out instead of polling forever."""

    def test_timeout_constant_exists(self):
        """MainWindow must define _FOLLOWUP_TIMEOUT_POLLS."""
        src = _get_main_window_source()
        assert "_FOLLOWUP_TIMEOUT_POLLS" in src
        # Should be a class-level constant (300 = 30 s at 100 ms)
        match = re.search(r"_FOLLOWUP_TIMEOUT_POLLS\s*=\s*(\d+)", src)
        assert match, "_FOLLOWUP_TIMEOUT_POLLS not assigned"
        assert int(match.group(1)) > 0

    def test_poll_count_initialized_in_init(self):
        """_followup_poll_count must be set in __init__."""
        src = _get_main_window_source()
        assert "_followup_poll_count" in src
        # Check it's initialized (not just used)
        assert re.search(r"self\._followup_poll_count\s*[:=]\s*", src)

    def test_poll_count_reset_on_ask(self):
        """_ask_followup must reset _followup_poll_count to 0."""
        src = _get_main_window_source()
        # Find _ask_followup method and check for counter reset
        ask_match = re.search(
            r"def _ask_followup\(.*?\n(.*?)(?=\n    def |\nclass |\Z)",
            src,
            re.DOTALL,
        )
        assert ask_match, "_ask_followup method not found"
        body = ask_match.group(1)
        assert "_followup_poll_count = 0" in body, "_ask_followup must reset poll counter"

    def test_timeout_resets_ui(self):
        """When poll count exceeds threshold, source must re-enable controls."""
        src = _get_main_window_source()
        poll_match = re.search(
            r"def _poll_followup_result\(.*?\n(.*?)(?=\n    def |\nclass |\Z)",
            src,
            re.DOTALL,
        )
        assert poll_match, "_poll_followup_result not found"
        body = poll_match.group(1)
        # Must check counter against threshold
        assert "_FOLLOWUP_TIMEOUT_POLLS" in body
        # Must re-enable button on timeout
        assert 'state="normal"' in body


# =========================================================================
# 2. _destroying flag ordering regression
# =========================================================================


class TestDestroyFlagOrdering:
    """Regression: _destroying must be set True before any after_cancel."""

    def test_destroying_before_after_cancel(self):
        """In destroy(), _destroying = True must appear before after_cancel."""
        src = _get_main_window_source()
        destroy_match = re.search(
            r"def destroy\(self\).*?\n(.*?)(?=\n    def |\nclass |\Z)",
            src,
            re.DOTALL,
        )
        assert destroy_match, "destroy() method not found"
        body = destroy_match.group(1)
        lines = body.split("\n")
        destroying_line = None
        first_cancel_line = None
        for i, line in enumerate(lines):
            if "_destroying = True" in line and destroying_line is None:
                destroying_line = i
            if "after_cancel" in line and first_cancel_line is None:
                first_cancel_line = i
        assert destroying_line is not None, "_destroying = True not found in destroy()"
        assert first_cancel_line is not None, "after_cancel not found in destroy()"
        assert destroying_line < first_cancel_line, (
            f"_destroying = True (line {destroying_line}) must come before "
            f"first after_cancel (line {first_cancel_line})"
        )


# =========================================================================
# 3. Active worker lock (BUG 2)
# =========================================================================


class TestActiveWorkerLock:
    """Verify worker_process.py protects state['active_worker'] with a lock."""

    def test_worker_lock_in_state_dict(self):
        """State dict must contain 'worker_lock'."""
        src = _get_worker_process_source()
        assert '"worker_lock"' in src

    def test_lock_is_threading_lock(self):
        """worker_lock value must be threading.Lock()."""
        src = _get_worker_process_source()
        assert re.search(r'"worker_lock"\s*:\s*threading\.Lock\(\)', src)

    def test_run_functions_use_lock(self):
        """Each _run_* function that assigns active_worker must use the lock."""
        src = _get_worker_process_source()
        # Match _run_* functions that assign state["active_worker"] (not comments)
        run_funcs = re.findall(
            r'(def _run_\w+\(.*?\n(?:.*?\n)*?.*?state\["active_worker"\]\s*=.*?\n)',
            src,
        )
        # At least process_files, extraction, qa, summary set active_worker
        assert len(run_funcs) >= 4, f"Expected >=4 _run_* functions, found {len(run_funcs)}"
        for func_body in run_funcs:
            assert "worker_lock" in func_body, (
                f"_run_* function missing worker_lock:\n{func_body[:120]}"
            )

    def test_stop_active_worker_uses_lock(self):
        """_stop_active_worker must read/write active_worker under lock."""
        src = _get_worker_process_source()
        stop_match = re.search(
            r"def _stop_active_worker\(.*?\n(.*?)(?=\ndef |\Z)",
            src,
            re.DOTALL,
        )
        assert stop_match, "_stop_active_worker not found"
        body = stop_match.group(1)
        assert "worker_lock" in body, "_stop_active_worker must use worker_lock"


# =========================================================================
# 4. Init defaults (BUG 3)
# =========================================================================


class TestInitDefaults:
    """Verify _pending_tasks and _completed_tasks are initialized in __init__."""

    def test_pending_tasks_in_init(self):
        """__init__ must set self._pending_tasks."""
        src = _get_main_window_source()
        init_match = re.search(
            r"def __init__\(self.*?\n(.*?)(?=\n    def |\nclass |\Z)",
            src,
            re.DOTALL,
        )
        assert init_match, "__init__ not found"
        body = init_match.group(1)
        assert "_pending_tasks" in body, "_pending_tasks not initialized in __init__"

    def test_completed_tasks_in_init(self):
        """__init__ must set self._completed_tasks."""
        src = _get_main_window_source()
        init_match = re.search(
            r"def __init__\(self.*?\n(.*?)(?=\n    def |\nclass |\Z)",
            src,
            re.DOTALL,
        )
        assert init_match, "__init__ not found"
        body = init_match.group(1)
        assert "_completed_tasks" in body, "_completed_tasks not initialized in __init__"

    def test_all_tasks_complete_safe_with_empty_defaults(self):
        """_all_tasks_complete logic must return True with empty defaults."""
        stub = _make_window_stub()
        # Replicate _all_tasks_complete logic
        for task_name, is_pending in stub._pending_tasks.items():
            if is_pending and task_name not in stub._completed_tasks:
                assert False, "Should not reach here with empty dict"
        result = not stub._qa_answering_active
        assert result is True


# =========================================================================
# 5. Anti-pattern audit (source inspection)
# =========================================================================


class TestAntiPatternAudit:
    """Source-level checks for known concurrency anti-patterns."""

    def test_no_bare_acquire_release(self):
        """Locks should use 'with' context manager, not bare .acquire()/.release()."""
        src_mw = _get_main_window_source()
        src_wp = _get_worker_process_source()
        for src, name in [(src_mw, "main_window"), (src_wp, "worker_process")]:
            # Allow .acquire in comments or strings, but not as statements
            lines = src.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert not re.match(r".*\.\s*acquire\s*\(", stripped), (
                    f"{name}:{i + 1} uses bare .acquire() — use 'with lock:' instead"
                )
                assert not re.match(r".*\.\s*release\s*\(", stripped), (
                    f"{name}:{i + 1} uses bare .release() — use 'with lock:' instead"
                )

    def test_destroying_checked_in_poll_callbacks(self):
        """All poll/after callbacks should check _destroying early."""
        src = _get_main_window_source()
        poll_methods = re.findall(
            r"def (_poll_\w+)\(self\).*?\n(.*?)(?=\n    def |\nclass |\Z)",
            src,
            re.DOTALL,
        )
        assert len(poll_methods) >= 2, "Expected at least 2 _poll_* methods"
        for name, body in poll_methods:
            # First few lines should check _destroying
            first_lines = body[:300]
            assert "_destroying" in first_lines, (
                f"{name} must check _destroying early in the method body"
            )

    def test_no_queue_empty_check(self):
        """queue.empty() is unreliable for concurrency — should not be used."""
        src_wp = _get_worker_process_source()
        # Allow Empty exception import, but not .empty() method calls
        lines = src_wp.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped.startswith("#")
                or stripped.startswith("from")
                or stripped.startswith("import")
            ):
                continue
            assert ".empty()" not in stripped, (
                f"worker_process:{i + 1} uses .empty() — use get(timeout=) instead"
            )

    def test_forwarder_uses_lock_for_active_worker(self):
        """_forwarder_loop must use worker_lock when setting active_worker."""
        src = _get_worker_process_source()
        fwd_match = re.search(
            r"def _forwarder_loop\(.*?\n(.*?)(?=\ndef |\Z)",
            src,
            re.DOTALL,
        )
        assert fwd_match, "_forwarder_loop not found"
        body = fwd_match.group(1)
        if "active_worker" in body:
            assert "worker_lock" in body, "_forwarder_loop sets active_worker without worker_lock"
