"""
Shared test fixtures for the CasePrepd test suite.

Resets module-level singletons before every test so that no test
inherits leftover state from a previous test.

IMPORTANT: Tests must be run with the project venv activated (.venv\\Scripts\\activate).
The system Python (3.13) differs from the venv Python (3.12) and is missing project
dependencies. Running with the wrong interpreter causes ModuleNotFoundError for
packages like nupunkt and tkinterdnd2.
"""

import pytest

# ---------------------------------------------------------------------------
# Progress milestones — writes to stderr so output is live even with -q
# Prints every 10% plus timestamps so you can tell if tests are hung
# ---------------------------------------------------------------------------


class ProgressMilestones:
    """Prints progress every ~10% to stderr with timestamps."""

    def __init__(self):
        """Initialize progress tracker."""
        self.total = 0
        self.count = 0
        self.next_pct = 10
        self.start_time = None
        self.last_test = ""

    def pytest_collection_modifyitems(self, items):
        """Record total test count and start time after collection."""
        import time

        self.total = len(items)
        self.start_time = time.time()
        self._log(f"Collected {self.total} tests")

    def pytest_runtest_logreport(self, report):
        """Print a milestone line when crossing each 10% boundary."""
        if report.when != "call" or self.total == 0:
            return
        self.count += 1
        self.last_test = report.nodeid
        pct = (self.count * 100) // self.total
        if pct >= self.next_pct:
            elapsed = self._elapsed()
            self._log(
                f"~{self.next_pct}% ({self.count}/{self.total}) "
                f"[{elapsed}] last: {report.nodeid.split('::')[-1]}"
            )
            self.next_pct += 10

    def _elapsed(self):
        """Return elapsed time as M:SS string."""
        import time

        if self.start_time is None:
            return "0:00"
        secs = int(time.time() - self.start_time)
        return f"{secs // 60}:{secs % 60:02d}"

    def _log(self, msg):
        """Write directly to stderr to bypass pytest capture."""
        import sys

        sys.stderr.write(f">>> {msg}\n")
        sys.stderr.flush()


def pytest_configure(config):
    """Register the progress milestone plugin."""
    config.pluginmanager.register(ProgressMilestones(), "progress_milestones")


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all module-level singletons before and after each test."""
    _do_reset()
    yield
    _do_reset()


def _do_reset():
    """Clear cached singleton instances across the codebase."""
    # UserPreferencesManager -- holds user settings (context size, GPU, etc.)
    try:
        from src.user_preferences import reset_singleton as reset_prefs

        reset_prefs()
    except ImportError:
        pass

    # ExportService -- singleton for Word/PDF/HTML export
    try:
        from src.services.export_service import reset_export_service

        reset_export_service()
    except ImportError:
        pass

    # DefaultQuestionsManager -- holds default semantic questions
    try:
        from src.core.semantic.default_questions_manager import reset_singleton as reset_questions

        reset_questions()
    except ImportError:
        pass
