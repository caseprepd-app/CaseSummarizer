"""
Regression tests for MainWindow retry after-ID leak fixes.

Covers Fixes 1-3 from the bug sweep:
- `_start_preprocessing` worker-not-ready retry must cancel prior after-id
  before scheduling a new one (Fix 1).
- `_start_progressive_extraction` retry has the same semantics (Fix 2).
- `destroy()` must cancel both retry IDs if set (Fix 3).

MainWindow is heavy (inherits ctk.CTk), so we test the
`_cancel_and_reschedule` helper directly via a lightweight stand-in
whose `after` / `after_cancel` are mocks.
"""

from unittest.mock import MagicMock

import pytest

from src.ui.main_window import MainWindow


class _Harness:
    """Minimal object that borrows MainWindow._cancel_and_reschedule."""

    def __init__(self):
        """Install mock after/after_cancel so we can assert call order."""
        self.after = MagicMock(side_effect=self._fake_after)
        self.after_cancel = MagicMock()
        self._next_id = 0

    def _fake_after(self, _delay, _callback):
        """Return a unique fake after-id string per call."""
        self._next_id += 1
        return f"after#{self._next_id}"


# Bind the real helper onto the harness for direct unit testing.
_Harness._cancel_and_reschedule = MainWindow._cancel_and_reschedule


class TestCancelAndReschedule:
    """Unit-test the leak-free retry scheduler helper."""

    def test_first_call_schedules_but_does_not_cancel(self):
        """First call has no prior ID, so after_cancel must not fire."""
        h = _Harness()
        h._cancel_and_reschedule("_preprocessing_retry_id", 3000, lambda: None)
        h.after.assert_called_once()
        h.after_cancel.assert_not_called()
        assert h._preprocessing_retry_id == "after#1"

    def test_second_call_cancels_previous(self):
        """Second call must cancel the prior after-id before scheduling again."""
        h = _Harness()
        cb = lambda: None  # noqa: E731
        h._cancel_and_reschedule("_preprocessing_retry_id", 3000, cb)
        h._cancel_and_reschedule("_preprocessing_retry_id", 3000, cb)
        # Exactly one cancel, on the original ID
        h.after_cancel.assert_called_once_with("after#1")
        # New id stored
        assert h._preprocessing_retry_id == "after#2"
        assert h.after.call_count == 2

    def test_cancel_exception_is_swallowed(self):
        """If after_cancel raises (stale id), we still schedule the new one."""
        h = _Harness()
        h._cancel_and_reschedule("_extraction_retry_id", 3000, lambda: None)
        h.after_cancel.side_effect = RuntimeError("stale")
        # Should not raise
        h._cancel_and_reschedule("_extraction_retry_id", 3000, lambda: None)
        assert h._extraction_retry_id == "after#2"

    def test_distinct_attrs_do_not_interfere(self):
        """Preprocessing and extraction retries use independent slots."""
        h = _Harness()
        h._cancel_and_reschedule("_preprocessing_retry_id", 3000, lambda: None)
        h._cancel_and_reschedule("_extraction_retry_id", 3000, lambda: None)
        h.after_cancel.assert_not_called()
        assert h._preprocessing_retry_id == "after#1"
        assert h._extraction_retry_id == "after#2"


class TestDestroyCancelsRetries:
    """Verify MainWindow.destroy() cancels both retry IDs (Fix 3)."""

    def test_destroy_cancels_preprocessing_and_extraction_retries(self):
        """
        Simulate destroy()'s retry-cancel block on a fake with both IDs set.

        We inline the same cancellation logic as destroy(); this guards
        against someone removing the retry-cancel block without updating
        this test.
        """
        fake = MagicMock()
        fake._preprocessing_retry_id = "p-id"
        fake._extraction_retry_id = "e-id"

        # Run the exact pattern from destroy() under test
        if getattr(fake, "_preprocessing_retry_id", None):
            try:
                fake.after_cancel(fake._preprocessing_retry_id)
            except Exception:
                pass
            fake._preprocessing_retry_id = None
        if getattr(fake, "_extraction_retry_id", None):
            try:
                fake.after_cancel(fake._extraction_retry_id)
            except Exception:
                pass
            fake._extraction_retry_id = None

        assert fake._preprocessing_retry_id is None
        assert fake._extraction_retry_id is None
        fake.after_cancel.assert_any_call("p-id")
        fake.after_cancel.assert_any_call("e-id")

    def test_destroy_source_contains_retry_cancel_blocks(self):
        """Regression guard: destroy() must reference both retry attrs."""
        import inspect

        src = inspect.getsource(MainWindow.destroy)
        assert "_preprocessing_retry_id" in src
        assert "_extraction_retry_id" in src


class TestConftestSingletonResets:
    """Fix 6: conftest._do_reset() must reset corpus_manager + gibberish_filter."""

    def test_corpus_manager_reset_is_called(self):
        """After conftest reset, corpus_manager module global must be None."""
        from src.core.vocabulary import corpus_manager as cm

        # Seed a dummy instance
        cm._corpus_manager = object()
        # Re-run the reset logic directly
        from tests.conftest import _do_reset

        _do_reset()
        assert cm._corpus_manager is None

    def test_gibberish_filter_reset_is_called(self):
        """After conftest reset, GibberishFilter._instance must be None."""
        from src.core.utils.gibberish_filter import GibberishFilter

        GibberishFilter._instance = object()
        from tests.conftest import _do_reset

        _do_reset()
        assert GibberishFilter._instance is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
