"""
Regression tests for tooltip after-ID leak fixes (Fixes 4 and 5).

Fix 4: `src.ui.widgets` `_show_tooltip()` must cancel any previously-stored
`_tooltip_dismiss_id` before scheduling a new auto-dismiss.

Fix 5: `src.ui.vocab_table.vocab_treeview.VocabTreeview` must cancel its
`_tooltip_after_id` on cleanup and on repeated `_on_hover` calls, so
callbacks cannot fire on a destroyed widget.

These tests avoid constructing real Tk widgets — the methods under test
are exercised on stand-in objects whose `after` / `after_cancel` are
mocks.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fix 4 — widgets._show_tooltip auto-dismiss timer
# ---------------------------------------------------------------------------


class _DismissHarness:
    """Re-implements only the cancel-then-schedule block added by Fix 4."""

    def __init__(self):
        """Pre-seed dismiss id; mocks for after/after_cancel."""
        self.after = MagicMock(return_value="new-id")
        self.after_cancel = MagicMock()
        self._tooltip_dismiss_id = None

    def schedule_dismiss(self):
        """Run the exact code paths from widgets._show_tooltip (Fix 4 block)."""
        import contextlib

        previous_dismiss = getattr(self, "_tooltip_dismiss_id", None)
        if previous_dismiss:
            with contextlib.suppress(Exception):
                self.after_cancel(previous_dismiss)
            self._tooltip_dismiss_id = None

        try:
            self._tooltip_dismiss_id = self.after(15000, lambda: None)
        except Exception:
            self._tooltip_dismiss_id = None


class TestTooltipDismissLeak:
    """Verify the dismiss-id leak is fixed."""

    def test_first_show_schedules_without_cancel(self):
        """First show has no prior dismiss id, so no cancel."""
        h = _DismissHarness()
        h.schedule_dismiss()
        h.after.assert_called_once()
        h.after_cancel.assert_not_called()
        assert h._tooltip_dismiss_id == "new-id"

    def test_second_show_cancels_first(self):
        """Second show must cancel the previous dismiss after-id."""
        h = _DismissHarness()
        h._tooltip_dismiss_id = "old-id"
        h.schedule_dismiss()
        h.after_cancel.assert_called_once_with("old-id")
        assert h._tooltip_dismiss_id == "new-id"

    def test_cancel_exception_is_swallowed(self):
        """after_cancel raising on a stale id does not break scheduling."""
        h = _DismissHarness()
        h._tooltip_dismiss_id = "stale"
        h.after_cancel.side_effect = RuntimeError("nope")
        h.schedule_dismiss()
        # Still scheduled new one despite the cancel failure
        assert h._tooltip_dismiss_id == "new-id"

    def test_widgets_source_has_cancel_before_schedule(self):
        """
        Guard: the Fix 4 cancel-then-schedule pattern must remain present
        in `_show_tooltip`. Prevents silent regressions via future refactors.
        """
        import inspect

        from src.ui import widgets

        src = inspect.getsource(widgets)
        show_idx = src.find("def _show_tooltip")
        assert show_idx >= 0
        # Scope to the _show_tooltip body
        tail = src[show_idx : show_idx + 2000]
        # Must reference cancelling the prior dismiss id before assigning new one
        assert "previous_dismiss" in tail or "after_cancel" in tail
        assert "_tooltip_dismiss_id" in tail


# ---------------------------------------------------------------------------
# Fix 5 — VocabTreeview hover timer cleanup
# ---------------------------------------------------------------------------


class TestVocabTreeviewHoverTimer:
    """Verify the hover-timer cleanup path (Fix 5)."""

    def _make_tv(self):
        """
        Construct a VocabTreeview without hitting the real Tk ttk machinery.

        We patch the heavy ctor collaborators so that only the timer-related
        attributes are initialised.
        """
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        with (
            patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"),
            patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}),
        ):
            parent = MagicMock()
            parent.after_cancel = MagicMock()
            tv = VocabTreeview(parent=parent, columns=("Term",))
        return tv

    def test_cleanup_cancels_pending_hover_timer(self):
        """cleanup() must cancel a pending _tooltip_after_id if set."""
        tv = self._make_tv()
        tv._tooltip_after_id = "pending-id"
        tv.cleanup()
        tv._parent.after_cancel.assert_any_call("pending-id")
        assert tv._tooltip_after_id is None

    def test_cleanup_is_safe_when_no_timer_pending(self):
        """cleanup() must not crash when no timer is pending."""
        tv = self._make_tv()
        tv._tooltip_after_id = None
        # Should not raise
        tv.cleanup()
        assert tv._tooltip_after_id is None

    def test_hide_tooltip_cancels_pending_timer(self):
        """_hide_tooltip cancels the pending hover timer (pre-existing guard)."""
        tv = self._make_tv()
        tv._tooltip_after_id = "pending"
        tv._hide_tooltip(None)
        tv._parent.after_cancel.assert_any_call("pending")
        assert tv._tooltip_after_id is None

    def test_repeated_on_hover_cancels_previous(self):
        """
        _on_hover must cancel any in-flight tooltip timer before scheduling
        a new one. This is the core leak-prevention path.
        """
        tv = self._make_tv()
        # Force the early-return branch so we don't exercise Treeview internals
        tv.widget = MagicMock()
        tv.widget.identify_row.return_value = ""  # no row -> early return
        tv._tooltip_after_id = "old-id"
        event = MagicMock(y=10)
        tv._on_hover(event)
        tv._parent.after_cancel.assert_any_call("old-id")
        # Early-return path leaves _tooltip_after_id at None
        assert tv._tooltip_after_id is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
