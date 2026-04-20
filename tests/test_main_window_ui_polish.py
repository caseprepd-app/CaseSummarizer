"""
UI polish regression tests for MainWindow.

Covers two small bugs:
- Fix A: Search button must disable itself while the search entry is empty,
  so the user can't click into the "no question" warning path.
- Fix B: The "no valid files" messagebox shown by _perform_tasks must
  distinguish (no files added) from (files still preparing) from
  (every file failed preprocessing).

Both fixes are tested without a real Tk root. Fix A's state-picker is
MainWindow._update_followup_btn_state called against a lightweight stand-in.
Fix B's message-picker is the pure module-level helper
`_pick_no_valid_files_msg`.
"""

from unittest.mock import MagicMock

import pytest

from src.ui.main_window import MainWindow, _pick_no_valid_files_msg

# ---------------------------------------------------------------------------
# Fix A — follow-up button state helper
# ---------------------------------------------------------------------------


class _FollowupHarness:
    """Stand-in MainWindow exposing only the attrs the helper touches."""

    def __init__(self, entry_text: str, entry_state: str = "normal"):
        """Install mock entry/button so configure() calls are observable."""
        self.followup_entry = MagicMock()
        self.followup_entry.get.return_value = entry_text
        self.followup_entry.cget.return_value = entry_state
        self.followup_btn = MagicMock()


# Borrow the real method for direct unit testing.
_FollowupHarness._update_followup_btn_state = MainWindow._update_followup_btn_state


class TestUpdateFollowupBtnState:
    """Fix A: entry emptiness drives the Search button's enabled state."""

    def test_empty_entry_disables_button(self):
        """An empty entry must force the button to 'disabled'."""
        h = _FollowupHarness(entry_text="")
        h._update_followup_btn_state()
        h.followup_btn.configure.assert_called_once_with(state="disabled")

    def test_text_entry_enables_button(self):
        """Non-empty entry text must set the button to 'normal'."""
        h = _FollowupHarness(entry_text="who signed the lease?")
        h._update_followup_btn_state()
        h.followup_btn.configure.assert_called_once_with(state="normal")

    def test_whitespace_only_counts_as_empty(self):
        """A field containing only spaces/tabs must still disable the button."""
        h = _FollowupHarness(entry_text="   \t  ")
        h._update_followup_btn_state()
        h.followup_btn.configure.assert_called_once_with(state="disabled")

    def test_disabled_entry_is_left_alone(self):
        """If the entry itself is disabled (no index yet), don't touch the button."""
        h = _FollowupHarness(entry_text="", entry_state="disabled")
        h._update_followup_btn_state()
        h.followup_btn.configure.assert_not_called()

    def test_missing_widgets_is_safe_noop(self):
        """Before widgets are created, calling the helper must not raise."""

        class _Bare:
            pass

        b = _Bare()
        # Should not raise even with no attrs at all.
        MainWindow._update_followup_btn_state(b)

    def test_accepts_event_arg(self):
        """Bindings pass an event; the helper must accept it as a positional arg."""
        h = _FollowupHarness(entry_text="x")
        fake_event = object()
        h._update_followup_btn_state(fake_event)
        h.followup_btn.configure.assert_called_once_with(state="normal")


# ---------------------------------------------------------------------------
# Fix B — _pick_no_valid_files_msg
# ---------------------------------------------------------------------------


class TestPickNoValidFilesMsg:
    """Fix B: pick the right title/message for each empty-results situation."""

    def test_no_files_selected(self):
        """Nothing added yet → prompt the user to add files."""
        title, msg = _pick_no_valid_files_msg([], [])
        assert title == "No Files"
        assert "add files" in msg.lower()

    def test_files_selected_but_no_results_yet(self):
        """Files added, preprocessing still running → tell user to wait."""
        title, msg = _pick_no_valid_files_msg(["a.pdf", "b.pdf"], [])
        assert title == "Still Preparing Files"
        assert "wait" in msg.lower()

    def test_files_selected_all_failed(self):
        """Results present but every file failed → count the failures."""
        results = [
            {"filename": "a.pdf", "status": "failed"},
            {"filename": "b.pdf", "status": "failed"},
        ]
        title, msg = _pick_no_valid_files_msg(["a.pdf", "b.pdf"], results)
        assert title == "All Files Failed"
        assert "2 of 2" in msg

    def test_mixed_statuses_still_reports_as_all_failed(self):
        """
        Helper only runs when caller already saw no successes, so any status
        that isn't "success" should be counted as failed in the message.
        """
        results = [
            {"filename": "a.pdf", "status": "failed"},
            {"filename": "b.pdf", "status": "pending"},
            {"filename": "c.pdf"},  # Missing status entirely
        ]
        title, msg = _pick_no_valid_files_msg(["a.pdf", "b.pdf", "c.pdf"], results)
        assert title == "All Files Failed"
        assert "3 of 3" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
