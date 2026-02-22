"""
Tests for resize debounce and _poll_queue behavior.

Covers:
- main_window._on_configure: debounce flag, after_cancel, child-widget filtering
- main_window._on_resize_complete: clears flag and debounce ID
- main_window._poll_queue: never calls update_idletasks (removed to prevent
  feedback loop with <Configure> events during window resize/maximize)
- dynamic_output._on_window_resize: filters child widget Configure events
- dynamic_output._refresh_display: batched summary insertion (single .insert call)
"""

import threading
from pathlib import Path
from unittest.mock import MagicMock, call

from src.ui.main_window import MainWindow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_source(module_path: str) -> str:
    """Read a source file relative to project root."""
    root = Path(__file__).parent.parent
    return (root / module_path).read_text(encoding="utf-8")


def _make_mock_window():
    """Create a mock MainWindow with the attributes needed by resize/poll logic."""
    w = MagicMock()
    # Resize debounce state
    w._resize_in_progress = False
    w._resize_debounce_id = None
    # Poll queue state
    w._destroying = False
    w._worker_manager = MagicMock()
    w._worker_manager.check_for_messages.return_value = []
    w._processing_active = False
    w._preprocessing_active = False
    w._queue_poll_id = None
    w._qa_results_lock = threading.Lock()
    w._qa_results = []
    return w


def _make_configure_event(widget, width=1200, height=750):
    """Create a fake <Configure> event."""
    event = MagicMock()
    event.widget = widget
    event.width = width
    event.height = height
    return event


# ===========================================================================
# _on_configure
# ===========================================================================


class TestOnConfigure:
    """Test the resize debounce handler."""

    def test_sets_resize_flag(self):
        """_on_configure sets _resize_in_progress to True."""
        w = _make_mock_window()
        event = _make_configure_event(widget=w)

        MainWindow._on_configure(w, event)

        assert w._resize_in_progress is True

    def test_schedules_debounce_callback(self):
        """_on_configure calls self.after(150, ...) to schedule completion."""
        w = _make_mock_window()
        w.after.return_value = "after_id_1"
        event = _make_configure_event(widget=w)

        MainWindow._on_configure(w, event)

        w.after.assert_called_once_with(150, w._on_resize_complete)
        assert w._resize_debounce_id == "after_id_1"

    def test_cancels_previous_debounce(self):
        """Rapid resizes cancel the previous after() before scheduling a new one."""
        w = _make_mock_window()
        w._resize_debounce_id = "old_id"
        w.after.return_value = "new_id"
        event = _make_configure_event(widget=w)

        MainWindow._on_configure(w, event)

        w.after_cancel.assert_called_once_with("old_id")
        assert w._resize_debounce_id == "new_id"

    def test_no_cancel_when_no_pending_debounce(self):
        """First resize event should not call after_cancel."""
        w = _make_mock_window()
        w._resize_debounce_id = None
        w.after.return_value = "first_id"
        event = _make_configure_event(widget=w)

        MainWindow._on_configure(w, event)

        w.after_cancel.assert_not_called()

    def test_ignores_child_widget_events(self):
        """Configure events from child widgets should be ignored."""
        w = _make_mock_window()
        child = MagicMock()  # Different widget
        event = _make_configure_event(widget=child)

        MainWindow._on_configure(w, event)

        assert w._resize_in_progress is False
        w.after.assert_not_called()

    def test_rapid_resize_only_one_pending(self):
        """Multiple rapid resizes should leave exactly one pending after()."""
        w = _make_mock_window()
        call_count = 0

        def mock_after(ms, callback):
            nonlocal call_count
            call_count += 1
            return f"id_{call_count}"

        w.after.side_effect = mock_after

        for i in range(10):
            event = _make_configure_event(widget=w, width=800 + i * 10)
            MainWindow._on_configure(w, event)

        # Should have cancelled 9 times (all except the first)
        assert w.after_cancel.call_count == 9
        # Final debounce ID should be the last one scheduled
        assert w._resize_debounce_id == "id_10"


# ===========================================================================
# _on_resize_complete
# ===========================================================================


class TestOnResizeComplete:
    """Test the debounce completion handler."""

    def test_clears_resize_flag(self):
        """_on_resize_complete sets _resize_in_progress to False."""
        w = _make_mock_window()
        w._resize_in_progress = True
        w._resize_debounce_id = "some_id"

        MainWindow._on_resize_complete(w)

        assert w._resize_in_progress is False

    def test_clears_debounce_id(self):
        """_on_resize_complete sets _resize_debounce_id to None."""
        w = _make_mock_window()
        w._resize_debounce_id = "some_id"

        MainWindow._on_resize_complete(w)

        assert w._resize_debounce_id is None


# ===========================================================================
# _poll_queue: must NEVER call update_idletasks
# ===========================================================================


class TestPollQueueNoIdletasks:
    """Verify _poll_queue never calls update_idletasks.

    Calling update_idletasks() from the poll loop caused a feedback loop:
    idletasks -> <Configure> events -> layout thrashing -> "(not responding)"
    during window resize/maximize. Tk already processes idle tasks between
    after() callbacks, so the forced call was both redundant and harmful.
    """

    def test_never_calls_update_idletasks(self):
        """_poll_queue should never call update_idletasks."""
        w = _make_mock_window()

        MainWindow._poll_queue(w)

        w.update_idletasks.assert_not_called()

    def test_no_update_idletasks_with_messages(self):
        """_poll_queue should not call update_idletasks even after processing messages."""
        w = _make_mock_window()
        w._worker_manager.check_for_messages.return_value = [("status", "Processing...")]
        w._processing_active = True

        MainWindow._poll_queue(w)

        w._handle_queue_message.assert_called_once()
        w.update_idletasks.assert_not_called()

    def test_queue_messages_processed_during_resize(self):
        """Messages in the queue should still be handled even during resize."""
        w = _make_mock_window()
        w._resize_in_progress = True
        w._worker_manager.check_for_messages.return_value = [("status", "Processing file 1...")]
        w._processing_active = True

        MainWindow._poll_queue(w)

        w._handle_queue_message.assert_called_once_with("status", "Processing file 1...")
        w.update_idletasks.assert_not_called()

    def test_no_update_idletasks_in_source(self):
        """The _poll_queue method should not call update_idletasks."""
        source = _read_source("src/ui/main_window.py")
        start = source.index("def _poll_queue(self)")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert ".update_idletasks()" not in body


# ===========================================================================
# No update_idletasks in activity indicator or queue handler
# ===========================================================================


class TestNoIdletasksInHotPaths:
    """Verify update_idletasks was removed from frequently-called code paths."""

    def test_no_idletasks_in_start_activity_indicator(self):
        """_start_activity_indicator should not call update_idletasks."""
        source = _read_source("src/ui/main_window.py")
        start = source.index("def _start_activity_indicator")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        # Check for actual call, not the word in a comment
        assert ".update_idletasks()" not in body

    def test_no_idletasks_in_timer_mixin_start(self):
        """Timer mixin _start_activity_indicator should not call update_idletasks."""
        source = _read_source("src/ui/main_window_helpers/timer_mixin.py")
        start = source.index("def _start_activity_indicator")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert ".update_idletasks()" not in body


# ===========================================================================
# DynamicOutputWidget: Configure event filtering
# ===========================================================================


class TestDynamicOutputConfigureFilter:
    """Verify DynamicOutputWidget filters child widget Configure events."""

    def test_on_window_resize_filters_child_events(self):
        """_on_window_resize should check event.widget is self."""
        source = _read_source("src/ui/dynamic_output.py")
        start = source.index("def _on_window_resize")
        next_def = source.index("\n    def ", start + 1)
        body = source[start:next_def]
        assert "event.widget is not self" in body

    def test_configure_binding_exists(self):
        """DynamicOutputWidget should bind <Configure> event."""
        source = _read_source("src/ui/dynamic_output.py")
        assert '"<Configure>", self._on_window_resize' in source


# ===========================================================================
# Init state: resize attributes exist
# ===========================================================================


class TestResizeInitState:
    """Verify __init__ sets the resize debounce attributes."""

    def test_init_has_resize_attributes(self):
        """MainWindow.__init__ should set resize debounce state variables."""
        import inspect

        source = inspect.getsource(MainWindow.__init__)

        assert "_resize_in_progress" in source
        assert "_resize_debounce_id" in source

    def test_init_does_not_have_last_idletasks(self):
        """MainWindow.__init__ should NOT have _last_idletasks (removed)."""
        import inspect

        source = inspect.getsource(MainWindow.__init__)

        assert "_last_idletasks" not in source

    def test_init_binds_configure(self):
        """MainWindow.__init__ should bind <Configure> event."""
        import inspect

        source = inspect.getsource(MainWindow.__init__)

        assert '"<Configure>"' in source or "'<Configure>'" in source
        assert "_on_configure" in source


# ===========================================================================
# Batched summary insertion in dynamic_output.py
# ===========================================================================


class TestBatchedSummaryInsertion:
    """Test that document summaries are inserted with a single .insert() call."""

    def _make_mock_output(self):
        """Create a mock DynamicOutputWidget with attributes for _refresh_tabs."""
        from src.ui.dynamic_output import DynamicOutputWidget

        w = DynamicOutputWidget.__new__(DynamicOutputWidget)
        w._outputs = {}
        w._document_summaries = {}
        w._extraction_source = None
        w.summary_text_display = MagicMock()
        w.tabview = MagicMock()
        w.tabview.get.return_value = "Summary"
        w._qa_status_label = MagicMock()
        w._summary_status_label = MagicMock()
        w._workflow_phase = MagicMock()
        w.winfo_toplevel = MagicMock()
        w.show_summary_content = MagicMock()
        w._display_csv = MagicMock()
        w._display_qa_results = MagicMock()
        w._update_progress_badge = MagicMock()
        return w

    def _call_refresh(self, w):
        """Call _refresh_tabs on the mock widget."""
        from src.ui.dynamic_output import DynamicOutputWidget

        DynamicOutputWidget._refresh_tabs(w)

    def test_single_insert_for_multiple_documents(self):
        """Multiple document summaries should use exactly one .insert() call."""
        w = self._make_mock_output()
        w._outputs = {"Summary": "Overall summary text."}
        w._document_summaries = {
            "complaint.pdf": "The plaintiff alleges negligence.",
            "answer.pdf": "The defendant denies all claims.",
            "motion.pdf": "Motion for summary judgment.",
        }

        self._call_refresh(w)

        # Summary tab: one delete + one insert for main summary + one insert for docs
        insert_calls = w.summary_text_display.insert.call_args_list

        # First insert: the main summary at "0.0"
        assert insert_calls[0] == call("0.0", "Overall summary text.")

        # Second insert: all document summaries batched into one call at "end"
        doc_insert = insert_calls[1]
        assert doc_insert[0][0] == "end"
        inserted_text = doc_insert[0][1]

        # Verify it's a single string containing all documents
        assert "INDIVIDUAL DOCUMENT SUMMARIES" in inserted_text
        assert "complaint.pdf:" in inserted_text
        assert "answer.pdf:" in inserted_text
        assert "motion.pdf:" in inserted_text
        assert "The plaintiff alleges negligence." in inserted_text

        # Only 2 insert calls total (main summary + batched docs)
        assert len(insert_calls) == 2

    def test_no_insert_when_no_document_summaries(self):
        """No doc summary insert when _document_summaries is empty."""
        w = self._make_mock_output()
        w._outputs = {"Summary": "Overall summary."}
        w._document_summaries = {}

        self._call_refresh(w)

        insert_calls = w.summary_text_display.insert.call_args_list
        # Only the main summary insert
        assert len(insert_calls) == 1
        assert insert_calls[0] == call("0.0", "Overall summary.")

    def test_document_summaries_sorted_alphabetically(self):
        """Document summaries should appear in alphabetical order."""
        w = self._make_mock_output()
        w._outputs = {}
        w._document_summaries = {
            "zebra.pdf": "Z summary",
            "alpha.pdf": "A summary",
            "middle.pdf": "M summary",
        }

        self._call_refresh(w)

        insert_calls = w.summary_text_display.insert.call_args_list
        assert len(insert_calls) == 1  # Only the batched doc insert (no main summary)
        inserted_text = insert_calls[0][0][1]

        # Verify alphabetical order
        alpha_pos = inserted_text.index("alpha.pdf")
        middle_pos = inserted_text.index("middle.pdf")
        zebra_pos = inserted_text.index("zebra.pdf")
        assert alpha_pos < middle_pos < zebra_pos

    def test_batched_text_has_separator_and_header(self):
        """Batched insert should contain separator lines and header."""
        w = self._make_mock_output()
        w._outputs = {}
        w._document_summaries = {"doc.pdf": "Summary content"}

        self._call_refresh(w)

        inserted_text = w.summary_text_display.insert.call_args_list[0][0][1]
        assert "=" * 50 in inserted_text
        assert "INDIVIDUAL DOCUMENT SUMMARIES" in inserted_text
        assert "doc.pdf:\nSummary content" in inserted_text
