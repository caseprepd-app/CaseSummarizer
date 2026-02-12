"""
Tests for GUI polish features (7 visual improvements).

Covers:
1. FileReviewTable empty state overlay (show/hide)
2. Determinate progress bar (_update_progress)
3. Drag-and-drop visual feedback (border highlight)
4. PipelineIndicator widget (state transitions)
5. Success celebration (status bar flash)
6. Hover previews (result data storage, tooltip text)
7. Tab transition animation
"""

from unittest.mock import MagicMock

# =========================================================================
# Feature 1: Empty State Overlay
# =========================================================================


class TestFileReviewTableEmptyState:
    """Test the empty state overlay in FileReviewTable."""

    def _make_table(self):
        """Create a FileReviewTable with manually set attributes (skips __init__)."""
        from src.ui.widgets import FileReviewTable

        table = FileReviewTable.__new__(FileReviewTable)
        table.column_map = {
            "include": ("Include", 50),
            "filename": ("Filename", 300),
            "status": ("Status", 100),
            "method": ("Method", 100),
            "confidence": ("Confidence", 100),
            "pages": ("Pages", 50),
            "size": ("Size", 80),
        }
        table.file_item_map = {}
        table._result_data = {}
        table._hovered_row = None
        table._tooltip_window = None
        table._drop_zone = MagicMock()
        table._drop_zone.winfo_ismapped.return_value = True
        table.tree = MagicMock()
        table.tree.identify_row.return_value = ""
        return table

    def test_drop_zone_hidden_on_first_file(self):
        """Empty state overlay should hide when first file is added."""
        table = self._make_table()
        table.tree.insert.return_value = "item1"

        result = {
            "filename": "test.pdf",
            "status": "success",
            "confidence": 90,
            "method": "digital",
            "page_count": 5,
            "file_size": 1024,
        }

        table.add_result(result)
        table._drop_zone.place_forget.assert_called_once()

    def test_drop_zone_not_hidden_on_update(self):
        """Overlay should not re-hide when updating existing file."""
        table = self._make_table()
        table.file_item_map["test.pdf"] = "item1"
        table._drop_zone.winfo_ismapped.return_value = False

        result = {
            "filename": "test.pdf",
            "status": "success",
            "confidence": 95,
            "method": "digital",
            "page_count": 5,
            "file_size": 1024,
        }

        table.add_result(result)
        table._drop_zone.place_forget.assert_not_called()

    def test_drop_zone_shown_on_clear(self):
        """Empty state overlay should reappear after clearing all files."""
        table = self._make_table()
        table.file_item_map["test.pdf"] = "item1"
        table.tree.get_children.return_value = ["item1"]

        table.clear()

        table._drop_zone.place.assert_called_once_with(
            relx=0.5, rely=0.5, anchor="center", relwidth=0.85, relheight=0.7
        )

    def test_clear_resets_result_data(self):
        """Clear should also reset hover preview data."""
        table = self._make_table()
        table._result_data["test.pdf"] = {"filename": "test.pdf"}
        table.tree.get_children.return_value = []

        table.clear()

        assert table._result_data == {}
        assert table.file_item_map == {}


# =========================================================================
# Feature 2: Determinate Progress Bar
# =========================================================================


class TestDeterminateProgressBar:
    """Test the _update_progress method in TimerMixin."""

    def _make_mixin(self):
        """Create a TimerMixin instance with mocked widgets."""
        from src.ui.main_window_helpers.timer_mixin import TimerMixin

        mixin = TimerMixin.__new__(TimerMixin)
        mixin.progress_bar = MagicMock()
        mixin._progress_bar_visible = False
        mixin.activity_indicator = MagicMock()
        mixin._activity_indicator_visible = False
        mixin.update_idletasks = MagicMock()
        return mixin

    def test_update_progress_sets_bar_value(self):
        """_update_progress should set progress bar to percentage/100."""
        mixin = self._make_mixin()
        mixin._update_progress(50.0)
        mixin.progress_bar.set.assert_called_once_with(0.5)

    def test_update_progress_clamps_to_zero(self):
        """Negative percentage should clamp to 0."""
        mixin = self._make_mixin()
        mixin._update_progress(-10.0)
        mixin.progress_bar.set.assert_called_once_with(0.0)

    def test_update_progress_clamps_to_one(self):
        """Percentage over 100 should clamp to 1.0."""
        mixin = self._make_mixin()
        mixin._update_progress(150.0)
        mixin.progress_bar.set.assert_called_once_with(1.0)

    def test_update_progress_zero(self):
        """Zero percentage should set bar to 0."""
        mixin = self._make_mixin()
        mixin._update_progress(0.0)
        mixin.progress_bar.set.assert_called_once_with(0.0)

    def test_update_progress_hundred(self):
        """100% should set bar to 1.0."""
        mixin = self._make_mixin()
        mixin._update_progress(100.0)
        mixin.progress_bar.set.assert_called_once_with(1.0)

    def test_start_activity_indicator_shows_progress_bar(self):
        """Starting activity should also show and reset the progress bar."""
        mixin = self._make_mixin()
        mixin._start_activity_indicator()

        mixin.progress_bar.set.assert_called_once_with(0)
        mixin.progress_bar.pack.assert_called_once()
        assert mixin._progress_bar_visible is True

    def test_stop_activity_indicator_hides_progress_bar(self):
        """Stopping activity should also hide the progress bar."""
        mixin = self._make_mixin()
        mixin._progress_bar_visible = True
        mixin._activity_indicator_visible = True

        mixin._stop_activity_indicator()

        mixin.progress_bar.pack_forget.assert_called_once()
        assert mixin._progress_bar_visible is False


# =========================================================================
# Feature 4: Pipeline Step Indicator
# =========================================================================


class TestPipelineIndicator:
    """Test the PipelineIndicator widget state management."""

    def _make_indicator(self):
        """Create a PipelineIndicator with manually set attributes (skips __init__)."""
        from src.ui.pipeline_indicator import PIPELINE_STEPS, PipelineIndicator

        indicator = PipelineIndicator.__new__(PipelineIndicator)
        indicator._step_labels = {}
        indicator._step_states = {}

        for step in PIPELINE_STEPS:
            indicator._step_labels[step] = MagicMock()
            indicator._step_states[step] = "pending"

        return indicator

    def test_initial_state_all_pending(self):
        """All steps should start as pending."""
        indicator = self._make_indicator()
        for step in ["Extract", "Vocabulary", "Q&A", "Summary"]:
            assert indicator._step_states[step] == "pending"

    def test_set_step_active(self):
        """Setting a step to active should update its state."""
        indicator = self._make_indicator()
        indicator.set_step_state("Extract", "active")

        assert indicator._step_states["Extract"] == "active"
        indicator._step_labels["Extract"].configure.assert_called()

    def test_set_step_done(self):
        """Setting a step to done should update state and show checkmark."""
        indicator = self._make_indicator()
        indicator.set_step_state("Extract", "done")

        assert indicator._step_states["Extract"] == "done"
        # Verify the label was configured with checkmark text
        call_kwargs = indicator._step_labels["Extract"].configure.call_args
        assert "\u2713 Extract" in str(call_kwargs)

    def test_set_step_skipped(self):
        """Setting a step to skipped should update its state."""
        indicator = self._make_indicator()
        indicator.set_step_state("Q&A", "skipped")
        assert indicator._step_states["Q&A"] == "skipped"

    def test_set_enabled_steps(self):
        """set_enabled_steps should mark enabled as pending, rest as skipped."""
        indicator = self._make_indicator()
        indicator.set_enabled_steps(["Extract", "Vocabulary"])

        assert indicator._step_states["Extract"] == "pending"
        assert indicator._step_states["Vocabulary"] == "pending"
        assert indicator._step_states["Q&A"] == "skipped"
        assert indicator._step_states["Summary"] == "skipped"

    def test_set_enabled_steps_all(self):
        """Enabling all steps should set all to pending."""
        indicator = self._make_indicator()
        # First set some to skipped
        indicator.set_step_state("Q&A", "skipped")
        # Now enable all
        indicator.set_enabled_steps(["Extract", "Vocabulary", "Q&A", "Summary"])

        for step in ["Extract", "Vocabulary", "Q&A", "Summary"]:
            assert indicator._step_states[step] == "pending"

    def test_reset(self):
        """reset() should set all steps back to pending."""
        indicator = self._make_indicator()
        indicator.set_step_state("Extract", "done")
        indicator.set_step_state("Vocabulary", "active")
        indicator.set_step_state("Q&A", "skipped")

        indicator.reset()

        for step in ["Extract", "Vocabulary", "Q&A", "Summary"]:
            assert indicator._step_states[step] == "pending"

    def test_set_invalid_step_no_error(self):
        """Setting state on non-existent step should not raise."""
        indicator = self._make_indicator()
        # Should not raise
        indicator.set_step_state("NonExistent", "active")

    def test_pipeline_steps_constant(self):
        """Verify the expected pipeline step names."""
        from src.ui.pipeline_indicator import PIPELINE_STEPS

        assert PIPELINE_STEPS == ["Extract", "Vocabulary", "Q&A", "Summary"]

    def test_state_transitions(self):
        """Test a typical pipeline progression."""
        indicator = self._make_indicator()

        # Start: Enable vocab + Q&A
        indicator.set_enabled_steps(["Extract", "Vocabulary", "Q&A"])
        assert indicator._step_states["Summary"] == "skipped"

        # Extract starts
        indicator.set_step_state("Extract", "active")
        assert indicator._step_states["Extract"] == "active"

        # Extract done, Vocabulary starts
        indicator.set_step_state("Extract", "done")
        indicator.set_step_state("Vocabulary", "active")
        assert indicator._step_states["Extract"] == "done"
        assert indicator._step_states["Vocabulary"] == "active"

        # Vocabulary done, Q&A starts
        indicator.set_step_state("Vocabulary", "done")
        indicator.set_step_state("Q&A", "active")
        assert indicator._step_states["Vocabulary"] == "done"
        assert indicator._step_states["Q&A"] == "active"

        # Q&A done
        indicator.set_step_state("Q&A", "done")
        assert indicator._step_states["Q&A"] == "done"


# =========================================================================
# Feature 5: Success Celebration
# =========================================================================


class TestSuccessCelebration:
    """Test the success flash on task completion."""

    def _make_task_mixin(self):
        """Create a TaskMixin with mocked widgets."""
        from src.ui.main_window_helpers.task_mixin import TaskMixin

        mixin = TaskMixin.__new__(TaskMixin)
        mixin.status_frame = MagicMock()
        mixin.status_label = MagicMock()
        mixin.add_files_btn = MagicMock()
        mixin.generate_btn = MagicMock()
        mixin.qa_check = MagicMock()
        mixin.qa_check.get.return_value = False
        mixin.vocab_check = MagicMock()
        mixin.vocab_check.get.return_value = True
        mixin.summary_check = MagicMock()
        mixin.summary_check.get.return_value = False
        mixin.processing_results = [{"filename": "test.pdf"}]
        mixin.export_all_btn = MagicMock()
        mixin._export_all_visible = False
        mixin._qa_ready = False
        mixin._qa_results = []
        mixin._processing_start_time = 100.0
        mixin.task_preview_label = MagicMock()
        mixin.output_display = MagicMock()
        mixin.output_display._vocab_csv_data = None
        mixin.followup_btn = MagicMock()
        mixin.followup_entry = MagicMock()
        mixin.after = MagicMock()
        mixin.stats_label = MagicMock()
        mixin.pipeline_indicator = MagicMock()
        mixin.pipeline_indicator._step_states = {}

        # Mock _stop_timer and _update_generate_button_state
        mixin._stop_timer = MagicMock()
        mixin._update_generate_button_state = MagicMock()
        mixin._gather_extraction_stats = MagicMock(return_value={})
        mixin._update_session_stats = MagicMock()

        return mixin

    def test_success_sets_green_background(self):
        """On success, status bar should flash green."""
        mixin = self._make_task_mixin()
        mixin._on_tasks_complete(True, "Completed 2 task(s)")

        # Status frame should have green background
        mixin.status_frame.configure.assert_called()
        call_kwargs = mixin.status_frame.configure.call_args[1]
        assert call_kwargs["fg_color"] == "#1a3a1a"  # monitor_bg

    def test_success_shows_checkmark(self):
        """On success, status text should have checkmark prefix."""
        mixin = self._make_task_mixin()
        mixin._on_tasks_complete(True, "Completed 2 task(s)")

        mixin.status_label.configure.assert_called()
        call_kwargs = mixin.status_label.configure.call_args[1]
        assert call_kwargs["text"] == "\u2713 Completed 2 task(s)"

    def test_success_schedules_restore(self):
        """On success, should schedule restore after 2 seconds."""
        mixin = self._make_task_mixin()
        mixin._on_tasks_complete(True, "Completed 2 task(s)")

        mixin.after.assert_called_once()
        delay = mixin.after.call_args[0][0]
        assert delay == 2000

    def test_failure_uses_set_status(self):
        """On failure, should call set_status without celebration."""
        mixin = self._make_task_mixin()
        mixin.set_status = MagicMock()
        mixin._on_tasks_complete(False, "No text to analyze")

        mixin.set_status.assert_called_once_with("No text to analyze")
        # Should NOT change status frame color
        mixin.status_frame.configure.assert_not_called()

    def test_restore_status_bar_color(self):
        """_restore_status_bar_color should reset to normal colors."""
        mixin = self._make_task_mixin()
        mixin._restore_status_bar_color("Completed 2 task(s)")

        mixin.status_frame.configure.assert_called_once()
        frame_kwargs = mixin.status_frame.configure.call_args[1]
        assert frame_kwargs["fg_color"] == "#1a1a2e"  # status_bar_bg

        mixin.status_label.configure.assert_called_once()
        label_kwargs = mixin.status_label.configure.call_args[1]
        assert label_kwargs["text"] == "Completed 2 task(s)"


# =========================================================================
# Feature 6: Hover Previews
# =========================================================================


class TestHoverPreviews:
    """Test the hover preview functionality in FileReviewTable."""

    def _make_table(self):
        """Create a FileReviewTable with mocked widgets."""
        from src.ui.widgets import FileReviewTable

        table = FileReviewTable.__new__(FileReviewTable)
        table.column_map = {
            "include": ("Include", 50),
            "filename": ("Filename", 300),
            "status": ("Status", 100),
            "method": ("Method", 100),
            "confidence": ("Confidence", 100),
            "pages": ("Pages", 50),
            "size": ("Size", 80),
        }
        table.file_item_map = {}
        table._result_data = {}
        table._hovered_row = None
        table._tooltip_window = None
        table._drop_zone = MagicMock()
        table._drop_zone.winfo_ismapped.return_value = False
        table.tree = MagicMock()
        return table

    def test_result_data_stored_on_add(self):
        """add_result should store full result dict for hover lookup."""
        table = self._make_table()
        table.tree.insert.return_value = "item1"

        result = {
            "filename": "test.pdf",
            "status": "success",
            "confidence": 90,
            "method": "digital",
            "page_count": 5,
            "file_size": 1024,
            "file_path": "C:\\Cases\\test.pdf",
            "word_count": 12345,
        }

        table.add_result(result)
        assert "test.pdf" in table._result_data
        assert table._result_data["test.pdf"]["word_count"] == 12345

    def test_result_data_cleared_on_clear(self):
        """clear() should reset result data."""
        table = self._make_table()
        table._result_data = {"test.pdf": {"filename": "test.pdf"}}
        table.tree.get_children.return_value = []

        table.clear()
        assert table._result_data == {}

    def test_on_hover_ignores_empty_row(self):
        """Hovering over empty area should not show tooltip."""
        table = self._make_table()
        table.tree.identify_row.return_value = ""

        event = MagicMock()
        event.y = 100
        table._show_tooltip = MagicMock()

        table._on_hover(event)
        table._show_tooltip.assert_not_called()

    def test_on_hover_ignores_same_row(self):
        """Hovering over same row should not re-show tooltip."""
        table = self._make_table()
        table._hovered_row = "row1"
        table.tree.identify_row.return_value = "row1"

        event = MagicMock()
        event.y = 100
        table._show_tooltip = MagicMock()

        table._on_hover(event)
        table._show_tooltip.assert_not_called()

    def test_on_hover_shows_tooltip_for_known_file(self):
        """Hovering over a row with stored data should show tooltip."""
        table = self._make_table()
        table._hovered_row = None
        table.tree.identify_row.return_value = "row1"
        # tree.item(row_id, "values") returns the values tuple directly
        table.tree.item.return_value = ("✓", "test.pdf", "Ready")
        table._result_data["test.pdf"] = {
            "filename": "test.pdf",
            "file_path": "C:\\Cases\\test.pdf",
            "word_count": 5000,
            "page_count": 20,
            "method": "digital_pdf",
        }
        table._show_tooltip = MagicMock()

        event = MagicMock()
        event.y = 50
        event.x_root = 200
        event.y_root = 300

        table._on_hover(event)
        table._show_tooltip.assert_called_once()

        # Verify tooltip text contains expected info
        tooltip_text = table._show_tooltip.call_args[0][2]
        assert "C:\\Cases\\test.pdf" in tooltip_text
        assert "5,000" in tooltip_text
        assert "20" in tooltip_text
        assert "Digital Pdf" in tooltip_text

    def test_on_hover_includes_case_numbers(self):
        """Tooltip should show case numbers if available."""
        table = self._make_table()
        table.tree.identify_row.return_value = "row1"
        table.tree.item.return_value = ("✓", "motion.pdf", "Ready")
        table._result_data["motion.pdf"] = {
            "filename": "motion.pdf",
            "case_numbers": ["21-CV-1234", "22-CV-5678"],
        }
        table._show_tooltip = MagicMock()

        event = MagicMock()
        event.y = 50
        event.x_root = 200
        event.y_root = 300

        table._on_hover(event)
        tooltip_text = table._show_tooltip.call_args[0][2]
        assert "21-CV-1234" in tooltip_text
        assert "22-CV-5678" in tooltip_text

    def test_on_leave_resets_hovered_row(self):
        """Leaving the treeview should reset hover state."""
        table = self._make_table()
        table._hovered_row = "row1"
        table._hide_tooltip = MagicMock()

        table._on_leave(MagicMock())
        assert table._hovered_row is None
        table._hide_tooltip.assert_called_once()

    def test_hide_tooltip_destroys_window(self):
        """_hide_tooltip should destroy the tooltip window."""
        table = self._make_table()
        mock_window = MagicMock()
        table._tooltip_window = mock_window

        table._hide_tooltip()
        mock_window.destroy.assert_called_once()
        assert table._tooltip_window is None

    def test_hide_tooltip_noop_when_no_tooltip(self):
        """_hide_tooltip should be safe when no tooltip exists."""
        table = self._make_table()
        table._tooltip_window = None
        # Should not raise
        table._hide_tooltip()


# =========================================================================
# Feature 3 & Queue Wiring: Progress percentage in queue messages
# =========================================================================


class TestProgressQueueWiring:
    """Test that progress messages update the determinate progress bar."""

    def _make_task_mixin(self):
        """Create a TaskMixin with mocked methods."""
        from src.ui.main_window_helpers.task_mixin import TaskMixin

        mixin = TaskMixin.__new__(TaskMixin)
        mixin._qa_ready = False
        mixin.set_status = MagicMock()
        mixin._update_progress = MagicMock()
        mixin.processing_results = []
        mixin.file_table = MagicMock()
        mixin.output_display = MagicMock()
        mixin._qa_results = []
        mixin._qa_results_lock = MagicMock()
        mixin._completed_tasks = set()
        mixin._pending_tasks = {}
        mixin.pipeline_indicator = MagicMock()
        mixin.pipeline_indicator._step_states = {}
        mixin.followup_btn = MagicMock()
        mixin.followup_entry = MagicMock()
        mixin.ask_default_questions_check = MagicMock()
        mixin.vocab_llm_check = MagicMock()
        return mixin

    def test_progress_message_calls_update_progress(self):
        """Progress queue message should call _update_progress with percentage."""
        mixin = self._make_task_mixin()
        mixin._handle_queue_message("progress", (45.0, "Processing file 3/7..."))

        mixin._update_progress.assert_called_once_with(45.0)
        mixin.set_status.assert_called_once_with("Processing file 3/7...")

    def test_progress_message_with_none_percentage(self):
        """Progress with None percentage should not call _update_progress."""
        mixin = self._make_task_mixin()
        mixin._handle_queue_message("progress", (None, "Starting..."))

        mixin._update_progress.assert_not_called()
        mixin.set_status.assert_called_once()

    def test_progress_message_with_zero_percentage(self):
        """Progress with 0 percentage (falsy) should not call _update_progress."""
        mixin = self._make_task_mixin()
        mixin._handle_queue_message("progress", (0, "Starting..."))

        # 0 is falsy, so _update_progress should NOT be called
        mixin._update_progress.assert_not_called()


# =========================================================================
# Pipeline indicator wiring in task_mixin
# =========================================================================


class TestPipelineIndicatorWiring:
    """Test that task_mixin updates pipeline indicator at the right times."""

    def _make_task_mixin(self):
        """Create a TaskMixin for pipeline indicator tests."""
        from src.ui.main_window_helpers.task_mixin import TaskMixin

        mixin = TaskMixin.__new__(TaskMixin)
        mixin.pipeline_indicator = MagicMock()
        mixin.pipeline_indicator._step_states = {
            "Extract": "active",
            "Vocabulary": "pending",
            "Q&A": "pending",
            "Summary": "skipped",
        }
        mixin._qa_ready = False
        mixin._qa_results = []
        mixin._qa_results_lock = MagicMock()
        mixin._completed_tasks = set()
        mixin._pending_tasks = {"vocab": True, "qa": True, "summary": False}
        mixin.set_status = MagicMock()
        mixin._update_progress = MagicMock()
        mixin.processing_results = []
        mixin.file_table = MagicMock()
        mixin.output_display = MagicMock()
        mixin.followup_btn = MagicMock()
        mixin.followup_entry = MagicMock()
        mixin.ask_default_questions_check = MagicMock()
        mixin.vocab_llm_check = MagicMock()
        mixin.vocab_llm_check.get.return_value = True
        mixin.vocab_llm_check.cget.return_value = "normal"
        return mixin

    def test_processing_finished_sets_extract_done(self):
        """processing_finished should mark Extract as done."""
        mixin = self._make_task_mixin()
        mixin._on_preprocessing_complete = MagicMock()

        mixin._handle_queue_message("processing_finished", [])

        mixin.pipeline_indicator.set_step_state.assert_any_call("Extract", "done")

    def test_processing_finished_activates_vocabulary(self):
        """processing_finished should activate Vocabulary when vocab is pending."""
        mixin = self._make_task_mixin()
        mixin._on_preprocessing_complete = MagicMock()

        mixin._handle_queue_message("processing_finished", [])

        mixin.pipeline_indicator.set_step_state.assert_any_call("Vocabulary", "active")

    def test_qa_ready_activates_qa(self):
        """qa_ready should activate Q&A step."""
        mixin = self._make_task_mixin()

        mixin._handle_queue_message(
            "qa_ready", {"chunk_count": 10, "vector_store_path": "/tmp", "embeddings": MagicMock()}
        )

        mixin.pipeline_indicator.set_step_state.assert_called_with("Q&A", "active")

    def test_qa_complete_sets_qa_done(self):
        """qa_complete should mark Q&A as done."""
        mixin = self._make_task_mixin()
        mixin._completed_tasks.add("vocab")
        mixin._on_tasks_complete = MagicMock()
        mixin._finalize_tasks = MagicMock()

        mixin._handle_queue_message("qa_complete", [])

        mixin.pipeline_indicator.set_step_state.assert_called_with("Q&A", "done")

    def test_llm_complete_sets_vocabulary_done(self):
        """llm_complete should mark Vocabulary as done."""
        mixin = self._make_task_mixin()
        mixin._finalize_tasks = MagicMock()

        mixin._handle_queue_message("llm_complete", [{"term": "test"}])

        mixin.pipeline_indicator.set_step_state.assert_called_with("Vocabulary", "done")

    def test_finalize_marks_active_steps_done(self):
        """_finalize_tasks should mark any remaining active steps as done."""
        mixin = self._make_task_mixin()
        mixin.pipeline_indicator._step_states = {
            "Extract": "done",
            "Vocabulary": "done",
            "Q&A": "active",
            "Summary": "skipped",
        }
        mixin._on_tasks_complete = MagicMock()

        mixin._finalize_tasks()

        mixin.pipeline_indicator.set_step_state.assert_called_with("Q&A", "done")


# =========================================================================
# Theme colors
# =========================================================================


class TestThemeColors:
    """Test that new theme colors are defined."""

    def test_progress_bar_color_exists(self):
        """progress_bar color should be defined in COLORS."""
        from src.ui.theme import COLORS

        assert "progress_bar" in COLORS
        assert COLORS["progress_bar"] == "#3d8bfd"

    def test_drop_zone_border_color_exists(self):
        """drop_zone_border color should be defined."""
        from src.ui.theme import COLORS

        assert "drop_zone_border" in COLORS

    def test_drop_zone_bg_color_exists(self):
        """drop_zone_bg color should be defined."""
        from src.ui.theme import COLORS

        assert "drop_zone_bg" in COLORS
