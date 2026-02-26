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
        table._remove_icon = MagicMock()
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
        table._remove_icon = MagicMock()
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
        table.tree.item.return_value = ("test.pdf", "Ready")
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
        table.tree.item.return_value = ("motion.pdf", "Ready")
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
# Feature 6b: Tooltip manager integration in FileReviewTable
# =========================================================================


class TestTooltipManagerIntegration:
    """Test that FileReviewTable coordinates with global TooltipManager."""

    def _make_table(self):
        """Create a FileReviewTable with mocked widgets."""
        from src.ui.widgets import FileReviewTable

        table = FileReviewTable.__new__(FileReviewTable)
        table._tooltip_window = None
        table._hovered_row = None
        table._result_data = {}
        table.file_item_map = {}
        table._drop_zone = MagicMock()
        table._drop_zone.winfo_ismapped.return_value = False
        table.tree = MagicMock()
        return table

    def test_show_tooltip_calls_close_active(self):
        """_show_tooltip should close any existing tooltip from other components."""
        from unittest.mock import patch

        table = self._make_table()
        table.winfo_toplevel = MagicMock()

        with (
            patch("src.ui.tooltip_manager.tooltip_manager") as mock_mgr,
            patch("src.ui.widgets.ctk.CTkToplevel"),
            patch("src.ui.widgets.ctk.CTkLabel"),
        ):
            table._show_tooltip(100, 200, "test text")
            mock_mgr.close_active.assert_called_once()

    def test_show_tooltip_registers_with_manager(self):
        """_show_tooltip should register the new window with tooltip_manager."""
        from unittest.mock import patch

        import customtkinter as ctk

        table = self._make_table()
        mock_toplevel = MagicMock(spec=ctk.CTkToplevel)

        with (
            patch("src.ui.tooltip_manager.tooltip_manager") as mock_mgr,
            patch("src.ui.widgets.ctk.CTkToplevel", return_value=mock_toplevel),
            patch("src.ui.widgets.ctk.CTkLabel"),
        ):
            table.winfo_toplevel = MagicMock()
            table._show_tooltip(100, 200, "test text")

            mock_mgr.register.assert_called_once_with(mock_toplevel, owner=table)

    def test_hide_tooltip_unregisters_from_manager(self):
        """_hide_tooltip should unregister from tooltip_manager before destroying."""
        from unittest.mock import patch

        table = self._make_table()
        mock_window = MagicMock()
        table._tooltip_window = mock_window

        with patch("src.ui.tooltip_manager.tooltip_manager") as mock_mgr:
            table._hide_tooltip()

            mock_mgr.unregister.assert_called_once_with(mock_window)
            mock_window.destroy.assert_called_once()

    def test_hide_tooltip_noop_skips_unregister(self):
        """_hide_tooltip with no tooltip should not call unregister."""
        from unittest.mock import patch

        table = self._make_table()
        table._tooltip_window = None

        with patch("src.ui.tooltip_manager.tooltip_manager") as mock_mgr:
            table._hide_tooltip()
            mock_mgr.unregister.assert_not_called()

    def test_show_tooltip_source_has_manager_calls(self):
        """_show_tooltip source should contain tooltip_manager integration."""
        import inspect

        from src.ui.widgets import FileReviewTable

        source = inspect.getsource(FileReviewTable._show_tooltip)
        assert "tooltip_manager.close_active()" in source
        assert "tooltip_manager.register(" in source

    def test_hide_tooltip_source_has_unregister(self):
        """_hide_tooltip source should call tooltip_manager.unregister."""
        import inspect

        from src.ui.widgets import FileReviewTable

        source = inspect.getsource(FileReviewTable._hide_tooltip)
        assert "tooltip_manager.unregister(" in source


# =========================================================================
# Tooltip auto-dismiss timer
# =========================================================================


class TestTooltipAutoDismiss:
    """Test that tooltips auto-dismiss after 15 seconds."""

    def test_show_tooltip_source_schedules_dismiss(self):
        """_show_tooltip should schedule auto-dismiss via self.after."""
        import inspect

        from src.ui.widgets import FileReviewTable

        source = inspect.getsource(FileReviewTable._show_tooltip)
        assert "15000" in source
        assert "_hide_tooltip" in source
        assert "_tooltip_dismiss_id" in source

    def test_hide_tooltip_source_cancels_dismiss(self):
        """_hide_tooltip should cancel pending auto-dismiss timer."""
        import inspect

        from src.ui.widgets import FileReviewTable

        source = inspect.getsource(FileReviewTable._hide_tooltip)
        assert "_tooltip_dismiss_id" in source
        assert "after_cancel" in source

    def test_hide_tooltip_clears_dismiss_id(self):
        """_hide_tooltip should set _tooltip_dismiss_id to None."""
        from src.ui.widgets import FileReviewTable

        table = FileReviewTable.__new__(FileReviewTable)
        table._tooltip_window = None
        table._tooltip_dismiss_id = "some_timer_id"

        table._hide_tooltip()

        assert table._tooltip_dismiss_id is None

    def test_hide_tooltip_without_dismiss_id(self):
        """_hide_tooltip should be safe when no dismiss timer is set."""
        from src.ui.widgets import FileReviewTable

        table = FileReviewTable.__new__(FileReviewTable)
        table._tooltip_window = None
        table._tooltip_dismiss_id = None

        table._hide_tooltip()  # Should not raise


# =========================================================================
# Orange ✕ icon in tree column
# =========================================================================


class TestOrangeRemoveIcon:
    """Test that the remove icon is an orange bitmap in the tree column."""

    def test_create_remove_icon_source_uses_orange(self):
        """_create_remove_icon should use orange foreground color."""
        import inspect

        from src.ui.widgets import FileReviewTable

        source = inspect.getsource(FileReviewTable._create_remove_icon)
        assert "#e67e22" in source
        assert "BitmapImage" in source

    def test_treeview_shows_tree_column(self):
        """_create_treeview should use show='tree headings' for icon column."""
        import inspect

        from src.ui.widgets import FileReviewTable

        source = inspect.getsource(FileReviewTable._create_treeview)
        assert '"tree headings"' in source

    def test_on_click_checks_tree_column(self):
        """_on_click should detect clicks on #0 (tree column), not #1."""
        import inspect

        from src.ui.widgets import FileReviewTable

        source = inspect.getsource(FileReviewTable._on_click)
        assert '"#0"' in source
        assert '"#1"' not in source


# =========================================================================
# MainWindow._on_tasks_complete failure path
# =========================================================================


class TestOnTasksCompleteFailurePath:
    """Test that _on_tasks_complete uses set_status_error for failures."""

    def _make_stub(self):
        from unittest.mock import MagicMock

        stub = MagicMock()
        stub._export_all_visible = False
        stub.qa_check = MagicMock()
        stub.qa_check.get.return_value = False
        stub.output_display = MagicMock()
        return stub

    def test_failure_calls_set_status_error(self):
        """On failure, should call set_status_error (orange text)."""
        from src.ui.main_window import MainWindow

        stub = self._make_stub()
        MainWindow._on_tasks_complete(stub, False, "No text to analyze")

        stub.set_status_error.assert_called_once_with("No text to analyze")
        stub.set_status.assert_not_called()

    def test_success_calls_set_status(self):
        """On success, should call set_status (normal text)."""
        from src.ui.main_window import MainWindow

        stub = self._make_stub()
        MainWindow._on_tasks_complete(stub, True, "Completed 2 task(s)")

        stub.set_status.assert_called_once_with("Completed 2 task(s)")
        stub.set_status_error.assert_not_called()

    def test_failure_does_not_show_export_button(self):
        """On failure, Export All button should not appear."""
        from src.ui.main_window import MainWindow

        stub = self._make_stub()
        MainWindow._on_tasks_complete(stub, False, "Processing failed")

        assert stub._export_all_visible is False


# =========================================================================
# Bug fix: _preprocessing_active flag (separate from _processing_active)
# =========================================================================


class TestPreprocessingActiveFlag:
    """Test that preprocessing uses its own flag, not _processing_active.

    Bug: _start_preprocessing set _processing_active = True, which caused
    _update_generate_button_state to exit early (button stuck disabled).
    Fix: Use separate _preprocessing_active flag for preprocessing polling.
    """

    def test_init_has_preprocessing_active(self):
        """MainWindow.__init__ should initialize _preprocessing_active."""
        import inspect

        from src.ui.main_window import MainWindow

        source = inspect.getsource(MainWindow.__init__)
        assert "_preprocessing_active" in source

    def test_start_preprocessing_sets_preprocessing_flag(self):
        """_start_preprocessing should set _preprocessing_active, not _processing_active."""
        import inspect

        from src.ui.main_window_helpers.file_mixin import FileMixin

        source = inspect.getsource(FileMixin._start_preprocessing)
        assert "self._preprocessing_active = True" in source
        assert "self._processing_active = True" not in source

    def test_on_preprocessing_complete_clears_flag(self):
        """_on_preprocessing_complete should clear _preprocessing_active."""
        import inspect

        from src.ui.main_window_helpers.file_mixin import FileMixin

        source = inspect.getsource(FileMixin._on_preprocessing_complete)
        assert "self._preprocessing_active = False" in source

    def test_poll_queue_checks_preprocessing_active(self):
        """_poll_queue should continue polling when _preprocessing_active is True."""
        import inspect

        from src.ui.main_window_helpers.file_mixin import FileMixin

        source = inspect.getsource(FileMixin._poll_queue)
        assert "self._preprocessing_active" in source

    def test_poll_queue_continues_during_preprocessing(self):
        """_poll_queue should schedule another poll when _preprocessing_active."""
        from src.ui.main_window import MainWindow

        w = MagicMock()
        w._destroying = False
        w._worker_manager = MagicMock()
        w._worker_manager.check_for_messages.return_value = []
        w._processing_active = False
        w._preprocessing_active = True
        w._queue_poll_id = None
        w._qa_results_lock = MagicMock()
        w._qa_results = []

        MainWindow._poll_queue(w)

        w.after.assert_called_once_with(33, w._poll_queue)

    def test_poll_queue_stops_when_all_flags_false(self):
        """_poll_queue should stop polling when all activity flags are False."""
        from src.ui.main_window import MainWindow

        w = MagicMock()
        w._destroying = False
        w._worker_manager = MagicMock()
        w._worker_manager.check_for_messages.return_value = []
        w._processing_active = False
        w._preprocessing_active = False
        w._qa_answering_active = False
        w._queue_poll_id = None
        w._qa_results_lock = MagicMock()
        w._qa_results = []

        MainWindow._poll_queue(w)

        w.after.assert_not_called()

    def test_button_enables_after_preprocessing(self):
        """_update_generate_button_state should enable button after preprocessing.

        This is the actual bug: _processing_active stayed True after preprocessing,
        causing the early return at the top of _update_generate_button_state.
        """
        from src.ui.main_window import MainWindow

        w = MagicMock()
        w._processing_active = False
        w.processing_results = [{"status": "success"}]
        w._get_task_count = MagicMock(return_value=2)

        MainWindow._update_generate_button_state(w)

        # Button should be enabled with "Perform 2 Tasks"
        w.generate_btn.configure.assert_called_with(text="Perform 2 Tasks", state="normal")

    def test_button_disabled_during_task_execution(self):
        """_update_generate_button_state should disable button during _processing_active."""
        from src.ui.main_window import MainWindow

        w = MagicMock()
        w._processing_active = True
        w.processing_results = [{"status": "success"}]

        MainWindow._update_generate_button_state(w)

        w.generate_btn.configure.assert_called_with(state="disabled")

    def test_main_window_start_preprocessing_mirrors_mixin(self):
        """main_window.py _start_preprocessing should also use _preprocessing_active."""
        import inspect

        from src.ui.main_window import MainWindow

        source = inspect.getsource(MainWindow._start_preprocessing)
        assert "self._preprocessing_active = True" in source
        assert "self._processing_active = True" not in source

    def test_main_window_on_preprocessing_complete_clears_flag(self):
        """main_window.py _on_preprocessing_complete should clear _preprocessing_active."""
        import inspect

        from src.ui.main_window import MainWindow

        source = inspect.getsource(MainWindow._on_preprocessing_complete)
        assert "self._preprocessing_active = False" in source


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

    def test_drop_zone_idle_border_color_exists(self):
        """drop_zone_idle_border color should be defined for empty-state rectangle."""
        from src.ui.theme import COLORS

        assert "drop_zone_idle_border" in COLORS
        assert isinstance(COLORS["drop_zone_idle_border"], str)

    def test_drop_zone_idle_bg_color_exists(self):
        """drop_zone_idle_bg color should be defined for empty-state rectangle."""
        from src.ui.theme import COLORS

        assert "drop_zone_idle_bg" in COLORS
        assert isinstance(COLORS["drop_zone_idle_bg"], str)

    def test_drop_zone_idle_colors_are_valid_hex(self):
        """Idle drop zone colors should be valid hex color strings."""
        from src.ui.theme import COLORS

        for key in ("drop_zone_idle_border", "drop_zone_idle_bg"):
            color = COLORS[key]
            assert color.startswith("#"), f"{key} should start with #"
            assert len(color) == 7, f"{key} should be 7 chars (#RRGGBB)"


# =========================================================================
# Feature 1b: Drop Zone Frame Structure
# =========================================================================


class TestDropZoneStructure:
    """Test the drop zone frame, labels, and placement in FileReviewTable."""

    def _make_table(self):
        """Create a FileReviewTable with manually set attributes (skips __init__)."""
        from src.ui.widgets import FileReviewTable

        table = FileReviewTable.__new__(FileReviewTable)
        table.column_map = {
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
        table._remove_icon = MagicMock()
        table._drop_zone = MagicMock()
        table._drop_zone.winfo_ismapped.return_value = True
        table._drop_zone_label = MagicMock()
        table._drop_zone_hint = MagicMock()
        table.tree = MagicMock()
        table.tree.identify_row.return_value = ""
        return table

    def test_drop_zone_hidden_after_multiple_files(self):
        """Drop zone should stay hidden when adding second file."""
        table = self._make_table()
        table.tree.insert.return_value = "item1"

        # Add first file -- hides drop zone
        table.add_result(
            {
                "filename": "a.pdf",
                "status": "success",
                "confidence": 90,
                "method": "digital",
                "page_count": 1,
                "file_size": 100,
            }
        )
        table._drop_zone.place_forget.assert_called_once()
        table._drop_zone.place_forget.reset_mock()

        # Add second file -- should NOT call place_forget again
        table.tree.insert.return_value = "item2"
        table._drop_zone.winfo_ismapped.return_value = False
        table.add_result(
            {
                "filename": "b.pdf",
                "status": "success",
                "confidence": 85,
                "method": "digital",
                "page_count": 2,
                "file_size": 200,
            }
        )
        table._drop_zone.place_forget.assert_not_called()

    def test_clear_then_add_shows_and_hides_drop_zone(self):
        """After clear + re-add, drop zone should show then hide again."""
        table = self._make_table()
        table.tree.insert.return_value = "item1"

        # Add file
        table.add_result(
            {
                "filename": "a.pdf",
                "status": "success",
                "confidence": 90,
                "method": "digital",
                "page_count": 1,
                "file_size": 100,
            }
        )
        table._drop_zone.place_forget.assert_called_once()

        # Clear
        table.tree.get_children.return_value = ["item1"]
        table.clear()
        table._drop_zone.place.assert_called_once()

        # Re-add -- file_item_map is empty after clear, drop zone is mapped
        table._drop_zone.winfo_ismapped.return_value = True
        table._drop_zone.place_forget.reset_mock()
        table.tree.insert.return_value = "item2"
        table.add_result(
            {
                "filename": "b.pdf",
                "status": "success",
                "confidence": 80,
                "method": "ocr",
                "page_count": 3,
                "file_size": 300,
            }
        )
        table._drop_zone.place_forget.assert_called_once()

    def test_clear_placement_uses_correct_dimensions(self):
        """clear() should restore drop zone at 85% width, 70% height."""
        table = self._make_table()
        table.tree.get_children.return_value = []

        table.clear()

        args = table._drop_zone.place.call_args
        assert args.kwargs["relwidth"] == 0.85
        assert args.kwargs["relheight"] == 0.7
        assert args.kwargs["anchor"] == "center"

    def test_drop_zone_not_hidden_when_not_mapped(self):
        """If drop zone is already hidden, add_result should not call place_forget."""
        table = self._make_table()
        table._drop_zone.winfo_ismapped.return_value = False
        table.tree.insert.return_value = "item1"

        table.add_result(
            {
                "filename": "a.pdf",
                "status": "success",
                "confidence": 90,
                "method": "digital",
                "page_count": 1,
                "file_size": 100,
            }
        )
        table._drop_zone.place_forget.assert_not_called()


# ---------------------------------------------------------------------------
# Status bar enforcement — no direct status_label.configure(text=...)
# ---------------------------------------------------------------------------


class TestStatusBarEnforcement:
    """Ensure all status updates go through set_status / set_status_error."""

    def test_no_direct_status_label_configure_in_main_window(self):
        """main_window.py must not call status_label.configure(text=...) directly.

        The only methods allowed to touch status_label are set_status,
        set_status_error, and _clear_status_to_default.
        """
        import ast
        import inspect

        from src.ui.main_window import MainWindow

        # Allowed methods that legitimately call status_label.configure
        allowed = {"set_status", "set_status_error", "_clear_status_to_default"}

        violations = []
        for method_name in dir(MainWindow):
            if method_name in allowed:
                continue
            method = getattr(MainWindow, method_name, None)
            if not callable(method):
                continue
            try:
                import textwrap

                source = textwrap.dedent(inspect.getsource(method))
                tree = ast.parse(source)
            except (TypeError, OSError, SyntaxError):
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute) or func.attr != "configure":
                    continue
                if not isinstance(func.value, ast.Attribute) or func.value.attr != "status_label":
                    continue
                if any(kw.arg == "text" for kw in node.keywords):
                    violations.append(f"{method_name} (line {node.lineno})")

        assert violations == [], (
            f"Direct status_label.configure(text=...) found in: {violations}. "
            f"Use set_status() or set_status_error() instead."
        )
