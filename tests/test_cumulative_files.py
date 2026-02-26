"""
Tests for cumulative file input and per-file removal.

Covers:
1. FileReviewTable.remove_result() — single file removal from table
2. FileReviewTable._on_click() — click handler for ✕ column
3. FileReviewTable column_map uses "remove" instead of "include"
4. MainWindow._remove_file() — remove file from session
5. MainWindow._select_files() — deduplication on add
6. MainWindow._on_file_drop() — deduplication on drop
7. MainWindow._start_preprocessing() — incremental (no clear, accepts file_paths)
8. MainWindow._on_preprocessing_complete() — merge instead of replace
"""

import inspect
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_table(on_remove=None):
    """Create a FileReviewTable via __new__ with mocked internals."""
    from src.ui.widgets import FileReviewTable

    table = FileReviewTable.__new__(FileReviewTable)
    table._on_remove = on_remove
    table.column_map = {
        "remove": ("", 36),
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
    table.tree.identify_row.return_value = ""
    return table


def _make_window_stub():
    """Create a MainWindow stub with attributes needed by file methods."""
    stub = MagicMock()
    stub.selected_files = []
    stub.processing_results = []
    stub.file_table = MagicMock()
    stub._export_all_visible = False
    stub._preprocessing_active = False
    stub._processing_active = False
    stub._qa_answering_active = False
    stub._qa_ready = False
    stub._queue_poll_id = None
    stub._worker_ready_retries = 0
    stub._worker_manager = MagicMock()
    stub._worker_manager.is_ready.return_value = True
    stub._destroying = False
    return stub


def _sample_result(filename, status="success", confidence=90):
    """Build a minimal processing result dict."""
    return {
        "filename": filename,
        "status": status,
        "confidence": confidence,
        "method": "digital",
        "page_count": 3,
        "file_size": 1024,
    }


# ===========================================================================
# 1. FileReviewTable.remove_result()
# ===========================================================================


class TestRemoveResult:
    """Tests for FileReviewTable.remove_result()."""

    def test_removes_item_from_treeview(self):
        """remove_result deletes the treeview item."""
        table = _make_table()
        table.file_item_map["test.pdf"] = "item1"
        table._result_data["test.pdf"] = {"filename": "test.pdf"}

        table.remove_result("test.pdf")

        table.tree.delete.assert_called_once_with("item1")
        assert "test.pdf" not in table.file_item_map
        assert "test.pdf" not in table._result_data

    def test_shows_drop_zone_when_last_file_removed(self):
        """Drop zone should reappear when table becomes empty."""
        table = _make_table()
        table.file_item_map["only.pdf"] = "item1"
        table._result_data["only.pdf"] = {"filename": "only.pdf"}

        table.remove_result("only.pdf")

        table._drop_zone.place.assert_called_once()
        kwargs = table._drop_zone.place.call_args.kwargs
        assert kwargs["relwidth"] == 0.85
        assert kwargs["relheight"] == 0.7

    def test_no_drop_zone_when_files_remain(self):
        """Drop zone should NOT appear if other files still exist."""
        table = _make_table()
        table.file_item_map["a.pdf"] = "item1"
        table.file_item_map["b.pdf"] = "item2"
        table._result_data["a.pdf"] = {"filename": "a.pdf"}
        table._result_data["b.pdf"] = {"filename": "b.pdf"}

        table.remove_result("a.pdf")

        table._drop_zone.place.assert_not_called()
        assert "b.pdf" in table.file_item_map

    def test_noop_for_unknown_filename(self):
        """remove_result should not crash for a filename not in the table."""
        table = _make_table()
        table.remove_result("nonexistent.pdf")
        table.tree.delete.assert_not_called()

    def test_hides_tooltip_on_remove(self):
        """remove_result should hide any active tooltip."""
        table = _make_table()
        table.file_item_map["test.pdf"] = "item1"
        table._result_data["test.pdf"] = {}
        table._tooltip_window = MagicMock()

        with patch.object(type(table), "_hide_tooltip", autospec=True) as mock_hide:
            table.remove_result("test.pdf")
            mock_hide.assert_called_once()


# ===========================================================================
# 2. FileReviewTable._on_click()
# ===========================================================================


class TestOnClick:
    """Tests for the ✕ column click handler."""

    def test_click_on_remove_column_calls_callback(self):
        """Clicking column #1 on a valid row invokes on_remove(filename)."""
        callback = MagicMock()
        table = _make_table(on_remove=callback)
        table.tree.identify_column.return_value = "#1"
        table.tree.identify_row.return_value = "row1"
        table.tree.item.return_value = ("✕", "report.pdf", "✓ Ready")

        event = MagicMock()
        event.x = 10
        event.y = 20
        table._on_click(event)

        callback.assert_called_once_with("report.pdf")

    def test_click_on_other_column_ignored(self):
        """Clicking a non-remove column should not call on_remove."""
        callback = MagicMock()
        table = _make_table(on_remove=callback)
        table.tree.identify_column.return_value = "#2"

        event = MagicMock()
        table._on_click(event)

        callback.assert_not_called()

    def test_click_on_empty_row_ignored(self):
        """Clicking empty space (no row) should not call on_remove."""
        callback = MagicMock()
        table = _make_table(on_remove=callback)
        table.tree.identify_column.return_value = "#1"
        table.tree.identify_row.return_value = ""

        event = MagicMock()
        table._on_click(event)

        callback.assert_not_called()

    def test_no_callback_set_does_not_crash(self):
        """If on_remove is None, clicks should be silently ignored."""
        table = _make_table(on_remove=None)
        event = MagicMock()
        table._on_click(event)  # Should not raise


# ===========================================================================
# 3. FileReviewTable column_map
# ===========================================================================


class TestColumnMapStructure:
    """Verify column_map uses 'remove' instead of 'include'."""

    def test_column_map_has_remove(self):
        """column_map should contain 'remove' key."""
        from src.ui.widgets import FileReviewTable

        table = FileReviewTable.__new__(FileReviewTable)
        table._on_remove = None
        table.column_map = {}
        # Re-read the class default by checking source
        source = inspect.getsource(FileReviewTable.__init__)
        assert '"remove"' in source
        assert '"include"' not in source

    def test_remove_column_is_narrow(self):
        """The remove column should be narrow (<=40px)."""
        source = inspect.getsource(
            __import__("src.ui.widgets", fromlist=["FileReviewTable"]).FileReviewTable.__init__
        )
        # The column definition: "remove": ("", 36)
        assert '"remove": ("", 36)' in source

    def test_prepare_result_uses_x_mark(self):
        """_prepare_result_for_display should put ✕ in first column."""
        table = _make_table()
        result = _sample_result("test.pdf")
        values, _tag = table._prepare_result_for_display(result)
        assert values[0] == "✕"


# ===========================================================================
# 4. MainWindow._remove_file()
# ===========================================================================


class TestRemoveFile:
    """Tests for MainWindow._remove_file()."""

    def test_removes_from_selected_files(self):
        """_remove_file should remove the matching path from selected_files."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\docs\report.pdf", r"C:\docs\brief.pdf"]
        stub.processing_results = [
            _sample_result("report.pdf"),
            _sample_result("brief.pdf"),
        ]

        MainWindow._remove_file(stub, "report.pdf")

        assert len(stub.selected_files) == 1
        assert stub.selected_files[0].endswith("brief.pdf")

    def test_removes_from_processing_results(self):
        """_remove_file should remove the result dict for that filename."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\docs\report.pdf"]
        stub.processing_results = [_sample_result("report.pdf")]

        MainWindow._remove_file(stub, "report.pdf")

        assert len(stub.processing_results) == 0

    def test_calls_file_table_remove(self):
        """_remove_file should call file_table.remove_result."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\docs\test.pdf"]
        stub.processing_results = [_sample_result("test.pdf")]

        MainWindow._remove_file(stub, "test.pdf")

        stub.file_table.remove_result.assert_called_once_with("test.pdf")

    def test_updates_ui_state(self):
        """_remove_file should refresh button state and session stats."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = []
        stub.processing_results = []

        MainWindow._remove_file(stub, "test.pdf")

        stub._update_generate_button_state.assert_called_once()
        stub._update_session_stats.assert_called_once()
        stub.set_status.assert_called_once()


# ===========================================================================
# 5. _select_files — deduplication
# ===========================================================================


class TestSelectFilesDedup:
    """Tests for cumulative file selection with dedup."""

    def test_new_files_appended(self):
        """New files should be appended to existing selected_files."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\docs\a.pdf"]
        stub._check_ocr_availability = MagicMock(return_value=False)

        with patch("src.ui.main_window.filedialog") as mock_dialog:
            mock_dialog.askopenfilenames.return_value = (r"C:\docs\b.pdf",)
            MainWindow._select_files(stub)

        assert len(stub.selected_files) == 2
        assert r"C:\docs\a.pdf" in stub.selected_files
        assert r"C:\docs\b.pdf" in stub.selected_files

    def test_duplicate_files_skipped(self):
        """Files already in selected_files should not be added again."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\docs\a.pdf"]
        stub._check_ocr_availability = MagicMock(return_value=False)

        with patch("src.ui.main_window.filedialog") as mock_dialog:
            mock_dialog.askopenfilenames.return_value = (r"C:\docs\a.pdf",)
            MainWindow._select_files(stub)

        assert len(stub.selected_files) == 1
        stub.set_status.assert_called_with("All selected files are already in the session")

    def test_mixed_new_and_existing(self):
        """Only new files from a mixed selection should be added."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\docs\a.pdf"]
        stub._check_ocr_availability = MagicMock(return_value=False)

        with patch("src.ui.main_window.filedialog") as mock_dialog:
            mock_dialog.askopenfilenames.return_value = (r"C:\docs\a.pdf", r"C:\docs\b.pdf")
            MainWindow._select_files(stub)

        assert len(stub.selected_files) == 2

    def test_empty_selection_returns_early(self):
        """Canceling the dialog should not modify state."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\docs\a.pdf"]

        with patch("src.ui.main_window.filedialog") as mock_dialog:
            mock_dialog.askopenfilenames.return_value = ()
            MainWindow._select_files(stub)

        assert len(stub.selected_files) == 1


# ===========================================================================
# 6. _on_file_drop — deduplication
# ===========================================================================


class TestOnFileDropDedup:
    """Tests for cumulative drag-drop with dedup."""

    def test_new_dropped_files_appended(self):
        """New dropped files should extend selected_files."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\docs\a.pdf"]
        stub._check_ocr_availability = MagicMock(return_value=False)
        stub._reset_drop_zone_border = MagicMock()

        event = MagicMock()
        event.data = r"C:\docs\b.pdf"

        with patch("src.ui.main_window.Path") as MockPath:
            mock_path = MagicMock()
            mock_path.suffix.lower.return_value = ".pdf"
            mock_path.is_file.return_value = True
            MockPath.return_value = mock_path

            MainWindow._on_file_drop(stub, event)

        assert len(stub.selected_files) == 2

    def test_duplicate_dropped_files_skipped(self):
        """Dropped files already in session should be skipped."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\docs\a.pdf"]
        stub._reset_drop_zone_border = MagicMock()

        event = MagicMock()
        event.data = r"C:\docs\a.pdf"

        with patch("src.ui.main_window.Path") as MockPath:
            mock_path = MagicMock()
            mock_path.suffix.lower.return_value = ".pdf"
            mock_path.is_file.return_value = True
            MockPath.return_value = mock_path

            MainWindow._on_file_drop(stub, event)

        assert len(stub.selected_files) == 1
        stub.set_status.assert_called_with("All dropped files are already in the session")


# ===========================================================================
# 7. _start_preprocessing — incremental
# ===========================================================================


class TestStartPreprocessingIncremental:
    """Tests for incremental preprocessing (no clear, accepts file_paths)."""

    def test_sends_only_new_files_to_worker(self):
        """_start_preprocessing(file_paths) should send only those paths."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\a.pdf", r"C:\b.pdf"]
        stub._check_ocr_availability = MagicMock(return_value=True)

        new_files = [r"C:\b.pdf"]
        MainWindow._start_preprocessing(stub, new_files)

        sent_paths = stub._worker_manager.send_command.call_args[0][1]["file_paths"]
        assert sent_paths == [r"C:\b.pdf"]

    def test_does_not_clear_file_table(self):
        """_start_preprocessing should NOT call file_table.clear()."""
        from src.ui.main_window import MainWindow

        source = inspect.getsource(MainWindow._start_preprocessing)
        assert "file_table.clear()" not in source
        assert "processing_results.clear()" not in source

    def test_falls_back_to_selected_files_when_no_arg(self):
        """Without file_paths arg, should use self.selected_files."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\only.pdf"]
        stub._check_ocr_availability = MagicMock(return_value=False)

        MainWindow._start_preprocessing(stub)

        sent_paths = stub._worker_manager.send_command.call_args[0][1]["file_paths"]
        assert sent_paths == [r"C:\only.pdf"]

    def test_disables_controls_during_preprocessing(self):
        """Add files and generate buttons should be disabled."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\a.pdf"]
        stub._check_ocr_availability = MagicMock(return_value=False)

        MainWindow._start_preprocessing(stub)

        stub.add_files_btn.configure.assert_called_with(state="disabled")
        stub.generate_btn.configure.assert_called_with(state="disabled")


# ===========================================================================
# 8. _on_preprocessing_complete — merge
# ===========================================================================


class TestOnPreprocessingCompleteMerge:
    """Tests for cumulative result merging."""

    def test_new_results_appended(self):
        """New filenames should be appended to processing_results."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.processing_results = [_sample_result("a.pdf")]

        new_results = [_sample_result("b.pdf")]
        MainWindow._on_preprocessing_complete(stub, new_results)

        assert len(stub.processing_results) == 2
        filenames = {r["filename"] for r in stub.processing_results}
        assert filenames == {"a.pdf", "b.pdf"}

    def test_existing_results_updated(self):
        """If a filename already exists, it should be updated (not duplicated)."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        old_result = _sample_result("a.pdf", confidence=50)
        stub.processing_results = [old_result]

        new_result = _sample_result("a.pdf", confidence=95)
        MainWindow._on_preprocessing_complete(stub, [new_result])

        assert len(stub.processing_results) == 1
        assert stub.processing_results[0]["confidence"] == 95

    def test_empty_results_leave_existing_intact(self):
        """Empty results list should not modify existing processing_results."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.processing_results = [_sample_result("a.pdf")]

        MainWindow._on_preprocessing_complete(stub, [])

        assert len(stub.processing_results) == 1

    def test_mixed_new_and_existing(self):
        """Batch with both new and existing filenames merges correctly."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.processing_results = [_sample_result("a.pdf", confidence=50)]

        batch = [
            _sample_result("a.pdf", confidence=99),
            _sample_result("b.pdf", confidence=80),
        ]
        MainWindow._on_preprocessing_complete(stub, batch)

        assert len(stub.processing_results) == 2
        by_name = {r["filename"]: r for r in stub.processing_results}
        assert by_name["a.pdf"]["confidence"] == 99
        assert by_name["b.pdf"]["confidence"] == 80

    def test_clears_preprocessing_flag(self):
        """_on_preprocessing_complete should set _preprocessing_active = False."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub._preprocessing_active = True

        MainWindow._on_preprocessing_complete(stub, [])

        assert stub._preprocessing_active is False

    def test_reenables_add_files_button(self):
        """Add files button should be re-enabled after completion."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        MainWindow._on_preprocessing_complete(stub, [_sample_result("a.pdf")])

        stub.add_files_btn.configure.assert_called_with(state="normal")


# ===========================================================================
# 9. _clear_files still wipes everything
# ===========================================================================


class TestClearFilesUnchanged:
    """Verify _clear_files still clears all state."""

    def test_clear_empties_all_lists(self):
        """_clear_files should empty selected_files, processing_results, and table."""
        from src.ui.main_window import MainWindow

        stub = _make_window_stub()
        stub.selected_files = [r"C:\a.pdf", r"C:\b.pdf"]
        stub.processing_results = [_sample_result("a.pdf"), _sample_result("b.pdf")]
        stub._qa_ready = True
        stub._qa_answering_active = True
        stub._qa_results = [{"q": "test"}]
        stub._vector_store_path = "/tmp/store"

        MainWindow._clear_files(stub)

        assert stub.selected_files == []
        assert stub.processing_results == []
        stub.file_table.clear.assert_called_once()
        assert stub._qa_ready is False
