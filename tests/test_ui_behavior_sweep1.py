"""
Regression tests for UI Behavior Sweep 1 fixes.

Bug 1: _clear_files allowed during _followup_pending — could corrupt session state.
Bug 2: _remove_file allowed during active processing — data inconsistency.
Bug 3: _refresh_tabs left Key Excerpts textbox editable by not managing state.
"""

import threading
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_main_window_stub():
    """Create a stub with MainWindow attributes used by handlers."""
    stub = MagicMock()
    stub._processing_active = False
    stub._preprocessing_active = False
    stub._semantic_ready = False
    stub._semantic_answering_active = False
    stub._semantic_failed = False
    stub._key_sentences_pending = False
    stub._followup_pending = False
    stub._vector_store_path = None
    stub._worker_ready_retries = 0
    stub._exporting_all = False
    stub._export_all_visible = False
    stub._semantic_results = []
    stub._semantic_results_lock = threading.Lock()
    stub.selected_files = ["test.pdf"]
    stub.processing_results = [{"filename": "test.pdf", "status": "success"}]
    # Widget mocks
    stub.add_files_btn = MagicMock()
    stub.clear_files_btn = MagicMock()
    stub.generate_btn = MagicMock()
    stub.followup_btn = MagicMock()
    stub.followup_entry = MagicMock()
    stub.export_all_btn = MagicMock()
    stub.file_table = MagicMock()
    stub.output_display = MagicMock()
    stub.output_display.document_preview_filename = None
    stub.stats_label = MagicMock()
    stub.doc_count_label = MagicMock()
    stub.complete_btn = MagicMock()
    stub.complete_btn.winfo_ismapped.return_value = False
    return stub


# ===========================================================================
# Bug 1: _clear_files should block during _followup_pending
# ===========================================================================


class TestClearFilesFollowupGuard:
    """_clear_files must not proceed while a follow-up search is pending."""

    def test_blocks_during_followup_pending(self):
        """Clearing files during a followup search is blocked."""
        stub = _make_main_window_stub()
        stub._followup_pending = True

        from src.ui.main_window import MainWindow

        MainWindow._clear_files(stub)

        # Should not have cleared the file list
        stub.file_table.clear.assert_not_called()

    def test_proceeds_when_no_followup(self):
        """Clearing files when no followup is pending succeeds."""
        stub = _make_main_window_stub()
        stub._followup_pending = False

        from src.ui.main_window import MainWindow

        with patch("src.ui.main_window.MainWindow._update_generate_button_state"):
            MainWindow._clear_files(stub)

        # Should have cleared the file list
        stub.file_table.clear.assert_called_once()


# ===========================================================================
# Bug 2: _remove_file should block during processing/preprocessing
# ===========================================================================


class TestRemoveFileProcessingGuard:
    """_remove_file must not allow removal during active processing."""

    def test_blocks_during_processing(self):
        """Removing a file during _processing_active is blocked."""
        stub = _make_main_window_stub()
        stub._processing_active = True

        from src.ui.main_window import MainWindow

        MainWindow._remove_file(stub, "test.pdf")

        stub.set_status.assert_called_once_with("Cannot remove files during processing")
        # File should still be in the list
        assert "test.pdf" in stub.selected_files

    def test_blocks_during_preprocessing(self):
        """Removing a file during _preprocessing_active is blocked."""
        stub = _make_main_window_stub()
        stub._preprocessing_active = True

        from src.ui.main_window import MainWindow

        MainWindow._remove_file(stub, "test.pdf")

        stub.set_status.assert_called_once_with("Cannot remove files during processing")

    def test_proceeds_when_idle(self):
        """Removing a file when idle succeeds."""
        stub = _make_main_window_stub()

        from src.ui.main_window import MainWindow

        MainWindow._remove_file(stub, "test.pdf")

        stub.file_table.remove_result.assert_called_once_with("test.pdf")


# ===========================================================================
# Bug 3: summary_text_display state management in _refresh_tabs
# ===========================================================================


class TestSummaryTextboxStateManagement:
    """Key Excerpts textbox must be set to disabled after content is inserted."""

    def test_summary_textbox_disabled_after_refresh(self):
        """After _refresh_tabs inserts summary text, textbox is read-only."""
        from src.ui.dynamic_output import DynamicOutputWidget

        stub = MagicMock(spec=DynamicOutputWidget)
        stub._outputs = {
            "Names & Vocabulary": [],
            "Search": [],
            "Key Excerpts": "Test summary text",
            "Meta-Summary": "",
            "Rare Word List (CSV)": [],
            "Semantic Results": [],
        }
        stub._document_summaries = {}
        stub._extraction_source = "none"
        stub._filtered_vocab_data_raw = []
        stub.summary_text_display = MagicMock()
        stub._semantic_panel = MagicMock()
        stub.tabview = MagicMock()
        stub.tabview.get.return_value = "Key Excerpts"

        DynamicOutputWidget._refresh_tabs(stub)

        # Verify state was set to normal before insert, then disabled after
        calls = stub.summary_text_display.configure.call_args_list
        state_calls = [c for c in calls if "state" in str(c)]
        # Should have at least 2 state calls: normal then disabled
        assert len(state_calls) >= 2
        # Last state call should be disabled
        last_state_call = state_calls[-1]
        assert last_state_call == ((), {"state": "disabled"})

    def test_doc_summaries_also_disable_textbox(self):
        """Textbox is disabled after individual document summaries are inserted."""
        from src.ui.dynamic_output import DynamicOutputWidget

        stub = MagicMock(spec=DynamicOutputWidget)
        stub._outputs = {
            "Names & Vocabulary": [],
            "Search": [],
            "Key Excerpts": "",
            "Meta-Summary": "",
            "Rare Word List (CSV)": [],
            "Semantic Results": [],
        }
        stub._document_summaries = {"doc1.pdf": "Summary of doc1"}
        stub._extraction_source = "none"
        stub._filtered_vocab_data_raw = []
        stub.summary_text_display = MagicMock()
        stub._semantic_panel = MagicMock()
        stub.tabview = MagicMock()
        stub.tabview.get.return_value = "Key Excerpts"

        DynamicOutputWidget._refresh_tabs(stub)

        # Verify state was set to disabled at end
        calls = stub.summary_text_display.configure.call_args_list
        state_calls = [c for c in calls if "state" in str(c)]
        assert len(state_calls) >= 2
        last_state_call = state_calls[-1]
        assert last_state_call == ((), {"state": "disabled"})
