"""
Tests for GUI button re-entrancy guards.

Validates that buttons and handlers are protected against double-click,
concurrent invocation, and keyboard shortcut bypass during active processing.
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub():
    """Create a stub with MainWindow attributes used by handlers."""
    stub = MagicMock()
    stub._processing_active = False
    stub._preprocessing_active = False
    stub._qa_ready = False
    stub._qa_answering_active = False
    stub._qa_failed = False
    stub._vector_store_path = None
    stub._worker_ready_retries = 0
    stub.selected_files = []
    stub.processing_results = []
    stub._export_all_visible = False
    stub._settings_dialog_open = False
    stub._exporting_all = False
    # Widget mocks
    stub.add_files_btn = MagicMock()
    stub.clear_files_btn = MagicMock()
    stub.generate_btn = MagicMock()
    stub.followup_btn = MagicMock()
    stub.followup_entry = MagicMock()
    stub.export_all_btn = MagicMock()
    stub.left_panel = MagicMock()
    return stub


# ===========================================================================
# _on_file_drop: guard during processing
# ===========================================================================


class TestFileDropGuard:
    """_on_file_drop should reject drops during active processing."""

    def test_rejects_during_processing(self):
        """Drop during _processing_active is rejected."""
        stub = _make_stub()
        stub._processing_active = True

        from src.ui.main_window import MainWindow

        event = MagicMock()
        event.data = "test.pdf"
        MainWindow._on_file_drop(stub, event)

        stub.set_status.assert_called_once_with("Cannot add files during active processing")

    def test_rejects_during_preprocessing(self):
        """Drop during _preprocessing_active is rejected."""
        stub = _make_stub()
        stub._preprocessing_active = True

        from src.ui.main_window import MainWindow

        event = MagicMock()
        event.data = "test.pdf"
        MainWindow._on_file_drop(stub, event)

        stub.set_status.assert_called_once_with("Cannot add files during active processing")

    def test_proceeds_when_idle(self):
        """Drop when idle should proceed past the guard (may hit later check)."""
        stub = _make_stub()

        from src.ui.main_window import MainWindow

        event = MagicMock()
        event.data = "nonexistent.pdf"
        MainWindow._on_file_drop(stub, event)

        # Should NOT have returned early with the guard message
        guard_calls = [c for c in stub.set_status.call_args_list if "Cannot add files" in str(c)]
        assert len(guard_calls) == 0


# ===========================================================================
# _select_files: guard during processing
# ===========================================================================


class TestSelectFilesGuard:
    """_select_files should reject during active processing (Ctrl+O bypass)."""

    def test_rejects_during_processing(self):
        """Ctrl+O during processing is silently ignored."""
        stub = _make_stub()
        stub._processing_active = True

        from src.ui.main_window import MainWindow

        with patch("src.ui.main_window.filedialog") as mock_dlg:
            MainWindow._select_files(stub)

        # File dialog should NOT be opened
        mock_dlg.askopenfilenames.assert_not_called()

    def test_rejects_during_preprocessing(self):
        """Ctrl+O during preprocessing is silently ignored."""
        stub = _make_stub()
        stub._preprocessing_active = True

        from src.ui.main_window import MainWindow

        with patch("src.ui.main_window.filedialog") as mock_dlg:
            MainWindow._select_files(stub)

        mock_dlg.askopenfilenames.assert_not_called()

    def test_proceeds_when_idle(self):
        """Ctrl+O when idle should open file dialog."""
        stub = _make_stub()

        from src.ui.main_window import MainWindow

        with patch("src.ui.main_window.filedialog") as mock_dlg:
            mock_dlg.askopenfilenames.return_value = ()  # User cancels
            MainWindow._select_files(stub)

        mock_dlg.askopenfilenames.assert_called_once()


# ===========================================================================
# _start_preprocessing: re-entrancy guard
# ===========================================================================


class TestStartPreprocessingGuard:
    """_start_preprocessing should reject if already preprocessing."""

    def test_rejects_when_already_preprocessing(self):
        """Second call during preprocessing is ignored."""
        stub = _make_stub()
        stub._preprocessing_active = True

        from src.ui.main_window import MainWindow

        MainWindow._start_preprocessing(stub, ["test.pdf"])

        # Should NOT disable buttons (guard returns early)
        stub.add_files_btn.configure.assert_not_called()

    def test_proceeds_when_not_preprocessing(self):
        """First call should proceed normally."""
        stub = _make_stub()
        stub._worker_manager = MagicMock()
        stub._worker_manager.is_ready.return_value = True

        from src.ui.main_window import MainWindow

        with patch("src.ui.main_window.MainWindow._check_ocr_availability", return_value=True):
            MainWindow._start_preprocessing(stub, ["test.pdf"])

        # Should disable buttons as part of normal flow
        stub.add_files_btn.configure.assert_called()


# ===========================================================================
# _export_all: re-entrancy guard
# ===========================================================================


class TestExportAllGuard:
    """_export_all should reject double-click."""

    def test_rejects_when_already_exporting(self):
        """Second click during export is ignored."""
        stub = _make_stub()
        stub._exporting_all = True

        from src.ui.main_window import MainWindow

        MainWindow._export_all(stub)

        # _export_all_impl should NOT be called
        # (if it were, it would try to import DocumentService etc.)
        # Check the flag wasn't changed
        assert stub._exporting_all is True

    def test_clears_flag_after_impl(self):
        """Flag should be cleared after _export_all_impl completes (via finally)."""
        stub = _make_stub()
        stub._exporting_all = False
        # Make _export_all_impl a no-op
        stub._export_all_impl.return_value = None

        from src.ui.main_window import MainWindow

        MainWindow._export_all(stub)

        # Flag should be cleared after (finally block)
        assert stub._exporting_all is False


# ===========================================================================
# _open_settings / _open_model_settings / _open_corpus_dialog: dialog guard
# ===========================================================================


class TestSettingsDialogGuard:
    """Settings dialog buttons use _settings_dialog_open guard."""

    def test_open_settings_rejects_when_open(self):
        """Double-click on settings is ignored when dialog already open."""
        stub = _make_stub()
        stub._settings_dialog_open = True

        from src.ui.main_window import MainWindow

        MainWindow._open_settings(stub)

        # Should not try to import SettingsDialog (guard returns early)
        # Verify no dialog-related calls happened
        stub._refresh_corpus_dropdown.assert_not_called()

    def test_open_model_settings_rejects_when_open(self):
        """Double-click on model configure is ignored."""
        stub = _make_stub()
        stub._settings_dialog_open = True

        from src.ui.main_window import MainWindow

        MainWindow._open_model_settings(stub)

        stub._refresh_corpus_dropdown.assert_not_called()

    def test_open_corpus_dialog_rejects_when_open(self):
        """Double-click on manage corpus is ignored."""
        stub = _make_stub()
        stub._settings_dialog_open = True

        from src.ui.main_window import MainWindow

        MainWindow._open_corpus_dialog(stub)

        stub._refresh_corpus_dropdown.assert_not_called()

    def test_open_settings_clears_flag_after(self):
        """Flag is cleared after dialog closes (via finally)."""
        stub = _make_stub()
        stub._settings_dialog_open = False

        from src.ui.main_window import MainWindow

        # SettingsDialog is imported inside the function body, so patch at source
        with (
            patch(
                "src.ui.settings.settings_dialog.SettingsDialog",
                side_effect=Exception("no Tk"),
            ),
            patch("src.user_preferences.get_user_preferences"),
        ):
            MainWindow._open_settings(stub)

        # Flag should be cleared even after exception
        assert stub._settings_dialog_open is False


# ===========================================================================
# Vocab export: re-entrancy guard
# ===========================================================================


class TestVocabExportGuard:
    """_export_vocab should reject double-click."""

    def test_rejects_when_already_exporting(self):
        """Second dropdown selection during export is ignored."""
        stub = MagicMock()
        stub._exporting_vocab = True

        from src.ui.dynamic_output import DynamicOutputWidget

        DynamicOutputWidget._export_vocab(stub, "csv")

        # _export_vocab_impl should NOT be called
        stub._export_vocab_impl.assert_not_called()


# ===========================================================================
# QA export: re-entrancy guard
# ===========================================================================


class TestQaExportGuard:
    """_export_qa should reject double-click."""

    def test_rejects_when_already_exporting(self):
        """Second dropdown selection during export is ignored."""
        stub = MagicMock()
        stub._exporting_qa = True

        from src.ui.qa_panel import QAPanel

        QAPanel._export_qa(stub, "csv")

        # _export_qa_impl should NOT be called
        stub._export_qa_impl.assert_not_called()
