"""
Tests for Export All format routing (HTML, Word, PDF).

Verifies that _export_all_impl routes to the correct ExportService method
based on the file extension chosen in the save dialog.
"""

from unittest.mock import MagicMock, patch

import pytest


def _make_stub(filepath="report.html"):
    """Create a MainWindow stub with export-related attributes."""
    stub = MagicMock()
    stub._exporting_all = False
    stub.export_all_btn = MagicMock()

    # Vocab data from output_display
    stub.output_display._get_filtered_vocab_data.return_value = [
        {"Term": "plaintiff", "Score": "80"}
    ]
    stub.output_display._get_visible_columns.return_value = ["Term", "Score"]

    # Q&A results
    qa = MagicMock()
    qa.is_exportable = True
    stub._semantic_results = [qa]
    stub.output_display._semantic_panel._results = [qa]

    # Summary text
    stub.output_display._outputs = {"Key Excerpts": "A test summary."}

    # after() for button reset
    stub.after = MagicMock()

    return stub


@pytest.fixture()
def mock_deps():
    """Patch external dependencies of _export_all_impl."""
    with (
        patch("src.ui.main_window.filedialog") as mock_dialog,
        patch("src.ui.main_window.messagebox") as mock_msgbox,
        patch("src.services.get_export_service") as mock_get_svc,
        patch("src.services.DocumentService") as mock_doc_svc,
        patch("src.user_preferences.get_user_preferences") as mock_prefs_fn,
    ):
        mock_prefs = MagicMock()
        mock_prefs.get.return_value = None
        mock_prefs_fn.return_value = mock_prefs

        mock_doc_svc.return_value.get_default_documents_folder.return_value = "/tmp"

        svc = MagicMock()
        svc.export_combined_html.return_value = (True, None)
        svc.export_combined_to_word.return_value = (True, None)
        svc.export_combined_to_pdf.return_value = (True, None)
        mock_get_svc.return_value = svc

        yield {
            "dialog": mock_dialog,
            "msgbox": mock_msgbox,
            "service": svc,
            "prefs": mock_prefs,
        }


class TestExportAllFormatRouting:
    """_export_all_impl routes to correct export method based on extension."""

    def test_html_extension_calls_combined_html(self, mock_deps):
        """Choosing .html routes to export_combined_html."""
        from src.ui.main_window import MainWindow

        mock_deps["dialog"].asksaveasfilename.return_value = "/tmp/report.html"
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        svc = mock_deps["service"]
        svc.export_combined_html.assert_called_once()
        svc.export_combined_to_word.assert_not_called()
        svc.export_combined_to_pdf.assert_not_called()

    def test_docx_extension_calls_combined_word(self, mock_deps):
        """Choosing .docx routes to export_combined_to_word."""
        from src.ui.main_window import MainWindow

        mock_deps["dialog"].asksaveasfilename.return_value = "/tmp/report.docx"
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        svc = mock_deps["service"]
        svc.export_combined_to_word.assert_called_once()
        svc.export_combined_html.assert_not_called()
        svc.export_combined_to_pdf.assert_not_called()

    def test_pdf_extension_calls_combined_pdf(self, mock_deps):
        """Choosing .pdf routes to export_combined_to_pdf."""
        from src.ui.main_window import MainWindow

        mock_deps["dialog"].asksaveasfilename.return_value = "/tmp/report.pdf"
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        svc = mock_deps["service"]
        svc.export_combined_to_pdf.assert_called_once()
        svc.export_combined_html.assert_not_called()
        svc.export_combined_to_word.assert_not_called()

    def test_unknown_extension_defaults_to_html(self, mock_deps):
        """Unknown extension falls through to HTML export."""
        from src.ui.main_window import MainWindow

        mock_deps["dialog"].asksaveasfilename.return_value = "/tmp/report.xyz"
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        mock_deps["service"].export_combined_html.assert_called_once()


class TestExportAllArguments:
    """Correct arguments are passed to each export method."""

    def test_word_receives_vocab_qa_and_summary(self, mock_deps):
        """Word export receives vocab_data, semantic_results, file_path, summary_text."""
        from src.ui.main_window import MainWindow

        mock_deps["dialog"].asksaveasfilename.return_value = "/tmp/report.docx"
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        call_kwargs = mock_deps["service"].export_combined_to_word.call_args
        assert call_kwargs.kwargs["vocab_data"] == [{"Term": "plaintiff", "Score": "80"}]
        assert len(call_kwargs.kwargs["semantic_results"]) == 1
        assert call_kwargs.kwargs["file_path"] == "/tmp/report.docx"
        assert call_kwargs.kwargs["summary_text"] == "A test summary."

    def test_pdf_receives_vocab_qa_and_summary(self, mock_deps):
        """PDF export receives vocab_data, semantic_results, file_path, summary_text."""
        from src.ui.main_window import MainWindow

        mock_deps["dialog"].asksaveasfilename.return_value = "/tmp/report.pdf"
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        call_kwargs = mock_deps["service"].export_combined_to_pdf.call_args
        assert call_kwargs.kwargs["vocab_data"] == [{"Term": "plaintiff", "Score": "80"}]
        assert len(call_kwargs.kwargs["semantic_results"]) == 1
        assert call_kwargs.kwargs["file_path"] == "/tmp/report.pdf"
        assert call_kwargs.kwargs["summary_text"] == "A test summary."

    def test_html_receives_summary_and_visible_columns(self, mock_deps):
        """HTML export receives summary_text and visible_columns."""
        from src.ui.main_window import MainWindow

        mock_deps["dialog"].asksaveasfilename.return_value = "/tmp/report.html"
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        call_kwargs = mock_deps["service"].export_combined_html.call_args
        assert call_kwargs.kwargs["summary_text"] == "A test summary."
        assert call_kwargs.kwargs["visible_columns"] == ["Term", "Score"]


class TestExportAllDialogOptions:
    """File dialog offers all three formats."""

    def test_dialog_includes_html_word_pdf(self, mock_deps):
        """Save dialog filetypes include HTML, Word, and PDF."""
        from src.ui.main_window import MainWindow

        mock_deps["dialog"].asksaveasfilename.return_value = ""
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        call_kwargs = mock_deps["dialog"].asksaveasfilename.call_args
        filetypes = call_kwargs.kwargs["filetypes"]
        extensions = [ft[1] for ft in filetypes]
        assert "*.html" in extensions
        assert "*.docx" in extensions
        assert "*.pdf" in extensions


class TestExportAllStatusAndErrors:
    """Status messages and error handling for each format."""

    def test_success_flashes_button(self, mock_deps):
        """Successful export flashes 'Exported!' on button."""
        from src.ui.main_window import MainWindow

        mock_deps["dialog"].asksaveasfilename.return_value = "/tmp/report.docx"
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        stub.export_all_btn.configure.assert_called_with(text="Exported!")

    def test_success_remembers_export_path(self, mock_deps):
        """Successful export saves last_export_path preference."""
        from pathlib import Path

        from src.ui.main_window import MainWindow

        mock_deps["dialog"].asksaveasfilename.return_value = "/tmp/exports/report.pdf"
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        expected_dir = str(Path("/tmp/exports"))  # OS-normalized
        mock_deps["prefs"].set.assert_called_with("last_export_path", expected_dir)

    def test_word_failure_shows_error(self, mock_deps):
        """Failed Word export shows error dialog with format name."""
        from src.ui.main_window import MainWindow

        mock_deps["service"].export_combined_to_word.return_value = (False, "Disk full")
        mock_deps["dialog"].asksaveasfilename.return_value = "/tmp/report.docx"
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        mock_deps["msgbox"].showerror.assert_called_once()
        error_msg = mock_deps["msgbox"].showerror.call_args[0][1]
        assert "Word" in error_msg
        assert "Disk full" in error_msg

    def test_pdf_failure_shows_error(self, mock_deps):
        """Failed PDF export shows error dialog with format name."""
        from src.ui.main_window import MainWindow

        mock_deps["service"].export_combined_to_pdf.return_value = (False, None)
        mock_deps["dialog"].asksaveasfilename.return_value = "/tmp/report.pdf"
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        mock_deps["msgbox"].showerror.assert_called_once()
        error_msg = mock_deps["msgbox"].showerror.call_args[0][1]
        assert "PDF" in error_msg

    def test_canceled_dialog_does_nothing(self, mock_deps):
        """Empty filepath from dialog means user canceled — no export called."""
        from src.ui.main_window import MainWindow

        mock_deps["dialog"].asksaveasfilename.return_value = ""
        stub = _make_stub()

        MainWindow._export_all_impl(stub)

        svc = mock_deps["service"]
        svc.export_combined_html.assert_not_called()
        svc.export_combined_to_word.assert_not_called()
        svc.export_combined_to_pdf.assert_not_called()

    def test_no_data_shows_warning(self, mock_deps):
        """No vocab, Q&A, or summary shows warning and skips dialog."""
        from src.ui.main_window import MainWindow

        stub = _make_stub()
        stub.output_display._get_filtered_vocab_data.return_value = []
        stub._semantic_results = []
        stub.output_display._outputs = {}

        MainWindow._export_all_impl(stub)

        mock_deps["msgbox"].showwarning.assert_called_once()
        mock_deps["dialog"].asksaveasfilename.assert_not_called()
