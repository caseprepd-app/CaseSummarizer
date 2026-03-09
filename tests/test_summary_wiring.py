"""
Tests for summary feature wiring (MultiDocSummaryWorker integration).

Covers:
- _start_summary_task() sends command to worker subprocess with correct params
- _start_summary_task() handles empty documents gracefully
- _start_summary_task() prefers preprocessed_text over extracted_text
- _on_summary_complete() extracts individual summaries and meta-summary
- _on_summary_complete() handles failed documents
- _on_summary_complete() clears worker reference and finalizes
- _handle_queue_message() routes multi_doc_result to _on_summary_complete
"""

import threading
from unittest.mock import MagicMock

from src.core.summarization.result_types import (
    DocumentSummaryResult,
    MultiDocumentSummaryResult,
)

# -------------------------------------------------------------------------
# Helper: create a mock MainWindow with the fields summary code needs
# -------------------------------------------------------------------------


def _make_mock_window():
    """Create a mock MainWindow-like object for summary tests."""
    window = MagicMock()
    window._destroying = False
    window._summary_worker = None
    window._pending_tasks = {"vocab": False, "qa": False, "summary": True}
    window._completed_tasks = set()
    window._qa_ready = False
    window._qa_results_lock = threading.Lock()
    window._qa_results = []
    window._queue_poll_id = None
    window._worker_manager = MagicMock()
    window._processing_active = False
    window._preprocessing_active = False
    window.processing_results = [
        {"filename": "complaint.pdf", "extracted_text": "The plaintiff alleges..."},
        {"filename": "answer.pdf", "preprocessed_text": "The defendant denies..."},
    ]
    window.model_manager = MagicMock()
    window.model_manager.model_name = "gemma3:4b"
    return window


# -------------------------------------------------------------------------
# _start_summary_task() tests
# -------------------------------------------------------------------------


class TestStartSummaryTask:
    """Test that _start_summary_task() correctly sends command to subprocess."""

    def test_sends_summary_command(self):
        """_start_summary_task() should send 'summary' command to worker subprocess."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()

        MainWindow._start_summary_task(window)

        window._worker_manager.send_command.assert_called_once()
        cmd_type = window._worker_manager.send_command.call_args[0][0]
        assert cmd_type == "summary"

    def test_passes_correct_documents(self):
        """Command should include documents with filename and extracted_text."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()

        MainWindow._start_summary_task(window)

        call_args = window._worker_manager.send_command.call_args[0][1]
        docs = call_args["documents"]

        assert len(docs) == 2
        assert docs[0]["filename"] == "complaint.pdf"
        assert docs[0]["extracted_text"] == "The plaintiff alleges..."
        # preprocessed_text should be preferred over extracted_text
        assert docs[1]["filename"] == "answer.pdf"
        assert docs[1]["extracted_text"] == "The defendant denies..."

    def test_prefers_preprocessed_text(self):
        """Should use preprocessed_text when both preprocessed and extracted exist."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        window.processing_results = [
            {
                "filename": "doc.pdf",
                "extracted_text": "raw text",
                "preprocessed_text": "clean text",
            }
        ]

        MainWindow._start_summary_task(window)

        call_args = window._worker_manager.send_command.call_args[0][1]
        docs = call_args["documents"]
        assert docs[0]["extracted_text"] == "clean text"

    def test_passes_ai_params_with_model(self):
        """Command should include ai_params with model_name."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()

        MainWindow._start_summary_task(window)

        call_args = window._worker_manager.send_command.call_args[0][1]
        ai_params = call_args["ai_params"]
        assert "model_name" in ai_params
        assert "summary_length" in ai_params
        assert "meta_length" in ai_params

    def test_empty_documents_skips_worker(self):
        """With no valid documents, should skip worker and finalize."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        window.processing_results = []

        MainWindow._start_summary_task(window)

        window._worker_manager.send_command.assert_not_called()
        assert "summary" in window._completed_tasks
        window._finalize_tasks.assert_called_once()

    def test_filters_empty_text_documents(self):
        """Documents with no text should be excluded."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        window.processing_results = [
            {"filename": "good.pdf", "extracted_text": "Some text"},
            {"filename": "empty.pdf", "extracted_text": ""},
            {"filename": "none.pdf"},
        ]

        MainWindow._start_summary_task(window)

        call_args = window._worker_manager.send_command.call_args[0][1]
        docs = call_args["documents"]
        assert len(docs) == 1
        assert docs[0]["filename"] == "good.pdf"

    def test_sets_workflow_phase(self):
        """Should set workflow phase to SUMMARY_RUNNING."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()

        MainWindow._start_summary_task(window)

        window.output_display.set_workflow_phase.assert_called_once()

    def test_starts_polling_if_not_active(self):
        """Should start polling via after() if not already polling."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        window._queue_poll_id = None

        MainWindow._start_summary_task(window)

        window.after.assert_called_once_with(33, window._poll_queue)

    def test_skips_polling_if_already_active(self):
        """Should not start a second poll loop if one is already running."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        window._queue_poll_id = "existing_poll_123"

        MainWindow._start_summary_task(window)

        window.after.assert_not_called()

    def test_updates_status_with_doc_count(self):
        """Status message should include the document count."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()

        MainWindow._start_summary_task(window)

        status_msg = window.set_status.call_args[0][0]
        assert "2" in status_msg  # 2 documents
        assert "summary" in status_msg.lower() or "Generating" in status_msg


# -------------------------------------------------------------------------
# _on_summary_complete() tests
# -------------------------------------------------------------------------


class TestOnSummaryComplete:
    """Test that _on_summary_complete() handles results correctly."""

    def _make_result(self, docs_processed=2, docs_failed=0, meta="Overall summary"):
        """Create a MultiDocumentSummaryResult for testing."""
        individual = {}
        for i in range(docs_processed):
            fname = f"doc_{i}.pdf"
            individual[fname] = DocumentSummaryResult(
                filename=fname,
                summary=f"Summary of doc {i}",
                word_count=50,
                chunk_count=3,
                processing_time_seconds=1.5,
                success=True,
            )
        for i in range(docs_failed):
            fname = f"failed_{i}.pdf"
            individual[fname] = DocumentSummaryResult(
                filename=fname,
                summary="",
                word_count=0,
                chunk_count=0,
                processing_time_seconds=0.1,
                success=False,
                error_message="Model connection failed",
            )

        return MultiDocumentSummaryResult(
            individual_summaries=individual,
            meta_summary=meta,
            total_processing_time_seconds=3.0,
            documents_processed=docs_processed,
            documents_failed=docs_failed,
        )

    def test_displays_meta_summary(self):
        """Should pass meta_summary to output_display.update_outputs()."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        result = self._make_result(meta="The case involves a contract dispute.")

        MainWindow._on_summary_complete(window, result)

        call_kwargs = window.output_display.update_outputs.call_args[1]
        assert call_kwargs["meta_summary"] == "The case involves a contract dispute."

    def test_displays_individual_summaries(self):
        """Should pass individual summaries dict to update_outputs()."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        result = self._make_result(docs_processed=2)

        MainWindow._on_summary_complete(window, result)

        call_kwargs = window.output_display.update_outputs.call_args[1]
        doc_summaries = call_kwargs["document_summaries"]
        assert len(doc_summaries) == 2
        assert "Summary of doc 0" in doc_summaries["doc_0.pdf"]
        assert "Summary of doc 1" in doc_summaries["doc_1.pdf"]

    def test_failed_docs_show_error(self):
        """Failed documents should show error message instead of summary."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        result = self._make_result(docs_processed=1, docs_failed=1)

        MainWindow._on_summary_complete(window, result)

        call_kwargs = window.output_display.update_outputs.call_args[1]
        doc_summaries = call_kwargs["document_summaries"]
        assert "[Error: Model connection failed]" in doc_summaries["failed_0.pdf"]

    def test_marks_summary_completed(self):
        """Should add 'summary' to _completed_tasks."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        result = self._make_result()

        MainWindow._on_summary_complete(window, result)

        assert "summary" in window._completed_tasks

    def test_calls_finalize_tasks(self):
        """Should call _finalize_tasks() after processing result."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        result = self._make_result()

        MainWindow._on_summary_complete(window, result)

        window._finalize_tasks.assert_called_once()

    def test_status_message_all_success(self):
        """Status should mention document count when all succeed."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        result = self._make_result(docs_processed=3, docs_failed=0)

        MainWindow._on_summary_complete(window, result)

        status = window.set_status.call_args[0][0]
        assert "3" in status
        assert "summarized" in status.lower()

    def test_status_message_with_failures(self):
        """Status should mention both succeeded and failed counts."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        result = self._make_result(docs_processed=2, docs_failed=1)

        MainWindow._on_summary_complete(window, result)

        status = window.set_status.call_args[0][0]
        assert "2" in status and "succeeded" in status
        assert "1" in status and "failed" in status


# -------------------------------------------------------------------------
# Message routing test
# -------------------------------------------------------------------------


class TestMultiDocResultRouting:
    """Test that multi_doc_result messages are routed correctly."""

    def test_handle_queue_message_routes_multi_doc_result(self):
        """_handle_queue_message should call _on_summary_complete for multi_doc_result."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()
        mock_result = MagicMock()

        MainWindow._handle_queue_message(window, "multi_doc_result", mock_result)

        window._on_summary_complete.assert_called_once_with(mock_result)

    def test_unhandled_messages_still_logged(self):
        """Unknown message types should not crash."""
        from src.ui.main_window import MainWindow

        window = _make_mock_window()

        # Should not raise
        MainWindow._handle_queue_message(window, "unknown_type", {})
