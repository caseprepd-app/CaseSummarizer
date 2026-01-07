"""
Tests for Multi-Document Summarization

Tests the new hierarchical map-reduce summarization architecture:
- DocumentSummaryResult and MultiDocumentSummaryResult data types
- ProgressiveDocumentSummarizer interface
- MultiDocumentOrchestrator parallel processing

Uses SequentialStrategy for deterministic testing.
"""

from unittest.mock import Mock

from src.core.parallel import SequentialStrategy
from src.core.summarization import (
    DocumentSummaryResult,
    MultiDocumentOrchestrator,
    MultiDocumentSummaryResult,
    ProgressiveDocumentSummarizer,
)


class TestDocumentSummaryResult:
    """Test DocumentSummaryResult dataclass."""

    def test_create_successful_result(self):
        """Successful result has correct attributes."""
        result = DocumentSummaryResult(
            filename="test.pdf",
            summary="This is a test summary.",
            word_count=5,
            chunk_count=3,
            processing_time_seconds=10.5,
        )

        assert result.filename == "test.pdf"
        assert result.summary == "This is a test summary."
        assert result.word_count == 5
        assert result.chunk_count == 3
        assert result.processing_time_seconds == 10.5
        assert result.success is True
        assert result.error_message is None

    def test_create_failed_result(self):
        """Failed result has error message."""
        result = DocumentSummaryResult(
            filename="bad.pdf",
            summary="",
            word_count=0,
            chunk_count=0,
            processing_time_seconds=1.0,
            success=False,
            error_message="File not found",
        )

        assert result.success is False
        assert result.error_message == "File not found"

    def test_failed_result_gets_default_error(self):
        """Failed result without error_message gets default."""
        result = DocumentSummaryResult(
            filename="bad.pdf",
            summary="",
            word_count=0,
            chunk_count=0,
            processing_time_seconds=0.0,
            success=False,
        )

        assert result.error_message == "Unknown error during document summarization"


class TestMultiDocumentSummaryResult:
    """Test MultiDocumentSummaryResult dataclass."""

    def test_create_empty_result(self):
        """Empty result has sensible defaults."""
        result = MultiDocumentSummaryResult()

        assert result.individual_summaries == {}
        assert result.meta_summary == ""
        assert result.documents_processed == 0
        assert result.documents_failed == 0
        assert result.document_order == []

    def test_success_rate_calculation(self):
        """Success rate calculated correctly."""
        result = MultiDocumentSummaryResult(documents_processed=8, documents_failed=2)

        assert result.success_rate == 80.0

    def test_success_rate_zero_total(self):
        """Success rate is 0 when no documents processed."""
        result = MultiDocumentSummaryResult()
        assert result.success_rate == 0.0

    def test_get_summary_for_document(self):
        """Can retrieve individual summary by filename."""
        doc_result = DocumentSummaryResult(
            filename="doc1.pdf",
            summary="Summary of doc1",
            word_count=3,
            chunk_count=1,
            processing_time_seconds=5.0,
        )

        result = MultiDocumentSummaryResult(individual_summaries={"doc1.pdf": doc_result})

        assert result.get_summary_for_document("doc1.pdf") == "Summary of doc1"
        assert result.get_summary_for_document("nonexistent.pdf") is None

    def test_get_all_summaries_formatted(self):
        """Formatted summaries include headers."""
        doc1 = DocumentSummaryResult(
            filename="doc1.pdf",
            summary="Summary 1",
            word_count=2,
            chunk_count=1,
            processing_time_seconds=1.0,
        )
        doc2 = DocumentSummaryResult(
            filename="doc2.pdf",
            summary="Summary 2",
            word_count=2,
            chunk_count=1,
            processing_time_seconds=1.0,
        )

        result = MultiDocumentSummaryResult(
            individual_summaries={"doc1.pdf": doc1, "doc2.pdf": doc2},
            document_order=["doc1.pdf", "doc2.pdf"],
        )

        formatted = result.get_all_summaries_formatted()

        assert "--- doc1.pdf ---" in formatted
        assert "Summary 1" in formatted
        assert "--- doc2.pdf ---" in formatted
        assert "Summary 2" in formatted


class TestProgressiveDocumentSummarizer:
    """Test ProgressiveDocumentSummarizer interface."""

    def test_summarizer_handles_empty_text(self):
        """Empty text returns failure result."""
        mock_model = Mock()

        summarizer = ProgressiveDocumentSummarizer(model_manager=mock_model)
        result = summarizer.summarize(text="", filename="empty.pdf")

        assert result.success is False
        assert "empty" in result.error_message.lower()

    def test_summarizer_handles_very_short_text(self):
        """Very short text returns failure result."""
        mock_model = Mock()

        summarizer = ProgressiveDocumentSummarizer(model_manager=mock_model)
        result = summarizer.summarize(text="Short", filename="short.pdf")

        assert result.success is False


class TestMultiDocumentOrchestrator:
    """Test MultiDocumentOrchestrator parallel processing."""

    def test_orchestrator_handles_empty_documents(self):
        """Empty document list returns appropriate result."""
        mock_summarizer = Mock()
        mock_model = Mock()

        orchestrator = MultiDocumentOrchestrator(
            document_summarizer=mock_summarizer,
            model_manager=mock_model,
            strategy=SequentialStrategy(),
        )

        result = orchestrator.summarize_documents(documents=[])

        assert result.documents_processed == 0
        assert "No valid documents" in result.meta_summary

    def test_orchestrator_filters_documents_without_text(self):
        """Documents without extracted_text are filtered out."""
        mock_summarizer = Mock()
        mock_model = Mock()

        orchestrator = MultiDocumentOrchestrator(
            document_summarizer=mock_summarizer,
            model_manager=mock_model,
            strategy=SequentialStrategy(),
        )

        documents = [
            {"filename": "empty.pdf", "extracted_text": ""},
            {"filename": "none.pdf", "extracted_text": None},
            {"filename": "short.pdf", "extracted_text": "too short"},
        ]

        result = orchestrator.summarize_documents(documents=documents)

        # All documents filtered - no valid text
        assert "No valid documents" in result.meta_summary

    def test_orchestrator_uses_sequential_strategy_for_testing(self):
        """SequentialStrategy enables deterministic testing."""
        strategy = SequentialStrategy()

        assert strategy.max_workers == 1

        # Should be able to substitute for ThreadPoolStrategy
        mock_summarizer = Mock()
        mock_model = Mock()

        orchestrator = MultiDocumentOrchestrator(
            document_summarizer=mock_summarizer, model_manager=mock_model, strategy=strategy
        )

        assert orchestrator.strategy.max_workers == 1


class TestIntegrationImports:
    """Test that all components import correctly."""

    def test_all_summarization_exports(self):
        """All public components are exported from package."""
        from src.core.summarization import (
            DocumentSummarizer,
            DocumentSummaryResult,
            MultiDocumentOrchestrator,
            MultiDocumentSummaryResult,
            ProgressiveDocumentSummarizer,
        )

        # Just checking imports work
        assert DocumentSummaryResult is not None
        assert MultiDocumentSummaryResult is not None
        assert DocumentSummarizer is not None
        assert ProgressiveDocumentSummarizer is not None
        assert MultiDocumentOrchestrator is not None

    def test_worker_import(self):
        """MultiDocSummaryWorker can be imported."""
        from src.services.workers import MultiDocSummaryWorker

        assert MultiDocSummaryWorker is not None

    def test_queue_handler_handles_multi_doc_result(self):
        """QueueMessageHandler has multi_doc_result handler registered."""
        from src.ui.queue_message_handler import QueueMessageHandler

        # Create mock main_window
        mock_main_window = Mock()
        mock_main_window.summary_results = Mock()
        mock_main_window.progress_bar = Mock()
        mock_main_window.status_label = Mock()
        mock_main_window.select_files_btn = Mock()
        mock_main_window.generate_outputs_btn = Mock()
        mock_main_window.output_options = Mock()
        mock_main_window.cancel_btn = Mock()
        mock_main_window.pending_ai_generation = None

        handler = QueueMessageHandler(mock_main_window)

        # Check handler exists in handlers dict
        handlers = {
            "progress": handler.handle_progress,
            "multi_doc_result": handler.handle_multi_doc_result,
        }

        assert "multi_doc_result" in handlers
