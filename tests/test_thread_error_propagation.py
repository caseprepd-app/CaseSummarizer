"""
Tests for thread error propagation fixes.

Covers:
- Q&A daemon thread signals success/failure to main thread via threading.Event
- Main thread detects Q&A thread crash vs. successful completion
- Main thread sends status_error on Q&A timeout
- (GLiNER warm-up timeout tests moved to tests/deprecated/ with algorithm)
"""

import threading
from queue import Queue
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Q&A thread error propagation (workers.py)
# ---------------------------------------------------------------------------


class TestQAThreadErrorPropagation:
    """Test that the main thread detects Q&A thread success vs. failure."""

    def _make_worker(self):
        """Create a ProgressiveExtractionWorker with minimal config."""
        from src.services.workers import ProgressiveExtractionWorker

        worker = ProgressiveExtractionWorker(
            documents=[{"filename": "test.pdf", "extracted_text": "Hello world."}],
            combined_text="Hello world.",
            ui_queue=Queue(),
            embeddings=MagicMock(),
            use_llm=False,
        )
        return worker

    @patch("src.services.workers.VocabularyExtractor")
    def test_qa_success_sets_event(self, mock_extractor_cls):
        """When _build_vector_store succeeds, _qa_succeeded should be set."""
        worker = self._make_worker()

        # Initialize the event and error fields (normally done in execute())
        worker._qa_succeeded = threading.Event()
        worker._qa_error_msg = None

        # Mock the heavy imports and operations inside _build_vector_store
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.return_value = [MagicMock(text="chunk1")]

        mock_result = MagicMock()
        mock_result.persist_dir = "/tmp/test"
        mock_result.chunk_count = 1
        mock_result.chunk_embeddings = None

        mock_builder = MagicMock()
        mock_builder.create_from_unified_chunks.return_value = mock_result

        with (
            patch("src.core.chunking.create_unified_chunker", return_value=mock_chunker),
            patch("src.core.vector_store.VectorStoreBuilder", return_value=mock_builder),
        ):
            worker._build_vector_store()

        assert worker._qa_succeeded.is_set(), "Success event should be set"
        assert worker._qa_error_msg is None, "No error message on success"

    @patch("src.services.workers.VocabularyExtractor")
    def test_qa_failure_stores_error_message(self, mock_extractor_cls):
        """When _build_vector_store crashes, _qa_error_msg should be set."""
        worker = self._make_worker()

        # Initialize the event and error fields
        worker._qa_succeeded = threading.Event()
        worker._qa_error_msg = None

        # Make the chunker import raise an exception
        with patch(
            "src.core.chunking.create_unified_chunker",
            side_effect=RuntimeError("embedding model failed"),
        ):
            worker._build_vector_store()

        assert not worker._qa_succeeded.is_set(), "Success event should NOT be set"
        assert worker._qa_error_msg == "embedding model failed"

    @patch("src.services.workers.VocabularyExtractor")
    def test_qa_failure_sends_error_to_queue(self, mock_extractor_cls):
        """When _build_vector_store crashes, error messages go to UI queue."""
        worker = self._make_worker()
        worker._qa_succeeded = threading.Event()
        worker._qa_error_msg = None

        with patch(
            "src.core.chunking.create_unified_chunker",
            side_effect=RuntimeError("disk full"),
        ):
            worker._build_vector_store()

        # Drain queue and check for error messages
        messages = []
        while not worker.ui_queue.empty():
            messages.append(worker.ui_queue.get_nowait())

        # Should have status_error and qa_error messages
        msg_types = [m[0] if isinstance(m, tuple) else m for m in messages]
        assert "status_error" in msg_types, f"Expected status_error in {msg_types}"
        assert "qa_error" in msg_types, f"Expected qa_error in {msg_types}"
