"""
Tests for thread error propagation fixes.

Covers:
- Q&A daemon thread signals success/failure to main thread via threading.Event
- Main thread detects Q&A thread crash vs. successful completion
- Main thread sends status_error on Q&A timeout
- GLiNER warm-up timeout sets _load_error to prevent repeated waits
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


# ---------------------------------------------------------------------------
# GLiNER warm-up timeout (gliner_algorithm.py)
# ---------------------------------------------------------------------------


class TestGLiNERWarmupTimeout:
    """Test that GLiNER warm-up timeout sets _load_error to prevent re-waits."""

    def test_timeout_sets_load_error(self):
        """After timeout, _load_error should be set so next call fails fast."""
        from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

        algo = GLiNERAlgorithm(labels=["person"])

        # Simulate a warm-up that never completes by NOT setting _model_ready
        # Use a very short timeout to avoid slow tests
        with patch("src.core.vocabulary.algorithms.gliner_algorithm._WARMUP_TIMEOUT_SEC", 0.01):
            result = algo._wait_for_model()

        assert result is False, "Should return False on timeout"
        assert algo._load_error is not None, "_load_error should be set after timeout"
        assert "timed out" in algo._load_error

    def test_second_call_after_timeout_fails_fast(self):
        """After timeout sets _load_error, next _wait_for_model returns immediately."""
        from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

        algo = GLiNERAlgorithm(labels=["person"])

        # First call: times out
        with patch("src.core.vocabulary.algorithms.gliner_algorithm._WARMUP_TIMEOUT_SEC", 0.01):
            algo._wait_for_model()

        # Second call: should return False immediately (not wait again)
        # because _load_error is now set, hitting the early-return branch
        import time

        start = time.time()
        result = algo._wait_for_model()
        elapsed = time.time() - start

        assert result is False, "Should still return False"
        assert elapsed < 1.0, f"Should return fast but took {elapsed:.1f}s"

    @patch(
        "src.core.vocabulary.algorithms.gliner_algorithm.GLiNERAlgorithm._load_model",
        side_effect=RuntimeError("model load blocked by test"),
    )
    def test_timeout_extract_returns_skipped_result(self, mock_load):
        """extract() should return a skipped result when warm-up timed out."""
        from src.core.vocabulary.algorithms.gliner_algorithm import GLiNERAlgorithm

        algo = GLiNERAlgorithm(labels=["person"])

        # Simulate: warm_up() was called, timed out, _load_error is set
        # _wait_for_model already ran and set _load_error.
        # Now extract() is called again — _model is None, _model_ready is NOT
        # set (warm_up never finished), _load_error IS set.
        algo._model_ready = threading.Event()  # not set
        algo._load_error = "warm-up timed out after 120s"

        result = algo.extract("John Smith went to the hospital.")

        assert len(result.candidates) == 0
        assert result.metadata.get("skipped") is True
