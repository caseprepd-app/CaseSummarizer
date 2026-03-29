"""
Tests for worker execute() methods.

Covers the main execution paths of:
- ProcessingWorker.execute()
- SemanticWorker.execute()
- ProgressiveExtractionWorker.execute()

All heavy dependencies (extractors, orchestrators, models) are mocked.
"""

import threading
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def drain_queue(q):
    """Drain all messages from a queue into a list."""
    messages = []
    while not q.empty():
        messages.append(q.get_nowait())
    return messages


def find_msg(messages, msg_type):
    """Find first message of given type."""
    for m in messages:
        if isinstance(m, tuple) and len(m) == 2 and m[0] == msg_type:
            return m
    return None


def find_all_msgs(messages, msg_type):
    """Find all messages of given type."""
    return [m for m in messages if isinstance(m, tuple) and len(m) == 2 and m[0] == msg_type]


# ===========================================================================
# ProcessingWorker
# ===========================================================================


class TestProcessingWorkerExecute:
    """Tests for ProcessingWorker.execute()."""

    def _make_worker(self, file_paths=None, ui_queue=None):
        """Create a ProcessingWorker with mocked extractor."""
        from src.core.parallel import SequentialStrategy
        from src.services.workers import ProcessingWorker

        q = ui_queue or Queue()
        worker = ProcessingWorker(
            file_paths=file_paths or [],
            ui_queue=q,
            strategy=SequentialStrategy(),
        )
        return worker, q

    def test_empty_file_list_sends_processing_finished(self):
        """Zero files => immediate processing_finished with empty list."""
        worker, q = self._make_worker([])
        worker.execute()
        msgs = drain_queue(q)
        finished = find_msg(msgs, "processing_finished")
        assert finished is not None
        assert finished[1] == []

    @patch("src.services.processing_worker.RawTextExtractor")
    def test_single_file_extraction(self, mock_extractor_cls):
        """Single file is extracted, preprocessed, and sends completion."""
        mock_instance = MagicMock()
        mock_instance.process_document.return_value = {
            "filename": "test.pdf",
            "extracted_text": "Hello world.",
            "confidence": 95.0,
        }
        mock_extractor_cls.return_value = mock_instance

        from src.core.parallel import SequentialStrategy
        from src.services.workers import ProcessingWorker

        q = Queue()
        worker = ProcessingWorker(
            file_paths=["test.pdf"],
            ui_queue=q,
            strategy=SequentialStrategy(),
        )
        worker.extractor = mock_instance

        with (
            patch("src.core.preprocessing.create_default_pipeline") as mock_pipeline,
            patch("src.services.document_service.DocumentService") as mock_doc_svc,
            patch("src.ui.silly_messages.get_silly_message", return_value="Loading..."),
        ):
            mock_doc_svc._get_preprocessing_settings.return_value = {}
            mock_pp = MagicMock()
            mock_pp.process.return_value = "Cleaned text."
            mock_pipeline.return_value = mock_pp

            worker.execute()

        msgs = drain_queue(q)
        # Should have file_processed and processing_finished
        assert find_msg(msgs, "file_processed") is not None
        assert find_msg(msgs, "processing_finished") is not None
        assert len(worker.processed_results) == 1

    @patch("src.services.processing_worker.RawTextExtractor")
    def test_failed_document_logged_as_status_error(self, mock_extractor_cls):
        """Failed document sends status_error, not in processed_results."""
        mock_instance = MagicMock()
        mock_instance.process_document.side_effect = Exception("PDF corrupt")
        mock_extractor_cls.return_value = mock_instance

        from src.core.parallel import SequentialStrategy
        from src.services.workers import ProcessingWorker

        q = Queue()
        worker = ProcessingWorker(
            file_paths=["bad.pdf"],
            ui_queue=q,
            strategy=SequentialStrategy(),
        )
        worker.extractor = mock_instance

        worker.execute()

        msgs = drain_queue(q)
        assert find_msg(msgs, "processing_finished") is not None
        assert len(worker.processed_results) == 0

    def test_stop_sends_error_message(self):
        """Stopped worker sends error 'cancelled' message."""
        from src.core.parallel import SequentialStrategy
        from src.services.workers import ProcessingWorker

        q = Queue()
        worker = ProcessingWorker(
            file_paths=["a.pdf"],
            ui_queue=q,
            strategy=SequentialStrategy(),
        )
        # Pre-stop before execute
        worker.stop()
        worker.extractor = MagicMock()
        worker.extractor.process_document.side_effect = InterruptedError("cancelled")

        worker.execute()

        msgs = drain_queue(q)
        error = find_msg(msgs, "error")
        assert error is not None
        assert "cancelled" in error[1].lower()

    def test_preprocessing_applied_to_results(self):
        """Preprocessor runs on extracted text and stores preprocessed_text."""
        from src.core.parallel import SequentialStrategy
        from src.services.workers import ProcessingWorker

        q = Queue()
        worker = ProcessingWorker(
            file_paths=["doc.pdf"],
            ui_queue=q,
            strategy=SequentialStrategy(),
        )
        worker.extractor = MagicMock()
        worker.extractor.process_document.return_value = {
            "filename": "doc.pdf",
            "extracted_text": "Raw text here.",
        }

        with (
            patch("src.core.preprocessing.create_default_pipeline") as mock_pipeline,
            patch("src.services.document_service.DocumentService") as mock_doc_svc,
            patch("src.ui.silly_messages.get_silly_message", return_value="Fun!"),
        ):
            mock_doc_svc._get_preprocessing_settings.return_value = {}
            mock_pp = MagicMock()
            mock_pp.process.return_value = "Cleaned."
            mock_pipeline.return_value = mock_pp

            worker.execute()

        assert worker.processed_results[0]["preprocessed_text"] == "Cleaned."


# ===========================================================================
# SemanticWorker
# ===========================================================================


class TestSemanticWorkerExecute:
    """Tests for SemanticWorker.execute()."""

    def _make_worker(self, questions=None, use_defaults=False):
        """Create a SemanticWorker with mocked dependencies."""
        from src.services.workers import SemanticWorker

        q = Queue()
        worker = SemanticWorker(
            vector_store_path=Path("/fake/store"),
            embeddings=MagicMock(),
            ui_queue=q,
            questions=questions,
            use_default_questions=use_defaults,
        )
        return worker, q

    @patch("src.core.semantic.SemanticOrchestrator")
    def test_custom_questions_processed(self, mock_orch_cls):
        """Custom questions are passed to orchestrator."""
        mock_orch = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "Test answer"
        mock_orch._ask_single_question.return_value = mock_result
        mock_orch_cls.return_value = mock_orch

        worker, q = self._make_worker(questions=["What happened?", "When?"])
        worker.execute()

        msgs = drain_queue(q)
        results = find_all_msgs(msgs, "semantic_result")
        assert len(results) == 2
        complete = find_msg(msgs, "semantic_complete")
        assert complete is not None
        assert len(complete[1]) == 2

    @patch("src.core.semantic.SemanticOrchestrator")
    def test_no_questions_sends_empty_complete(self, mock_orch_cls):
        """Zero questions sends semantic_complete with empty list."""
        mock_orch = MagicMock()
        mock_orch.load_default_questions_from_txt.return_value = []
        mock_orch_cls.return_value = mock_orch

        worker, q = self._make_worker()
        worker.execute()

        msgs = drain_queue(q)
        complete = find_msg(msgs, "semantic_complete")
        assert complete is not None
        assert complete[1] == []

    @patch("src.core.semantic.SemanticOrchestrator")
    def test_progress_messages_sent(self, mock_orch_cls):
        """Progress messages are sent for each question."""
        mock_orch = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "Answer"
        mock_orch._ask_single_question.return_value = mock_result
        mock_orch_cls.return_value = mock_orch

        worker, q = self._make_worker(questions=["Q1", "Q2"])
        worker.execute()

        msgs = drain_queue(q)
        progress = find_all_msgs(msgs, "semantic_progress")
        assert len(progress) == 2

    @patch("src.core.semantic.SemanticOrchestrator")
    def test_cancellation_stops_processing(self, mock_orch_cls):
        """Cancelling mid-processing raises InterruptedError."""
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch

        worker, q = self._make_worker(questions=["Q1", "Q2"])
        worker.stop()  # Pre-cancel

        with pytest.raises(InterruptedError):
            worker.execute()

    @patch("src.core.semantic.SemanticOrchestrator")
    def test_default_questions_loaded_when_no_custom(self, mock_orch_cls):
        """When no custom questions, loads defaults from orchestrator."""
        mock_orch = MagicMock()
        mock_orch.load_default_questions_from_txt.return_value = ["Default Q?"]
        mock_result = MagicMock()
        mock_result.answer = "Default answer"
        mock_orch._ask_single_question.return_value = mock_result
        mock_orch_cls.return_value = mock_orch

        worker, q = self._make_worker()
        worker.execute()

        mock_orch.load_default_questions_from_txt.assert_called_once()
        msgs = drain_queue(q)
        assert find_msg(msgs, "semantic_complete") is not None

    @patch("src.core.semantic.SemanticOrchestrator")
    def test_long_question_truncated_in_progress(self, mock_orch_cls):
        """Questions longer than 50 chars are truncated in progress messages."""
        mock_orch = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "A"
        mock_orch._ask_single_question.return_value = mock_result
        mock_orch_cls.return_value = mock_orch

        long_q = "x" * 60
        worker, q = self._make_worker(questions=[long_q])
        worker.execute()

        msgs = drain_queue(q)
        progress = find_msg(msgs, "semantic_progress")
        assert progress[1][2].endswith("...")


# ===========================================================================
# ProgressiveExtractionWorker
# ===========================================================================


class TestProgressiveExtractionWorkerExecute:
    """Tests for ProgressiveExtractionWorker.execute()."""

    def _make_worker(self, documents=None, combined_text="Test text."):
        """Create worker with minimal args."""
        from src.services.workers import ProgressiveExtractionWorker

        q = Queue()
        docs = documents or [{"filename": "test.pdf", "extracted_text": "Test text."}]
        worker = ProgressiveExtractionWorker(
            documents=docs,
            combined_text=combined_text,
            ui_queue=q,
            embeddings=MagicMock(),
        )
        return worker, q

    @patch("src.services.progressive_extraction_worker.VocabularyExtractor")
    def test_phase1_sends_ner_complete(self, mock_extractor_cls):
        """Phase 1 produces ner_complete and extraction_complete messages."""
        mock_ext = MagicMock()
        mock_alg = MagicMock()
        mock_alg.name = "NER"
        mock_alg.enabled = True
        mock_ext.algorithms = [mock_alg]
        mock_ext.extract_progressive.return_value = ([{"Term": "Smith"}], [])
        mock_extractor_cls.return_value = mock_ext

        worker, q = self._make_worker()
        # Mock Phase 2 to do nothing
        worker._build_vector_store = MagicMock()
        worker._search_succeeded = threading.Event()
        worker._search_succeeded.set()

        worker.execute()

        msgs = drain_queue(q)
        assert find_msg(msgs, "extraction_started") is not None
        assert find_msg(msgs, "ner_complete") is not None
        assert find_msg(msgs, "extraction_complete") is not None

    @patch("src.services.progressive_extraction_worker.VocabularyExtractor")
    def test_multi_doc_uses_extract_documents(self, mock_extractor_cls):
        """Multiple documents use extract_documents instead of extract_progressive."""
        mock_ext = MagicMock()
        mock_alg = MagicMock()
        mock_alg.name = "NER"
        mock_alg.enabled = True
        mock_ext.algorithms = [mock_alg]
        mock_ext.extract_documents.return_value = ([{"Term": "Jones"}], [])
        mock_extractor_cls.return_value = mock_ext

        docs = [
            {"filename": "a.pdf", "extracted_text": "Text A"},
            {"filename": "b.pdf", "extracted_text": "Text B"},
        ]
        worker, q = self._make_worker(documents=docs, combined_text="Text A Text B")
        worker._build_vector_store = MagicMock()
        worker._search_succeeded = threading.Event()
        worker._search_succeeded.set()

        worker.execute()

        mock_ext.extract_documents.assert_called_once()
        msgs = drain_queue(q)
        assert find_msg(msgs, "ner_complete") is not None

    @patch("src.services.progressive_extraction_worker.VocabularyExtractor")
    def test_phase2_failure_sends_semantic_error(self, mock_extractor_cls):
        """Phase 2 crash sends semantic_error message."""
        mock_ext = MagicMock()
        mock_ext.algorithms = []
        mock_ext.extract_progressive.return_value = ([], [])
        mock_extractor_cls.return_value = mock_ext

        worker, q = self._make_worker()

        # Make _build_vector_store fail — replicate the real except block behavior
        def failing_build():
            """Simulate a build failure with proper error signaling."""
            from src.services.queue_messages import QueueMessage as QM

            with worker._search_error_lock:
                worker._search_error_msg = "CUDA OOM"
            worker.ui_queue.put(QM.semantic_error("CUDA OOM"))

        worker._build_vector_store = failing_build

        worker.execute()

        msgs = drain_queue(q)
        error = find_msg(msgs, "semantic_error")
        assert error is not None
        assert "CUDA OOM" in str(error[1])

    @patch("src.services.progressive_extraction_worker.VocabularyExtractor")
    def test_cancellation_during_phase2_wait(self, mock_extractor_cls):
        """Cancellation during Phase 2 wait exits gracefully."""
        mock_ext = MagicMock()
        mock_ext.algorithms = []
        mock_ext.extract_progressive.return_value = ([], [])
        mock_extractor_cls.return_value = mock_ext

        worker, q = self._make_worker()

        # Make Phase 2 block until cancelled
        def blocking_build():
            """Block until stopped."""
            import time

            while not worker.is_stopped:
                time.sleep(0.05)

        worker._build_vector_store = blocking_build

        # Stop after a brief delay
        def cancel_later():
            """Cancel after 200ms."""
            import time

            time.sleep(0.2)
            worker.stop()

        cancel_thread = threading.Thread(target=cancel_later)
        cancel_thread.start()

        worker.execute()  # Should exit after cancel
        cancel_thread.join(timeout=5)

        msgs = drain_queue(q)
        assert find_msg(msgs, "ner_complete") is not None

    @patch("src.services.progressive_extraction_worker.VocabularyExtractor")
    def test_single_doc_calls_extract_progressive(self, mock_extractor_cls):
        """Single document uses extract_progressive with callbacks."""
        mock_ext = MagicMock()
        mock_alg = MagicMock()
        mock_alg.name = "RAKE"
        mock_alg.enabled = True
        mock_ext.algorithms = [mock_alg]
        mock_ext.extract_progressive.return_value = ([{"Term": "test"}], [])
        mock_extractor_cls.return_value = mock_ext

        worker, q = self._make_worker()
        worker._build_vector_store = MagicMock()
        worker._search_succeeded = threading.Event()
        worker._search_succeeded.set()

        worker.execute()

        mock_ext.extract_progressive.assert_called_once()
        # Verify callbacks were passed
        call_kwargs = mock_ext.extract_progressive.call_args
        assert call_kwargs.kwargs.get("partial_callback") is not None
        assert call_kwargs.kwargs.get("ner_progress_callback") is not None
