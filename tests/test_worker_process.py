"""
Tests for Worker Subprocess Architecture

Tests the WorkerProcessManager (GUI side) and worker_process_main
(subprocess side) without requiring actual Tkinter or heavy pipeline imports.
"""

import pickle
import threading
import time
from queue import Queue
from unittest.mock import MagicMock

import pytest

# =========================================================================
# WorkerProcessManager Tests
# =========================================================================


class TestWorkerProcessManagerInit:
    """Test WorkerProcessManager initialization."""

    def test_init_creates_queues(self):
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        assert manager.command_queue is not None
        assert manager.result_queue is not None
        assert manager.process is None
        assert manager._started is False

    def test_is_alive_before_start(self):
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        assert manager.is_alive() is False

    def test_check_for_messages_empty(self):
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        assert manager.check_for_messages() == []


class TestWorkerProcessManagerLifecycle:
    """Test subprocess start/shutdown/restart."""

    def test_start_creates_process(self):
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            assert manager.is_alive()
            assert manager._started is True
            assert manager.process is not None
            assert manager.process.pid > 0
        finally:
            manager.shutdown(blocking=True)

    def test_double_start_is_safe(self):
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        pid1 = manager.process.pid
        try:
            manager.start()  # Should not create a new process
            assert manager.process.pid == pid1
        finally:
            manager.shutdown(blocking=True)

    def test_shutdown_blocking(self):
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        manager.shutdown(blocking=True)
        assert not manager.is_alive()
        assert manager.process is None
        assert manager._started is False

    def test_shutdown_nonblocking(self):
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        manager.shutdown(blocking=False)
        # Give it a moment to shut down
        time.sleep(1.0)
        assert not manager.is_alive()

    def test_shutdown_without_start(self):
        """Shutdown before start should not crash."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.shutdown(blocking=True)  # Should be a no-op

    def test_restart_if_dead(self):
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        manager.shutdown(blocking=True)
        assert not manager.is_alive()

        # Now restart
        manager.restart_if_dead()
        try:
            assert manager.is_alive()
        finally:
            manager.shutdown(blocking=True)


class TestWorkerProcessManagerSendCommand:
    """Test send_command behavior."""

    def test_send_command_auto_starts(self):
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        try:
            manager.send_command("test", {"key": "value"})
            assert manager.is_alive()
        finally:
            manager.shutdown(blocking=True)

    def test_cancel_sends_cancel(self):
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            manager.cancel()
            # Should not crash
        finally:
            manager.shutdown(blocking=True)


class TestWorkerProcessManagerMessages:
    """Test message passing through queues."""

    def test_messages_pass_through(self):
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            # Put a message directly into result_queue (simulating subprocess output)
            manager.result_queue.put(("test_msg", {"key": "value"}))
            time.sleep(0.1)
            messages = manager.check_for_messages()
            assert len(messages) == 1
            assert messages[0] == ("test_msg", {"key": "value"})
        finally:
            manager.shutdown(blocking=True)

    def test_check_for_messages_drains_all(self):
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            for i in range(5):
                manager.result_queue.put(("msg", i))
            time.sleep(0.1)
            messages = manager.check_for_messages()
            assert len(messages) == 5
        finally:
            manager.shutdown(blocking=True)

    def test_clear_queue_helper(self):
        from queue import Queue

        from src.services.worker_manager import WorkerProcessManager

        # Use thread Queue for deterministic behavior (no internal buffering)
        q = Queue()
        q.put("a")
        q.put("b")
        WorkerProcessManager._clear_queue(q)
        assert q.empty()


# =========================================================================
# Worker Process Internal Tests (unit tests without subprocess)
# =========================================================================


class TestForwarderLoop:
    """Test the forwarder loop message interception logic."""

    def test_regular_message_forwarded(self):
        """Non-special messages should be forwarded as-is."""
        from src.worker_process import _forwarder_loop

        internal_q = Queue()
        result_q = Queue()
        command_q = Queue()
        state = {
            "shutdown": threading.Event(),
            "embeddings": None,
            "vector_store_path": None,
            "chunk_scores": None,
            "active_worker": None,
        }

        # Put a regular message
        internal_q.put(("progress", (50, "Working...")))

        # Run forwarder in a thread, let it process, then shut down
        t = threading.Thread(target=_forwarder_loop, args=(internal_q, result_q, command_q, state))
        t.start()
        msg = result_q.get(timeout=5)
        state["shutdown"].set()
        t.join(timeout=5)

        assert msg == ("progress", (50, "Working..."))

    def test_qa_ready_strips_embeddings(self):
        """qa_ready messages should have embeddings stripped and saved to state."""
        from src.worker_process import _forwarder_loop

        internal_q = Queue()
        result_q = Queue()
        command_q = Queue()
        mock_embeddings = MagicMock()
        state = {
            "shutdown": threading.Event(),
            "embeddings": None,
            "vector_store_path": None,
            "chunk_scores": None,
            "active_worker": None,
        }

        internal_q.put(
            (
                "qa_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                    "chunk_count": 42,
                    "chunk_scores": None,
                },
            )
        )

        # Run forwarder in a thread, wait for output, then shut down
        t = threading.Thread(target=_forwarder_loop, args=(internal_q, result_q, command_q, state))
        t.start()
        msg_type, data = result_q.get(timeout=5)
        state["shutdown"].set()
        t.join(timeout=5)

        # Check state was updated
        assert state["embeddings"] is mock_embeddings
        assert state["vector_store_path"] == "/tmp/vs"

        # Check forwarded message has no embeddings
        assert msg_type == "qa_ready"
        assert "embeddings" not in data
        assert data["vector_store_path"] == "/tmp/vs"
        assert data["chunk_count"] == 42

    def test_trigger_default_qa_intercepted(self):
        """trigger_default_qa should be intercepted and trigger_default_qa_started sent."""
        from src.worker_process import _forwarder_loop

        internal_q = Queue()
        result_q = Queue()
        command_q = Queue()
        state = {
            "shutdown": threading.Event(),
            "embeddings": None,
            "vector_store_path": None,
            "chunk_scores": None,
            "active_worker": None,
        }

        internal_q.put(
            (
                "trigger_default_qa",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": MagicMock(),
                },
            )
        )

        # Run forwarder in a thread, wait for output, then shut down
        t = threading.Thread(target=_forwarder_loop, args=(internal_q, result_q, command_q, state))
        t.start()
        msg_type, data = result_q.get(timeout=5)
        state["shutdown"].set()
        t.join(timeout=5)

        assert msg_type == "trigger_default_qa_started"


class TestCommandDispatch:
    """Test command dispatch without actually running workers."""

    def test_unknown_command(self):
        """Unknown commands should produce error message."""
        from src.worker_process import _dispatch_command

        internal_q = Queue()
        state = {"active_worker": None}

        _dispatch_command("unknown_cmd", {}, internal_q, state)

        msg = internal_q.get_nowait()
        assert msg[0] == "error"
        assert "Unknown command" in msg[1]

    def test_run_qa_without_embeddings(self):
        """run_qa without embeddings should produce error."""
        from src.worker_process import _dispatch_command

        internal_q = Queue()
        state = {"active_worker": None, "embeddings": None, "vector_store_path": None}

        _dispatch_command("run_qa", {"answer_mode": "extraction"}, internal_q, state)

        msg = internal_q.get_nowait()
        assert msg[0] == "error"
        assert "not ready" in msg[1].lower()

    def test_followup_without_embeddings(self):
        """followup without embeddings should return None result."""
        from src.worker_process import _dispatch_command

        internal_q = Queue()
        state = {"active_worker": None, "embeddings": None, "vector_store_path": None}

        _dispatch_command("followup", {"question": "test?"}, internal_q, state)

        msg = internal_q.get_nowait()
        assert msg == ("qa_followup_result", None)


class TestStopActiveWorker:
    """Test active worker stopping."""

    def test_stop_none_worker(self):
        """Stopping when no worker is active should be a no-op."""
        from src.worker_process import _stop_active_worker

        state = {"active_worker": None}
        _stop_active_worker(state)  # Should not crash
        assert state["active_worker"] is None

    def test_stop_running_worker(self):
        """Should call stop() and join() on the active worker."""
        from src.worker_process import _stop_active_worker

        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        state = {"active_worker": mock_worker}

        _stop_active_worker(state)

        mock_worker.stop.assert_called_once()
        mock_worker.join.assert_called_once_with(timeout=2.0)
        assert state["active_worker"] is None

    def test_stop_dead_worker(self):
        """Should just clear the reference for dead workers."""
        from src.worker_process import _stop_active_worker

        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = False
        state = {"active_worker": mock_worker}

        _stop_active_worker(state)

        mock_worker.stop.assert_not_called()
        assert state["active_worker"] is None


# =========================================================================
# Pickling Tests (critical for multiprocessing.Queue)
# =========================================================================


class TestQAResultPickling:
    """Verify QAResult and related dataclasses survive pickling."""

    def test_qa_result_roundtrip(self):
        """QAResult should survive pickle roundtrip."""
        from src.core.qa.qa_orchestrator import QAResult

        result = QAResult(
            question="What happened?",
            quick_answer="The defendant testified.",
            citation="Page 3: The defendant stated...",
            confidence=0.85,
            is_followup=True,
        )

        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)

        assert unpickled.question == result.question
        assert unpickled.quick_answer == result.quick_answer
        assert unpickled.citation == result.citation
        assert unpickled.confidence == result.confidence
        assert unpickled.is_followup is True

    def test_qa_result_with_verification_roundtrip(self):
        """QAResult with VerificationResult should survive pickle."""
        from src.core.qa.hallucination_verifier import VerificationResult, VerifiedSpan
        from src.core.qa.qa_orchestrator import QAResult

        verification = VerificationResult(
            spans=[
                VerifiedSpan(text="The defendant", start=0, end=13, hallucination_prob=0.1),
                VerifiedSpan(text="testified", start=14, end=23, hallucination_prob=0.05),
            ],
            overall_reliability=0.92,
            answer_rejected=False,
        )

        result = QAResult(
            question="What happened?",
            quick_answer="The defendant testified.",
            citation="Page 3",
            confidence=0.85,
            verification=verification,
        )

        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)

        assert unpickled.verification is not None
        assert len(unpickled.verification.spans) == 2
        assert unpickled.verification.overall_reliability == 0.92

    def test_chunk_scores_roundtrip(self):
        """ChunkScores should survive pickle roundtrip."""
        from src.core.utils.chunk_scoring import ChunkScores

        scores = ChunkScores(
            skip=[False, True, False],
            skip_reason=["", "redundant with chunk 0", ""],
        )

        pickled = pickle.dumps(scores)
        unpickled = pickle.loads(pickled)

        assert unpickled.skip == [False, True, False]
        assert unpickled.skip_reason[1] == "redundant with chunk 0"

    def test_multi_doc_result_roundtrip(self):
        """MultiDocumentSummaryResult should survive pickle."""
        from src.core.summarization.result_types import (
            DocumentSummaryResult,
            MultiDocumentSummaryResult,
        )

        result = MultiDocumentSummaryResult(
            individual_summaries={
                "doc1.pdf": DocumentSummaryResult(
                    filename="doc1.pdf",
                    summary="Summary of doc1.",
                    word_count=4,
                    chunk_count=1,
                    processing_time_seconds=1.0,
                    success=True,
                ),
            },
            meta_summary="Combined summary.",
            total_processing_time_seconds=5.2,
            documents_processed=1,
            documents_failed=0,
            document_order=["doc1.pdf"],
        )

        pickled = pickle.dumps(result)
        unpickled = pickle.loads(pickled)

        assert unpickled.meta_summary == "Combined summary."
        assert unpickled.documents_processed == 1

    def test_queue_message_tuples_picklable(self):
        """All standard queue message formats should be picklable."""
        messages = [
            ("progress", (50, "Working...")),
            ("error", "Something went wrong"),
            ("status_error", "Minor issue"),
            ("file_processed", {"filename": "test.pdf", "status": "success"}),
            ("processing_finished", [{"filename": "test.pdf"}]),
            ("ner_complete", [{"term": "defendant"}]),
            ("qa_ready", {"vector_store_path": "/tmp/vs", "chunk_count": 10}),
            ("llm_progress", (3, 10)),
            ("llm_complete", [{"term": "plaintiff"}]),
            ("qa_progress", (1, 5, "What happened?")),
            ("extraction_started", None),
            ("extraction_complete", None),
            ("trigger_default_qa_started", None),
        ]

        for msg in messages:
            pickled = pickle.dumps(msg)
            unpickled = pickle.loads(pickled)
            assert unpickled == msg, f"Failed roundtrip for {msg[0]}"


# =========================================================================
# Integration Tests (with real subprocess)
# =========================================================================


class TestSubprocessIntegration:
    """Integration tests that start a real subprocess."""

    def test_subprocess_responds_to_shutdown(self):
        """Subprocess should shut down cleanly on shutdown command."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        assert manager.is_alive()

        manager.shutdown(blocking=True)
        assert not manager.is_alive()

    def test_subprocess_crash_recovery(self):
        """Manager should detect and recover from subprocess crash."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        pid = manager.process.pid

        # Kill the subprocess
        manager.process.terminate()
        manager.process.join(timeout=2.0)
        assert not manager.is_alive()

        # Restart
        manager.restart_if_dead()
        try:
            assert manager.is_alive()
            assert manager.process.pid != pid
        finally:
            manager.shutdown(blocking=True)

    def test_error_message_through_subprocess(self):
        """Error messages from unknown commands should reach the GUI side."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            # Wait for subprocess to be fully ready before sending commands
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                manager.check_for_messages()
                if manager._worker_ready:
                    break
                time.sleep(0.2)
            manager.send_command("nonexistent_command", {})
            # Error travels: internal_queue -> forwarder (0.5s poll) -> result_queue
            # Must poll repeatedly rather than single sleep+check
            messages = []
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                messages.extend(manager.check_for_messages())
                if any(m[0] == "error" for m in messages):
                    break
                time.sleep(0.2)
            error_msgs = [m for m in messages if m[0] == "error"]
            assert len(error_msgs) > 0
            assert "Unknown command" in error_msgs[0][1]
        finally:
            manager.shutdown(blocking=True)


class TestQueueMessageFactory:
    """Test that QueueMessage factory creates picklable messages."""

    def test_all_factory_methods_picklable(self):
        """Every QueueMessage factory method should produce picklable output."""
        from src.ui.queue_messages import QueueMessage

        messages = [
            QueueMessage.progress(50, "Working..."),
            QueueMessage.error("Something failed"),
            QueueMessage.status_error("Minor issue"),
            QueueMessage.file_processed({"filename": "test.pdf"}),
            QueueMessage.processing_finished([]),
            QueueMessage.vocab_csv_generated([{"term": "test"}]),
            QueueMessage.summary_result("Summary text"),
            QueueMessage.qa_progress(1, 5, "Question?"),
            QueueMessage.qa_complete([]),
            QueueMessage.qa_error("Error"),
            QueueMessage.ner_complete([]),
            QueueMessage.qa_ready(
                vector_store_path="/tmp/vs",
                embeddings=None,  # Real embeddings not picklable, but None is
                chunk_count=10,
            ),
            QueueMessage.llm_progress(3, 10),
            QueueMessage.llm_complete([]),
            QueueMessage.extraction_started(),
            QueueMessage.extraction_complete(),
            QueueMessage.partial_vocab_complete([]),
            QueueMessage.ner_progress([], 1, 5),
        ]

        for msg in messages:
            try:
                pickled = pickle.dumps(msg)
                unpickled = pickle.loads(pickled)
                assert unpickled[0] == msg[0], f"Type mismatch for {msg[0]}"
            except Exception as e:
                pytest.fail(f"Failed to pickle {msg[0]}: {e}")
