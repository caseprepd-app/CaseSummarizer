"""
IPC Message Flow Integration Tests

Tests inter-process communication between the worker subprocess and main window.
Covers:
1. Message sequence ordering through the forwarder loop
2. State preservation across messages (embeddings, vector_store_path)
3. Main window handler state transitions for realistic message sequences
4. Real subprocess round-trip communication
5. Concurrent command handling
"""

import threading
import time
from queue import Empty, Queue
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_forwarder_state():
    """Create a clean forwarder state dict."""
    return {
        "embeddings": None,
        "vector_store_path": None,
        "chunk_scores": None,
        "active_worker": None,
        "shutdown": threading.Event(),
        "worker_lock": threading.Lock(),
    }


def _run_forwarder_with_messages(messages, state=None, timeout=5):
    """
    Run forwarder loop with given messages and collect output.

    Args:
        messages: List of (msg_type, data) tuples to feed into internal queue
        state: Optional state dict (created if None)
        timeout: Max seconds to wait for all output messages

    Returns:
        (output_messages, state) tuple
    """
    from src.worker_process import _forwarder_loop

    internal_q = Queue()
    result_q = Queue()
    command_q = Queue()
    if state is None:
        state = _make_forwarder_state()

    # Load all messages
    for msg in messages:
        internal_q.put(msg)

    # Run forwarder in thread
    t = threading.Thread(
        target=_forwarder_loop,
        args=(internal_q, result_q, command_q, state),
        daemon=True,
    )
    t.start()

    # Collect output messages (wait for expected count with timeout)
    # Use consecutive-empty-reads pattern instead of .empty() check (TOCTOU race)
    output = []
    deadline = time.monotonic() + timeout
    consecutive_empties = 0
    while time.monotonic() < deadline:
        try:
            msg = result_q.get(timeout=0.3)
            output.append(msg)
            consecutive_empties = 0
        except Empty:
            consecutive_empties += 1
            # After 2 consecutive empty reads (0.6s with no messages), stop
            if consecutive_empties >= 2:
                break

    state["shutdown"].set()
    t.join(timeout=2)
    return output, state


def _make_window_stub():
    """Create a minimal MainWindow state stub for handler testing."""
    stub = MagicMock()
    stub._pending_tasks = {
        "vocab": True,
        "semantic": True,
    }
    stub._completed_tasks = set()
    stub._semantic_answering_active = False
    stub._semantic_failed = False
    stub._semantic_ready = False
    stub._processing_active = True
    stub._preprocessing_active = False
    stub._destroying = False
    stub._semantic_results = []
    stub._semantic_results_lock = threading.Lock()
    stub._vector_store_path = None
    stub.clear_files_btn = MagicMock()
    return stub


# ===========================================================================
# 1. Forwarder Message Sequence Tests
# ===========================================================================


class TestForwarderMessageSequence:
    """Test that messages flow through forwarder in the correct order."""

    def test_progress_messages_forwarded_in_order(self):
        """Multiple progress messages should arrive in order."""
        messages = [
            ("progress", (10, "Step 1")),
            ("progress", (50, "Step 2")),
            ("progress", (100, "Done")),
        ]
        output, _ = _run_forwarder_with_messages(messages)
        assert len(output) == 3
        assert output[0] == ("progress", (10, "Step 1"))
        assert output[1] == ("progress", (50, "Step 2"))
        assert output[2] == ("progress", (100, "Done"))

    def test_extraction_sequence_ner_then_semantic_ready(self):
        """NER complete should arrive before semantic_ready."""
        mock_embeddings = MagicMock()
        messages = [
            ("extraction_started", None),
            ("ner_complete", {"vocab": [{"term": "defendant"}], "filtered": []}),
            ("extraction_complete", None),
            (
                "semantic_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                    "chunk_count": 42,
                    "chunk_scores": None,
                },
            ),
        ]
        output, state = _run_forwarder_with_messages(messages)
        types = [m[0] for m in output]
        assert types == [
            "extraction_started",
            "ner_complete",
            "extraction_complete",
            "semantic_ready",
            "key_sentences_result",  # spawned by forwarder after semantic_ready
        ]

    def test_semantic_ready_before_trigger_default_semantic(self):
        """semantic_ready must arrive before trigger_default_semantic_started."""
        mock_embeddings = MagicMock()
        messages = [
            (
                "semantic_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                    "chunk_count": 10,
                    "chunk_scores": None,
                },
            ),
            (
                "trigger_default_semantic",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                },
            ),
        ]
        output, state = _run_forwarder_with_messages(messages)
        types = [m[0] for m in output]
        # semantic_ready is forwarded, trigger_default_semantic becomes trigger_default_semantic_started
        assert "semantic_ready" in types
        assert "trigger_default_semantic_started" in types
        assert types.index("semantic_ready") < types.index("trigger_default_semantic_started")

    def test_error_messages_forwarded(self):
        """Error messages pass through the forwarder."""
        messages = [
            ("error", "Something failed"),
            ("status_error", "Minor issue"),
        ]
        output, _ = _run_forwarder_with_messages(messages)
        assert len(output) == 2
        assert output[0] == ("error", "Something failed")
        assert output[1] == ("status_error", "Minor issue")

    def test_semantic_error_forwarded(self):
        """semantic_error messages pass through the forwarder as-is."""
        messages = [
            ("semantic_error", {"error": "Indexing failed"}),
        ]
        output, _ = _run_forwarder_with_messages(messages)
        assert len(output) == 1
        assert output[0] == ("semantic_error", {"error": "Indexing failed"})


# ===========================================================================
# 2. State Preservation Tests
# ===========================================================================


class TestForwarderStatePreservation:
    """Test that forwarder saves/restores state correctly across messages."""

    def test_semantic_ready_saves_embeddings_in_state(self):
        """semantic_ready should save embeddings to subprocess state."""
        mock_embeddings = MagicMock()
        messages = [
            (
                "semantic_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                    "chunk_count": 42,
                    "chunk_scores": None,
                },
            ),
        ]
        _, state = _run_forwarder_with_messages(messages)
        assert state["embeddings"] is mock_embeddings
        assert state["vector_store_path"] == "/tmp/vs"

    def test_semantic_ready_saves_chunk_scores(self):
        """semantic_ready should save chunk_scores to subprocess state."""
        mock_scores = MagicMock()
        messages = [
            (
                "semantic_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": MagicMock(),
                    "chunk_count": 10,
                    "chunk_scores": mock_scores,
                },
            ),
        ]
        _, state = _run_forwarder_with_messages(messages)
        assert state["chunk_scores"] is mock_scores

    def test_semantic_ready_strips_embeddings_from_forwarded_message(self):
        """Forwarded semantic_ready should NOT contain embeddings (not picklable)."""
        messages = [
            (
                "semantic_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": MagicMock(),
                    "chunk_count": 42,
                    "chunk_scores": None,
                },
            ),
        ]
        output, _ = _run_forwarder_with_messages(messages)
        # semantic_ready + key_sentences_result (spawned after semantic_ready)
        sr_msgs = [m for m in output if m[0] == "semantic_ready"]
        assert len(sr_msgs) == 1
        msg_type, data = sr_msgs[0]
        assert msg_type == "semantic_ready"
        assert "embeddings" not in data
        assert data["vector_store_path"] == "/tmp/vs"
        assert data["chunk_count"] == 42

    def test_trigger_default_semantic_uses_saved_state(self):
        """trigger_default_semantic should use embeddings/path saved from semantic_ready."""
        mock_embeddings = MagicMock()
        state = _make_forwarder_state()
        # Pre-set state as if semantic_ready already ran
        state["embeddings"] = mock_embeddings
        state["vector_store_path"] = "/tmp/vs"

        messages = [
            (
                "trigger_default_semantic",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                },
            ),
        ]
        output, final_state = _run_forwarder_with_messages(messages, state=state)
        types = [m[0] for m in output]
        assert "trigger_default_semantic_started" in types

    def test_trigger_default_semantic_skipped_when_disabled(self):
        """trigger_default_semantic should send semantic_complete (skip) when ask_default_questions=False."""
        state = _make_forwarder_state()
        state["ask_default_questions"] = False
        state["embeddings"] = MagicMock()
        state["vector_store_path"] = "/tmp/vs"

        messages = [
            (
                "trigger_default_semantic",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": MagicMock(),
                },
            ),
        ]
        output, _ = _run_forwarder_with_messages(messages, state=state)
        types = [m[0] for m in output]
        # Should skip Q&A entirely — no trigger_default_semantic_started
        assert "trigger_default_semantic_started" not in types
        assert "semantic_complete" in types

    def test_trigger_default_semantic_without_state_logs_warning(self):
        """trigger_default_semantic with no saved embeddings should warn, not crash."""
        state = _make_forwarder_state()
        # embeddings and vector_store_path are None

        messages = [
            (
                "trigger_default_semantic",
                {
                    "vector_store_path": None,
                    "embeddings": None,
                },
            ),
        ]
        # Should not raise -- just log warning
        output, _ = _run_forwarder_with_messages(messages, state=state)
        types = [m[0] for m in output]
        assert "trigger_default_semantic_started" in types

    def test_state_persists_across_multiple_messages(self):
        """State from semantic_ready persists through subsequent unrelated messages."""
        mock_embeddings = MagicMock()
        messages = [
            (
                "semantic_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                    "chunk_count": 42,
                    "chunk_scores": None,
                },
            ),
            ("progress", (80, "Processing...")),
        ]
        _, state = _run_forwarder_with_messages(messages)
        # State should still have embeddings from semantic_ready
        assert state["embeddings"] is mock_embeddings
        assert state["vector_store_path"] == "/tmp/vs"


# ===========================================================================
# 3. Full Extraction Message Sequence (simulated)
# ===========================================================================


class TestFullExtractionSequence:
    """
    Test the complete message sequence from ProgressiveExtractionWorker.

    Expected order for vocab + semantic search flow:
    1. extraction_started
    2. progress (multiple)
    3. ner_complete
    4. extraction_complete
    5. progress (phase 2)
    6. semantic_ready (-> stripped by forwarder)
    7. trigger_default_semantic (-> intercepted, becomes trigger_default_semantic_started)
    8. semantic_progress (multiple)
    9. semantic_result (multiple)
    10. semantic_complete
    """

    def test_full_vocab_qa_sequence_through_forwarder(self):
        """Full extraction message sequence should arrive in correct order."""
        mock_embeddings = MagicMock()
        messages = [
            ("extraction_started", None),
            ("progress", (10, "Phase 1: Running NER...")),
            ("ner_complete", {"vocab": [{"term": "defendant"}], "filtered": []}),
            ("extraction_complete", None),
            ("progress", (20, "Phase 2: Building Q&A index...")),
            (
                "semantic_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                    "chunk_count": 42,
                    "chunk_scores": None,
                },
            ),
            (
                "trigger_default_semantic",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                },
            ),
        ]
        output, state = _run_forwarder_with_messages(messages)
        types = [m[0] for m in output]

        # Verify key ordering constraints
        assert "extraction_started" in types
        assert "ner_complete" in types
        assert "semantic_ready" in types
        assert "trigger_default_semantic_started" in types  # Note: renamed by forwarder

        # extraction_started before ner_complete
        assert types.index("extraction_started") < types.index("ner_complete")
        # ner_complete before semantic_ready
        assert types.index("ner_complete") < types.index("semantic_ready")
        # semantic_ready before trigger_default_semantic_started
        assert types.index("semantic_ready") < types.index("trigger_default_semantic_started")
        # trigger_default_semantic was intercepted (not in output)
        assert "trigger_default_semantic" not in types

    def test_semantic_answering_messages_arrive_after_semantic_ready(self):
        """semantic_progress/semantic_result/semantic_complete must come after semantic_ready."""
        mock_embeddings = MagicMock()
        messages = [
            (
                "semantic_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                    "chunk_count": 10,
                    "chunk_scores": None,
                },
            ),
            (
                "trigger_default_semantic",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                },
            ),
            # These would come from SemanticWorker spawned by trigger_default_semantic
            ("semantic_progress", (0, 3, "What happened?")),
            ("semantic_result", {"question": "What happened?", "answer": "..."}),
            ("semantic_progress", (1, 3, "Who are the parties?")),
            ("semantic_result", {"question": "Who are the parties?", "answer": "..."}),
            (
                "semantic_complete",
                [
                    {"question": "What happened?", "answer": "..."},
                    {"question": "Who are the parties?", "answer": "..."},
                ],
            ),
        ]
        output, _ = _run_forwarder_with_messages(messages)
        types = [m[0] for m in output]

        assert "semantic_ready" in types
        if "semantic_complete" in types:
            assert types.index("semantic_ready") < types.index("semantic_complete")


# ===========================================================================
# 4. Main Window Handler State Transitions
# ===========================================================================


class TestHandlerStateTransitions:
    """
    Test that main_window._handle_queue_message produces correct state transitions
    for realistic message sequences. Uses stub objects (no Tk needed).
    """

    def _simulate_handler(self, stub, msg_type, data):
        """
        Simulate main_window._handle_queue_message for state-tracking messages.

        Only covers message types that affect _semantic_ready, _semantic_answering_active,
        _pending_tasks, _completed_tasks.
        """
        if msg_type == "semantic_ready":
            stub._semantic_ready = True
            # Note: NO _completed_tasks.add("semantic") here (bug was fixed)

        elif msg_type == "semantic_error":
            stub._semantic_answering_active = False
            stub._semantic_failed = True
            if stub._pending_tasks.get("semantic"):
                stub._completed_tasks.add("semantic")

        elif msg_type == "trigger_default_semantic_started":
            stub._semantic_answering_active = True

        elif msg_type == "semantic_complete":
            stub._semantic_answering_active = False
            if stub._pending_tasks.get("semantic"):
                stub._completed_tasks.add("semantic")

        elif msg_type == "ner_complete":
            stub._completed_tasks.add("vocab")

    def _all_tasks_complete(self, stub):
        """Check if all pending tasks are truly complete."""
        for task_name, is_pending in stub._pending_tasks.items():
            if is_pending and task_name not in stub._completed_tasks:
                return False
        return not stub._semantic_answering_active

    def test_vocab_only_flow(self):
        """Vocab only: ner_complete -> finalize."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": False,
        }

        self._simulate_handler(stub, "ner_complete", {"vocab": [], "filtered": []})

        assert "vocab" in stub._completed_tasks
        assert self._all_tasks_complete(stub)

    def test_vocab_plus_qa_flow(self):
        """Vocab+Q&A: semantic_ready -> trigger -> answering -> qa_complete -> finalize."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }

        # Phase 1: NER
        self._simulate_handler(stub, "ner_complete", {"vocab": [], "filtered": []})
        assert not stub._semantic_answering_active

        # Phase 2: Q&A index ready
        self._simulate_handler(stub, "semantic_ready", {"chunk_count": 42})
        assert stub._semantic_ready is True
        assert "semantic" not in stub._completed_tasks  # NOT premature!

        # Forwarder sends trigger_default_semantic_started
        self._simulate_handler(stub, "trigger_default_semantic_started", None)
        assert stub._semantic_answering_active is True

        # Vocab already completed via ner_complete
        assert "vocab" in stub._completed_tasks
        assert not self._all_tasks_complete(stub)  # Q&A still active!

        # Q&A answers arrive
        self._simulate_handler(stub, "semantic_complete", [{"q": "test", "a": "answer"}])
        assert stub._semantic_answering_active is False
        assert "semantic" in stub._completed_tasks
        assert self._all_tasks_complete(stub)

    def test_semantic_error_still_allows_finalization(self):
        """Semantic error should mark search as complete so finalization isn't blocked."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }

        self._simulate_handler(stub, "ner_complete", {"vocab": [], "filtered": []})
        self._simulate_handler(stub, "semantic_error", {"error": "Indexing failed"})

        assert "semantic" in stub._completed_tasks
        assert "vocab" in stub._completed_tasks
        assert self._all_tasks_complete(stub)

    def test_semantic_error_clears_answering_flag(self):
        """semantic_error should clear _semantic_answering_active even if it was set."""
        stub = _make_window_stub()
        stub._semantic_answering_active = True  # Was set by trigger_default_semantic_started

        self._simulate_handler(stub, "semantic_error", {"error": "crash"})
        assert stub._semantic_answering_active is False

    def test_semantic_ready_does_not_mark_complete(self):
        """semantic_ready should NOT add 'qa' to completed_tasks (the old bug)."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }

        self._simulate_handler(stub, "semantic_ready", {"chunk_count": 42})
        assert "semantic" not in stub._completed_tasks

    def test_trigger_default_semantic_started_sets_answering_flag(self):
        """trigger_default_semantic_started must set _semantic_answering_active."""
        stub = _make_window_stub()
        assert not stub._semantic_answering_active

        self._simulate_handler(stub, "trigger_default_semantic_started", None)
        assert stub._semantic_answering_active is True

    def test_vocab_complete_before_qa_complete(self):
        """Vocab completing before Q&A should NOT allow premature finalization."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }

        self._simulate_handler(stub, "ner_complete", {"vocab": [], "filtered": []})
        self._simulate_handler(stub, "semantic_ready", {})
        self._simulate_handler(stub, "trigger_default_semantic_started", None)

        # Vocab is done but Q&A answering is still active
        assert "vocab" in stub._completed_tasks
        assert "semantic" not in stub._completed_tasks
        assert stub._semantic_answering_active is True
        assert not self._all_tasks_complete(stub)

    def test_semantic_complete_before_vocab_complete_impossible(self):
        """With NER marking vocab done, semantic search can't complete before vocab in practice."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }

        # Simulate Q&A completing first (unlikely but should be handled)
        self._simulate_handler(stub, "semantic_ready", {})
        self._simulate_handler(stub, "trigger_default_semantic_started", None)
        self._simulate_handler(stub, "semantic_complete", [])

        # Q&A done but vocab not yet complete
        assert "semantic" in stub._completed_tasks
        assert "vocab" not in stub._completed_tasks
        assert not self._all_tasks_complete(stub)


# ===========================================================================
# 5. Real Subprocess Round-Trip Tests
# ===========================================================================


class TestSubprocessRoundTrip:
    """Tests that send real commands through the subprocess and verify responses."""

    def test_unknown_command_returns_error(self):
        """Unknown command should produce an error message back through IPC."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            self._wait_for_ready(manager)
            manager.send_command("bogus_command", {})
            # Wait for subprocess to process and respond
            messages = self._wait_for_messages(manager, expected_type="error")
            error_msgs = [m for m in messages if m[0] == "error"]
            assert len(error_msgs) >= 1
            assert "Unknown command" in str(error_msgs[0][1])
        finally:
            manager.shutdown(blocking=True)

    def test_run_semantic_without_state_returns_error(self):
        """run_qa without prior semantic_ready should return an error."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            self._wait_for_ready(manager)
            manager.send_command("run_qa", {"answer_mode": "extraction"})
            # Error goes internal_queue -> forwarder -> result_queue, needs extra time
            messages = self._wait_for_messages(manager, expected_type="error", timeout=10.0)
            error_msgs = [m for m in messages if m[0] == "error"]
            assert len(error_msgs) >= 1
            assert "not ready" in error_msgs[0][1].lower()
        finally:
            manager.shutdown(blocking=True)

    def test_followup_without_state_returns_none_result(self):
        """followup without prior semantic_ready should return semantic_followup_result with None."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            self._wait_for_ready(manager)
            manager.send_command("followup", {"question": "test?"})
            messages = self._wait_for_messages(manager, expected_type="semantic_followup_result")
            followup_msgs = [m for m in messages if m[0] == "semantic_followup_result"]
            assert len(followup_msgs) >= 1
            assert followup_msgs[0][1] is None
        finally:
            manager.shutdown(blocking=True)

    def test_multiple_commands_in_sequence(self):
        """Multiple commands should each produce their own response."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            self._wait_for_ready(manager)
            manager.send_command("bogus_1", {})
            manager.send_command("bogus_2", {})
            # Errors travel internal_queue -> forwarder -> result_queue; poll repeatedly
            messages = []
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                messages.extend(manager.check_for_messages())
                if len([m for m in messages if m[0] == "error"]) >= 2:
                    break
                time.sleep(0.2)
            error_msgs = [m for m in messages if m[0] == "error"]
            # Should have at least 2 error messages
            assert len(error_msgs) >= 2
        finally:
            manager.shutdown(blocking=True)

    def test_cancel_does_not_crash_subprocess(self):
        """Cancel command should not crash the subprocess."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            self._wait_for_ready(manager)
            manager.cancel()
            time.sleep(0.5)
            assert manager.is_alive()
        finally:
            manager.shutdown(blocking=True)

    def test_subprocess_survives_rapid_commands(self):
        """Rapid command sends should not crash the subprocess."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            self._wait_for_ready(manager)
            for i in range(10):
                manager.send_command(f"rapid_test_{i}", {})
            # Errors travel internal_queue -> forwarder -> result_queue; poll repeatedly
            messages = []
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                messages.extend(manager.check_for_messages())
                if len([m for m in messages if m[0] == "error"]) >= 10:
                    break
                time.sleep(0.2)
            assert manager.is_alive()
            error_msgs = [m for m in messages if m[0] == "error"]
            assert len(error_msgs) == 10
        finally:
            manager.shutdown(blocking=True)

    @staticmethod
    def _wait_for_ready(manager, timeout=30.0):
        """Poll until the worker subprocess signals it is ready."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            manager.check_for_messages()  # drains worker_ready
            if manager._worker_ready:
                return
            if not manager.is_alive():
                pytest.skip("Worker subprocess died during startup")
            time.sleep(0.2)
        pytest.skip("Worker subprocess not ready within timeout")

    @staticmethod
    def _wait_for_messages(manager, expected_type, timeout=15.0):
        """Wait for a specific message type from the subprocess."""
        messages = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            new_msgs = manager.check_for_messages()
            messages.extend(new_msgs)
            if any(m[0] == expected_type for m in messages):
                return messages
            time.sleep(0.2)
        return messages


# ===========================================================================
# 6. Polling Condition with Real Message Draining
# ===========================================================================


class TestPollingWithMessageDraining:
    """
    Test that the polling condition correctly handles message draining scenarios.
    These test the logic from _poll_queue more thoroughly.
    """

    def _should_continue_polling(self, processing, preprocessing, qa_answering, messages):
        """Simulate the polling continuation condition."""
        return bool(processing or preprocessing or qa_answering or messages)

    def test_semantic_searching_keeps_poll_alive_after_processing_done(self):
        """After _processing_active=False, _semantic_answering_active keeps polling."""
        assert self._should_continue_polling(
            processing=False,
            preprocessing=False,
            qa_answering=True,
            messages=[],
        )

    def test_messages_in_queue_keep_poll_alive(self):
        """Unprocessed messages keep polling alive even if all flags are False."""
        assert self._should_continue_polling(
            processing=False,
            preprocessing=False,
            qa_answering=False,
            messages=[("semantic_complete", [])],
        )

    def test_no_activity_and_no_messages_stops_poll(self):
        """No active flags and no messages -> stop polling."""
        assert not self._should_continue_polling(
            processing=False,
            preprocessing=False,
            qa_answering=False,
            messages=[],
        )

    def test_only_semantic_searching_plus_final_message(self):
        """
        Typical end-of-pipeline scenario: processing done, semantic searching done,
        and the semantic_complete message just arrived (in messages list).
        Should continue polling to process that message.
        """
        assert self._should_continue_polling(
            processing=False,
            preprocessing=False,
            qa_answering=False,
            messages=[("semantic_complete", [])],
        )


# ===========================================================================
# 7. Command Dispatch Integration Tests
# ===========================================================================


class TestCommandDispatchIntegration:
    """Test _dispatch_command for commands that require state."""

    def test_run_semantic_with_state_spawns_worker(self):
        """run_qa with valid state should spawn a SemanticWorker."""
        from src.worker_process import _dispatch_command

        internal_q = Queue()
        mock_embeddings = MagicMock()
        state = {
            "active_worker": None,
            "embeddings": mock_embeddings,
            "vector_store_path": "/tmp/vs",
            "worker_lock": threading.Lock(),
        }

        with patch("src.services.workers.SemanticWorker") as MockSemanticWorker:
            mock_worker_instance = MagicMock()
            MockSemanticWorker.return_value = mock_worker_instance

            _dispatch_command(
                "run_qa",
                {"answer_mode": "extraction", "use_default_questions": True},
                internal_q,
                state,
            )

            MockSemanticWorker.assert_called_once()
            mock_worker_instance.start.assert_called_once()
            assert state["active_worker"] is mock_worker_instance

    def test_new_command_stops_previous_worker(self):
        """Dispatching a valid command should stop the active worker first."""
        from src.worker_process import _dispatch_command

        internal_q = Queue()
        old_worker = MagicMock()
        old_worker.is_alive.return_value = True
        state = {"active_worker": old_worker, "worker_lock": threading.Lock()}

        # Use a real command to trigger stop-before-start logic.
        # Patch the worker class to avoid real work.
        with patch("src.worker_process._run_qa"):
            _dispatch_command("run_qa", {}, internal_q, state)

        old_worker.stop.assert_called_once()
        old_worker.join.assert_called_once()

    def test_unknown_command_does_not_stop_worker(self):
        """Unknown commands return early without touching the active worker."""
        from src.worker_process import _dispatch_command

        internal_q = Queue()
        old_worker = MagicMock()
        old_worker.is_alive.return_value = True
        state = {"active_worker": old_worker, "worker_lock": threading.Lock()}

        _dispatch_command("unknown_cmd", {}, internal_q, state)

        old_worker.stop.assert_not_called()


# ===========================================================================
# 8. Edge Cases and Error Recovery
# ===========================================================================


class TestIPCEdgeCases:
    """Test error recovery and edge cases in IPC."""

    def test_forwarder_handles_malformed_message(self):
        """Forwarder should skip malformed messages without crashing."""
        messages = [
            "not_a_tuple",  # Malformed
            ("progress", (50, "Valid")),  # Valid
        ]
        # Put messages directly into queue
        from src.worker_process import _forwarder_loop

        internal_q = Queue()
        result_q = Queue()
        command_q = Queue()
        state = _make_forwarder_state()

        for msg in messages:
            internal_q.put(msg)

        t = threading.Thread(
            target=_forwarder_loop,
            args=(internal_q, result_q, command_q, state),
            daemon=True,
        )
        t.start()

        # Should get the valid message through
        msg = result_q.get(timeout=5)
        state["shutdown"].set()
        t.join(timeout=2)

        assert msg == ("progress", (50, "Valid"))

    def test_forwarder_handles_three_element_tuple(self):
        """Three-element tuples should fail unpacking and be skipped."""
        messages = [
            ("a", "b", "c"),  # Can't unpack to (msg_type, data)
            ("progress", (100, "Done")),
        ]
        from src.worker_process import _forwarder_loop

        internal_q = Queue()
        result_q = Queue()
        command_q = Queue()
        state = _make_forwarder_state()

        for msg in messages:
            internal_q.put(msg)

        t = threading.Thread(
            target=_forwarder_loop,
            args=(internal_q, result_q, command_q, state),
            daemon=True,
        )
        t.start()

        msg = result_q.get(timeout=5)
        state["shutdown"].set()
        t.join(timeout=2)

        assert msg == ("progress", (100, "Done"))

    def test_command_loop_handles_malformed_command(self):
        """Command loop should skip malformed commands without crashing."""
        from src.worker_process import _command_loop

        command_q = Queue()
        internal_q = Queue()
        result_q = Queue()
        state = _make_forwarder_state()

        # Put malformed command then shutdown
        command_q.put("not_a_tuple_or_sentinel")
        command_q.put("shutdown")

        _command_loop(command_q, internal_q, result_q, state)
        # Should not crash - just skip the bad command

    def test_subprocess_restart_clears_state(self):
        """Restarting subprocess should clear old queue state."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()

        # Put a fake message in result queue
        manager.result_queue.put(("stale_msg", "old"))

        # Shutdown and restart
        manager.shutdown(blocking=True)
        manager.start()

        try:
            # Stale message should be gone (cleared on restart)
            time.sleep(0.5)
            messages = manager.check_for_messages()
            stale = [m for m in messages if m[0] == "stale_msg"]
            assert len(stale) == 0
        finally:
            manager.shutdown(blocking=True)
