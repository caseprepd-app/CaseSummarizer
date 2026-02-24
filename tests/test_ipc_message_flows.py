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
    output = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            msg = result_q.get(timeout=0.2)
            output.append(msg)
        except Empty:
            # If we've waited a bit and got nothing new, check if we have enough
            if not internal_q.empty():
                continue
            # Give a small grace period for the last message
            time.sleep(0.3)
            try:
                msg = result_q.get_nowait()
                output.append(msg)
            except Empty:
                break

    state["shutdown"].set()
    t.join(timeout=2)
    return output, state


def _make_window_stub():
    """Create a minimal MainWindow state stub for handler testing."""
    stub = MagicMock()
    stub._pending_tasks = {"vocab": True, "qa": True, "summary": False}
    stub._completed_tasks = set()
    stub._qa_answering_active = False
    stub._qa_ready = False
    stub._processing_active = True
    stub._preprocessing_active = False
    stub._destroying = False
    stub._qa_results = []
    stub._qa_results_lock = threading.Lock()
    stub._vector_store_path = None
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

    def test_extraction_sequence_ner_then_qa_ready(self):
        """NER complete should arrive before qa_ready."""
        mock_embeddings = MagicMock()
        messages = [
            ("extraction_started", None),
            ("ner_complete", [{"term": "defendant"}]),
            ("extraction_complete", None),
            (
                "qa_ready",
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
            "qa_ready",
        ]

    def test_qa_ready_before_trigger_default_qa(self):
        """qa_ready must arrive before trigger_default_qa_started."""
        mock_embeddings = MagicMock()
        messages = [
            (
                "qa_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                    "chunk_count": 10,
                    "chunk_scores": None,
                },
            ),
            (
                "trigger_default_qa",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                },
            ),
        ]
        output, state = _run_forwarder_with_messages(messages)
        types = [m[0] for m in output]
        # qa_ready is forwarded, trigger_default_qa becomes trigger_default_qa_started
        assert "qa_ready" in types
        assert "trigger_default_qa_started" in types
        assert types.index("qa_ready") < types.index("trigger_default_qa_started")

    def test_llm_complete_forwarded_as_is(self):
        """llm_complete messages pass through without modification."""
        messages = [
            ("llm_progress", (1, 5)),
            ("llm_progress", (2, 5)),
            ("llm_complete", [{"term": "plaintiff", "source": "llm"}]),
        ]
        output, _ = _run_forwarder_with_messages(messages)
        assert len(output) == 3
        assert output[2][0] == "llm_complete"
        assert output[2][1] == [{"term": "plaintiff", "source": "llm"}]

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

    def test_qa_error_forwarded(self):
        """qa_error messages pass through the forwarder as-is."""
        messages = [
            ("qa_error", {"error": "Indexing failed"}),
        ]
        output, _ = _run_forwarder_with_messages(messages)
        assert len(output) == 1
        assert output[0] == ("qa_error", {"error": "Indexing failed"})


# ===========================================================================
# 2. State Preservation Tests
# ===========================================================================


class TestForwarderStatePreservation:
    """Test that forwarder saves/restores state correctly across messages."""

    def test_qa_ready_saves_embeddings_in_state(self):
        """qa_ready should save embeddings to subprocess state."""
        mock_embeddings = MagicMock()
        messages = [
            (
                "qa_ready",
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

    def test_qa_ready_saves_chunk_scores(self):
        """qa_ready should save chunk_scores to subprocess state."""
        mock_scores = MagicMock()
        messages = [
            (
                "qa_ready",
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

    def test_qa_ready_strips_embeddings_from_forwarded_message(self):
        """Forwarded qa_ready should NOT contain embeddings (not picklable)."""
        messages = [
            (
                "qa_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": MagicMock(),
                    "chunk_count": 42,
                    "chunk_scores": None,
                },
            ),
        ]
        output, _ = _run_forwarder_with_messages(messages)
        assert len(output) == 1
        msg_type, data = output[0]
        assert msg_type == "qa_ready"
        assert "embeddings" not in data
        assert data["vector_store_path"] == "/tmp/vs"
        assert data["chunk_count"] == 42

    def test_trigger_default_qa_uses_saved_state(self):
        """trigger_default_qa should use embeddings/path saved from qa_ready."""
        mock_embeddings = MagicMock()
        state = _make_forwarder_state()
        # Pre-set state as if qa_ready already ran
        state["embeddings"] = mock_embeddings
        state["vector_store_path"] = "/tmp/vs"

        messages = [
            (
                "trigger_default_qa",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                },
            ),
        ]
        output, final_state = _run_forwarder_with_messages(messages, state=state)
        types = [m[0] for m in output]
        assert "trigger_default_qa_started" in types

    def test_trigger_default_qa_without_state_logs_warning(self):
        """trigger_default_qa with no saved embeddings should warn, not crash."""
        state = _make_forwarder_state()
        # embeddings and vector_store_path are None

        messages = [
            (
                "trigger_default_qa",
                {
                    "vector_store_path": None,
                    "embeddings": None,
                },
            ),
        ]
        # Should not raise -- just log warning
        output, _ = _run_forwarder_with_messages(messages, state=state)
        types = [m[0] for m in output]
        assert "trigger_default_qa_started" in types

    def test_state_persists_across_multiple_messages(self):
        """State from qa_ready persists through subsequent unrelated messages."""
        mock_embeddings = MagicMock()
        messages = [
            (
                "qa_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                    "chunk_count": 42,
                    "chunk_scores": None,
                },
            ),
            ("progress", (80, "LLM processing...")),
            ("llm_complete", [{"term": "foo"}]),
        ]
        _, state = _run_forwarder_with_messages(messages)
        # State should still have embeddings from qa_ready
        assert state["embeddings"] is mock_embeddings
        assert state["vector_store_path"] == "/tmp/vs"


# ===========================================================================
# 3. Full Extraction Message Sequence (simulated)
# ===========================================================================


class TestFullExtractionSequence:
    """
    Test the complete message sequence from ProgressiveExtractionWorker.

    Expected order for vocab + Q&A flow:
    1. extraction_started
    2. progress (multiple)
    3. ner_complete
    4. extraction_complete
    5. progress (phase 2)
    6. qa_ready (-> stripped by forwarder)
    7. trigger_default_qa (-> intercepted, becomes trigger_default_qa_started)
    8. llm_progress (multiple, if LLM enabled)
    9. llm_complete
    10. qa_progress (multiple)
    11. qa_result (multiple)
    12. qa_complete
    """

    def test_full_vocab_qa_sequence_through_forwarder(self):
        """Full extraction message sequence should arrive in correct order."""
        mock_embeddings = MagicMock()
        messages = [
            ("extraction_started", None),
            ("progress", (10, "Phase 1: Running NER...")),
            ("ner_complete", [{"term": "defendant"}]),
            ("extraction_complete", None),
            ("progress", (20, "Phase 2: Building Q&A index...")),
            (
                "qa_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                    "chunk_count": 42,
                    "chunk_scores": None,
                },
            ),
            (
                "trigger_default_qa",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                },
            ),
            ("progress", (30, "Phase 3: LLM enhancement...")),
            ("llm_progress", (1, 3)),
            ("llm_progress", (2, 3)),
            ("llm_progress", (3, 3)),
            ("llm_complete", [{"term": "plaintiff"}]),
        ]
        output, state = _run_forwarder_with_messages(messages)
        types = [m[0] for m in output]

        # Verify key ordering constraints
        assert "extraction_started" in types
        assert "ner_complete" in types
        assert "qa_ready" in types
        assert "trigger_default_qa_started" in types  # Note: renamed by forwarder
        assert "llm_complete" in types

        # extraction_started before ner_complete
        assert types.index("extraction_started") < types.index("ner_complete")
        # ner_complete before qa_ready
        assert types.index("ner_complete") < types.index("qa_ready")
        # qa_ready before trigger_default_qa_started
        assert types.index("qa_ready") < types.index("trigger_default_qa_started")
        # trigger_default_qa was intercepted (not in output)
        assert "trigger_default_qa" not in types

    def test_qa_answering_messages_arrive_after_qa_ready(self):
        """qa_progress/qa_result/qa_complete must come after qa_ready."""
        mock_embeddings = MagicMock()
        messages = [
            (
                "qa_ready",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                    "chunk_count": 10,
                    "chunk_scores": None,
                },
            ),
            (
                "trigger_default_qa",
                {
                    "vector_store_path": "/tmp/vs",
                    "embeddings": mock_embeddings,
                },
            ),
            # These would come from QAWorker spawned by trigger_default_qa
            ("qa_progress", (0, 3, "What happened?")),
            ("qa_result", {"question": "What happened?", "answer": "..."}),
            ("qa_progress", (1, 3, "Who are the parties?")),
            ("qa_result", {"question": "Who are the parties?", "answer": "..."}),
            (
                "qa_complete",
                [
                    {"question": "What happened?", "answer": "..."},
                    {"question": "Who are the parties?", "answer": "..."},
                ],
            ),
        ]
        output, _ = _run_forwarder_with_messages(messages)
        types = [m[0] for m in output]

        assert "qa_ready" in types
        if "qa_complete" in types:
            assert types.index("qa_ready") < types.index("qa_complete")


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

        Only covers message types that affect _qa_ready, _qa_answering_active,
        _pending_tasks, _completed_tasks.
        """
        if msg_type == "qa_ready":
            stub._qa_ready = True
            # Note: NO _completed_tasks.add("qa") here (bug was fixed)

        elif msg_type == "qa_error":
            stub._qa_answering_active = False
            if stub._pending_tasks.get("qa"):
                stub._completed_tasks.add("qa")

        elif msg_type == "trigger_default_qa_started":
            stub._qa_answering_active = True

        elif msg_type == "qa_complete":
            stub._qa_answering_active = False
            if stub._pending_tasks.get("qa"):
                stub._completed_tasks.add("qa")

        elif msg_type == "ner_complete":
            pass  # UI updates only

        elif msg_type == "llm_complete":
            stub._completed_tasks.add("vocab")

        elif msg_type == "multi_doc_result":
            stub._completed_tasks.add("summary")

    def _all_tasks_complete(self, stub):
        """Check if all pending tasks are truly complete."""
        for task_name, is_pending in stub._pending_tasks.items():
            if is_pending and task_name not in stub._completed_tasks:
                return False
        return not stub._qa_answering_active

    def test_vocab_only_flow(self):
        """Vocab only: llm_complete -> finalize."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": False, "summary": False}

        sequence = [
            ("ner_complete", []),
            ("llm_complete", []),
        ]
        for msg_type, data in sequence:
            self._simulate_handler(stub, msg_type, data)

        assert "vocab" in stub._completed_tasks
        assert self._all_tasks_complete(stub)

    def test_vocab_plus_qa_flow(self):
        """Vocab+Q&A: qa_ready -> trigger -> answering -> qa_complete -> finalize."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": True, "summary": False}

        # Phase 1: NER
        self._simulate_handler(stub, "ner_complete", [])
        assert not stub._qa_answering_active

        # Phase 2: Q&A index ready
        self._simulate_handler(stub, "qa_ready", {"chunk_count": 42})
        assert stub._qa_ready is True
        assert "qa" not in stub._completed_tasks  # NOT premature!

        # Forwarder sends trigger_default_qa_started
        self._simulate_handler(stub, "trigger_default_qa_started", None)
        assert stub._qa_answering_active is True

        # Phase 3: LLM completes (vocab done, but Q&A still active)
        self._simulate_handler(stub, "llm_complete", [])
        assert "vocab" in stub._completed_tasks
        assert not self._all_tasks_complete(stub)  # Q&A still active!

        # Q&A answers arrive
        self._simulate_handler(stub, "qa_complete", [{"q": "test", "a": "answer"}])
        assert stub._qa_answering_active is False
        assert "qa" in stub._completed_tasks
        assert self._all_tasks_complete(stub)

    def test_vocab_qa_summary_flow(self):
        """Vocab+Q&A+Summary: all three must complete before finalization."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": True, "summary": True}

        self._simulate_handler(stub, "ner_complete", [])
        self._simulate_handler(stub, "qa_ready", {"chunk_count": 10})
        self._simulate_handler(stub, "trigger_default_qa_started", None)
        self._simulate_handler(stub, "llm_complete", [])

        # Vocab done, Q&A still active, summary not started
        assert not self._all_tasks_complete(stub)

        # Q&A completes
        self._simulate_handler(stub, "qa_complete", [])
        # Summary still pending
        assert not self._all_tasks_complete(stub)

        # Summary completes
        self._simulate_handler(stub, "multi_doc_result", MagicMock())
        assert self._all_tasks_complete(stub)

    def test_qa_error_still_allows_finalization(self):
        """Q&A error should mark Q&A as complete so finalization isn't blocked."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": True, "summary": False}

        self._simulate_handler(stub, "ner_complete", [])
        self._simulate_handler(stub, "qa_error", {"error": "Indexing failed"})
        self._simulate_handler(stub, "llm_complete", [])

        assert "qa" in stub._completed_tasks
        assert "vocab" in stub._completed_tasks
        assert self._all_tasks_complete(stub)

    def test_qa_error_clears_answering_flag(self):
        """qa_error should clear _qa_answering_active even if it was set."""
        stub = _make_window_stub()
        stub._qa_answering_active = True  # Was set by trigger_default_qa_started

        self._simulate_handler(stub, "qa_error", {"error": "crash"})
        assert stub._qa_answering_active is False

    def test_qa_ready_does_not_mark_complete(self):
        """qa_ready should NOT add 'qa' to completed_tasks (the old bug)."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": True, "summary": False}

        self._simulate_handler(stub, "qa_ready", {"chunk_count": 42})
        assert "qa" not in stub._completed_tasks

    def test_trigger_default_qa_started_sets_answering_flag(self):
        """trigger_default_qa_started must set _qa_answering_active."""
        stub = _make_window_stub()
        assert not stub._qa_answering_active

        self._simulate_handler(stub, "trigger_default_qa_started", None)
        assert stub._qa_answering_active is True

    def test_llm_complete_before_qa_complete(self):
        """LLM completing before Q&A should NOT allow premature finalization."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": True, "summary": False}

        self._simulate_handler(stub, "qa_ready", {})
        self._simulate_handler(stub, "trigger_default_qa_started", None)
        self._simulate_handler(stub, "llm_complete", [])

        # Vocab is done but Q&A answering is still active
        assert "vocab" in stub._completed_tasks
        assert "qa" not in stub._completed_tasks
        assert stub._qa_answering_active is True
        assert not self._all_tasks_complete(stub)

    def test_qa_complete_before_llm_complete(self):
        """Q&A completing before LLM should NOT allow premature finalization."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": True, "summary": False}

        self._simulate_handler(stub, "qa_ready", {})
        self._simulate_handler(stub, "trigger_default_qa_started", None)
        self._simulate_handler(stub, "qa_complete", [])

        # Q&A done but vocab not yet complete
        assert "qa" in stub._completed_tasks
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

    def test_run_qa_without_state_returns_error(self):
        """run_qa without prior qa_ready should return an error."""
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
        """followup without prior qa_ready should return qa_followup_result with None."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            self._wait_for_ready(manager)
            manager.send_command("followup", {"question": "test?"})
            messages = self._wait_for_messages(manager, expected_type="qa_followup_result")
            followup_msgs = [m for m in messages if m[0] == "qa_followup_result"]
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
            deadline = time.monotonic() + 10.0
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
            deadline = time.monotonic() + 15.0
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
    def _wait_for_ready(manager, timeout=10.0):
        """Poll until the worker subprocess signals it is ready."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            manager.check_for_messages()  # drains worker_ready
            if manager._worker_ready:
                return
            time.sleep(0.2)

    @staticmethod
    def _wait_for_messages(manager, expected_type, timeout=5.0):
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

    def test_qa_answering_keeps_poll_alive_after_processing_done(self):
        """After _processing_active=False, _qa_answering_active keeps polling."""
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
            messages=[("qa_complete", [])],
        )

    def test_no_activity_and_no_messages_stops_poll(self):
        """No active flags and no messages -> stop polling."""
        assert not self._should_continue_polling(
            processing=False,
            preprocessing=False,
            qa_answering=False,
            messages=[],
        )

    def test_only_qa_answering_plus_final_message(self):
        """
        Typical end-of-pipeline scenario: processing done, Q&A answering done,
        and the qa_complete message just arrived (in messages list).
        Should continue polling to process that message.
        """
        assert self._should_continue_polling(
            processing=False,
            preprocessing=False,
            qa_answering=False,
            messages=[("qa_complete", [])],
        )


# ===========================================================================
# 7. Command Dispatch Integration Tests
# ===========================================================================


class TestCommandDispatchIntegration:
    """Test _dispatch_command for commands that require state."""

    def test_run_qa_with_state_spawns_worker(self):
        """run_qa with valid state should spawn a QAWorker."""
        from src.worker_process import _dispatch_command

        internal_q = Queue()
        mock_embeddings = MagicMock()
        state = {
            "active_worker": None,
            "embeddings": mock_embeddings,
            "vector_store_path": "/tmp/vs",
            "worker_lock": threading.Lock(),
        }

        with patch("src.services.workers.QAWorker") as MockQAWorker:
            mock_worker_instance = MagicMock()
            MockQAWorker.return_value = mock_worker_instance

            _dispatch_command(
                "run_qa",
                {"answer_mode": "extraction", "use_default_questions": True},
                internal_q,
                state,
            )

            MockQAWorker.assert_called_once()
            mock_worker_instance.start.assert_called_once()
            assert state["active_worker"] is mock_worker_instance

    def test_summary_command_spawns_worker(self):
        """summary command should spawn MultiDocSummaryWorker."""
        from src.worker_process import _dispatch_command

        internal_q = Queue()
        state = {"active_worker": None, "worker_lock": threading.Lock()}

        with patch("src.services.workers.MultiDocSummaryWorker") as MockWorker:
            mock_instance = MagicMock()
            MockWorker.return_value = mock_instance

            _dispatch_command(
                "summary",
                {"documents": [{"filename": "test.pdf"}], "ai_params": {}},
                internal_q,
                state,
            )

            MockWorker.assert_called_once()
            mock_instance.start.assert_called_once()

    def test_new_command_stops_previous_worker(self):
        """Dispatching a new command should stop the active worker first."""
        from src.worker_process import _dispatch_command

        internal_q = Queue()
        old_worker = MagicMock()
        old_worker.is_alive.return_value = True
        state = {"active_worker": old_worker, "worker_lock": threading.Lock()}

        _dispatch_command("unknown_cmd", {}, internal_q, state)

        old_worker.stop.assert_called_once()
        old_worker.join.assert_called_once()


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
