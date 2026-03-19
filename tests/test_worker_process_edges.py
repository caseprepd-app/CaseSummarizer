"""
Edge case tests for worker_process command loop and forwarder loop.

Covers malformed queue messages, shutdown races, and error resilience
without spawning real subprocesses.
"""

import threading
import time
from queue import Queue
from unittest.mock import MagicMock, patch

from src.worker_process import _command_loop, _forwarder_loop


def _make_state():
    """Build a minimal state dict for command loop tests."""
    return {
        "embeddings": None,
        "vector_store_path": None,
        "chunk_scores": None,
        "documents": None,
        "active_worker": None,
        "auto_semantic_worker": None,
        "ask_default_questions": True,
        "shutdown": threading.Event(),
        "worker_lock": threading.Lock(),
    }


class TestMalformedQueueMessages:
    """Verify the command loop survives non-tuple messages."""

    def test_non_tuple_string_message(self):
        """A bare string on the command queue is logged and skipped."""
        cmd_q = Queue()
        internal_q = Queue()
        result_q = Queue()
        state = _make_state()

        cmd_q.put("random_string_not_a_sentinel")
        cmd_q.put("shutdown")

        _command_loop(cmd_q, internal_q, result_q, state)

        assert state["shutdown"].is_set()

    def test_non_tuple_int_message(self):
        """An integer on the command queue is logged and skipped."""
        cmd_q = Queue()
        internal_q = Queue()
        result_q = Queue()
        state = _make_state()

        cmd_q.put(42)
        cmd_q.put("shutdown")

        _command_loop(cmd_q, internal_q, result_q, state)

        assert state["shutdown"].is_set()

    def test_non_tuple_dict_message(self):
        """A dict on the command queue is logged and skipped."""
        cmd_q = Queue()
        internal_q = Queue()
        result_q = Queue()
        state = _make_state()

        cmd_q.put({"bad": "message"})
        cmd_q.put("shutdown")

        _command_loop(cmd_q, internal_q, result_q, state)

        assert state["shutdown"].is_set()


class TestThreeElementTuple:
    """Verify the command loop handles tuples with wrong arity."""

    def test_three_element_tuple_skipped(self):
        """A 3-element tuple cannot unpack to (cmd, args), so it is skipped."""
        cmd_q = Queue()
        internal_q = Queue()
        result_q = Queue()
        state = _make_state()

        cmd_q.put(("cmd", "arg1", "extra"))
        cmd_q.put("shutdown")

        _command_loop(cmd_q, internal_q, result_q, state)

        assert state["shutdown"].is_set()


class TestQueueEOFDuringOperation:
    """Simulate EOFError from command_queue.get()."""

    def test_eof_error_continues_loop(self):
        """EOFError from queue.get is caught; loop continues to shutdown."""
        cmd_q = MagicMock()
        internal_q = Queue()
        result_q = Queue()
        state = _make_state()

        # First call raises EOFError, second returns shutdown
        cmd_q.get.side_effect = [EOFError("pipe closed"), "shutdown"]

        _command_loop(cmd_q, internal_q, result_q, state)

        assert state["shutdown"].is_set()
        assert cmd_q.get.call_count == 2


class TestShutdownWhileProcessing:
    """Send shutdown while a command is being dispatched."""

    @patch("src.worker_process._dispatch_command")
    def test_shutdown_interrupts_processing(self, mock_dispatch):
        """Shutdown sentinel after a command ends the loop promptly."""
        cmd_q = Queue()
        internal_q = Queue()
        result_q = Queue()
        state = _make_state()

        # Simulate a slow dispatch that takes a moment
        def slow_dispatch(*_args, **_kwargs):
            """Simulate work that takes some time."""
            time.sleep(0.1)

        mock_dispatch.side_effect = slow_dispatch

        cmd_q.put(("extract", {}))
        cmd_q.put("shutdown")

        start = time.monotonic()
        _command_loop(cmd_q, internal_q, result_q, state)
        elapsed = time.monotonic() - start

        assert state["shutdown"].is_set()
        assert elapsed < 5.0, "Command loop should not hang"


class TestRapidCommandSequence:
    """Send many commands quickly; verify no deadlock."""

    @patch("src.worker_process._dispatch_command")
    def test_twenty_commands_all_processed(self, mock_dispatch):
        """20 rapid commands are each dispatched before shutdown."""
        cmd_q = Queue()
        internal_q = Queue()
        result_q = Queue()
        state = _make_state()

        for i in range(20):
            cmd_q.put((f"cmd_{i}", {"i": i}))
        cmd_q.put("shutdown")

        _command_loop(cmd_q, internal_q, result_q, state)

        assert mock_dispatch.call_count == 20
        assert state["shutdown"].is_set()


class TestResultQueuePutAfterShutdown:
    """Verify the forwarder survives a broken result queue."""

    def test_forwarder_survives_closed_result_queue(self):
        """If result_queue.put raises, forwarder logs and continues."""
        internal_q = Queue()
        result_q = MagicMock()
        command_q = Queue()
        state = _make_state()

        # result_queue.put always raises (simulating closed pipe)
        result_q.put.side_effect = OSError("queue closed")

        internal_q.put(("progress", (50, "Working...")))

        # Run forwarder in a thread; let it process, then shut down
        t = threading.Thread(
            target=_forwarder_loop,
            args=(internal_q, result_q, command_q, state),
        )
        t.start()
        # Give it time to attempt the put and hit the error
        time.sleep(1.0)
        state["shutdown"].set()
        t.join(timeout=5)

        assert not t.is_alive(), "Forwarder should not hang"
