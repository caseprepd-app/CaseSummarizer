"""
Edge case tests for WorkerProcessManager lifecycle and thread safety.

Covers startup/shutdown races, dead-process recovery, unpicklable results,
and concurrent command sending — all without spawning real subprocesses.
"""

import threading
import time
from queue import Empty
from unittest.mock import MagicMock, patch

from src.services.worker_manager import WorkerProcessManager


def _mock_alive_process():
    """Create a mock process that reports is_alive=True."""
    proc = MagicMock()
    proc.is_alive.return_value = True
    proc.exitcode = None
    proc.pid = 99999
    return proc


class TestShutdownDuringStartup:
    """Verify shutdown right after start does not race or hang."""

    def test_immediate_shutdown_after_start(self):
        """Calling shutdown immediately after start is safe."""
        mgr = WorkerProcessManager()
        mgr._started = True
        proc = _mock_alive_process()
        # After join, simulate process exiting
        proc.is_alive.side_effect = [True, False]
        proc.exitcode = 0
        mgr.process = proc

        mgr.shutdown(blocking=True)

        assert mgr.process is None
        assert mgr._started is False

    def test_shutdown_during_startup_no_hang(self):
        """Shutdown that must force-terminate still completes."""
        mgr = WorkerProcessManager()
        mgr._started = True
        proc = _mock_alive_process()
        # Process stays alive through graceful shutdown, dies after terminate
        proc.is_alive.side_effect = [True, True, True, False]
        proc.exitcode = -15
        mgr.process = proc

        start = time.monotonic()
        mgr.shutdown(blocking=True)
        elapsed = time.monotonic() - start

        proc.terminate.assert_called_once()
        assert elapsed < 15.0, "Shutdown should not hang"


class TestRapidRestart:
    """Start/shutdown cycles should not leak resources."""

    def test_five_rapid_cycles(self):
        """Five start/shutdown cycles leave manager in clean state."""
        mgr = WorkerProcessManager()

        for _ in range(5):
            mgr._started = True
            proc = _mock_alive_process()
            proc.is_alive.side_effect = [True, False]
            proc.exitcode = 0
            mgr.process = proc

            mgr.shutdown(blocking=True)

            assert mgr.process is None
            assert mgr._started is False
            assert mgr._worker_ready is False


class TestSendCommandToDeadProcess:
    """Verify send_command handles a dead process gracefully."""

    def test_send_to_dead_triggers_restart(self):
        """send_command on dead process calls restart_if_dead."""
        mgr = WorkerProcessManager()
        proc = _mock_alive_process()
        proc.is_alive.return_value = False
        mgr.process = proc

        with patch.object(mgr, "restart_if_dead") as mock_restart:
            mgr.send_command("extract", {"documents": []})
            mock_restart.assert_called_once()

    def test_send_command_queue_error_does_not_raise(self):
        """If command_queue.put fails, send_command logs, not raises."""
        mgr = WorkerProcessManager()
        proc = _mock_alive_process()
        mgr.process = proc

        with patch.object(mgr.command_queue, "put", side_effect=OSError("broken")):
            # Should not raise
            mgr.send_command("extract", {"documents": []})


class TestWorkerNeverSendsReady:
    """Verify is_ready handles a worker that never signals readiness."""

    def test_is_ready_false_when_no_signal(self):
        """is_ready returns False when no worker_ready is on the queue."""
        mgr = WorkerProcessManager()
        proc = _mock_alive_process()
        mgr.process = proc
        mgr._started = True

        # Queue is empty -- no worker_ready signal
        assert mgr.is_ready() is False
        assert mgr._worker_ready is False

    def test_is_ready_preserves_other_messages(self):
        """Draining for ready signal re-queues non-ready messages."""
        mgr = WorkerProcessManager()
        proc = _mock_alive_process()
        mgr.process = proc
        mgr._started = True

        # Put a non-ready message
        mgr.result_queue.put(("progress", (10, "Loading...")))
        time.sleep(0.2)  # mp.Queue flush

        assert mgr.is_ready() is False

        # The progress message should still be retrievable
        msg = mgr.result_queue.get(timeout=2)
        assert msg[0] == "progress"


class TestUnpicklableResult:
    """Verify check_for_messages handles garbled queue entries."""

    def test_non_tuple_on_result_queue(self):
        """A non-tuple result is appended as-is, not crash."""
        mgr = WorkerProcessManager()
        mgr.result_queue.put("bare_string")
        time.sleep(0.2)  # mp.Queue flush

        messages = mgr.check_for_messages()

        # check_for_messages catches TypeError on unpack and appends raw
        assert len(messages) == 1
        assert messages[0] == "bare_string"

    def test_single_element_tuple_on_result_queue(self):
        """A 1-element tuple survives check_for_messages."""
        mgr = WorkerProcessManager()
        mgr.result_queue.put(("lonely",))
        time.sleep(0.2)

        messages = mgr.check_for_messages()

        assert len(messages) == 1


class TestConcurrentSendCommands:
    """Verify thread safety when two threads send commands."""

    def test_two_threads_send_simultaneously(self):
        """Two threads calling send_command do not corrupt the queue."""
        mgr = WorkerProcessManager()
        proc = _mock_alive_process()
        mgr.process = proc

        errors = []
        barrier = threading.Barrier(2)

        def sender(label):
            """Send 10 commands, collecting any exceptions."""
            try:
                barrier.wait(timeout=5)
                for i in range(10):
                    mgr.send_command(f"{label}_{i}", {"i": i})
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=sender, args=("A",))
        t2 = threading.Thread(target=sender, args=("B",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Threads raised: {errors}"

        # Drain and verify all 20 commands arrived
        commands = []
        while True:
            try:
                commands.append(mgr.command_queue.get_nowait())
            except Empty:
                break
        assert len(commands) == 20
