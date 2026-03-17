"""
Tests for WorkerProcessManager crash recovery and lifecycle edge cases.

Covers restart_if_dead(), _cleanup_dead_process(), and crash detection paths.
"""

import multiprocessing
from queue import Empty
from unittest.mock import MagicMock, patch

from src.services.worker_manager import WorkerProcessManager


class TestWorkerManagerCrashRecovery:
    """Tests for crash recovery in WorkerProcessManager."""

    def _make_manager(self):
        """Create a manager without starting a real subprocess."""
        mgr = WorkerProcessManager()
        return mgr

    def test_restart_if_dead_when_never_started(self):
        """restart_if_dead starts a new process when none exists."""
        mgr = self._make_manager()
        assert mgr.process is None

        with patch.object(mgr, "start") as mock_start:
            mgr.restart_if_dead()
            mock_start.assert_called_once()

    def test_restart_if_dead_when_process_dead(self):
        """restart_if_dead restarts when process has exited."""
        mgr = self._make_manager()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mgr.process = mock_proc
        mgr._started = True

        with (
            patch.object(mgr, "start") as mock_start,
            patch.object(mgr, "_cleanup_dead_process") as mock_cleanup,
        ):
            mgr.restart_if_dead()
            mock_cleanup.assert_called_once()
            mock_start.assert_called_once()

    def test_restart_if_dead_noop_when_alive(self):
        """restart_if_dead does nothing when process is alive."""
        mgr = self._make_manager()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mgr.process = mock_proc
        mgr._started = True

        with patch.object(mgr, "start") as mock_start:
            mgr.restart_if_dead()
            mock_start.assert_not_called()

    def test_cleanup_dead_process_resets_state(self):
        """_cleanup_dead_process clears process, flags, and queues."""
        import time

        mgr = self._make_manager()
        mgr.process = MagicMock()
        mgr._started = True
        mgr._worker_ready = True
        # Put some items on the queues and wait for them to flush
        mgr.command_queue.put("test_cmd")
        mgr.result_queue.put(("test", None))
        time.sleep(0.2)

        mgr._cleanup_dead_process()

        assert mgr.process is None
        assert mgr._started is False
        assert mgr._worker_ready is False

    def test_send_command_auto_restarts_dead_process(self):
        """send_command auto-restarts when process is dead."""
        mgr = self._make_manager()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mgr.process = mock_proc

        with patch.object(mgr, "restart_if_dead") as mock_restart:
            mgr.send_command("process_files", {"file_paths": []})
            mock_restart.assert_called_once()

    def test_send_command_puts_on_queue(self):
        """send_command puts (cmd_type, args) tuple on command_queue."""
        mgr = self._make_manager()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mgr.process = mock_proc

        mgr.send_command("extract", {"documents": []})

        msg = mgr.command_queue.get(timeout=2)
        assert msg == ("extract", {"documents": []})

    def test_check_for_messages_intercepts_worker_ready(self):
        """check_for_messages intercepts worker_ready and sets flag."""
        import time

        mgr = self._make_manager()
        mgr.result_queue.put(("worker_ready", None))
        mgr.result_queue.put(("progress", (50, "Working...")))
        time.sleep(0.2)  # mp.Queue needs time to flush

        messages = mgr.check_for_messages()

        assert mgr._worker_ready is True
        # worker_ready should NOT be forwarded
        assert len(messages) == 1
        assert messages[0] == ("progress", (50, "Working..."))

    def test_is_ready_drains_for_ready_signal(self):
        """is_ready peeks at queue for worker_ready signal."""
        import time

        mgr = self._make_manager()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mgr.process = mock_proc
        mgr._started = True

        # Put worker_ready and another message
        mgr.result_queue.put(("worker_ready", None))
        mgr.result_queue.put(("progress", (10, "Loading...")))
        time.sleep(0.2)  # mp.Queue needs time to flush

        assert mgr.is_ready() is True
        assert mgr._worker_ready is True

        # The progress message should still be on the queue (requeued)
        time.sleep(0.1)
        msg = mgr.result_queue.get(timeout=2)
        assert msg[0] == "progress"

    def test_shutdown_sends_shutdown_command(self):
        """shutdown puts 'shutdown' on command_queue before cleanup."""

        mgr = self._make_manager()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mock_proc.exitcode = 0
        mgr.process = mock_proc
        mgr._started = True

        # Intercept command_queue.put to verify shutdown was sent
        sent_commands = []
        original_put = mgr.command_queue.put

        def capturing_put(item):
            """Capture items sent to the queue."""
            sent_commands.append(item)
            original_put(item)

        mgr.command_queue.put = capturing_put

        # After join, process is no longer alive
        mock_proc.is_alive.side_effect = [True, False]

        mgr.shutdown(blocking=True)

        assert "shutdown" in sent_commands

    def test_shutdown_force_terminates_if_stuck(self):
        """shutdown force-terminates process that doesn't exit gracefully."""
        mgr = self._make_manager()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mock_proc.exitcode = None
        mgr.process = mock_proc
        mgr._started = True

        mgr.shutdown(blocking=True)

        mock_proc.terminate.assert_called_once()

    def test_cancel_sends_cancel_string(self):
        """cancel sends 'cancel' to command_queue."""
        mgr = self._make_manager()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mgr.process = mock_proc

        mgr.cancel()

        cmd = mgr.command_queue.get(timeout=2)
        assert cmd == "cancel"

    def test_cancel_noop_when_not_alive(self):
        """cancel does nothing when process isn't alive."""
        mgr = self._make_manager()
        mgr.cancel()
        # Queue should be empty
        with __import__("pytest").raises(Empty):
            mgr.command_queue.get_nowait()

    def test_clear_queue_drains_all(self):
        """_clear_queue drains all items from a queue."""
        import time

        q = multiprocessing.Queue()
        q.put("a")
        q.put("b")
        q.put("c")
        time.sleep(0.2)  # mp.Queue needs time to flush

        WorkerProcessManager._clear_queue(q)

        with __import__("pytest").raises(Empty):
            q.get_nowait()
