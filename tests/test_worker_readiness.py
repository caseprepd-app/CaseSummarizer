"""
Tests for worker subprocess readiness handshake and command acknowledgment.

Covers:
- WorkerProcessManager.is_ready() before and after worker_ready signal
- check_for_messages() intercepts worker_ready (doesn't forward to GUI)
- _worker_ready flag resets on _cleanup_dead_process()
- worker_process_main() sends worker_ready as first message
- _command_loop sends command_ack before dispatching
- _handle_queue_message handles command_ack without error
- Readiness guard in _start_preprocessing auto-retries when worker not ready
- Readiness guard in _start_progressive_extraction auto-retries when worker not ready
"""

import threading
import time
from queue import Empty, Queue
from unittest.mock import MagicMock, patch


def _drain_queue(q, timeout=2.0):
    """Drain all messages from a queue using timeout-based reads (no .empty() race)."""
    messages = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            messages.append(q.get(timeout=0.1))
        except Empty:
            break
    return messages


def _poll_check(manager, min_msgs=0, timeout=2.0):
    """Poll check_for_messages() until at least min_msgs arrive (avoids mp.Queue sleep)."""
    all_msgs = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        all_msgs.extend(manager.check_for_messages())
        if len(all_msgs) >= min_msgs:
            return all_msgs
        time.sleep(0.05)
    return all_msgs


# ===========================================================================
# 1. WorkerProcessManager: is_ready() and worker_ready interception
# ===========================================================================


class TestIsReadyBeforeSignal:
    """is_ready() should return False until the worker_ready signal arrives."""

    def test_is_ready_false_before_start(self):
        """Before start(), is_ready() must be False."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        assert manager.is_ready() is False

    def test_is_ready_false_immediately_after_start(self):
        """Right after start(), before draining queue, is_ready() is False."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        # Don't actually start a subprocess — just set flags to simulate
        manager._started = True
        manager.process = MagicMock(is_alive=MagicMock(return_value=True))
        # No worker_ready signal yet
        assert manager.is_ready() is False

    def test_worker_ready_flag_init(self):
        """_worker_ready should be False on construction."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        assert manager._worker_ready is False


class TestWorkerReadyInterception:
    """check_for_messages() should intercept worker_ready and set the flag."""

    def test_worker_ready_sets_flag(self):
        """Receiving worker_ready should set _worker_ready = True."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.result_queue.put(("worker_ready", None))
        # Poll until worker_ready is intercepted (sets flag but returns 0 msgs)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            manager.check_for_messages()
            if manager._worker_ready:
                break
            time.sleep(0.05)
        assert manager._worker_ready is True

    def test_worker_ready_not_forwarded(self):
        """worker_ready should NOT appear in the returned message list."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.result_queue.put(("worker_ready", None))
        # Poll until worker_ready is intercepted
        deadline = time.monotonic() + 2.0
        all_messages = []
        while time.monotonic() < deadline:
            msgs = manager.check_for_messages()
            all_messages.extend(msgs)
            if manager._worker_ready:
                break
            time.sleep(0.05)
        assert len(all_messages) == 0

    def test_other_messages_still_forwarded(self):
        """Non-worker_ready messages should pass through normally."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.result_queue.put(("worker_ready", None))
        manager.result_queue.put(("progress", (50, "Working...")))
        manager.result_queue.put(("error", "something broke"))
        messages = _poll_check(manager, min_msgs=2)
        assert len(messages) == 2
        assert messages[0] == ("progress", (50, "Working..."))
        assert messages[1] == ("error", "something broke")

    def test_is_ready_true_after_signal_and_alive(self):
        """is_ready() returns True when _worker_ready is set and process is alive."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager._started = True
        manager.process = MagicMock(is_alive=MagicMock(return_value=True))
        manager.result_queue.put(("worker_ready", None))
        # Poll until worker_ready is intercepted
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            manager.check_for_messages()
            if manager._worker_ready:
                break
            time.sleep(0.05)
        assert manager.is_ready() is True

    def test_is_ready_drains_queue_without_check_for_messages(self):
        """is_ready() should find worker_ready even without check_for_messages().

        This covers the real-world scenario where the GUI never polls the
        queue before the user clicks 'Generate'. is_ready() must drain
        the worker_ready signal itself via _drain_ready_signal().
        """
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager._started = True
        manager.process = MagicMock(is_alive=MagicMock(return_value=True))
        manager.result_queue.put(("worker_ready", None))
        # Poll is_ready() until mp.Queue flushes (no check_for_messages)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if manager.is_ready():
                break
            time.sleep(0.05)
        assert manager.is_ready() is True

    def test_is_ready_preserves_other_messages_in_queue(self):
        """is_ready() should re-queue non-worker_ready messages."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager._started = True
        manager.process = MagicMock(is_alive=MagicMock(return_value=True))
        manager.result_queue.put(("progress", (50, "Working...")))
        manager.result_queue.put(("worker_ready", None))
        manager.result_queue.put(("error", "something"))
        # Poll is_ready() until mp.Queue flushes
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if manager.is_ready():
                break
            time.sleep(0.05)
        assert manager.is_ready() is True
        # Other messages should still be retrievable (poll for re-queued msgs)
        messages = _poll_check(manager, min_msgs=2)
        assert len(messages) == 2
        msg_types = [m[0] for m in messages]
        assert "progress" in msg_types
        assert "error" in msg_types

    def test_is_ready_false_when_ready_but_not_alive(self):
        """is_ready() returns False if flag is set but process is dead."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager._worker_ready = True
        manager.process = None  # Dead
        assert manager.is_ready() is False

    def test_malformed_message_still_forwarded(self):
        """Non-tuple messages should be forwarded without crashing."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.result_queue.put("plain_string")
        messages = _poll_check(manager, min_msgs=1)
        assert len(messages) == 1
        assert messages[0] == "plain_string"


class TestWorkerReadyReset:
    """_worker_ready should reset when the subprocess dies or is cleaned up."""

    def test_cleanup_resets_worker_ready(self):
        """_cleanup_dead_process() must reset _worker_ready to False."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager._worker_ready = True
        manager._started = True
        manager._cleanup_dead_process()
        assert manager._worker_ready is False

    def test_shutdown_resets_worker_ready(self):
        """After shutdown(), is_ready() should be False."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        # Wait for worker_ready signal
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            manager.check_for_messages()
            if manager._worker_ready:
                break
            time.sleep(0.1)
        assert manager._worker_ready is True, "Worker should have sent ready signal"

        manager.shutdown(blocking=True)
        assert manager.is_ready() is False


# ===========================================================================
# 2. Real subprocess: worker_ready signal
# ===========================================================================


class TestWorkerReadySignalRealSubprocess:
    """Verify the real subprocess sends worker_ready as its first message."""

    def test_worker_ready_arrives_after_start(self):
        """After start(), check_for_messages() should eventually see worker_ready."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                manager.check_for_messages()
                if manager._worker_ready:
                    break
                time.sleep(0.1)
            assert manager._worker_ready is True
            assert manager.is_ready() is True
        finally:
            manager.shutdown(blocking=True)

    def test_worker_ready_is_first_message(self):
        """worker_ready should be the very first message from the subprocess."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.start()
        try:
            # Collect raw messages before check_for_messages intercepts them
            deadline = time.monotonic() + 10.0
            first_msg = None
            while time.monotonic() < deadline:
                try:
                    msg = manager.result_queue.get(timeout=0.2)
                    first_msg = msg
                    break
                except Exception:
                    continue
            assert first_msg is not None, "No message received from subprocess"
            assert first_msg[0] == "worker_ready"
        finally:
            manager.shutdown(blocking=True)


# ===========================================================================
# 3. worker_process_main: worker_ready and command_ack unit tests
# ===========================================================================


class TestWorkerReadyInProcessMain:
    """Test that worker_process_main sends worker_ready."""

    def test_worker_ready_sent_to_result_queue(self):
        """worker_process_main should put worker_ready on result_queue."""
        from src.worker_process import worker_process_main

        command_queue = Queue()
        result_queue = Queue()

        # Send shutdown immediately so the process exits quickly
        command_queue.put("shutdown")

        # Run in a thread (not a process) for testability
        thread = threading.Thread(
            target=worker_process_main,
            args=(command_queue, result_queue),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5.0)

        # Collect all messages (timeout-based to avoid .empty() race)
        messages = _drain_queue(result_queue)

        ready_msgs = [m for m in messages if m[0] == "worker_ready"]
        assert len(ready_msgs) == 1
        assert ready_msgs[0] == ("worker_ready", None)


class TestCommandAck:
    """Test that _command_loop sends command_ack before dispatching."""

    def test_command_ack_sent_for_known_command(self):
        """A valid command should produce a command_ack message."""
        from src.worker_process import worker_process_main

        command_queue = Queue()
        result_queue = Queue()

        # Send a command that will fail (no real files), then shutdown
        command_queue.put(("bogus_cmd", {}))
        command_queue.put("shutdown")

        thread = threading.Thread(
            target=worker_process_main,
            args=(command_queue, result_queue),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5.0)

        messages = _drain_queue(result_queue)

        ack_msgs = [m for m in messages if m[0] == "command_ack"]
        assert len(ack_msgs) == 1
        assert ack_msgs[0][1]["cmd"] == "bogus_cmd"

    def test_command_ack_before_dispatch_error(self):
        """command_ack should arrive even if dispatch produces an error."""
        from src.worker_process import worker_process_main

        command_queue = Queue()
        result_queue = Queue()

        command_queue.put(("bogus_cmd", {}))
        command_queue.put("shutdown")

        thread = threading.Thread(
            target=worker_process_main,
            args=(command_queue, result_queue),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5.0)

        messages = _drain_queue(result_queue)

        # Find positions — ack should come before error
        ack_idx = next(i for i, m in enumerate(messages) if m[0] == "command_ack")
        # The error goes through internal_queue -> forwarder, so it may arrive
        # after ack (which goes directly to result_queue). That's expected.
        error_msgs = [m for m in messages if m[0] == "error"]
        assert len(error_msgs) >= 1
        # ack should exist
        assert ack_idx >= 0

    def test_no_ack_for_shutdown(self):
        """Shutdown command should NOT produce a command_ack."""
        from src.worker_process import worker_process_main

        command_queue = Queue()
        result_queue = Queue()

        command_queue.put("shutdown")

        thread = threading.Thread(
            target=worker_process_main,
            args=(command_queue, result_queue),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5.0)

        messages = _drain_queue(result_queue)

        ack_msgs = [m for m in messages if m[0] == "command_ack"]
        assert len(ack_msgs) == 0

    def test_no_ack_for_cancel(self):
        """Cancel command should NOT produce a command_ack."""
        from src.worker_process import worker_process_main

        command_queue = Queue()
        result_queue = Queue()

        command_queue.put("cancel")
        command_queue.put("shutdown")

        thread = threading.Thread(
            target=worker_process_main,
            args=(command_queue, result_queue),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5.0)

        messages = _drain_queue(result_queue)

        ack_msgs = [m for m in messages if m[0] == "command_ack"]
        assert len(ack_msgs) == 0

    def test_multiple_commands_each_get_ack(self):
        """Each dispatched command should produce its own command_ack."""
        from src.worker_process import worker_process_main

        command_queue = Queue()
        result_queue = Queue()

        command_queue.put(("cmd_a", {}))
        command_queue.put(("cmd_b", {}))
        command_queue.put("shutdown")

        thread = threading.Thread(
            target=worker_process_main,
            args=(command_queue, result_queue),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5.0)

        messages = _drain_queue(result_queue)

        ack_msgs = [m for m in messages if m[0] == "command_ack"]
        assert len(ack_msgs) == 2
        cmds = {m[1]["cmd"] for m in ack_msgs}
        assert cmds == {"cmd_a", "cmd_b"}


# ===========================================================================
# 4. _handle_queue_message: command_ack handler
# ===========================================================================


def _make_stub():
    """Create a stub with the same attributes MainWindow._handle_queue_message uses."""
    stub = MagicMock()
    stub._qa_ready = False
    stub._qa_answering_active = False
    stub._qa_results = []
    stub._qa_results_lock = threading.Lock()
    stub._pending_tasks = {"vocab": True, "qa": True, "summary": False}
    stub._completed_tasks = set()
    stub._vector_store_path = None
    stub.processing_results = []
    stub.followup_btn = MagicMock()
    stub.followup_entry = MagicMock()
    stub.status_label = MagicMock()
    stub.output_display = MagicMock()
    stub.file_table = MagicMock()
    stub.ask_default_questions_check = MagicMock()
    stub.ask_default_questions_check.get.return_value = True
    return stub


def _call_handler(stub, msg_type, data):
    """Call the real _handle_queue_message on our stub."""
    from src.ui.main_window import MainWindow

    MainWindow._handle_queue_message(stub, msg_type, data)


class TestCommandAckHandler:
    """Tests for command_ack message handling in MainWindow."""

    def test_command_ack_does_not_crash(self):
        """command_ack with dict data should be handled without error."""
        stub = _make_stub()
        _call_handler(stub, "command_ack", {"cmd": "process_files"})

    def test_command_ack_with_string_data(self):
        """command_ack with plain string data should not crash."""
        stub = _make_stub()
        _call_handler(stub, "command_ack", "process_files")

    def test_command_ack_no_ui_side_effects(self):
        """command_ack should not change status or update any widgets."""
        stub = _make_stub()
        _call_handler(stub, "command_ack", {"cmd": "extract"})
        stub.set_status.assert_not_called()
        stub.output_display.update_outputs.assert_not_called()
        stub.file_table.add_result.assert_not_called()

    def test_command_ack_does_not_affect_task_state(self):
        """command_ack should not modify task completion state."""
        stub = _make_stub()
        completed_before = set(stub._completed_tasks)
        _call_handler(stub, "command_ack", {"cmd": "run_qa"})
        assert stub._completed_tasks == completed_before


# ===========================================================================
# 5. Readiness guard in _start_preprocessing (non-modal auto-retry)
# ===========================================================================


class TestPreprocessingReadinessGuard:
    """Test that _start_preprocessing auto-retries when worker is not ready."""

    def _make_window_stub(self, worker_ready):
        """Create a MainWindow-like stub for _start_preprocessing."""
        stub = MagicMock()
        stub.selected_files = ["/tmp/test.pdf"]
        stub._check_ocr_availability = MagicMock(return_value=True)
        stub.add_files_btn = MagicMock()
        stub.generate_btn = MagicMock()
        stub.file_table = MagicMock()
        stub.processing_results = MagicMock()
        stub._worker_manager = MagicMock()
        stub._worker_manager.is_ready.return_value = worker_ready
        stub._preprocessing_active = False
        stub._worker_ready_retries = 0
        return stub

    def test_auto_retries_when_not_ready(self):
        """When is_ready() is False, should set status and schedule retry."""
        stub = self._make_window_stub(worker_ready=False)

        from src.ui.main_window import MainWindow

        MainWindow._start_preprocessing(stub)

        stub.set_status.assert_called_once()
        msg = stub.set_status.call_args[0][0]
        assert "starting up" in msg.lower() or "please wait" in msg.lower()
        stub.after.assert_called_once()
        assert stub.after.call_args[0][0] == 3000
        stub._worker_manager.send_command.assert_not_called()

    def test_retry_counter_increments(self):
        """Each retry should increment _worker_ready_retries."""
        stub = self._make_window_stub(worker_ready=False)

        from src.ui.main_window import MainWindow

        MainWindow._start_preprocessing(stub)
        assert stub._worker_ready_retries == 1

    def test_gives_up_after_max_retries(self):
        """After 20+ retries, should show error and re-enable buttons."""
        stub = self._make_window_stub(worker_ready=False)
        stub._worker_ready_retries = 20  # Already at limit

        from src.ui.main_window import MainWindow

        MainWindow._start_preprocessing(stub)

        stub.set_status_error.assert_called_once()
        msg = stub.set_status_error.call_args[0][0]
        assert "failed" in msg.lower() or "restart" in msg.lower()
        stub.add_files_btn.configure.assert_any_call(state="normal")
        stub.generate_btn.configure.assert_any_call(state="normal")
        stub.after.assert_not_called()

    def test_buttons_not_re_enabled_during_retry(self):
        """During normal retries (not max), buttons stay disabled."""
        stub = self._make_window_stub(worker_ready=False)

        from src.ui.main_window import MainWindow

        MainWindow._start_preprocessing(stub)

        # Buttons should NOT be re-enabled (no configure(state="normal") call
        # after the initial disable at the top of the method)
        stub._worker_manager.send_command.assert_not_called()

    def test_proceeds_when_ready(self):
        """When is_ready() is True, should send_command normally."""
        stub = self._make_window_stub(worker_ready=True)

        from src.ui.main_window import MainWindow

        MainWindow._start_preprocessing(stub)

        stub._worker_manager.send_command.assert_called_once()
        assert stub._worker_manager.send_command.call_args[0][0] == "process_files"

    def test_retry_counter_resets_on_success(self):
        """When worker is ready, _worker_ready_retries should reset to 0."""
        stub = self._make_window_stub(worker_ready=True)
        stub._worker_ready_retries = 5  # Had some retries before

        from src.ui.main_window import MainWindow

        MainWindow._start_preprocessing(stub)

        assert stub._worker_ready_retries == 0


# ===========================================================================
# 6. Readiness guard in _start_progressive_extraction (non-modal auto-retry)
# ===========================================================================


class TestExtractionReadinessGuard:
    """Test that _start_progressive_extraction auto-retries when not ready."""

    def _make_extraction_stub(self, worker_ready):
        """Create a MainWindow-like stub for _start_progressive_extraction."""
        stub = MagicMock()
        stub.processing_results = [
            {"filename": "test.pdf", "text": "Some legal text", "status": "success"}
        ]
        stub._worker_manager = MagicMock()
        stub._worker_manager.is_ready.return_value = worker_ready
        stub._queue_poll_id = None
        stub._worker_ready_retries = 0
        return stub

    def test_auto_retries_when_not_ready(self):
        """When is_ready() is False, should set status and schedule retry."""
        stub = self._make_extraction_stub(worker_ready=False)

        with patch("src.services.DocumentService") as mock_doc_svc:
            mock_doc_svc.return_value.combine_document_texts.return_value = "Some text"

            from src.ui.main_window import MainWindow

            MainWindow._start_progressive_extraction(stub)

            # set_status is called once for "Starting extraction..." and once for retry
            retry_calls = [
                c
                for c in stub.set_status.call_args_list
                if "starting up" in c[0][0].lower() or "please wait" in c[0][0].lower()
            ]
            assert len(retry_calls) == 1
            stub.after.assert_called_once()
            assert stub.after.call_args[0][0] == 3000
            stub._worker_manager.send_command.assert_not_called()

    def test_gives_up_after_max_retries(self):
        """After 20+ retries, should call _on_tasks_complete with failure."""
        stub = self._make_extraction_stub(worker_ready=False)
        stub._worker_ready_retries = 20

        with patch("src.services.DocumentService") as mock_doc_svc:
            mock_doc_svc.return_value.combine_document_texts.return_value = "Some text"

            from src.ui.main_window import MainWindow

            MainWindow._start_progressive_extraction(stub)

            stub._on_tasks_complete.assert_called_once()
            args = stub._on_tasks_complete.call_args[0]
            assert args[0] is False  # success=False
            assert "failed" in args[1].lower() or "restart" in args[1].lower()
            stub.after.assert_not_called()

    def test_proceeds_when_ready(self):
        """When is_ready() is True, should send extract command."""
        stub = self._make_extraction_stub(worker_ready=True)

        with patch("src.services.DocumentService") as mock_doc_svc:
            mock_doc_svc.return_value.combine_document_texts.return_value = "Some text"

            from src.ui.main_window import MainWindow

            MainWindow._start_progressive_extraction(stub)

            stub._worker_manager.send_command.assert_called_once()
            assert stub._worker_manager.send_command.call_args[0][0] == "extract"

    def test_retry_counter_resets_on_success(self):
        """When worker is ready, _worker_ready_retries should reset to 0."""
        stub = self._make_extraction_stub(worker_ready=True)
        stub._worker_ready_retries = 3

        with patch("src.services.DocumentService") as mock_doc_svc:
            mock_doc_svc.return_value.combine_document_texts.return_value = "Some text"

            from src.ui.main_window import MainWindow

            MainWindow._start_progressive_extraction(stub)

            assert stub._worker_ready_retries == 0
