"""
Tests for worker subprocess readiness handshake and command acknowledgment.

Covers:
- WorkerProcessManager.is_ready() before and after worker_ready signal
- check_for_messages() intercepts worker_ready (doesn't forward to GUI)
- _worker_ready flag resets on _cleanup_dead_process()
- worker_process_main() sends worker_ready as first message
- _command_loop sends command_ack before dispatching
- _handle_queue_message handles command_ack without error
- Readiness guard in _start_preprocessing blocks when worker not ready
- Readiness guard in _start_progressive_extraction blocks when worker not ready
"""

import threading
import time
from queue import Queue
from unittest.mock import MagicMock, patch

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
        time.sleep(0.1)  # mp.Queue needs time to flush
        manager.check_for_messages()
        assert manager._worker_ready is True

    def test_worker_ready_not_forwarded(self):
        """worker_ready should NOT appear in the returned message list."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.result_queue.put(("worker_ready", None))
        time.sleep(0.1)  # mp.Queue needs time to flush
        messages = manager.check_for_messages()
        assert len(messages) == 0

    def test_other_messages_still_forwarded(self):
        """Non-worker_ready messages should pass through normally."""
        from src.services.worker_manager import WorkerProcessManager

        manager = WorkerProcessManager()
        manager.result_queue.put(("worker_ready", None))
        manager.result_queue.put(("progress", (50, "Working...")))
        manager.result_queue.put(("error", "something broke"))
        time.sleep(0.1)  # mp.Queue needs time to flush
        messages = manager.check_for_messages()
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
        time.sleep(0.1)  # mp.Queue needs time to flush
        manager.check_for_messages()
        assert manager.is_ready() is True

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
        time.sleep(0.1)  # mp.Queue needs time to flush
        messages = manager.check_for_messages()
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

        # Collect all messages
        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get_nowait())

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

        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get_nowait())

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

        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get_nowait())

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

        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get_nowait())

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

        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get_nowait())

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

        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get_nowait())

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
# 5. Readiness guard in _start_preprocessing
# ===========================================================================


class TestPreprocessingReadinessGuard:
    """Test that _start_preprocessing blocks when worker is not ready."""

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
        return stub

    def test_blocks_when_not_ready(self):
        """When is_ready() is False, should show dialog and NOT send_command."""
        stub = self._make_window_stub(worker_ready=False)

        with patch("src.ui.main_window.messagebox") as mock_mb:
            from src.ui.main_window import MainWindow

            MainWindow._start_preprocessing(stub)

            mock_mb.showinfo.assert_called_once()
            title, msg = mock_mb.showinfo.call_args[0]
            assert "wait" in title.lower() or "wait" in msg.lower()
            stub._worker_manager.send_command.assert_not_called()

    def test_re_enables_buttons_when_not_ready(self):
        """When blocked, buttons should be re-enabled so user can try again."""
        stub = self._make_window_stub(worker_ready=False)

        with patch("src.ui.main_window.messagebox"):
            from src.ui.main_window import MainWindow

            MainWindow._start_preprocessing(stub)

            stub.add_files_btn.configure.assert_any_call(state="normal")
            stub.generate_btn.configure.assert_any_call(state="normal")

    def test_proceeds_when_ready(self):
        """When is_ready() is True, should send_command normally."""
        stub = self._make_window_stub(worker_ready=True)

        with patch("src.ui.main_window.messagebox") as mock_mb:
            from src.ui.main_window import MainWindow

            MainWindow._start_preprocessing(stub)

            mock_mb.showinfo.assert_not_called()
            stub._worker_manager.send_command.assert_called_once()
            assert stub._worker_manager.send_command.call_args[0][0] == "process_files"


# ===========================================================================
# 6. Readiness guard in _start_progressive_extraction
# ===========================================================================


class TestExtractionReadinessGuard:
    """Test that _start_progressive_extraction blocks when worker not ready."""

    def _make_extraction_stub(self, worker_ready):
        """Create a MainWindow-like stub for _start_progressive_extraction."""
        stub = MagicMock()
        stub.processing_results = [
            {"filename": "test.pdf", "text": "Some legal text", "status": "success"}
        ]
        stub._worker_manager = MagicMock()
        stub._worker_manager.is_ready.return_value = worker_ready
        stub._queue_poll_id = None
        return stub

    def test_blocks_when_not_ready(self):
        """When is_ready() is False, should show dialog and NOT send_command."""
        stub = self._make_extraction_stub(worker_ready=False)

        with (
            patch("src.ui.main_window.messagebox") as mock_mb,
            patch("src.services.DocumentService") as mock_doc_svc,
            patch("src.user_preferences.get_user_preferences") as mock_prefs,
        ):
            mock_doc_svc.return_value.combine_document_texts.return_value = "Some text"
            mock_prefs.return_value.is_vocab_llm_enabled.return_value = False

            from src.ui.main_window import MainWindow

            MainWindow._start_progressive_extraction(stub)

            mock_mb.showinfo.assert_called_once()
            stub._worker_manager.send_command.assert_not_called()

    def test_proceeds_when_ready(self):
        """When is_ready() is True, should send extract command."""
        stub = self._make_extraction_stub(worker_ready=True)

        with (
            patch("src.ui.main_window.messagebox") as mock_mb,
            patch("src.services.DocumentService") as mock_doc_svc,
            patch("src.user_preferences.get_user_preferences") as mock_prefs,
        ):
            mock_doc_svc.return_value.combine_document_texts.return_value = "Some text"
            mock_prefs.return_value.is_vocab_llm_enabled.return_value = False

            from src.ui.main_window import MainWindow

            MainWindow._start_progressive_extraction(stub)

            mock_mb.showinfo.assert_not_called()
            stub._worker_manager.send_command.assert_called_once()
            assert stub._worker_manager.send_command.call_args[0][0] == "extract"
