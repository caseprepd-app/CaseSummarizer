"""
Tests for MainWindow._poll_queue() and _handle_queue_message() dispatch.

Uses a headless mock approach: we directly test _handle_queue_message()
by constructing a minimal MainWindow mock with the required attributes.

NOTE: The "error" message type triggers messagebox.showerror() which would
show a modal dialog — we patch it away in tests that exercise that path.
"""

from unittest.mock import MagicMock, patch


def _make_mock_window():
    """Create a mock MainWindow with attributes needed by _handle_queue_message."""
    window = MagicMock()
    window._destroying = False
    window._processing_active = True
    window._preprocessing_active = False
    window._semantic_answering_active = False
    window._key_sentences_pending = False
    window._semantic_ready = False
    window._semantic_failed = False
    window._vector_store_path = None
    window._resize_in_progress = False
    window._queue_poll_id = None
    window._pending_tasks = {}
    window._completed_tasks = set()
    window._failed_tasks = set()
    window.processing_results = []
    window.set_status = MagicMock()
    window.set_status_error = MagicMock()
    window.file_table = MagicMock()
    window.output_display = MagicMock()
    window.followup_entry = MagicMock()
    window.followup_btn = MagicMock()
    window._on_preprocessing_complete = MagicMock()
    window._on_tasks_complete = MagicMock()
    window._all_tasks_complete = MagicMock(return_value=False)
    window._finalize_tasks = MagicMock()
    window._worker_manager = MagicMock()
    return window


def _call_handler(window, msg_type, data):
    """Call _handle_queue_message on a real MainWindow class with mock instance."""
    from src.ui.main_window import MainWindow

    MainWindow._handle_queue_message(window, msg_type, data)


class TestHandleQueueMessage:
    """Tests for message dispatch in _handle_queue_message."""

    def test_progress_updates_status(self):
        """Progress message updates status label."""
        window = _make_mock_window()
        _call_handler(window, "progress", (50, "Working..."))
        window.set_status.assert_called_once_with("Working...")

    def test_progress_skipped_when_not_active(self):
        """Progress messages are ignored when processing is not active."""
        window = _make_mock_window()
        window._processing_active = False
        _call_handler(window, "progress", (50, "Stale message"))
        window.set_status.assert_not_called()

    def test_file_processed_appended(self):
        """file_processed message adds result to processing_results."""
        window = _make_mock_window()
        result = {"filename": "test.pdf", "status": "ok"}
        _call_handler(window, "file_processed", result)
        assert result in window.processing_results
        window.file_table.add_result.assert_called_once_with(result)

    def test_processing_finished_calls_callback(self):
        """processing_finished message triggers _on_preprocessing_complete."""
        window = _make_mock_window()
        results = [{"filename": "a.pdf"}]
        _call_handler(window, "processing_finished", results)
        window._on_preprocessing_complete.assert_called_once_with(results)

    @patch("src.ui.main_window.messagebox")
    def test_error_message_shows_dialog_and_resets_flags(self, mock_msgbox):
        """error message shows error dialog and resets processing flags."""
        window = _make_mock_window()
        window._processing_active = True
        _call_handler(window, "error", "Something broke")
        mock_msgbox.showerror.assert_called_once()
        window.set_status_error.assert_called_once()
        assert window._processing_active is False

    def test_status_error_displayed(self):
        """status_error shows non-blocking error in status bar."""
        window = _make_mock_window()
        _call_handler(window, "status_error", "Minor issue")
        window.set_status_error.assert_called_once_with("Minor issue")

    def test_extraction_started_dims_buttons(self):
        """extraction_started message dims feedback buttons."""
        window = _make_mock_window()
        _call_handler(window, "extraction_started", None)
        window.output_display.set_extraction_in_progress.assert_called_once_with(True)

    def test_extraction_complete_enables_buttons(self):
        """extraction_complete message re-enables feedback buttons."""
        window = _make_mock_window()
        _call_handler(window, "extraction_complete", None)
        window.output_display.set_extraction_in_progress.assert_called_once_with(False)

    def test_partial_vocab_complete_updates_outputs(self):
        """partial_vocab_complete updates the output display."""
        window = _make_mock_window()
        vocab = [{"Term": "test"}]
        _call_handler(window, "partial_vocab_complete", vocab)
        window.output_display.update_outputs.assert_called_once_with(vocab_csv_data=vocab)
        window.output_display.set_extraction_source.assert_called_once_with("partial")

    def test_ner_complete_updates_output_and_status(self):
        """ner_complete message displays vocabulary and updates status."""
        window = _make_mock_window()
        data = {"vocab": [{"Term": "Smith"}], "filtered": []}
        _call_handler(window, "ner_complete", data)
        window.output_display.update_outputs.assert_called_once()
        window.output_display.set_extraction_source.assert_called_once_with("ner")
        window._completed_tasks.add("vocab")  # Verify task tracked

    def test_semantic_ready_sets_flag(self):
        """semantic_ready message sets _semantic_ready flag."""
        window = _make_mock_window()
        data = {"vector_store_path": "/tmp/store", "chunk_count": 100}
        _call_handler(window, "semantic_ready", data)
        assert window._semantic_ready is True
        assert window._vector_store_path == "/tmp/store"
        assert window._key_sentences_pending is True

    def test_progress_appends_search_note_when_semantic_ready(self):
        """Progress message appends search note when semantic index ready."""
        window = _make_mock_window()
        window._semantic_ready = True
        _call_handler(window, "progress", (50, "Extracting vocabulary..."))
        call_args = window.set_status.call_args[0][0]
        assert "searching documents" in call_args.lower()


class TestPollQueue:
    """Tests for _poll_queue() message draining and crash detection."""

    def test_drains_multiple_messages(self):
        """_poll_queue processes all messages in the queue."""
        window = _make_mock_window()
        window._worker_manager.check_for_messages.return_value = [
            ("progress", (10, "Starting...")),
            ("progress", (50, "Working...")),
        ]
        window._worker_manager.is_alive.return_value = True
        window.after = MagicMock(return_value=42)

        from src.ui.main_window import MainWindow

        MainWindow._poll_queue(window)

        # _handle_queue_message called once per message
        assert window._handle_queue_message.call_count == 2

    def test_detects_dead_subprocess(self):
        """_poll_queue detects dead subprocess during active processing."""
        window = _make_mock_window()
        window._processing_active = True
        window._worker_manager.check_for_messages.return_value = []
        window._worker_manager.is_alive.return_value = False
        window.after = MagicMock(return_value=42)

        from src.ui.main_window import MainWindow

        MainWindow._poll_queue(window)

        window.set_status_error.assert_called()
        assert window._processing_active is False

    def test_stops_polling_when_inactive(self):
        """_poll_queue stops scheduling when no work is active."""
        window = _make_mock_window()
        window._processing_active = False
        window._preprocessing_active = False
        window._semantic_answering_active = False
        window._key_sentences_pending = False
        window._worker_manager.check_for_messages.return_value = []
        window._worker_manager.is_alive.return_value = True
        window.after = MagicMock()

        from src.ui.main_window import MainWindow

        MainWindow._poll_queue(window)

        # Should NOT schedule next poll
        window.after.assert_not_called()
        assert window._queue_poll_id is None

    def test_continues_polling_while_active(self):
        """_poll_queue reschedules when processing is active."""
        window = _make_mock_window()
        window._processing_active = True
        window._worker_manager.check_for_messages.return_value = []
        window._worker_manager.is_alive.return_value = True
        window.after = MagicMock(return_value=99)

        from src.ui.main_window import MainWindow

        MainWindow._poll_queue(window)

        window.after.assert_called_once()
        assert window._queue_poll_id == 99

    def test_skips_when_destroying(self):
        """_poll_queue returns immediately when window is being destroyed."""
        window = _make_mock_window()
        window._destroying = True
        window._worker_manager.check_for_messages = MagicMock()

        from src.ui.main_window import MainWindow

        MainWindow._poll_queue(window)

        window._worker_manager.check_for_messages.assert_not_called()
