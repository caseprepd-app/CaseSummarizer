"""
Tests for MainWindow._handle_queue_message() handlers.

Validates all message types are handled correctly with proper state
transitions, UI updates, and task completion logic. Uses a stub
MainWindow to avoid needing a Tk root.
"""

import threading
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers: create a MainWindow stub with attributes needed by handlers
# ---------------------------------------------------------------------------


def _make_stub():
    """Create a stub with the same attributes MainWindow._handle_queue_message uses."""
    stub = MagicMock()
    # State attributes
    stub._semantic_ready = False
    stub._semantic_answering_active = False
    stub._semantic_results = []
    stub._semantic_results_lock = threading.Lock()
    stub._pending_tasks = {
        "vocab": True,
        "semantic": True,
    }
    stub._completed_tasks = set()
    stub._vector_store_path = None
    stub.processing_results = []
    stub._processing_active = False
    stub._preprocessing_active = False
    stub._semantic_failed = False
    # Widget mocks
    stub.followup_btn = MagicMock()
    stub.followup_entry = MagicMock()
    stub.clear_files_btn = MagicMock()
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


# ---------------------------------------------------------------------------
# progress handler
# ---------------------------------------------------------------------------


class TestProgressHandler:
    """Tests for 'progress' message handler."""

    def test_sets_status_message(self):
        """Progress handler updates status with message text."""
        stub = _make_stub()
        stub._processing_active = True
        _call_handler(stub, "progress", (50, "Processing files..."))
        stub.set_status.assert_called_once_with("Processing files...")

    def test_ignored_when_processing_inactive(self):
        """Progress messages are ignored after processing completes."""
        stub = _make_stub()
        stub._processing_active = False
        _call_handler(stub, "progress", (90, "Stale progress"))
        stub.set_status.assert_not_called()

    def test_appends_qa_note_when_semantic_ready(self):
        """When Q&A index is ready, append answering note to status."""
        stub = _make_stub()
        stub._processing_active = True
        stub._semantic_ready = True
        _call_handler(stub, "progress", (75, "LLM chunk 3/5"))
        call_args = stub.set_status.call_args[0][0]
        assert "(searching documents...)" in call_args

    def test_no_search_note_when_message_mentions_search(self):
        """Don't append search note if message already mentions search."""
        stub = _make_stub()
        stub._processing_active = True
        stub._semantic_ready = True
        _call_handler(stub, "progress", (75, "Running search 2/5..."))
        call_args = stub.set_status.call_args[0][0]
        assert "(searching documents...)" not in call_args

    def test_no_search_note_when_message_contains_search_capitalized(self):
        """Don't append search note if message already mentions Search (capitalized)."""
        stub = _make_stub()
        stub._processing_active = True
        stub._semantic_ready = True
        _call_handler(stub, "progress", (75, "Search indexing..."))
        call_args = stub.set_status.call_args[0][0]
        assert "(searching documents...)" not in call_args


# ---------------------------------------------------------------------------
# file_processed handler
# ---------------------------------------------------------------------------


class TestFileProcessedHandler:
    """Tests for 'file_processed' message handler."""

    def test_appends_to_processing_results(self):
        """file_processed adds data to processing_results list."""
        stub = _make_stub()
        doc_data = {"filename": "test.pdf", "status": "success"}
        _call_handler(stub, "file_processed", doc_data)
        assert doc_data in stub.processing_results

    def test_updates_file_table(self):
        """file_processed calls file_table.add_result."""
        stub = _make_stub()
        doc_data = {"filename": "test.pdf", "status": "success"}
        _call_handler(stub, "file_processed", doc_data)
        stub.file_table.add_result.assert_called_once_with(doc_data)

    def test_multiple_files_accumulate(self):
        """Multiple file_processed messages accumulate in processing_results."""
        stub = _make_stub()
        _call_handler(stub, "file_processed", {"filename": "a.pdf"})
        _call_handler(stub, "file_processed", {"filename": "b.pdf"})
        assert len(stub.processing_results) == 2


# ---------------------------------------------------------------------------
# processing_finished handler
# ---------------------------------------------------------------------------


class TestProcessingFinishedHandler:
    """Tests for 'processing_finished' message handler."""

    def test_calls_preprocessing_complete(self):
        """processing_finished triggers _on_preprocessing_complete."""
        stub = _make_stub()
        results = [{"filename": "test.pdf"}]
        _call_handler(stub, "processing_finished", results)
        stub._on_preprocessing_complete.assert_called_once_with(results)


# ---------------------------------------------------------------------------
# error handler
# ---------------------------------------------------------------------------


class TestErrorHandler:
    """Tests for 'error' message handler."""

    @patch("src.ui.main_window.messagebox")
    def test_shows_error_dialog(self, mock_msgbox):
        """Error handler shows error dialog."""
        stub = _make_stub()
        _call_handler(stub, "error", "Something went wrong")
        mock_msgbox.showerror.assert_called_once()

    @patch("src.ui.main_window.messagebox")
    def test_sets_error_status(self, mock_msgbox):
        """Error handler updates status with error prefix."""
        stub = _make_stub()
        _call_handler(stub, "error", "Something went wrong")
        stub.set_status_error.assert_called_once_with("Error: Something went wrong")

    @patch("src.ui.main_window.messagebox")
    def test_calls_preprocessing_complete_with_empty(self, mock_msgbox):
        """Error during preprocessing calls _on_preprocessing_complete with empty list."""
        stub = _make_stub()
        stub._preprocessing_active = True
        _call_handler(stub, "error", "fail")
        stub._on_preprocessing_complete.assert_called_once_with([])


# ---------------------------------------------------------------------------
# extraction_started handler
# ---------------------------------------------------------------------------


class TestExtractionStartedHandler:
    """Tests for 'extraction_started' message handler."""

    def test_dims_feedback_buttons(self):
        """extraction_started sets extraction_in_progress to True."""
        stub = _make_stub()
        _call_handler(stub, "extraction_started", None)
        stub.output_display.set_extraction_in_progress.assert_called_once_with(True)


# ---------------------------------------------------------------------------
# extraction_complete handler
# ---------------------------------------------------------------------------


class TestExtractionCompleteHandler:
    """Tests for 'extraction_complete' message handler."""

    def test_enables_feedback_buttons(self):
        """extraction_complete sets extraction_in_progress to False."""
        stub = _make_stub()
        _call_handler(stub, "extraction_complete", None)
        stub.output_display.set_extraction_in_progress.assert_called_once_with(False)


# ---------------------------------------------------------------------------
# partial_vocab_complete handler
# ---------------------------------------------------------------------------


class TestPartialVocabCompleteHandler:
    """Tests for 'partial_vocab_complete' message handler."""

    def test_updates_vocab_display(self):
        """partial_vocab_complete updates output display with vocab data."""
        stub = _make_stub()
        terms = [{"term": "plaintiff"}, {"term": "defendant"}]
        _call_handler(stub, "partial_vocab_complete", terms)
        stub.output_display.update_outputs.assert_called_once_with(vocab_csv_data=terms)

    def test_sets_extraction_source_partial(self):
        """partial_vocab_complete sets extraction source to 'partial'."""
        stub = _make_stub()
        _call_handler(stub, "partial_vocab_complete", [{"term": "a"}])
        stub.output_display.set_extraction_source.assert_called_once_with("partial")

    def test_status_shows_term_count(self):
        """partial_vocab_complete shows term count in status."""
        stub = _make_stub()
        terms = [{"term": "a"}, {"term": "b"}, {"term": "c"}]
        _call_handler(stub, "partial_vocab_complete", terms)
        call_args = stub.set_status.call_args[0][0]
        # Non-debug: "Found 3 key terms. Now scanning for names..."
        # Debug: "Found 3 terms (BM25+RAKE). Running NER..."
        assert "3" in call_args
        assert "terms" in call_args.lower()

    def test_handles_empty_data(self):
        """partial_vocab_complete handles empty/None data."""
        stub = _make_stub()
        _call_handler(stub, "partial_vocab_complete", [])
        call_args = stub.set_status.call_args[0][0]
        assert "0" in call_args and "terms" in call_args.lower()


# ---------------------------------------------------------------------------
# ner_progress handler
# ---------------------------------------------------------------------------


class TestNERProgressHandler:
    """Tests for 'ner_progress' message handler."""

    def test_updates_status_with_progress(self):
        """ner_progress updates status with chunk progress percentage."""
        stub = _make_stub()
        _call_handler(stub, "ner_progress", {"chunk_num": 3, "total_chunks": 10})
        call_args = stub.set_status.call_args[0][0]
        assert "30%" in call_args

    def test_handles_single_chunk(self):
        """ner_progress handles single-chunk document."""
        stub = _make_stub()
        _call_handler(stub, "ner_progress", {"chunk_num": 1, "total_chunks": 1})
        call_args = stub.set_status.call_args[0][0]
        assert "100%" in call_args

    def test_handles_missing_keys_gracefully(self):
        """ner_progress uses defaults for missing keys."""
        stub = _make_stub()
        _call_handler(stub, "ner_progress", {})
        # Should not crash - uses defaults 0/1
        stub.set_status.assert_called_once()


# ---------------------------------------------------------------------------
# ner_complete handler
# ---------------------------------------------------------------------------


class TestNERCompleteHandler:
    """Tests for 'ner_complete' message handler."""

    def test_updates_vocab_display(self):
        """ner_complete updates output display with NER results."""
        stub = _make_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": False,
        }
        terms = [{"term": "John Smith"}, {"term": "plaintiff"}]
        _call_handler(stub, "ner_complete", {"vocab": terms, "filtered": []})
        stub.output_display.update_outputs.assert_called_once_with(
            vocab_csv_data=terms, filtered_vocab_data=[]
        )

    def test_sets_extraction_source_ner(self):
        """ner_complete sets extraction source to 'ner'."""
        stub = _make_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": False,
        }
        _call_handler(stub, "ner_complete", {"vocab": [{"term": "a"}], "filtered": []})
        stub.output_display.set_extraction_source.assert_called_once_with("ner")

    def test_marks_vocab_completed(self):
        """ner_complete adds 'vocab' to _completed_tasks."""
        stub = _make_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": False,
        }
        _call_handler(stub, "ner_complete", {"vocab": [{"term": "a"}], "filtered": []})
        assert "vocab" in stub._completed_tasks

    def test_status_mentions_search_index_when_qa_pending(self):
        """ner_complete shows search index message when Q&A is pending."""
        stub = _make_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }
        _call_handler(stub, "ner_complete", {"vocab": [{"term": "a"}], "filtered": []})
        call_args = stub.set_status.call_args[0][0]
        assert "search index" in call_args.lower() or "index" in call_args.lower()

    def test_status_shows_complete_when_vocab_only(self):
        """ner_complete shows completion message when no Q&A pending."""
        stub = _make_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": False,
        }
        _call_handler(stub, "ner_complete", {"vocab": [{"term": "a"}], "filtered": []})
        call_args = stub.set_status.call_args[0][0]
        assert "complete" in call_args.lower()


# ---------------------------------------------------------------------------
# semantic_ready handler
# ---------------------------------------------------------------------------


class TestQAReadyHandler:
    """Tests for 'semantic_ready' message handler."""

    def test_sets_semantic_ready_flag(self):
        """semantic_ready sets _semantic_ready to True."""
        stub = _make_stub()
        _call_handler(stub, "semantic_ready", {"chunk_count": 50, "vector_store_path": "/tmp/vs"})
        assert stub._semantic_ready is True

    def test_stores_vector_store_path(self):
        """semantic_ready stores the vector store path."""
        stub = _make_stub()
        _call_handler(stub, "semantic_ready", {"chunk_count": 50, "vector_store_path": "/tmp/vs"})
        assert stub._vector_store_path == "/tmp/vs"

    def test_enables_followup_controls(self):
        """semantic_ready enables follow-up button and entry with white placeholder."""
        stub = _make_stub()
        _call_handler(stub, "semantic_ready", {"chunk_count": 50})
        stub.followup_btn.configure.assert_called_with(state="normal")
        stub.followup_entry.configure.assert_any_call(
            state="normal",
            placeholder_text="Search your documents...",
            placeholder_text_color="white",
        )

    def test_does_not_mark_qa_complete(self):
        """semantic_ready should NOT add 'qa' to _completed_tasks."""
        stub = _make_stub()
        _call_handler(stub, "semantic_ready", {"chunk_count": 50})
        assert "semantic" not in stub._completed_tasks


# ---------------------------------------------------------------------------
# qa_error handler
# ---------------------------------------------------------------------------


class TestQAErrorHandler:
    """Tests for 'qa_error' message handler."""

    def test_clears_qa_answering_flag(self):
        """qa_error clears _semantic_answering_active."""
        stub = _make_stub()
        stub._semantic_answering_active = True
        _call_handler(stub, "semantic_error", {"error": "Model failed"})
        assert stub._semantic_answering_active is False

    def test_marks_qa_failed_when_pending(self):
        """qa_error marks Q&A as failed when it was a pending task."""
        stub = _make_stub()
        stub._failed_tasks = set()
        _call_handler(stub, "semantic_error", {"error": "fail"})
        assert "semantic" in stub._failed_tasks

    def test_shows_error_in_status(self):
        """qa_error shows error message in status bar."""
        stub = _make_stub()
        _call_handler(stub, "semantic_error", {"error": "Connection failed"})
        stub.set_status_error.assert_called_once()
        call_args = stub.set_status_error.call_args[0][0]
        assert "Connection failed" in call_args

    def test_handles_string_error_data(self):
        """qa_error handles plain string error data."""
        stub = _make_stub()
        _call_handler(stub, "semantic_error", "plain error string")
        stub.set_status_error.assert_called_once()

    def test_triggers_finalization_when_all_complete(self):
        """qa_error triggers finalization if all other tasks are done."""
        stub = _make_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }
        stub._completed_tasks = {"vocab"}
        stub._all_tasks_complete = MagicMock(return_value=True)
        _call_handler(stub, "semantic_error", {"error": "fail"})
        stub._finalize_tasks.assert_called_once()

    def test_sets_semantic_failed_flag(self):
        """qa_error should set _semantic_failed = True."""
        stub = _make_stub()
        _call_handler(stub, "semantic_error", {"error": "Embedding model not available"})
        assert stub._semantic_failed is True

    def test_disables_followup_entry_with_message(self):
        """qa_error should disable followup_entry with explanatory placeholder."""
        stub = _make_stub()
        _call_handler(stub, "semantic_error", {"error": "Embedding model not available"})
        stub.followup_entry.configure.assert_called_once()
        call_kwargs = stub.followup_entry.configure.call_args[1]
        assert call_kwargs["state"] == "disabled"
        assert "unavailable" in call_kwargs["placeholder_text"].lower()

    def test_disables_followup_button(self):
        """qa_error should disable followup_btn."""
        stub = _make_stub()
        _call_handler(stub, "semantic_error", {"error": "fail"})
        stub.followup_btn.configure.assert_called_with(state="disabled")


# ---------------------------------------------------------------------------
# trigger_default_semantic_started handler
# ---------------------------------------------------------------------------


class TestTriggerDefaultQAStartedHandler:
    """Tests for 'trigger_default_semantic_started' message handler."""

    def test_sets_semantic_answering_active(self):
        """trigger_default_semantic_started sets _semantic_answering_active = True."""
        stub = _make_stub()
        _call_handler(stub, "trigger_default_semantic_started", {})
        assert stub._semantic_answering_active is True

    def test_updates_workflow_phase_when_defaults_enabled(self):
        """When default questions enabled, sets workflow phase to SEMANTIC_SEARCHING."""
        stub = _make_stub()
        stub.ask_default_questions_check.get.return_value = True
        _call_handler(stub, "trigger_default_semantic_started", {})
        stub.output_display.set_workflow_phase.assert_called_once()

    def test_shows_ready_message_when_defaults_disabled(self):
        """When default questions disabled, shows ready-to-search message via set_status."""
        stub = _make_stub()
        stub.ask_default_questions_check.get.return_value = False
        _call_handler(stub, "trigger_default_semantic_started", {})
        stub.set_status.assert_called_once()
        msg = stub.set_status.call_args[0][0]
        assert "search" in msg.lower()


# ---------------------------------------------------------------------------
# qa_progress handler
# ---------------------------------------------------------------------------


class TestQAProgressHandler:
    """Tests for 'qa_progress' message handler."""

    def test_updates_status_with_progress(self):
        """qa_progress shows completed count in status."""
        stub = _make_stub()
        _call_handler(stub, "semantic_progress", (2, 5, "Who is the plaintiff?"))
        call_args = stub.set_status.call_args[0][0]
        assert "3/5" in call_args  # answered 3 of 5

    def test_first_question(self):
        """qa_progress shows 1/N after first answer."""
        stub = _make_stub()
        _call_handler(stub, "semantic_progress", (0, 3, "Question 1"))
        call_args = stub.set_status.call_args[0][0]
        assert "1/3" in call_args

    def test_last_question(self):
        """qa_progress shows 'Completed N/N searches' on the last question."""
        stub = _make_stub()
        _call_handler(stub, "semantic_progress", (4, 5, "Last question"))
        call_args = stub.set_status.call_args[0][0]
        assert "5/5" in call_args
        assert "Completed" in call_args


# ---------------------------------------------------------------------------
# qa_result handler
# ---------------------------------------------------------------------------


class TestSemanticResultHandler:
    """Tests for 'qa_result' message handler."""

    def test_appends_result_to_list(self):
        """qa_result adds individual result to _semantic_results."""
        stub = _make_stub()
        result = MagicMock(question="Who?", quick_answer="John")
        _call_handler(stub, "semantic_result", result)
        assert result in stub._semantic_results

    def test_updates_output_display(self):
        """qa_result triggers output display update."""
        stub = _make_stub()
        result = MagicMock()
        _call_handler(stub, "semantic_result", result)
        stub.output_display.update_outputs.assert_called_once()

    def test_multiple_results_accumulate(self):
        """Multiple qa_result messages accumulate in _semantic_results."""
        stub = _make_stub()
        _call_handler(stub, "semantic_result", MagicMock())
        _call_handler(stub, "semantic_result", MagicMock())
        _call_handler(stub, "semantic_result", MagicMock())
        assert len(stub._semantic_results) == 3

    def test_thread_safe_access(self):
        """qa_result uses lock for thread-safe list access."""
        stub = _make_stub()
        # Replace lock with a tracking mock
        real_lock = threading.Lock()
        lock_entered = []

        class TrackingLock:
            def __enter__(self):
                lock_entered.append(True)
                return real_lock.__enter__()

            def __exit__(self, *args):
                return real_lock.__exit__(*args)

        stub._semantic_results_lock = TrackingLock()
        _call_handler(stub, "semantic_result", MagicMock())
        assert len(lock_entered) == 1


# ---------------------------------------------------------------------------
# qa_complete handler
# ---------------------------------------------------------------------------


class TestQACompleteHandler:
    """Tests for 'qa_complete' message handler."""

    def test_replaces_semantic_results(self):
        """qa_complete replaces _semantic_results with final list."""
        stub = _make_stub()
        stub._semantic_results = [MagicMock()]  # Pre-existing
        final_results = [MagicMock(), MagicMock()]
        stub._all_tasks_complete = MagicMock(return_value=False)
        _call_handler(stub, "semantic_complete", final_results)
        assert stub._semantic_results == final_results

    def test_clears_qa_answering_flag(self):
        """qa_complete clears _semantic_answering_active."""
        stub = _make_stub()
        stub._semantic_answering_active = True
        stub._all_tasks_complete = MagicMock(return_value=False)
        _call_handler(stub, "semantic_complete", [MagicMock()])
        assert stub._semantic_answering_active is False

    def test_marks_qa_completed(self):
        """qa_complete adds 'qa' to _completed_tasks."""
        stub = _make_stub()
        stub._all_tasks_complete = MagicMock(return_value=False)
        _call_handler(stub, "semantic_complete", [MagicMock()])
        assert "semantic" in stub._completed_tasks

    def test_enables_followup_button(self):
        """qa_complete enables the follow-up button."""
        stub = _make_stub()
        stub._all_tasks_complete = MagicMock(return_value=False)
        _call_handler(stub, "semantic_complete", [MagicMock()])
        stub.followup_btn.configure.assert_called_with(state="normal")

    def test_updates_display_with_results(self):
        """qa_complete updates output display when results exist."""
        stub = _make_stub()
        results = [MagicMock()]
        stub._all_tasks_complete = MagicMock(return_value=False)
        _call_handler(stub, "semantic_complete", results)
        stub.output_display.update_outputs.assert_called_once_with(semantic_results=results)

    def test_no_display_update_with_empty_results(self):
        """qa_complete skips display update when no results."""
        stub = _make_stub()
        stub._all_tasks_complete = MagicMock(return_value=False)
        _call_handler(stub, "semantic_complete", [])
        stub.output_display.update_outputs.assert_not_called()

    def test_triggers_finalization_when_all_complete(self):
        """qa_complete triggers finalization when all tasks are done."""
        stub = _make_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }
        stub._completed_tasks = {"vocab"}
        stub._all_tasks_complete = MagicMock(return_value=True)
        _call_handler(stub, "semantic_complete", [MagicMock()])
        stub._finalize_tasks.assert_called_once()

    def test_handles_none_data(self):
        """qa_complete handles None data gracefully."""
        stub = _make_stub()
        stub._all_tasks_complete = MagicMock(return_value=False)
        _call_handler(stub, "semantic_complete", None)
        assert stub._semantic_results == []


# ---------------------------------------------------------------------------
# Unknown message type
# ---------------------------------------------------------------------------


class TestUnknownMessageType:
    """Tests for unhandled message types."""

    def test_does_not_crash(self):
        """Unknown message types are logged but don't crash."""
        stub = _make_stub()
        _call_handler(stub, "some_unknown_type", {"data": 123})
        # Should not crash


# ---------------------------------------------------------------------------
# Full message sequence tests
# ---------------------------------------------------------------------------


class TestMessageSequences:
    """Test realistic multi-message sequences."""

    def test_extraction_started_then_partial_then_ner_complete(self):
        """Full extraction sequence: started -> partial -> ner_complete."""
        stub = _make_stub()
        _call_handler(stub, "extraction_started", None)
        stub.output_display.set_extraction_in_progress.assert_called_with(True)

        _call_handler(stub, "partial_vocab_complete", [{"term": "a"}])
        stub.output_display.set_extraction_source.assert_called_with("partial")

        _call_handler(
            stub, "ner_complete", {"vocab": [{"term": "a"}, {"term": "b"}], "filtered": []}
        )
        stub.output_display.set_extraction_source.assert_called_with("ner")

        _call_handler(stub, "extraction_complete", None)
        stub.output_display.set_extraction_in_progress.assert_called_with(False)

    def test_semantic_ready_then_trigger_then_progress_then_complete(self):
        """Full Q&A sequence: semantic_ready -> trigger -> progress -> complete."""
        stub = _make_stub()
        stub._all_tasks_complete = MagicMock(return_value=False)

        # semantic_ready
        _call_handler(stub, "semantic_ready", {"chunk_count": 50, "vector_store_path": "/tmp"})
        assert stub._semantic_ready is True
        assert "semantic" not in stub._completed_tasks

        # trigger_default_semantic_started
        _call_handler(stub, "trigger_default_semantic_started", {})
        assert stub._semantic_answering_active is True

        # qa_progress
        _call_handler(stub, "semantic_progress", (0, 3, "Question 1"))

        # qa_result (individual)
        _call_handler(stub, "semantic_result", MagicMock())
        assert len(stub._semantic_results) == 1

        # qa_complete
        stub._all_tasks_complete.return_value = True
        final = [MagicMock(), MagicMock(), MagicMock()]
        _call_handler(stub, "semantic_complete", final)
        assert stub._semantic_answering_active is False
        assert "semantic" in stub._completed_tasks

    def test_vocab_plus_qa_finalization_order(self):
        """Vocab+Q&A: ner_complete before qa_complete marks vocab done."""
        stub = _make_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }

        # NER complete first - should mark vocab done
        _call_handler(stub, "ner_complete", {"vocab": [{"term": "a"}], "filtered": []})
        assert "vocab" in stub._completed_tasks

    def test_error_during_extraction_resets_state(self):
        """Error during preprocessing triggers preprocessing complete."""
        stub = _make_stub()
        stub._preprocessing_active = True
        with patch("src.ui.main_window.messagebox"):
            _call_handler(stub, "error", "Extraction failed")
        stub._on_preprocessing_complete.assert_called_once_with([])
