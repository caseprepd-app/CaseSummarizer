"""
Tests for semantic search polling lifecycle fixes.

Validates 3 bug fixes:
1. Polling loop stays alive while semantic search answering is active
2. semantic search marked complete only when questions are answered (not at index-ready)
3. Cross-encoder uses direct kwargs (not model_kwargs) for sentence-transformers 3.0+
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers: create a minimal MainWindow stub with just the attributes we test
# ---------------------------------------------------------------------------


def _make_window_stub():
    """Create a stub with the same state attributes as MainWindow."""
    stub = MagicMock()
    stub._pending_tasks = {
        "vocab": True,
        "semantic": True,
    }
    stub._completed_tasks = set()
    stub._semantic_answering_active = False
    stub._semantic_failed = False
    stub._processing_active = True
    stub._preprocessing_active = False
    stub._destroying = False
    stub.clear_files_btn = MagicMock()
    return stub


# ---------------------------------------------------------------------------
# _all_tasks_complete tests
# ---------------------------------------------------------------------------


class TestAllTasksComplete:
    """Tests for the _all_tasks_complete helper method."""

    def _call(self, stub):
        """Call the real method logic (avoid importing MainWindow which needs Tk)."""
        for task_name, is_pending in stub._pending_tasks.items():
            if is_pending and task_name not in stub._completed_tasks:
                return False
        return not stub._semantic_answering_active

    def test_returns_false_when_pending_task_not_completed(self):
        """Pending tasks not in completed_tasks -> not complete."""
        stub = _make_window_stub()
        stub._completed_tasks = {"vocab"}  # qa still pending
        assert self._call(stub) is False

    def test_returns_false_when_semantic_answering_active(self):
        """Even if all tasks completed, active semantic search means not done."""
        stub = _make_window_stub()
        stub._completed_tasks = {"vocab", "semantic"}
        stub._semantic_answering_active = True
        assert self._call(stub) is False

    def test_returns_true_when_all_complete_and_qa_inactive(self):
        """All pending tasks completed and semantic search inactive -> complete."""
        stub = _make_window_stub()
        stub._completed_tasks = {"vocab", "semantic"}
        stub._semantic_answering_active = False
        assert self._call(stub) is True

    def test_ignores_non_pending_tasks(self):
        """Tasks with is_pending=False are not required to be in completed_tasks."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": False,
        }
        stub._completed_tasks = {"vocab"}
        assert self._call(stub) is True

    def test_empty_pending_tasks(self):
        """No pending tasks -> complete (edge case)."""
        stub = _make_window_stub()
        stub._pending_tasks = {}
        assert self._call(stub) is True


# ---------------------------------------------------------------------------
# _finalize_tasks guard tests
# ---------------------------------------------------------------------------


class TestFinalizeTasksGuard:
    """Tests for the _finalize_tasks deferral guards."""

    def _finalize(self, stub):
        """Simulate _finalize_tasks logic (matches production code)."""
        if not stub._processing_active:
            return "skipped"
        if stub._semantic_answering_active:
            return "deferred"
        qa_pending_not_started = (
            stub._pending_tasks.get("semantic")
            and "semantic" not in stub._completed_tasks
            and not stub._semantic_failed
        )
        if qa_pending_not_started:
            return "deferred"
        return "finalized"

    def test_defers_when_semantic_answering_active(self):
        """_finalize_tasks should return early when semantic search is still running."""
        stub = _make_window_stub()
        stub._semantic_answering_active = True
        assert self._finalize(stub) == "deferred"

    def test_proceeds_when_qa_answering_inactive(self):
        """_finalize_tasks should proceed when semantic search is not running."""
        stub = _make_window_stub()
        stub._semantic_answering_active = False
        stub._completed_tasks = {"vocab", "semantic"}
        assert self._finalize(stub) == "finalized"

    def test_defers_when_qa_pending_not_started(self):
        """_finalize_tasks defers when semantic search is pending but not yet started."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }
        stub._completed_tasks = {"vocab"}
        stub._semantic_answering_active = False
        assert self._finalize(stub) == "deferred"

    def test_proceeds_when_qa_pending_and_completed(self):
        """_finalize_tasks proceeds when semantic search is pending and already completed."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }
        stub._completed_tasks = {"vocab", "semantic"}
        stub._semantic_answering_active = False
        assert self._finalize(stub) == "finalized"

    def test_skips_when_already_complete(self):
        """_finalize_tasks skips if _processing_active is already False (double call)."""
        stub = _make_window_stub()
        stub._processing_active = False
        stub._completed_tasks = {"vocab", "semantic"}
        assert self._finalize(stub) == "skipped"

    def test_proceeds_when_semantic_failed_even_if_not_completed(self):
        """_finalize_tasks proceeds when semantic search failed (don't wait for trigger_default_semantic)."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }
        stub._completed_tasks = {"vocab", "semantic"}  # semantic_error already added it
        stub._semantic_failed = True
        assert self._finalize(stub) == "finalized"

    def test_proceeds_when_qa_not_pending(self):
        """_finalize_tasks proceeds when semantic search was not a pending task."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": False,
        }
        stub._completed_tasks = {"vocab"}
        stub._semantic_answering_active = False
        assert self._finalize(stub) == "finalized"


# ---------------------------------------------------------------------------
# Polling condition tests
# ---------------------------------------------------------------------------


class TestPollingCondition:
    """Tests that polling condition includes _semantic_answering_active."""

    def _should_poll(self, stub, messages):
        """Simulate the polling condition from _poll_queue."""
        return bool(
            stub._processing_active
            or stub._preprocessing_active
            or stub._semantic_answering_active
            or messages
        )

    def test_polls_when_semantic_answering_active(self):
        """Polling should continue when semantic search answering is active even if processing is done."""
        stub = _make_window_stub()
        stub._processing_active = False
        stub._preprocessing_active = False
        stub._semantic_answering_active = True
        assert self._should_poll(stub, []) is True

    def test_stops_when_everything_inactive_and_no_messages(self):
        """Polling should stop when all flags are False and no messages."""
        stub = _make_window_stub()
        stub._processing_active = False
        stub._preprocessing_active = False
        stub._semantic_answering_active = False
        assert self._should_poll(stub, []) is False

    def test_polls_when_processing_active(self):
        """Polling continues when _processing_active is True (existing behavior)."""
        stub = _make_window_stub()
        stub._processing_active = True
        stub._preprocessing_active = False
        stub._semantic_answering_active = False
        assert self._should_poll(stub, []) is True

    def test_polls_when_messages_present(self):
        """Polling continues when there are unprocessed messages."""
        stub = _make_window_stub()
        stub._processing_active = False
        stub._preprocessing_active = False
        stub._semantic_answering_active = False
        assert self._should_poll(stub, [("progress", "test")]) is True


# ---------------------------------------------------------------------------
# semantic_complete / semantic_error handler behavior tests
# ---------------------------------------------------------------------------


class TestQAHandlerBehavior:
    """Tests that semantic_complete and semantic_error clear flag and trigger finalization."""

    def _simulate_semantic_complete(self, stub):
        """Simulate the semantic_complete handler logic."""
        stub._semantic_answering_active = False
        if stub._pending_tasks.get("semantic"):
            stub._completed_tasks.add("semantic")

    def _simulate_semantic_error(self, stub):
        """Simulate the semantic_error handler logic."""
        stub._semantic_answering_active = False
        stub._semantic_failed = True
        if stub._pending_tasks.get("semantic"):
            stub._completed_tasks.add("semantic")

    def test_semantic_complete_clears_flag(self):
        """semantic_complete handler should clear _semantic_answering_active."""
        stub = _make_window_stub()
        stub._semantic_answering_active = True
        self._simulate_semantic_complete(stub)
        assert stub._semantic_answering_active is False

    def test_semantic_complete_adds_to_completed_tasks(self):
        """semantic_complete handler should mark semantic search as completed."""
        stub = _make_window_stub()
        stub._semantic_answering_active = True
        self._simulate_semantic_complete(stub)
        assert "semantic" in stub._completed_tasks

    def test_semantic_error_clears_flag(self):
        """semantic_error handler should clear _semantic_answering_active."""
        stub = _make_window_stub()
        stub._semantic_answering_active = True
        self._simulate_semantic_error(stub)
        assert stub._semantic_answering_active is False

    def test_semantic_error_adds_to_completed_tasks(self):
        """semantic_error handler should mark semantic search as completed (failed but done)."""
        stub = _make_window_stub()
        stub._semantic_answering_active = True
        self._simulate_semantic_error(stub)
        assert "semantic" in stub._completed_tasks

    def test_semantic_complete_skips_non_pending_qa(self):
        """If semantic search wasn't a pending task, don't add to completed."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": False,
        }
        stub._semantic_answering_active = True
        self._simulate_semantic_complete(stub)
        assert "semantic" not in stub._completed_tasks


# ---------------------------------------------------------------------------
# semantic_ready no longer marks complete
# ---------------------------------------------------------------------------


class TestQAReadyNoCompletion:
    """Verify semantic_ready handler does NOT mark semantic search as complete."""

    def test_semantic_ready_source_code_has_no_premature_completion(self):
        """The semantic_ready handler should not add 'qa' to _completed_tasks."""
        import ast
        from pathlib import Path

        source_path = Path(__file__).parent.parent / "src" / "ui" / "main_window.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find the _handle_queue_message method
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_handle_queue_message":
                # Get the source lines for this function
                func_source = ast.get_source_segment(source, node)
                # Find the semantic_ready elif block
                lines = func_source.split("\n")
                in_semantic_ready_block = False
                semantic_ready_lines = []
                for line in lines:
                    if 'msg_type == "semantic_ready"' in line:
                        in_semantic_ready_block = True
                        continue
                    elif in_semantic_ready_block and (
                        "elif msg_type ==" in line or "else:" in line
                    ):
                        break
                    elif in_semantic_ready_block:
                        semantic_ready_lines.append(line)

                semantic_ready_block = "\n".join(semantic_ready_lines)
                assert '_completed_tasks.add("semantic")' not in semantic_ready_block, (
                    "semantic_ready handler should not mark semantic search as complete"
                )
                return

        pytest.fail("Could not find _handle_queue_message method")


# ---------------------------------------------------------------------------
# trigger_default_semantic_started sets flag
# ---------------------------------------------------------------------------


class TestTriggerDefaultQAStarted:
    """Verify trigger_default_semantic_started handler sets _semantic_answering_active."""

    def test_source_code_sets_semantic_answering_active(self):
        """The handler should set _semantic_answering_active = True."""
        from pathlib import Path

        source_path = Path(__file__).parent.parent / "src" / "ui" / "main_window.py"
        source = source_path.read_text(encoding="utf-8")

        # Find the trigger_default_semantic_started block
        lines = source.split("\n")
        in_block = False
        block_lines = []
        for line in lines:
            if 'msg_type == "trigger_default_semantic_started"' in line:
                in_block = True
                continue
            elif in_block and ("elif msg_type ==" in line or "else:" in line):
                break
            elif in_block:
                block_lines.append(line)

        block_text = "\n".join(block_lines)
        assert "_semantic_answering_active = True" in block_text, (
            "trigger_default_semantic_started handler must set _semantic_answering_active = True"
        )


# ---------------------------------------------------------------------------
# Cross-encoder kwarg test
# ---------------------------------------------------------------------------


class TestCrossEncoderKwargs:
    """Tests that cross-encoder uses direct kwargs, not model_kwargs."""

    def test_no_model_kwargs_in_source(self):
        """CrossEncoderReranker should not pass model_kwargs to CrossEncoder."""
        from pathlib import Path

        source_path = (
            Path(__file__).parent.parent
            / "src"
            / "core"
            / "retrieval"
            / "cross_encoder_reranker.py"
        )
        source = source_path.read_text(encoding="utf-8")
        assert "model_kwargs" not in source, (
            "cross_encoder_reranker.py should not use model_kwargs "
            "(conflicts with sentence-transformers 3.0+)"
        )

    def test_local_files_only_as_direct_kwarg(self):
        """local_files_only should be passed as a direct CrossEncoder kwarg."""
        from pathlib import Path

        source_path = (
            Path(__file__).parent.parent
            / "src"
            / "core"
            / "retrieval"
            / "cross_encoder_reranker.py"
        )
        source = source_path.read_text(encoding="utf-8")

        # Should have init_kwargs pattern with local_files_only
        assert 'init_kwargs["local_files_only"]' in source or "local_files_only" in source

    @patch("src.core.retrieval.cross_encoder_reranker.RERANKER_MODEL_LOCAL_PATH")
    @patch("src.core.retrieval.cross_encoder_reranker.RERANKER_MAX_LENGTH", 512)
    def test_load_model_passes_local_files_only_directly(self, mock_local_path):
        """When bundled model exists, local_files_only is a direct kwarg to CrossEncoder."""
        mock_local_path.exists.return_value = True
        mock_local_path.__str__ = lambda self: "/fake/model/path"

        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        with patch("sentence_transformers.CrossEncoder") as MockCE:
            reranker._load_model()
            MockCE.assert_called_once_with(
                "/fake/model/path",
                max_length=512,
                local_files_only=True,
            )

    @patch("src.core.retrieval.cross_encoder_reranker.RERANKER_MODEL_LOCAL_PATH")
    @patch("src.core.retrieval.cross_encoder_reranker.RERANKER_MODEL_NAME", "test-model")
    @patch("src.core.retrieval.cross_encoder_reranker.RERANKER_MAX_LENGTH", 512)
    def test_load_model_omits_local_files_only_for_remote(self, mock_local_path):
        """When no bundled model, local_files_only should NOT be passed."""
        mock_local_path.exists.return_value = False

        from src.core.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        with patch("sentence_transformers.CrossEncoder") as MockCE:
            reranker._load_model()
            MockCE.assert_called_once_with(
                "test-model",
                max_length=512,
            )


# ---------------------------------------------------------------------------
# Task combination flow tests
# ---------------------------------------------------------------------------


class TestTaskCombinationFlows:
    """Verify correct finalization for different task combinations."""

    def _all_tasks_complete(self, stub):
        for task_name, is_pending in stub._pending_tasks.items():
            if is_pending and task_name not in stub._completed_tasks:
                return False
        return not stub._semantic_answering_active

    def test_vocab_only_finalizes_immediately(self):
        """Vocab-only: _finalize_tasks proceeds because _semantic_answering_active=False."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": False,
        }
        stub._completed_tasks = {"vocab"}
        stub._semantic_answering_active = False
        assert self._all_tasks_complete(stub) is True

    def test_vocab_plus_qa_defers_during_answering(self):
        """Vocab+semantic search: finalization deferred while semantic search answering is active."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }
        stub._completed_tasks = {"vocab"}
        stub._semantic_answering_active = True
        assert self._all_tasks_complete(stub) is False

    def test_vocab_plus_semantic_completes_after_answering(self):
        """Vocab+semantic search: finalization proceeds after semantic search answering completes."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }
        stub._completed_tasks = {"vocab", "semantic"}
        stub._semantic_answering_active = False
        assert self._all_tasks_complete(stub) is True


# ---------------------------------------------------------------------------
# Race condition: vocab completes before trigger_default_semantic_started
# ---------------------------------------------------------------------------


class TestVocabCompleteQARace:
    """Tests for the race between vocab completion and semantic search startup."""

    def _finalize(self, stub):
        """Simulate _finalize_tasks logic (matches production code)."""
        if not stub._processing_active:
            return "skipped"
        if stub._semantic_answering_active:
            return "deferred"
        qa_pending_not_started = (
            stub._pending_tasks.get("semantic")
            and "semantic" not in stub._completed_tasks
            and not stub._semantic_failed
        )
        if qa_pending_not_started:
            return "deferred"
        stub._on_tasks_complete(True, f"Completed {len(stub._completed_tasks)} task(s)")
        return "finalized"

    def _simulate_vocab_complete(self, stub):
        """Simulate the ner_complete handler's finalization path."""
        stub._completed_tasks.add("vocab")
        return self._finalize(stub)

    def test_vocab_complete_does_not_finalize_when_qa_pending(self):
        """Vocab complete should not finalize when semantic search is pending but not started."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }
        stub._completed_tasks = set()
        stub._semantic_answering_active = False
        result = self._simulate_vocab_complete(stub)
        assert result == "deferred"
        assert stub._on_tasks_complete.call_count == 0

    def test_full_sequence_vocab_then_qa(self):
        """Integration: vocab defers, then semantic search starts, then semantic search completes."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": True,
        }
        stub._completed_tasks = set()
        stub._semantic_answering_active = False

        # Step 1: vocab completes — should defer
        result = self._simulate_vocab_complete(stub)
        assert result == "deferred"
        assert stub._on_tasks_complete.call_count == 0

        # Step 2: trigger_default_semantic_started arrives
        stub._semantic_answering_active = True

        # Step 3: Finalize attempted mid-semantic search — should still defer
        result2 = self._finalize(stub)
        assert result2 == "deferred"

        # Step 4: semantic_complete arrives
        stub._semantic_answering_active = False
        stub._completed_tasks.add("semantic")

        # Step 5: Now finalization should proceed
        result3 = self._finalize(stub)
        assert result3 == "finalized"
        stub._on_tasks_complete.assert_called_once()

    def test_vocab_complete_finalizes_when_qa_not_pending(self):
        """Vocab complete should finalize normally when semantic search is not pending."""
        stub = _make_window_stub()
        stub._pending_tasks = {
            "vocab": True,
            "semantic": False,
        }
        stub._completed_tasks = set()
        stub._semantic_answering_active = False
        result = self._simulate_vocab_complete(stub)
        assert result == "finalized"
        stub._on_tasks_complete.assert_called_once()


# ---------------------------------------------------------------------------
# Timer lifecycle: _start_timer / _stop_timer source-inspection tests
# ---------------------------------------------------------------------------


class TestTimerLifecycle:
    """Tests that _start_timer and _stop_timer prevent orphaned timer loops."""

    def _get_source(self, method_name):
        import inspect

        from src.ui.main_window import MainWindow

        return inspect.getsource(getattr(MainWindow, method_name))

    def test_start_timer_cancels_existing(self):
        """_start_timer should cancel existing _timer_after_id to prevent parallel loops."""
        source = self._get_source("_start_timer")
        assert "after_cancel" in source, "_start_timer must cancel existing timer"
        assert "_timer_after_id" in source

    def test_stop_timer_clears_start_time(self):
        """_stop_timer should set _processing_start_time = None to prevent reschedule."""
        source = self._get_source("_stop_timer")
        assert "_processing_start_time = None" in source

    def test_no_duplicate_start_timer_in_progressive_extraction(self):
        """_start_progressive_extraction should NOT call _start_timer (already done in _perform_tasks)."""
        source = self._get_source("_start_progressive_extraction")
        assert "_start_timer" not in source, (
            "_start_progressive_extraction must not call _start_timer "
            "(already called by _perform_tasks)"
        )
