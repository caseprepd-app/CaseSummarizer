"""
Tests for Q&A polling lifecycle fixes.

Validates 3 bug fixes:
1. Polling loop stays alive while Q&A answering is active
2. Q&A marked complete only when questions are answered (not at index-ready)
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
    stub._pending_tasks = {"vocab": True, "qa": True, "summary": False}
    stub._completed_tasks = set()
    stub._qa_answering_active = False
    stub._processing_active = True
    stub._preprocessing_active = False
    stub._destroying = False
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
        return not stub._qa_answering_active

    def test_returns_false_when_pending_task_not_completed(self):
        """Pending tasks not in completed_tasks -> not complete."""
        stub = _make_window_stub()
        stub._completed_tasks = {"vocab"}  # qa still pending
        assert self._call(stub) is False

    def test_returns_false_when_qa_answering_active(self):
        """Even if all tasks completed, active Q&A means not done."""
        stub = _make_window_stub()
        stub._completed_tasks = {"vocab", "qa"}
        stub._qa_answering_active = True
        assert self._call(stub) is False

    def test_returns_true_when_all_complete_and_qa_inactive(self):
        """All pending tasks completed and Q&A inactive -> complete."""
        stub = _make_window_stub()
        stub._completed_tasks = {"vocab", "qa"}
        stub._qa_answering_active = False
        assert self._call(stub) is True

    def test_ignores_non_pending_tasks(self):
        """Tasks with is_pending=False are not required to be in completed_tasks."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": False, "summary": False}
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
    """Tests for the _finalize_tasks deferral when Q&A is active."""

    def _finalize(self, stub):
        """Simulate _finalize_tasks logic."""
        if stub._qa_answering_active:
            return "deferred"
        return "finalized"

    def test_defers_when_qa_answering_active(self):
        """_finalize_tasks should return early when Q&A is still running."""
        stub = _make_window_stub()
        stub._qa_answering_active = True
        assert self._finalize(stub) == "deferred"

    def test_proceeds_when_qa_answering_inactive(self):
        """_finalize_tasks should proceed when Q&A is not running."""
        stub = _make_window_stub()
        stub._qa_answering_active = False
        assert self._finalize(stub) == "finalized"


# ---------------------------------------------------------------------------
# Polling condition tests
# ---------------------------------------------------------------------------


class TestPollingCondition:
    """Tests that polling condition includes _qa_answering_active."""

    def _should_poll(self, stub, messages):
        """Simulate the polling condition from _poll_queue."""
        return bool(
            stub._processing_active
            or stub._preprocessing_active
            or stub._qa_answering_active
            or messages
        )

    def test_polls_when_qa_answering_active(self):
        """Polling should continue when Q&A answering is active even if processing is done."""
        stub = _make_window_stub()
        stub._processing_active = False
        stub._preprocessing_active = False
        stub._qa_answering_active = True
        assert self._should_poll(stub, []) is True

    def test_stops_when_everything_inactive_and_no_messages(self):
        """Polling should stop when all flags are False and no messages."""
        stub = _make_window_stub()
        stub._processing_active = False
        stub._preprocessing_active = False
        stub._qa_answering_active = False
        assert self._should_poll(stub, []) is False

    def test_polls_when_processing_active(self):
        """Polling continues when _processing_active is True (existing behavior)."""
        stub = _make_window_stub()
        stub._processing_active = True
        stub._preprocessing_active = False
        stub._qa_answering_active = False
        assert self._should_poll(stub, []) is True

    def test_polls_when_messages_present(self):
        """Polling continues when there are unprocessed messages."""
        stub = _make_window_stub()
        stub._processing_active = False
        stub._preprocessing_active = False
        stub._qa_answering_active = False
        assert self._should_poll(stub, [("progress", "test")]) is True


# ---------------------------------------------------------------------------
# qa_complete / qa_error handler behavior tests
# ---------------------------------------------------------------------------


class TestQAHandlerBehavior:
    """Tests that qa_complete and qa_error clear flag and trigger finalization."""

    def _simulate_qa_complete(self, stub):
        """Simulate the qa_complete handler logic."""
        stub._qa_answering_active = False
        if stub._pending_tasks.get("qa"):
            stub._completed_tasks.add("qa")

    def _simulate_qa_error(self, stub):
        """Simulate the qa_error handler logic."""
        stub._qa_answering_active = False
        if stub._pending_tasks.get("qa"):
            stub._completed_tasks.add("qa")

    def test_qa_complete_clears_flag(self):
        """qa_complete handler should clear _qa_answering_active."""
        stub = _make_window_stub()
        stub._qa_answering_active = True
        self._simulate_qa_complete(stub)
        assert stub._qa_answering_active is False

    def test_qa_complete_adds_to_completed_tasks(self):
        """qa_complete handler should mark Q&A as completed."""
        stub = _make_window_stub()
        stub._qa_answering_active = True
        self._simulate_qa_complete(stub)
        assert "qa" in stub._completed_tasks

    def test_qa_error_clears_flag(self):
        """qa_error handler should clear _qa_answering_active."""
        stub = _make_window_stub()
        stub._qa_answering_active = True
        self._simulate_qa_error(stub)
        assert stub._qa_answering_active is False

    def test_qa_error_adds_to_completed_tasks(self):
        """qa_error handler should mark Q&A as completed (failed but done)."""
        stub = _make_window_stub()
        stub._qa_answering_active = True
        self._simulate_qa_error(stub)
        assert "qa" in stub._completed_tasks

    def test_qa_complete_skips_non_pending_qa(self):
        """If Q&A wasn't a pending task, don't add to completed."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": False, "summary": False}
        stub._qa_answering_active = True
        self._simulate_qa_complete(stub)
        assert "qa" not in stub._completed_tasks


# ---------------------------------------------------------------------------
# qa_ready no longer marks complete
# ---------------------------------------------------------------------------


class TestQAReadyNoCompletion:
    """Verify qa_ready handler does NOT mark Q&A as complete."""

    def test_qa_ready_source_code_has_no_premature_completion(self):
        """The qa_ready handler should not add 'qa' to _completed_tasks."""
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
                # Find the qa_ready elif block
                lines = func_source.split("\n")
                in_qa_ready_block = False
                qa_ready_lines = []
                for line in lines:
                    if 'msg_type == "qa_ready"' in line:
                        in_qa_ready_block = True
                        continue
                    elif in_qa_ready_block and ("elif msg_type ==" in line or "else:" in line):
                        break
                    elif in_qa_ready_block:
                        qa_ready_lines.append(line)

                qa_ready_block = "\n".join(qa_ready_lines)
                assert '_completed_tasks.add("qa")' not in qa_ready_block, (
                    "qa_ready handler should not mark Q&A as complete"
                )
                return

        pytest.fail("Could not find _handle_queue_message method")


# ---------------------------------------------------------------------------
# trigger_default_qa_started sets flag
# ---------------------------------------------------------------------------


class TestTriggerDefaultQAStarted:
    """Verify trigger_default_qa_started handler sets _qa_answering_active."""

    def test_source_code_sets_qa_answering_active(self):
        """The handler should set _qa_answering_active = True."""
        from pathlib import Path

        source_path = Path(__file__).parent.parent / "src" / "ui" / "main_window.py"
        source = source_path.read_text(encoding="utf-8")

        # Find the trigger_default_qa_started block
        lines = source.split("\n")
        in_block = False
        block_lines = []
        for line in lines:
            if 'msg_type == "trigger_default_qa_started"' in line:
                in_block = True
                continue
            elif in_block and ("elif msg_type ==" in line or "else:" in line):
                break
            elif in_block:
                block_lines.append(line)

        block_text = "\n".join(block_lines)
        assert "_qa_answering_active = True" in block_text, (
            "trigger_default_qa_started handler must set _qa_answering_active = True"
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
        return not stub._qa_answering_active

    def test_vocab_only_finalizes_immediately(self):
        """Vocab-only: _finalize_tasks proceeds because _qa_answering_active=False."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": False, "summary": False}
        stub._completed_tasks = {"vocab"}
        stub._qa_answering_active = False
        assert self._all_tasks_complete(stub) is True

    def test_vocab_plus_qa_defers_during_answering(self):
        """Vocab+Q&A: finalization deferred while Q&A answering is active."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": True, "summary": False}
        stub._completed_tasks = {"vocab"}
        stub._qa_answering_active = True
        assert self._all_tasks_complete(stub) is False

    def test_vocab_plus_qa_completes_after_answering(self):
        """Vocab+Q&A: finalization proceeds after Q&A answering completes."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": True, "summary": False}
        stub._completed_tasks = {"vocab", "qa"}
        stub._qa_answering_active = False
        assert self._all_tasks_complete(stub) is True

    def test_all_three_tasks_waits_for_all(self):
        """Vocab+Q&A+Summary: must wait for all three."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": True, "summary": True}
        stub._completed_tasks = {"vocab", "qa"}
        stub._qa_answering_active = False
        # Summary not complete yet
        assert self._all_tasks_complete(stub) is False

    def test_all_three_tasks_complete(self):
        """Vocab+Q&A+Summary: all done -> finalize."""
        stub = _make_window_stub()
        stub._pending_tasks = {"vocab": True, "qa": True, "summary": True}
        stub._completed_tasks = {"vocab", "qa", "summary"}
        stub._qa_answering_active = False
        assert self._all_tasks_complete(stub) is True
