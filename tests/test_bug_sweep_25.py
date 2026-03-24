"""
Tests for bug sweep fixes (v1.0.25).

Covers:
1. Exclusion list removal — un-excluding a term on rating clear
2. Unknown command validation — reject before stopping active worker
3. HTML escape order — truncate before escaping to preserve entities
"""

import threading
from dataclasses import dataclass  # used by TestHtmlEscapeOrder.MockResult
from queue import Queue
from unittest.mock import MagicMock, patch

# =========================================================================
# Bug #1: Exclusion list removal
# =========================================================================


class TestExclusionListRemoval:
    """Clearing a Skip (-1) rating should remove term from exclusion file."""

    def _call_remove(self, term, tmp_path):
        """Call _remove_from_user_exclusion_list with patched path."""
        from src.ui.dynamic_output import DynamicOutputWidget

        exclude_path = tmp_path / "user_vocab_exclude.txt"
        mock_frame = MagicMock(spec=DynamicOutputWidget)

        with patch("src.ui.dynamic_output.USER_VOCAB_EXCLUDE_PATH", exclude_path):
            DynamicOutputWidget._remove_from_user_exclusion_list(mock_frame, term)

        return exclude_path

    def test_removes_term_from_file(self, tmp_path):
        """Term is removed from the exclusion file."""
        exclude_file = tmp_path / "user_vocab_exclude.txt"
        exclude_file.write_text("apple\nbanana\ncherry\n", encoding="utf-8")

        self._call_remove("banana", tmp_path)

        lines = exclude_file.read_text(encoding="utf-8").splitlines()
        assert "banana" not in lines
        assert "apple" in lines
        assert "cherry" in lines

    def test_case_insensitive_removal(self, tmp_path):
        """Removal is case-insensitive."""
        exclude_file = tmp_path / "user_vocab_exclude.txt"
        exclude_file.write_text("Apple\nBanana\n", encoding="utf-8")

        self._call_remove("BANANA", tmp_path)

        lines = exclude_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert "Apple" in lines

    def test_noop_when_term_not_present(self, tmp_path):
        """File is unchanged when term is not in the exclusion list."""
        exclude_file = tmp_path / "user_vocab_exclude.txt"
        exclude_file.write_text("apple\nbanana\n", encoding="utf-8")

        self._call_remove("cherry", tmp_path)

        lines = exclude_file.read_text(encoding="utf-8").splitlines()
        assert lines == ["apple", "banana"]

    def test_noop_when_file_missing(self, tmp_path):
        """No error when exclusion file doesn't exist."""
        path = self._call_remove("anything", tmp_path)
        assert not path.exists()

    def test_noop_on_empty_term(self, tmp_path):
        """Empty string term is a no-op."""
        exclude_file = tmp_path / "user_vocab_exclude.txt"
        exclude_file.write_text("apple\n", encoding="utf-8")

        self._call_remove("", tmp_path)

        assert exclude_file.read_text(encoding="utf-8") == "apple\n"

    def test_strips_whitespace_for_match(self, tmp_path):
        """Leading/trailing whitespace is ignored during matching."""
        exclude_file = tmp_path / "user_vocab_exclude.txt"
        exclude_file.write_text("  apple  \nbanana\n", encoding="utf-8")

        self._call_remove("apple", tmp_path)

        lines = exclude_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert "banana" in lines


# =========================================================================
# Bug #2: Unknown command doesn't stop active worker
# =========================================================================


class TestUnknownCommandNoWorkerStop:
    """Unknown commands should NOT stop the active worker."""

    def test_active_worker_survives_unknown_command(self):
        """Active worker thread is not joined/stopped for unknown cmd."""
        from src.worker_process import _dispatch_command

        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True

        internal_q = Queue()
        state = {
            "active_worker": mock_worker,
            "worker_lock": threading.Lock(),
        }

        _dispatch_command("bogus_cmd", {}, internal_q, state)

        # Worker should NOT have been stopped
        mock_worker.join.assert_not_called()
        # State should still reference the original worker
        assert state["active_worker"] is mock_worker
        # Error message should be queued
        msg = internal_q.get_nowait()
        assert msg[0] == "error"

    def test_valid_commands_accepted(self):
        """All four valid command types are accepted (not rejected)."""
        from src.worker_process import _dispatch_command

        valid = ["process_files", "extract", "run_qa", "followup"]
        for cmd in valid:
            internal_q = Queue()
            state = {
                "active_worker": None,
                "worker_lock": threading.Lock(),
                "embeddings": None,
                "vector_store_path": None,
            }
            # Commands will fail downstream (missing args/embeddings),
            # but should NOT produce "Unknown command" errors
            try:
                _dispatch_command(cmd, {}, internal_q, state)
            except (KeyError, TypeError):
                pass  # expected — missing required args
            messages = []
            while not internal_q.empty():
                messages.append(internal_q.get_nowait())
            for msg_type, msg_text in messages:
                assert "Unknown command" not in str(msg_text), f"'{cmd}' was rejected as unknown"


# =========================================================================
# Bug #3: HTML escape order — truncate before escaping
# =========================================================================


class TestHtmlEscapeOrder:
    """Truncation must happen before escaping to avoid splitting entities."""

    @dataclass
    class MockResult:
        """Minimal mock matching export_semantic_html expectations."""

        question: str
        quick_answer: str
        citation: str
        source_summary: str = ""
        verification: object = None

    def test_truncate_then_escape_in_semantic_html(self, tmp_path):
        """Question with HTML chars near truncation point stays valid."""
        from src.core.export.html_builder import export_semantic_html

        # Create a question where '&' sits right at the 80-char boundary
        # If escape happens first, '&' becomes '&amp;' (5 chars), then
        # truncation at 80 could split '&amp;' into '&am' (broken entity)
        question = "A" * 79 + "&B"  # 81 chars, '&' at position 80
        assert len(question) > 80  # triggers truncation

        result = self.MockResult(
            question=question,
            quick_answer="test answer",
            citation="test citation",
        )

        output_file = tmp_path / "test.html"
        export_semantic_html([result], str(output_file))

        content = output_file.read_text(encoding="utf-8")

        # The truncated+escaped version should NOT contain broken entities
        # like '&am' (from splitting '&amp;' mid-entity)
        assert "&am..." not in content
        # Should contain properly escaped '&amp;' (truncation before escape)
        assert "&amp;" in content
