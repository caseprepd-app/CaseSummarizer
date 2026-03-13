"""
Tests for the 16-bug audit fixes.

Covers gaps identified after the bug audit:
1. _d() user prefs lookup + float precision guard (Fix 1)
2. Dead subprocess detection in _poll_queue (Fix 3)
3. clear_rating cache consistency on file write failure (Fix 5)
4. safe_replace retry on PermissionError (Fix 6)
5. DefaultQuestionsManager.replace_all() (Fix 7)
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# =========================================================================
# Fix 1: _d() user prefs lookup + float precision guard
# =========================================================================


class TestConfigPrefsLookup:
    """Tests for _d() preferring user preferences over factory defaults."""

    def test_returns_user_pref_over_factory_default(self):
        """_d() returns value from user prefs when key exists."""
        from src.config import _d

        mock_prefs = MagicMock()
        mock_prefs.get.return_value = 2.5  # Different from factory default

        with patch("src.user_preferences.get_user_preferences", return_value=mock_prefs):
            result = _d("bm25_k1")

        assert result == 2.5
        mock_prefs.get.assert_called_once_with("bm25_k1")

    def test_falls_back_to_factory_when_pref_is_none(self):
        """_d() returns factory default when user pref key not found."""
        from src.config import _d, _factory_default

        mock_prefs = MagicMock()
        mock_prefs.get.return_value = None

        with patch("src.user_preferences.get_user_preferences", return_value=mock_prefs):
            result = _d("bm25_k1")

        assert result == _factory_default("bm25_k1")

    def test_falls_back_on_import_error(self):
        """_d() returns factory default when get_user_preferences fails."""
        from src.config import _d, _factory_default

        with patch("src.user_preferences.get_user_preferences", side_effect=ImportError("test")):
            result = _d("bm25_k1")

        assert result == _factory_default("bm25_k1")

    def test_falls_back_on_corrupted_prefs(self):
        """_d() returns factory default when prefs raise an exception."""
        from src.config import _d, _factory_default

        with patch(
            "src.user_preferences.get_user_preferences", side_effect=RuntimeError("corrupt")
        ):
            result = _d("bm25_k1")

        assert result == _factory_default("bm25_k1")

    def test_float_precision_guard_returns_factory_for_near_equal(self):
        """_d() returns factory float when stored value has JSON drift."""
        from src.config import _d, _factory_default

        factory_val = _factory_default("bm25_k1")  # 1.5
        drifted = factory_val + 1e-15  # Tiny JSON roundtrip drift

        mock_prefs = MagicMock()
        mock_prefs.get.return_value = drifted

        with patch("src.user_preferences.get_user_preferences", return_value=mock_prefs):
            result = _d("bm25_k1")

        # Should return exact factory value, not drifted
        assert result == factory_val
        assert result is not drifted or factory_val == drifted

    def test_float_precision_guard_allows_real_changes(self):
        """_d() returns user float when it genuinely differs from factory."""
        from src.config import _d, _factory_default

        factory_val = _factory_default("bm25_k1")  # 1.5
        user_val = 2.0  # Clearly different

        mock_prefs = MagicMock()
        mock_prefs.get.return_value = user_val

        with patch("src.user_preferences.get_user_preferences", return_value=mock_prefs):
            result = _d("bm25_k1")

        assert result == 2.0

    def test_int_prefs_returned_directly(self):
        """_d() returns int prefs without float precision guard."""
        from src.config import _d

        mock_prefs = MagicMock()
        mock_prefs.get.return_value = 42

        with patch("src.user_preferences.get_user_preferences", return_value=mock_prefs):
            result = _d("ml_min_samples")

        assert result == 42

    def test_string_prefs_returned_directly(self):
        """_d() returns string prefs without float precision guard."""
        from src.config import _d

        mock_prefs = MagicMock()
        mock_prefs.get.return_value = "pymupdf_only"

        with patch("src.user_preferences.get_user_preferences", return_value=mock_prefs):
            result = _d("pdf_extraction_mode")

        assert result == "pymupdf_only"

    def test_bool_prefs_returned_directly(self):
        """_d() returns bool prefs directly."""
        from src.config import _d

        mock_prefs = MagicMock()
        mock_prefs.get.return_value = False

        with patch("src.user_preferences.get_user_preferences", return_value=mock_prefs):
            result = _d("pdf_voting_enabled")

        assert result is False


# =========================================================================
# Fix 3: Dead subprocess detection in _poll_queue
# =========================================================================


def _make_poll_stub():
    """Create a MainWindow stub for _poll_queue tests."""
    stub = MagicMock()
    stub._destroying = False
    stub._processing_active = False
    stub._preprocessing_active = False
    stub._semantic_answering_active = False
    stub._queue_poll_id = None
    stub._worker_manager = MagicMock()
    stub._worker_manager.check_for_messages.return_value = []
    stub._worker_manager.is_alive.return_value = True
    stub.output_display = MagicMock()
    stub.after = MagicMock(return_value="after_id")
    return stub


class TestDeadSubprocessDetection:
    """Tests for _poll_queue subprocess health check (Fix 3)."""

    def test_no_recovery_when_subprocess_alive(self):
        """No error recovery when subprocess is alive and processing."""
        stub = _make_poll_stub()
        stub._processing_active = True
        stub._worker_manager.is_alive.return_value = True

        from src.ui.main_window import MainWindow

        MainWindow._poll_queue(stub)

        stub.set_status_error.assert_not_called()
        stub.after.assert_called_once()  # Continues polling

    def test_recovery_triggered_when_subprocess_dead(self):
        """Error recovery when subprocess is dead but processing flag active."""
        stub = _make_poll_stub()
        stub._processing_active = True
        stub._worker_manager.is_alive.return_value = False

        from src.ui.main_window import MainWindow

        MainWindow._poll_queue(stub)

        stub.set_status_error.assert_called_once()
        assert "crashed" in stub.set_status_error.call_args[0][0].lower()
        assert stub._processing_active is False
        assert stub._preprocessing_active is False
        assert stub._semantic_answering_active is False

    def test_recovery_clears_all_active_flags(self):
        """All three active flags are cleared on subprocess death."""
        stub = _make_poll_stub()
        stub._preprocessing_active = True
        stub._semantic_answering_active = True
        stub._processing_active = True
        stub._worker_manager.is_alive.return_value = False

        from src.ui.main_window import MainWindow

        MainWindow._poll_queue(stub)

        assert stub._preprocessing_active is False
        assert stub._semantic_answering_active is False
        assert stub._processing_active is False

    def test_no_recovery_when_no_active_flags(self):
        """No recovery when subprocess dies but nothing was active."""
        stub = _make_poll_stub()
        stub._worker_manager.is_alive.return_value = False

        from src.ui.main_window import MainWindow

        MainWindow._poll_queue(stub)

        stub.set_status_error.assert_not_called()

    def test_calls_on_tasks_complete_with_failure(self):
        """Recovery calls _on_tasks_complete(False, ...) when processing was active."""
        stub = _make_poll_stub()
        stub._processing_active = True
        stub._worker_manager.is_alive.return_value = False

        from src.ui.main_window import MainWindow

        MainWindow._poll_queue(stub)

        stub._on_tasks_complete.assert_called_once()
        args = stub._on_tasks_complete.call_args[0]
        assert args[0] is False  # success=False
        assert "crashed" in args[1].lower()

    def test_preprocessing_dead_no_on_tasks_complete(self):
        """When only preprocessing was active, _on_tasks_complete is NOT called."""
        stub = _make_poll_stub()
        stub._preprocessing_active = True
        stub._processing_active = False
        stub._worker_manager.is_alive.return_value = False

        from src.ui.main_window import MainWindow

        MainWindow._poll_queue(stub)

        stub.set_status_error.assert_called_once()
        stub._on_tasks_complete.assert_not_called()


# =========================================================================
# Fix 5: clear_rating cache consistency on file write failure
# =========================================================================


class TestClearRatingCacheConsistency:
    """Tests for clear_rating keeping cache consistent with disk on failure."""

    @pytest.fixture
    def feedback_setup(self, tmp_path):
        """Create a FeedbackManager with a rated term."""
        from src.core.vocabulary.feedback_manager import FeedbackManager

        fm = FeedbackManager(
            feedback_dir=tmp_path,
            default_feedback_file=tmp_path / "nonexistent_default.csv",
        )
        term_data = {"Term": "test_term", "Occurrences": 3}
        fm.record_feedback(term_data, +1)
        assert fm.get_rating("test_term") == 1
        return fm

    def test_cache_cleared_after_successful_write(self, feedback_setup):
        """Cache should be cleared after successful file write."""
        fm = feedback_setup
        result = fm.clear_rating("test_term")
        assert result is True
        assert fm.get_rating("test_term") == 0  # Unrated

    def test_cache_intact_when_write_fails(self, feedback_setup):
        """Cache should remain intact when file write raises an exception."""
        fm = feedback_setup

        with patch("src.core.vocabulary.feedback_manager.csv.DictWriter") as mock_writer:
            mock_writer.return_value.writeheader.side_effect = OSError("disk full")
            result = fm.clear_rating("test_term")

        assert result is False  # Write failed
        assert fm.get_rating("test_term") == 1  # Cache still has old rating

    def test_cache_cleared_when_file_not_found(self, feedback_setup):
        """Cache should be cleared when feedback file doesn't exist."""
        fm = feedback_setup
        # Remove the feedback file
        for f in fm.user_feedback_file.parent.glob("*.csv"):
            f.unlink()

        result = fm.clear_rating("test_term")
        assert result is True  # FileNotFoundError path
        assert fm.get_rating("test_term") == 0  # Cache cleared


# =========================================================================
# Fix 6: safe_replace retry on PermissionError
# =========================================================================


class TestSafeReplace:
    """Tests for safe_replace retry logic."""

    def test_succeeds_on_first_attempt(self, tmp_path):
        """Normal case: os.replace succeeds immediately."""
        from src.file_utils import safe_replace

        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("hello")

        safe_replace(src, dst)

        assert dst.read_text() == "hello"
        assert not src.exists()

    def test_succeeds_on_second_attempt(self, tmp_path):
        """Retries after first PermissionError, succeeds on second."""
        from src.file_utils import safe_replace

        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("hello")

        call_count = 0
        original_replace = os.replace

        def flaky_replace(s, d):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PermissionError("[WinError 5] Access is denied")
            return original_replace(s, d)

        with patch("src.file_utils.os.replace", side_effect=flaky_replace):
            safe_replace(src, dst)

        assert call_count == 2

    def test_raises_after_max_attempts(self, tmp_path):
        """Raises PermissionError after exhausting all retry attempts."""
        from src.file_utils import safe_replace

        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("hello")

        with (
            patch(
                "src.file_utils.os.replace",
                side_effect=PermissionError("[WinError 5] Access is denied"),
            ),
            pytest.raises(PermissionError),
        ):
            safe_replace(src, dst)

    def test_does_not_retry_on_non_permission_error(self, tmp_path):
        """Non-PermissionError OSErrors are raised immediately."""
        from src.file_utils import safe_replace

        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("hello")

        with (
            patch(
                "src.file_utils.os.replace",
                side_effect=FileNotFoundError("not found"),
            ),
            pytest.raises(FileNotFoundError),
        ):
            safe_replace(src, dst)

    def test_retry_delay_uses_sleep(self, tmp_path):
        """Retries use time.sleep between attempts."""
        from src.file_utils import safe_replace

        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("hello")

        call_count = 0
        original_replace = os.replace

        def flaky_replace(s, d):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise PermissionError("[WinError 5]")
            return original_replace(s, d)

        with (
            patch("src.file_utils.os.replace", side_effect=flaky_replace),
            patch("src.file_utils.time.sleep") as mock_sleep,
        ):
            safe_replace(src, dst)

        assert mock_sleep.call_count == 2  # Slept twice before 3rd attempt


# =========================================================================
# Fix 7: DefaultQuestionsManager.replace_all()
# =========================================================================


class TestReplaceAll:
    """Tests for DefaultQuestionsManager.replace_all()."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a DefaultQuestionsManager with temp file."""
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager

        config_path = tmp_path / "questions.json"
        return DefaultQuestionsManager(config_path=config_path)

    def test_replaces_all_questions(self, manager):
        """replace_all replaces all questions with new list."""
        new_questions = [
            {"text": "Question A", "enabled": True},
            {"text": "Question B", "enabled": False},
            {"text": "Question C", "enabled": True},
        ]
        manager.replace_all(new_questions)

        all_q = manager.get_all_questions()
        assert len(all_q) == 3
        assert all_q[0].text == "Question A"
        assert all_q[0].enabled is True
        assert all_q[1].text == "Question B"
        assert all_q[1].enabled is False
        assert all_q[2].text == "Question C"
        assert all_q[2].enabled is True

    def test_persists_to_disk(self, manager, tmp_path):
        """replace_all saves to disk (new instance reads same data)."""
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager

        manager.replace_all(
            [
                {"text": "Persisted Q", "enabled": True},
            ]
        )

        # Create new instance from same file
        manager2 = DefaultQuestionsManager(config_path=manager.config_path)
        all_q = manager2.get_all_questions()
        assert len(all_q) == 1
        assert all_q[0].text == "Persisted Q"

    def test_single_save_call(self, manager):
        """replace_all calls _save exactly once (not N+M times)."""
        with patch.object(manager, "_save", wraps=manager._save) as mock_save:
            manager.replace_all(
                [
                    {"text": "Q1"},
                    {"text": "Q2"},
                    {"text": "Q3"},
                ]
            )
        mock_save.assert_called_once()

    def test_skips_empty_text(self, manager):
        """replace_all skips questions with empty or whitespace-only text."""
        manager.replace_all(
            [
                {"text": "Valid", "enabled": True},
                {"text": "", "enabled": True},
                {"text": "   ", "enabled": True},
            ]
        )
        all_q = manager.get_all_questions()
        assert len(all_q) == 1
        assert all_q[0].text == "Valid"

    def test_defaults_enabled_to_true(self, manager):
        """replace_all defaults 'enabled' to True when not specified."""
        manager.replace_all([{"text": "No enabled key"}])
        assert manager.get_all_questions()[0].enabled is True

    def test_empty_list_clears_all(self, manager):
        """replace_all with empty list clears all questions."""
        assert manager.get_total_count() > 0  # Has defaults
        manager.replace_all([])
        assert manager.get_total_count() == 0
