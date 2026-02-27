"""Tests for moving LLM vocabulary enhancement from main GUI to Settings.

Covers:
- Default preference changed from "auto" to "no"
- Legacy/corrupted values fall back to "no" (not "auto")
- Task preview reads preference directly (no checkbox)
- NER-complete status message reads preference directly
- Worker launch reads preference directly
- Setting registered under Vocabulary category (not Performance)
- vocab_llm_check no longer exists in window_layout
"""

from unittest.mock import MagicMock, patch

import pytest

# =========================================================================
# Default preference is now "no"
# =========================================================================


class TestDefaultIsNo:
    """Verify the default LLM mode changed from 'auto' to 'no'."""

    @pytest.fixture
    def prefs(self, tmp_path):
        from src.user_preferences import UserPreferencesManager

        return UserPreferencesManager(tmp_path / "prefs.json")

    def test_fresh_prefs_default_no(self, prefs):
        """Fresh preferences should default to 'no' (was 'auto')."""
        assert prefs.get_vocab_llm_mode() == "no"

    def test_fresh_prefs_llm_disabled(self, prefs):
        """Fresh preferences should resolve is_vocab_llm_enabled() to False."""
        assert prefs.is_vocab_llm_enabled() is False

    def test_corrupted_value_falls_back_to_no(self, prefs):
        """Corrupted stored value should fall back to 'no' (was 'auto')."""
        prefs._preferences["experimental"] = {"vocab_use_llm": "garbage"}
        assert prefs.get_vocab_llm_mode() == "no"

    def test_explicit_auto_still_works(self, prefs):
        """User who explicitly sets 'auto' should get 'auto'."""
        prefs.set_vocab_llm_mode("auto")
        assert prefs.get_vocab_llm_mode() == "auto"

    def test_explicit_yes_still_works(self, prefs):
        """User who explicitly sets 'yes' should get 'yes'."""
        prefs.set_vocab_llm_mode("yes")
        assert prefs.get_vocab_llm_mode() == "yes"
        assert prefs.is_vocab_llm_enabled() is True

    def test_legacy_true_maps_to_yes(self, prefs):
        """Legacy boolean True should map to 'yes'."""
        prefs._preferences["experimental"] = {"vocab_use_llm": True}
        assert prefs.get_vocab_llm_mode() == "yes"

    def test_legacy_false_maps_to_no(self, prefs):
        """Legacy boolean False should map to 'no'."""
        prefs._preferences["experimental"] = {"vocab_use_llm": False}
        assert prefs.get_vocab_llm_mode() == "no"


# =========================================================================
# Task preview reads preference directly
# =========================================================================


class TestTaskPreviewUsesPreference:
    """Task preview shows NER+LLM vs NER based on preference, not checkbox."""

    def _make_stub(self):
        """Create MainWindow stub with mocked widgets."""
        stub = MagicMock()
        stub.vocab_check = MagicMock()
        stub.qa_check = MagicMock()
        stub.summary_check = MagicMock()
        stub.ask_default_questions_check = MagicMock()
        stub.task_preview_label = MagicMock()
        return stub

    @patch("src.user_preferences.get_user_preferences")
    def test_preview_shows_ner_when_llm_disabled(self, mock_prefs_fn):
        """Preview should show 'Vocabulary (NER)' when LLM is disabled."""
        from src.ui.main_window import MainWindow

        mock_prefs = MagicMock()
        mock_prefs.is_vocab_llm_enabled.return_value = False
        mock_prefs_fn.return_value = mock_prefs

        stub = self._make_stub()
        stub.vocab_check.get.return_value = True
        stub.qa_check.get.return_value = False
        stub.summary_check.get.return_value = False

        MainWindow._update_task_preview(stub)

        text = stub.task_preview_label.configure.call_args[1]["text"]
        assert "Vocabulary (NER)" in text
        assert "LLM" not in text

    @patch("src.user_preferences.get_user_preferences")
    def test_preview_shows_ner_llm_when_enabled(self, mock_prefs_fn):
        """Preview should show 'Vocabulary (NER+LLM)' when LLM is enabled."""
        from src.ui.main_window import MainWindow

        mock_prefs = MagicMock()
        mock_prefs.is_vocab_llm_enabled.return_value = True
        mock_prefs_fn.return_value = mock_prefs

        stub = self._make_stub()
        stub.vocab_check.get.return_value = True
        stub.qa_check.get.return_value = False
        stub.summary_check.get.return_value = False

        MainWindow._update_task_preview(stub)

        text = stub.task_preview_label.configure.call_args[1]["text"]
        assert "NER+LLM" in text


# =========================================================================
# NER-complete status reads preference
# =========================================================================


class TestNerCompleteStatusUsesPreference:
    """ner_complete handler shows correct status based on preference."""

    def _make_stub(self):
        """Create a MainWindow stub for _handle_queue_message."""
        import threading

        stub = MagicMock()
        stub._qa_ready = False
        stub._qa_answering_active = False
        stub._qa_results = []
        stub._qa_results_lock = threading.Lock()
        stub._pending_tasks = {}
        stub._completed_tasks = set()
        stub._vector_store_path = None
        stub.processing_results = []
        return stub

    @patch("src.user_preferences.get_user_preferences")
    def test_ner_complete_shows_llm_starting_when_enabled(self, mock_prefs_fn):
        """When LLM enabled, ner_complete status says 'LLM enhancement starting'."""
        from src.ui.main_window import MainWindow

        mock_prefs = MagicMock()
        mock_prefs.is_vocab_llm_enabled.return_value = True
        mock_prefs_fn.return_value = mock_prefs

        stub = self._make_stub()
        MainWindow._handle_queue_message(stub, "ner_complete", [{"term": "test"}])

        status_text = stub.set_status.call_args[0][0]
        assert "LLM enhancement starting" in status_text

    @patch("src.user_preferences.get_user_preferences")
    def test_ner_complete_shows_index_when_disabled(self, mock_prefs_fn):
        """When LLM disabled, ner_complete status says 'Building search index'."""
        from src.ui.main_window import MainWindow

        mock_prefs = MagicMock()
        mock_prefs.is_vocab_llm_enabled.return_value = False
        mock_prefs_fn.return_value = mock_prefs

        stub = self._make_stub()
        MainWindow._handle_queue_message(stub, "ner_complete", [{"term": "test"}])

        status_text = stub.set_status.call_args[0][0]
        assert "Building search index" in status_text


# =========================================================================
# No vocab_llm_check in layout
# =========================================================================


class TestNoLlmCheckboxInLayout:
    """Verify the LLM checkbox was removed from window_layout."""

    def test_no_vocab_llm_check_attribute(self):
        """WindowLayoutMixin should not reference vocab_llm_check."""
        import inspect

        from src.ui.window_layout import WindowLayoutMixin

        source = inspect.getsource(WindowLayoutMixin)
        assert "vocab_llm_check" not in source

    def test_no_llm_check_changed_callback(self):
        """WindowLayoutMixin should not reference _on_vocab_llm_check_changed."""
        import inspect

        from src.ui.window_layout import WindowLayoutMixin

        source = inspect.getsource(WindowLayoutMixin)
        assert "_on_vocab_llm_check_changed" not in source


# =========================================================================
# No LLM checkbox methods on MainWindow
# =========================================================================


class TestNoLlmMethodsOnMainWindow:
    """Verify the LLM checkbox methods were removed from MainWindow."""

    def test_no_update_vocab_llm_method(self):
        """MainWindow should not have _update_vocab_llm_checkbox_state."""
        from src.ui.main_window import MainWindow

        assert not hasattr(MainWindow, "_update_vocab_llm_checkbox_state")

    def test_no_set_vocab_llm_tooltip_method(self):
        """MainWindow should not have _set_vocab_llm_tooltip."""
        from src.ui.main_window import MainWindow

        assert not hasattr(MainWindow, "_set_vocab_llm_tooltip")

    def test_no_on_vocab_llm_check_changed_method(self):
        """MainWindow should not have _on_vocab_llm_check_changed."""
        from src.ui.main_window import MainWindow

        assert not hasattr(MainWindow, "_on_vocab_llm_check_changed")


# =========================================================================
# Setting registered under Vocabulary category
# =========================================================================


class TestSettingInVocabularyCategory:
    """Verify vocab_use_llm is in Vocabulary tab, not Performance."""

    def test_setting_in_vocabulary_category(self):
        """vocab_use_llm should be registered under 'Vocabulary' category."""
        from src.ui.settings.settings_registry import SettingsRegistry

        setting = SettingsRegistry.get_setting("vocab_use_llm")
        assert setting is not None, "vocab_use_llm setting not found"
        assert setting.category == "Vocabulary"

    def test_setting_default_is_no(self):
        """vocab_use_llm default should be 'no'."""
        from src.ui.settings.settings_registry import SettingsRegistry

        setting = SettingsRegistry.get_setting("vocab_use_llm")
        assert setting.default == "no"

    def test_ner_only_is_first_option(self):
        """NER-only should be the first dropdown option (since it's the default)."""
        from src.ui.settings.settings_registry import SettingsRegistry

        setting = SettingsRegistry.get_setting("vocab_use_llm")
        first_label, first_value = setting.options[0]
        assert first_value == "no"
        assert "NER" in first_label
