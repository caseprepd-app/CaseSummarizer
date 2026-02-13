"""
Tests for Ollama prerequisite gating on LLM Enhancement, Q&A, and Summary checkboxes.

Verifies that all three LLM-dependent toggles are disabled when Ollama is not
connected or has no model configured, and that they fall through to their
existing checks when Ollama IS ready.

Uses MagicMock widgets (no Tk needed), following the pattern from test_gui_polish.py.
"""

from unittest.mock import MagicMock, patch


def _make_window():
    """Create a MainWindow with mocked attributes (skips __init__)."""
    from src.ui.main_window import MainWindow

    win = MainWindow.__new__(MainWindow)
    win.model_manager = MagicMock()
    win.vocab_check = MagicMock()
    win.vocab_llm_check = MagicMock()
    win.qa_check = MagicMock()
    win.ask_default_questions_check = MagicMock()
    win.summary_check = MagicMock()
    win._set_vocab_llm_tooltip = MagicMock()
    win._set_qa_tooltip = MagicMock()
    win._set_summary_tooltip = MagicMock()
    return win


# =========================================================================
# _is_ollama_ready() helper
# =========================================================================


class TestIsOllamaReady:
    """Test the _is_ollama_ready() helper method."""

    def test_ready_when_model_loaded(self):
        """Returns (True, '') when Ollama is connected and has a model."""
        win = _make_window()
        win.model_manager.is_model_loaded.return_value = True

        ready, reason = win._is_ollama_ready()

        assert ready is True
        assert reason == ""

    def test_not_ready_when_disconnected(self):
        """Returns (False, ...) with connection instructions when Ollama is down."""
        win = _make_window()
        win.model_manager.is_model_loaded.return_value = False
        win.model_manager.is_connected = False

        ready, reason = win._is_ollama_ready()

        assert ready is False
        assert "not running" in reason
        assert "ollama serve" in reason

    def test_not_ready_when_no_model(self):
        """Returns (False, ...) pointing to Settings when connected but no model."""
        win = _make_window()
        win.model_manager.is_model_loaded.return_value = False
        win.model_manager.is_connected = True

        ready, reason = win._is_ollama_ready()

        assert ready is False
        assert "No Ollama model" in reason
        assert "Settings" in reason


# =========================================================================
# LLM Enhancement checkbox gating
# =========================================================================


class TestVocabLlmOllamaGate:
    """Test that LLM Enhancement checkbox is disabled when Ollama is not ready."""

    def test_disabled_when_ollama_not_ready(self):
        """LLM checkbox disabled with Ollama tooltip when not connected."""
        win = _make_window()
        win.vocab_check.get.return_value = True  # Vocab IS checked
        win.model_manager.is_model_loaded.return_value = False
        win.model_manager.is_connected = False

        win._update_vocab_llm_checkbox_state()

        win.vocab_llm_check.deselect.assert_called()
        win.vocab_llm_check.configure.assert_called_with(state="disabled")
        tooltip_text = win._set_vocab_llm_tooltip.call_args[0][0]
        assert "not running" in tooltip_text

    @patch("src.user_preferences.get_user_preferences")
    @patch("src.services.AIService")
    def test_falls_through_when_ollama_ready(self, mock_ai_cls, mock_prefs_fn):
        """LLM checkbox proceeds to GPU check when Ollama is connected."""
        win = _make_window()
        win.vocab_check.get.return_value = True
        win.model_manager.is_model_loaded.return_value = True

        # Set up mocks for the existing checks
        mock_ai = MagicMock()
        mock_ai.has_dedicated_gpu.return_value = True
        mock_ai_cls.return_value = mock_ai
        mock_prefs = MagicMock()
        mock_prefs.get_vocab_llm_mode.return_value = "auto"
        mock_prefs.is_vocab_llm_enabled.return_value = True
        mock_prefs_fn.return_value = mock_prefs

        win._update_vocab_llm_checkbox_state()

        # Should have reached Case 4 (enabled)
        win.vocab_llm_check.configure.assert_called_with(state="normal")


# =========================================================================
# Q&A checkbox gating
# =========================================================================


class TestQaOllamaGate:
    """Test that Q&A checkbox is disabled when Ollama is not ready."""

    def test_disabled_when_ollama_not_ready(self):
        """Q&A checkbox disabled with Ollama tooltip when not connected."""
        win = _make_window()
        win.model_manager.is_model_loaded.return_value = False
        win.model_manager.is_connected = False

        win._update_qa_checkbox_state()

        win.qa_check.deselect.assert_called()
        win.qa_check.configure.assert_called_with(state="disabled")
        tooltip_text = win._set_qa_tooltip.call_args[0][0]
        assert "not running" in tooltip_text

    def test_default_questions_disabled_when_ollama_not_ready(self):
        """Default questions sub-checkbox also disabled when Ollama is down."""
        win = _make_window()
        win.model_manager.is_model_loaded.return_value = False
        win.model_manager.is_connected = False
        # Q&A is unchecked (deselected above), so sub-checkbox should disable
        win.qa_check.get.return_value = False

        win._update_qa_checkbox_state()

        win.ask_default_questions_check.deselect.assert_called()
        win.ask_default_questions_check.configure.assert_called_with(state="disabled")

    def test_enabled_when_ollama_ready(self):
        """Q&A checkbox enabled when Ollama is connected (no model-size gate)."""
        win = _make_window()
        win.model_manager.is_model_loaded.return_value = True

        win._update_qa_checkbox_state()

        win.qa_check.configure.assert_called_with(state="normal")


# =========================================================================
# Summary checkbox gating
# =========================================================================


class TestSummaryOllamaGate:
    """Test that Summary checkbox is disabled when Ollama is not ready."""

    def test_disabled_when_ollama_not_ready(self):
        """Summary checkbox disabled with Ollama tooltip when not connected."""
        win = _make_window()
        win.model_manager.is_model_loaded.return_value = False
        win.model_manager.is_connected = False

        win._update_summary_checkbox_state()

        win.summary_check.deselect.assert_called()
        win.summary_check.configure.assert_called_with(state="disabled")
        tooltip_text = win._set_summary_tooltip.call_args[0][0]
        assert "not running" in tooltip_text

    @patch("src.user_preferences.get_user_preferences")
    def test_falls_through_when_ollama_ready(self, mock_prefs_fn):
        """Summary checkbox proceeds to GPU check when Ollama is connected."""
        win = _make_window()
        win.model_manager.is_model_loaded.return_value = True

        mock_prefs = MagicMock()
        mock_prefs.is_summary_allowed.return_value = (True, "")
        mock_prefs_fn.return_value = mock_prefs

        win._update_summary_checkbox_state()

        # Should have reached the "allowed" branch
        win.summary_check.configure.assert_called_with(state="normal")


# =========================================================================
# get_model_param_count utility
# =========================================================================


class TestGetModelParamCount:
    """Test parameter count parsing from model names."""

    def test_parses_8b(self):
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        assert prefs.get_model_param_count("llama3:8b") == 8.0

    def test_parses_1b(self):
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        assert prefs.get_model_param_count("gemma3:1b") == 1.0

    def test_parses_decimal(self):
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        assert prefs.get_model_param_count("llama3.2:3.5b") == 3.5

    def test_returns_none_for_unparseable(self):
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        assert prefs.get_model_param_count("custom-model") is None

    def test_returns_none_for_empty(self):
        from src.user_preferences import get_user_preferences

        prefs = get_user_preferences()
        assert prefs.get_model_param_count("") is None


# =========================================================================
# Small model warning popup
# =========================================================================


class TestSmallModelWarning:
    """Test _check_small_model_warning() logic (no actual dialog shown)."""

    @patch("src.user_preferences.get_user_preferences")
    def test_no_warning_when_large_model(self, mock_prefs_fn):
        """8B+ model should not trigger warning."""
        win = _make_window()
        win.qa_check.get.return_value = True
        win.summary_check.get.return_value = False
        mock_prefs = MagicMock()
        mock_prefs.get.return_value = "llama3:8b"
        mock_prefs.get_model_param_count.return_value = 8.0
        mock_prefs.has_dismissed_small_model_warning.return_value = False
        mock_prefs_fn.return_value = mock_prefs

        result = win._check_small_model_warning()
        assert result is True

    @patch("src.user_preferences.get_user_preferences")
    def test_no_warning_when_no_llm_tasks(self, mock_prefs_fn):
        """No warning when neither Q&A nor Summary is enabled."""
        win = _make_window()
        win.qa_check.get.return_value = False
        win.summary_check.get.return_value = False
        mock_prefs = MagicMock()
        mock_prefs_fn.return_value = mock_prefs

        result = win._check_small_model_warning()
        assert result is True
        # Should not even check model size
        mock_prefs.get_model_param_count.assert_not_called()

    @patch("src.user_preferences.get_user_preferences")
    def test_no_warning_when_already_dismissed(self, mock_prefs_fn):
        """No warning when user previously dismissed it."""
        win = _make_window()
        win.qa_check.get.return_value = True
        win.summary_check.get.return_value = False
        mock_prefs = MagicMock()
        mock_prefs.has_dismissed_small_model_warning.return_value = True
        mock_prefs_fn.return_value = mock_prefs

        result = win._check_small_model_warning()
        assert result is True

    @patch("src.user_preferences.get_user_preferences")
    def test_no_warning_when_unparseable_model(self, mock_prefs_fn):
        """No warning when model name can't be parsed (benefit of the doubt)."""
        win = _make_window()
        win.qa_check.get.return_value = True
        win.summary_check.get.return_value = False
        mock_prefs = MagicMock()
        mock_prefs.get.return_value = "custom-model"
        mock_prefs.get_model_param_count.return_value = None
        mock_prefs.has_dismissed_small_model_warning.return_value = False
        mock_prefs_fn.return_value = mock_prefs

        result = win._check_small_model_warning()
        assert result is True

    @patch("src.user_preferences.get_user_preferences")
    def test_no_warning_for_summary_with_large_model(self, mock_prefs_fn):
        """Summary with 12B model should not trigger warning."""
        win = _make_window()
        win.qa_check.get.return_value = False
        win.summary_check.get.return_value = True
        mock_prefs = MagicMock()
        mock_prefs.get.return_value = "gemma3:12b"
        mock_prefs.get_model_param_count.return_value = 12.0
        mock_prefs.has_dismissed_small_model_warning.return_value = False
        mock_prefs_fn.return_value = mock_prefs

        result = win._check_small_model_warning()
        assert result is True
