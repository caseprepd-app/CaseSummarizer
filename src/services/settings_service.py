"""
Settings Service for CasePrepd.

Provides a clean interface for application settings and preferences.
Wraps UserPreferencesManager and related config components.

Usage:
    from src.services import SettingsService

    service = SettingsService()
    value = service.get("summary_word_count")
    service.set("summary_word_count", 300)
"""

from typing import Any

from src.config import DEBUG_MODE
from src.logging_config import debug_log
from src.user_preferences import UserPreferencesManager


class SettingsService:
    """
    Service layer for application settings.

    Wraps the UserPreferencesManager and provides a simplified interface.
    Settings are persisted to disk automatically.
    """

    def __init__(self):
        """Initialize the settings service."""
        self._manager = UserPreferencesManager()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.

        Args:
            key: Setting key name
            default: Default value if key not found

        Returns:
            Setting value or default
        """
        return self._manager.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a setting value.

        Args:
            key: Setting key name
            value: Value to set
        """
        self._manager.set(key, value)

        if DEBUG_MODE:
            debug_log(f"[SettingsService] Set {key} = {value}")

    def get_all(self) -> dict:
        """
        Get all settings as a dictionary.

        Returns:
            Dict of all settings
        """
        return self._manager.get_all_preferences()

    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values."""
        self._manager.reset_to_defaults()

        if DEBUG_MODE:
            debug_log("[SettingsService] Reset to defaults")

    # Convenience properties for common settings

    @property
    def summary_word_count(self) -> int:
        """Get target summary word count."""
        return self.get("summary_words", 200)

    @summary_word_count.setter
    def summary_word_count(self, value: int) -> None:
        """Set target summary word count."""
        self.set("summary_words", value)

    @property
    def ollama_model(self) -> str:
        """Get selected Ollama model."""
        return self.get("ollama_model", "gemma3:1b")

    @ollama_model.setter
    def ollama_model(self, value: str) -> None:
        """Set selected Ollama model."""
        self.set("ollama_model", value)

    @property
    def vocabulary_sort_method(self) -> str:
        """Get vocabulary sort method."""
        return self.get("vocab_sort_method", "quality_score")

    @vocabulary_sort_method.setter
    def vocabulary_sort_method(self, value: str) -> None:
        """Set vocabulary sort method."""
        self.set("vocab_sort_method", value)

    @property
    def qa_answer_mode(self) -> str:
        """Get Q&A answer mode (extraction/ollama)."""
        return self.get("qa_answer_mode", "extraction")

    @qa_answer_mode.setter
    def qa_answer_mode(self, value: str) -> None:
        """Set Q&A answer mode."""
        self.set("qa_answer_mode", value)

    @property
    def selected_corpus(self) -> str | None:
        """Get currently selected corpus name."""
        return self.get("selected_corpus", None)

    @selected_corpus.setter
    def selected_corpus(self, value: str | None) -> None:
        """Set currently selected corpus name."""
        self.set("selected_corpus", value)
